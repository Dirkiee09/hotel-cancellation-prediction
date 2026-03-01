"""Thesis-grade model analysis: baselines, CIs, tuning, SHAP, temporal stability."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    ARTIFACTS_DIR,
    BOOKING_TIME_FEATURES,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_N_ITERATIONS,
    EXPANDING_WINDOW_N_SPLITS,
    FN_RECOVERY_NIGHTS,
    FP_INTERVENTION_COST,
    LATE_WINDOW_MAX_LEAD_DAYS,
    LEAKAGE_COLS,
    LEARNING_CURVE_FRACTIONS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SECONDS,
    RANDOM_STATE,
    REPORTS_DIR,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
    TARGET_COL,
    TEMPORAL_STABILITY_BUCKETS,
    THRESHOLD_STEP,
)
from src.data.load import load_raw_data
from src.eval.statistical import bootstrap_all_metrics, paired_bootstrap_test
from src.features.build import add_arrival_date, build_preprocessor, split_time_aware
from src.models.baselines import train_decision_tree, train_dummy, train_logistic, train_naive_bayes
from src.models.metrics import safe_pr_auc, safe_roc_auc
from src.models.train import is_lightgbm_available, train_gb, train_lgbm, train_xgb
from src.utils.business import (
    assign_risk_tiers,
    compute_cost_threshold_policy,
    compute_fn_cost_vector,
    safe_threshold_metrics,
)
from src.utils.reproducibility import set_global_seed
from src.utils.validate_data import assert_no_leakage_columns, clean_raw, validate_raw

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThesisOutputs:
    """Structured summary of thesis analysis outputs."""

    reports_dir: Path
    sections: dict[str, Path]


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _prepare_data(
    data_path: str | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and split data identically to run_training_pipeline."""
    df = load_raw_data(path=data_path)
    df, _ = clean_raw(df)
    validation = validate_raw(df)
    if not validation.passed:
        raise ValueError(f"Data validation failed: {validation.messages}")

    feature_cols = BOOKING_TIME_FEATURES.copy()
    assert_no_leakage_columns(feature_cols)

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    df = df[feature_cols + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL])
    if max_rows is not None:
        df = df.head(max_rows).copy()

    train_df, val_df, test_df = split_time_aware(df)
    return train_df, val_df, test_df, feature_cols


def _train_model_family(
    model_family: str,
    X_train_t: np.ndarray,
    y_train: pd.Series,
    X_val_t: np.ndarray | None = None,
    y_val: pd.Series | None = None,
):
    if model_family == "xgboost":
        return train_xgb(X_train_t, y_train, X_val=X_val_t, y_val=y_val)
    if model_family == "gradient_boosting":
        return train_gb(X_train_t, y_train)
    if model_family == "lightgbm":
        return train_lgbm(X_train_t, y_train, X_val=X_val_t, y_val=y_val)
    raise ValueError(f"Unsupported model family: {model_family}")


def _choose_champion_family(
    X_train_t: np.ndarray,
    y_train: pd.Series,
    X_val_t: np.ndarray,
    y_val: pd.Series,
) -> tuple[str, dict[str, dict[str, float]], Any]:
    candidates = ["xgboost", "gradient_boosting"]
    if is_lightgbm_available():
        candidates.append("lightgbm")

    y_val_np = y_val.to_numpy().astype(int)
    results: list[tuple[str, dict[str, float], Any]] = []
    for family in candidates:
        model = _train_model_family(family, X_train_t, y_train, X_val_t, y_val)
        probs = model.predict_proba(X_val_t)[:, 1]
        scores = {
            "pr_auc": safe_pr_auc(y_val_np, probs),
            "roc_auc": safe_roc_auc(y_val_np, probs),
        }
        results.append((family, scores, model))

    ranked = sorted(
        results,
        key=lambda item: (-float(item[1]["pr_auc"]), -float(item[1]["roc_auc"]), item[0]),
    )
    champion_family, _, champion_model = ranked[0]
    score_map = {family: scores for family, scores, _ in results}
    return champion_family, score_map, champion_model


# ---------------------------------------------------------------------------
# Section 1: Baseline comparison
# ---------------------------------------------------------------------------


def _run_baseline_comparison(
    X_train_t: np.ndarray,
    y_train: pd.Series,
    X_test_t: np.ndarray,
    y_test: pd.Series,
    champion_probs: np.ndarray,
    threshold: float,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Train baselines and compare to champion via paired bootstrap.

    Trains four baselines ordered by complexity:
    - Dummy (most_frequent) — trivial floor
    - Naive Bayes — independent-feature probabilistic model
    - Logistic Regression — linear interpretable model
    - Decision Tree (max_depth=5) — interpretable non-linear model

    The pruned Decision Tree is also saved to ``artifacts_dir`` if provided
    so that ``plot_thesis_dt()`` in notebook_utils can load and visualise it.
    """
    dummy = train_dummy(X_train_t, y_train, strategy="most_frequent")
    dummy_probs = dummy.predict_proba(X_test_t)[:, 1]

    nb = train_naive_bayes(X_train_t, y_train)
    nb_probs = nb.predict_proba(X_test_t)[:, 1]

    lr = train_logistic(X_train_t, y_train)
    lr_probs = lr.predict_proba(X_test_t)[:, 1]

    dt = train_decision_tree(X_train_t, y_train)
    dt_probs = dt.predict_proba(X_test_t)[:, 1]
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(dt, artifacts_dir / "thesis_baseline_dt.pkl")

    y_test_np = y_test.to_numpy().astype(int)
    results: dict[str, Any] = {}

    for name, probs in [
        ("dummy_most_frequent", dummy_probs),
        ("naive_bayes", nb_probs),
        ("logistic_regression", lr_probs),
        ("decision_tree_depth5", dt_probs),
        ("champion", champion_probs),
    ]:
        results[name] = {
            "roc_auc": safe_roc_auc(y_test_np, probs),
            "pr_auc": safe_pr_auc(y_test_np, probs),
        }

    # Paired bootstrap: champion vs each baseline
    for baseline_name, baseline_probs in [
        ("lr", lr_probs),
        ("dt", dt_probs),
    ]:
        for metric_name, metric_fn in [
            ("roc_auc", roc_auc_score),
            ("pr_auc", average_precision_score),
        ]:
            test_result = paired_bootstrap_test(
                y_test_np, champion_probs, baseline_probs, metric_fn
            )
            results[f"champion_vs_{baseline_name}_{metric_name}"] = test_result

    return results


# ---------------------------------------------------------------------------
# Section 2: Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _run_confidence_intervals(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Bootstrap CIs for all key metrics."""
    cis = bootstrap_all_metrics(
        y_test,
        test_probs,
        threshold,
        n_bootstraps=BOOTSTRAP_N_ITERATIONS,
        alpha=BOOTSTRAP_ALPHA,
    )
    return {
        name: {
            "point_estimate": ci.point_estimate,
            "ci_lower": ci.ci_lower,
            "ci_upper": ci.ci_upper,
            "alpha": ci.alpha,
            "n_bootstraps": ci.n_bootstraps,
        }
        for name, ci in cis.items()
    }


# ---------------------------------------------------------------------------
# Section 3: Optuna hyperparameter tuning
# ---------------------------------------------------------------------------


def _run_tuning(
    selection_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
) -> dict[str, Any]:
    """Run Optuna tuning (gracefully skips if not installed)."""
    try:
        from src.models.tuning import tune_model
    except ImportError:
        logger.warning("optuna not installed, skipping tuning. pip install optuna")
        return {"skipped": True, "reason": "optuna not installed"}

    result = tune_model(
        selection_df,
        feature_cols,
        model_family,
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT_SECONDS,
    )
    return {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "study_summary": result.study_summary,
        "n_trials_completed": len(result.all_trials),
        "all_trials": result.all_trials,
    }


# ---------------------------------------------------------------------------
# Section 4: SHAP feature importance
# ---------------------------------------------------------------------------


def _run_shap_analysis(
    model: Any,
    X_test_t: np.ndarray,
    reports_dir: Path,
) -> dict[str, Any]:
    """SHAP feature importance (gracefully skips if not installed)."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed, skipping. pip install shap")
        return {"skipped": True, "reason": "shap not installed"}

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_t)
    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature_index": range(len(mean_abs_shap)),
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(reports_dir / "shap_feature_importance.csv", index=False)

    # Summary plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X_test_t, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(reports_dir / "shap_summary_plot.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning("SHAP plot generation failed: %s", exc)

    return {
        "n_features": len(mean_abs_shap),
        "top_10_indices": importance_df.head(10)["feature_index"].tolist(),
        "top_10_mean_abs_shap": importance_df.head(10)["mean_abs_shap"].tolist(),
    }


# ---------------------------------------------------------------------------
# Section 5: Expanding-window cross-validation
# ---------------------------------------------------------------------------


def _run_expanding_window_cv(
    selection_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    n_splits: int = EXPANDING_WINDOW_N_SPLITS,
) -> dict[str, Any]:
    """Expanding-window CV with more folds than the 3-fold rolling origin."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = selection_df[feature_cols].reset_index(drop=True)
    y = selection_df[TARGET_COL].reset_index(drop=True).astype(int)

    rows: list[dict[str, Any]] = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), start=1):
        preprocessor = build_preprocessor()
        X_tr_t = preprocessor.fit_transform(X.iloc[tr_idx])
        X_val_t = preprocessor.transform(X.iloc[val_idx])
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx].to_numpy()

        model = _train_model_family(model_family, X_tr_t, y_tr)

        probs = model.predict_proba(X_val_t)[:, 1]
        rows.append(
            {
                "fold": fold,
                "n_train": len(tr_idx),
                "n_val": len(val_idx),
                "roc_auc": safe_roc_auc(y_val, probs),
                "pr_auc": safe_pr_auc(y_val, probs),
            }
        )

    roc_values = [r["roc_auc"] for r in rows if not np.isnan(r["roc_auc"])]
    pr_values = [r["pr_auc"] for r in rows if not np.isnan(r["pr_auc"])]

    return {
        "folds": rows,
        "n_splits": n_splits,
        "roc_auc_mean": float(np.mean(roc_values)) if roc_values else None,
        "roc_auc_std": float(np.std(roc_values)) if roc_values else None,
        "pr_auc_mean": float(np.mean(pr_values)) if pr_values else None,
        "pr_auc_std": float(np.std(pr_values)) if pr_values else None,
    }


# ---------------------------------------------------------------------------
# Section 6: Learning curves
# ---------------------------------------------------------------------------


def _run_learning_curve_analysis(
    selection_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    fractions: list[float] | None = None,
) -> dict[str, Any]:
    """Performance vs training size."""
    if fractions is None:
        fractions = LEARNING_CURVE_FRACTIONS

    total = len(selection_df)
    # Hold out last 10% as validation
    val_start = int(total * 0.9)
    val_df = selection_df.iloc[val_start:]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    for frac in fractions:
        n_train = int(val_start * frac)
        if n_train < 100:
            continue
        train_sub = selection_df.iloc[:n_train]
        X_tr = train_sub[feature_cols]
        y_tr = train_sub[TARGET_COL].astype(int)

        preprocessor = build_preprocessor()
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_val_t = preprocessor.transform(X_val)

        model = _train_model_family(model_family, X_tr_t, y_tr)

        probs = model.predict_proba(X_val_t)[:, 1]
        rows.append(
            {
                "fraction": frac,
                "n_train": n_train,
                "roc_auc": safe_roc_auc(y_val, probs),
                "pr_auc": safe_pr_auc(y_val, probs),
            }
        )

    return {"points": rows}


# ---------------------------------------------------------------------------
# Section 7: Temporal stability
# ---------------------------------------------------------------------------


def _run_temporal_stability(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    test_probs: np.ndarray,
    threshold: float,
    n_buckets: int = TEMPORAL_STABILITY_BUCKETS,
) -> dict[str, Any]:
    """Per-time-period metrics on the test set."""
    test_copy = test_df.copy()
    test_copy["_arrival_date"] = add_arrival_date(test_copy)
    test_copy["_prob"] = test_probs
    test_copy["_y"] = y_test.to_numpy().astype(int)
    test_copy = test_copy.sort_values("_arrival_date")
    buckets = np.array_split(np.arange(len(test_copy)), n_buckets)

    rows: list[dict[str, Any]] = []
    for bucket_id, idx in enumerate(buckets, start=1):
        sub = test_copy.iloc[idx]
        y_b = sub["_y"].to_numpy()
        p_b = sub["_prob"].to_numpy()
        rows.append(
            {
                "bucket": bucket_id,
                "n_rows": len(idx),
                "cancel_rate": float(np.mean(y_b)),
                "roc_auc": safe_roc_auc(y_b, p_b),
                "pr_auc": safe_pr_auc(y_b, p_b),
                "date_min": str(sub["_arrival_date"].min()),
                "date_max": str(sub["_arrival_date"].max()),
            }
        )

    return {"buckets": rows, "n_buckets": n_buckets}


def _run_cost_sensitive_threshold(
    val_df: pd.DataFrame,
    y_val: pd.Series,
    val_probs: np.ndarray,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    test_probs: np.ndarray,
    reports_dir: Path,
) -> dict[str, Any]:
    cost_summary, cost_sweep_df = compute_cost_threshold_policy(
        y_val.to_numpy().astype(int),
        val_probs,
        compute_fn_cost_vector(val_df, fn_recovery_nights=FN_RECOVERY_NIGHTS),
        fp_cost=FP_INTERVENTION_COST,
        step=THRESHOLD_STEP,
    )
    threshold = float(cost_summary["threshold"])
    cost_sweep_df.to_csv(reports_dir / "cost_threshold_sweep.csv", index=False)

    risk_tiers = assign_risk_tiers(
        test_probs,
        medium_threshold=RISK_TIER_MEDIUM_THRESHOLD,
        high_threshold=RISK_TIER_HIGH_THRESHOLD,
    )
    export_df = test_df.copy()
    export_df["cancel_probability"] = test_probs
    export_df["risk_tier"] = risk_tiers
    export_df["predicted_cancel_cost_sensitive"] = (
        export_df["cancel_probability"] >= threshold
    ).astype(int)
    export_df.to_csv(reports_dir / "test_predictions_for_powerbi.csv", index=False)

    y_test_np = y_test.to_numpy().astype(int)
    fn_cost_test = compute_fn_cost_vector(test_df, fn_recovery_nights=FN_RECOVERY_NIGHTS)
    preds = (test_probs >= threshold).astype(int)
    fp_total = float(((preds == 1) & (y_test_np == 0)).sum() * FP_INTERVENTION_COST)
    fn_total = float(fn_cost_test[(preds == 0) & (y_test_np == 1)].sum())

    return {
        **cost_summary,
        "threshold": threshold,
        "test_total_cost": float(fp_total + fn_total),
        "test_fp_cost_total": fp_total,
        "test_fn_cost_total": fn_total,
        "risk_tier_counts": {
            "low": int(np.sum(risk_tiers == "low")),
            "medium": int(np.sum(risk_tiers == "medium")),
            "high": int(np.sum(risk_tiers == "high")),
        },
        "risk_tier_medium_threshold": float(RISK_TIER_MEDIUM_THRESHOLD),
        "risk_tier_high_threshold": float(RISK_TIER_HIGH_THRESHOLD),
    }


def _run_late_window_analysis(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    test_probs: np.ndarray,
    *,
    threshold_max_f1: float,
    threshold_cost_sensitive: float,
) -> dict[str, Any]:
    mask = pd.to_numeric(test_df["lead_time"], errors="coerce").fillna(np.inf) <= float(
        LATE_WINDOW_MAX_LEAD_DAYS
    )
    late = test_df.loc[mask]
    late_y = y_test.loc[mask].to_numpy().astype(int)
    late_probs = np.asarray(test_probs)[mask.to_numpy()]
    overall_y = y_test.to_numpy().astype(int)

    return {
        "lead_time_days_max": int(LATE_WINDOW_MAX_LEAD_DAYS),
        "n_rows_late_window": int(len(late)),
        "n_rows_test_total": int(len(test_df)),
        "late_window_share": float(len(late) / len(test_df)) if len(test_df) else 0.0,
        "cancel_rate_overall_test": float(np.mean(overall_y)) if len(overall_y) else 0.0,
        "cancel_rate_late_window": float(np.mean(late_y)) if len(late_y) else 0.0,
        "metrics_max_f1": safe_threshold_metrics(late_y, late_probs, threshold_max_f1)
        if len(late_y)
        else None,
        "metrics_cost_sensitive": safe_threshold_metrics(
            late_y, late_probs, threshold_cost_sensitive
        )
        if len(late_y)
        else None,
    }


def _run_hypothesis_mapping(
    *,
    champion_family: str,
    family_scores: dict[str, dict[str, float]],
    cost_summary: dict[str, Any],
    shap_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    h2_supported = champion_family in {"xgboost", "lightgbm"}
    h4_supported = float(cost_summary.get("savings_vs_050", 0.0)) > 0

    h1_h3_status = "deferred"
    top_shap = []
    if shap_summary and not shap_summary.get("skipped", False):
        top_shap = shap_summary.get("top_10_indices", [])
        h1_h3_status = "evaluated"

    return {
        "H1": {
            "statement": "Lead time, deposit type, and previous cancellations are significant predictors.",
            "status": h1_h3_status,
            "supporting_artifact": "shap_feature_importance.csv",
            "top_shap_feature_indices": top_shap,
        },
        "H2": {
            "statement": "Gradient-boosted trees outperform baseline alternatives.",
            "status": "supported" if h2_supported else "not_supported",
            "champion_family": champion_family,
            "validation_scores": family_scores,
        },
        "H3": {
            "statement": "Lead time has greatest SHAP importance.",
            "status": h1_h3_status,
            "supporting_artifact": "shap_feature_importance.csv",
            "top_shap_feature_indices": top_shap,
        },
        "H4": {
            "statement": "Cost-sensitive thresholding lowers expected revenue loss.",
            "status": "supported" if h4_supported else "not_supported",
            "savings_vs_050": float(cost_summary.get("savings_vs_050", 0.0)),
            "savings_vs_no_model": float(cost_summary.get("savings_vs_no_model", 0.0)),
        },
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_thesis_analysis(
    *,
    reports_dir: Path = REPORTS_DIR,
    data_path: str | None = None,
    max_rows: int | None = None,
    skip_tuning: bool = False,
    skip_shap: bool = False,
) -> ThesisOutputs:
    """Run all thesis-grade analyses. Call AFTER run_training_pipeline."""
    set_global_seed(RANDOM_STATE)
    thesis_dir = reports_dir / "thesis"
    thesis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("thesis_analysis_start reports_dir=%s", thesis_dir)

    # Prepare data identically to the training pipeline
    train_df, val_df, test_df, feature_cols = _prepare_data(data_path, max_rows)
    selection_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Run champion/challenger comparison for family_scores (used in H2 hypothesis and summary).
    # We discard the fresh model object and instead load the saved pipeline artifact so that
    # champion_probs, val_probs, and threshold are identical to what metrics.json reports.
    champion_family, family_scores, _ = _choose_champion_family(
        X_train_t,
        y_train,
        X_val_t,
        y_val,
    )

    # Load saved pipeline + calibrator to get calibrated probabilities consistent with the
    # training pipeline (thresholds in artifacts/thresholds.json are computed on calibrated probs).
    _pipeline = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
    _calibrator = joblib.load(ARTIFACTS_DIR / "probability_calibrator.pkl")
    _inner_model = _pipeline.named_steps["model"]

    champion_probs = np.clip(_calibrator.predict(_inner_model.predict_proba(X_test_t)[:, 1]), 0.0, 1.0)
    val_probs = np.clip(_calibrator.predict(_inner_model.predict_proba(X_val_t)[:, 1]), 0.0, 1.0)

    # Load threshold directly from artifact so model_family_summary matches thresholds.json.
    _raw_thr: dict[str, Any] = json.loads((ARTIFACTS_DIR / "thresholds.json").read_text())
    _max_f1_payload = _raw_thr.get("max_f1", {})
    threshold = float(_max_f1_payload.get("threshold", 0.35)) if isinstance(_max_f1_payload, dict) else 0.35

    sections: dict[str, Path] = {}

    # 1. Baselines
    logger.info("thesis: running baseline comparison")
    baselines = _run_baseline_comparison(
        X_train_t,
        y_train,
        X_test_t,
        y_test,
        champion_probs,
        threshold,
        artifacts_dir=ARTIFACTS_DIR,
    )
    path = thesis_dir / "baseline_comparison.json"
    _save_json(path, baselines)
    sections["baselines"] = path

    # 2. Confidence intervals
    logger.info("thesis: running bootstrap CIs")
    cis = _run_confidence_intervals(y_test.to_numpy().astype(int), champion_probs, threshold)
    path = thesis_dir / "confidence_intervals.json"
    _save_json(path, cis)
    sections["confidence_intervals"] = path

    # 3. Optuna tuning
    if not skip_tuning:
        logger.info("thesis: running Optuna tuning")
        if champion_family in {"xgboost", "gradient_boosting"}:
            tuning = _run_tuning(selection_df, feature_cols, champion_family)
        else:
            tuning = {
                "skipped": True,
                "reason": "Optuna tuning currently implemented for xgboost/gradient_boosting only",
            }
        path = thesis_dir / "optuna_tuning.json"
        _save_json(path, tuning)
        sections["tuning"] = path

    # 4. SHAP
    shap_results: dict[str, Any] | None = None
    if not skip_shap:
        logger.info("thesis: running SHAP analysis")
        shap_results = _run_shap_analysis(_inner_model, X_test_t, thesis_dir)
        path = thesis_dir / "shap_analysis.json"
        _save_json(path, shap_results)
        sections["shap"] = path

    # 5. Expanding-window CV
    logger.info("thesis: running expanding-window CV")
    cv_results = _run_expanding_window_cv(selection_df, feature_cols, champion_family)
    path = thesis_dir / "expanding_window_cv.json"
    _save_json(path, cv_results)
    sections["expanding_window_cv"] = path

    # 6. Learning curves
    logger.info("thesis: running learning curves")
    lc_results = _run_learning_curve_analysis(selection_df, feature_cols, champion_family)
    path = thesis_dir / "learning_curves.json"
    _save_json(path, lc_results)
    sections["learning_curves"] = path

    # 7. Temporal stability
    logger.info("thesis: running temporal stability")
    stability = _run_temporal_stability(test_df, y_test, champion_probs, threshold)
    path = thesis_dir / "temporal_stability.json"
    _save_json(path, stability)
    sections["temporal_stability"] = path

    # 8. Cost-sensitive threshold and Power BI export
    logger.info("thesis: running cost-sensitive threshold analysis")
    cost_summary = _run_cost_sensitive_threshold(
        val_df,
        y_val,
        val_probs,
        test_df,
        y_test,
        champion_probs,
        thesis_dir,
    )
    path = thesis_dir / "cost_sensitive_threshold.json"
    _save_json(path, cost_summary)
    sections["cost_sensitive_threshold"] = path

    # 9. Late-window analysis
    logger.info("thesis: running late-window analysis")
    late_window = _run_late_window_analysis(
        test_df,
        y_test,
        champion_probs,
        threshold_max_f1=threshold,
        threshold_cost_sensitive=float(cost_summary["threshold"]),
    )
    path = thesis_dir / "late_window_analysis.json"
    _save_json(path, late_window)
    sections["late_window_analysis"] = path

    # 10. Hypothesis mapping summary
    logger.info("thesis: building hypothesis mapping")
    hypotheses = _run_hypothesis_mapping(
        champion_family=champion_family,
        family_scores=family_scores,
        cost_summary=cost_summary,
        shap_summary=shap_results,
    )
    path = thesis_dir / "hypothesis_mapping.json"
    _save_json(path, hypotheses)
    sections["hypothesis_mapping"] = path

    model_summary = {
        "champion_family": champion_family,
        "family_validation_scores": family_scores,
        "max_f1_threshold": threshold,
        "cost_sensitive_threshold": float(cost_summary["threshold"]),
    }
    path = thesis_dir / "model_family_summary.json"
    _save_json(path, model_summary)
    sections["model_family_summary"] = path

    logger.info("thesis_analysis_complete sections=%s", list(sections.keys()))
    return ThesisOutputs(reports_dir=thesis_dir, sections=sections)
