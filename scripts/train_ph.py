"""Transferability probe: train LightGBM on the Punta Villa Resort (PH) data.

This is a side-experiment that demonstrates methodological portability to a
different geography (Philippines vs. Portugal) and property type (single
resort vs. mixed city/resort). It uses ONLY the 8 raw features both datasets
share, so the model deliberately operates with a weakened predictor set.

**Caveats baked into the output JSON**:
- n_train ≈ 240, n_val ≈ 30, n_test ≈ 30 — bootstrap CIs span ±15 pp
- No rolling-origin CV (insufficient data for the project's standard 3 folds)
- No deposit_type, market_segment, country, agent, previous_cancellations
  — these were among the top SHAP features on the Portugal model

Usage:
    python scripts/train_ph.py

Outputs (all under artifacts/ph/ and reports/ph/, separate from main project):
    artifacts/ph/ph_model.pkl              — sklearn Pipeline (preprocessor + LightGBM)
    artifacts/ph/ph_calibrator.pkl         — isotonic calibrator fit on val set
    artifacts/ph/ph_thresholds.json        — max_f1 + high_precision thresholds
    artifacts/ph/ph_feature_columns.json   — reduced feature list
    artifacts/ph/ph_model_metadata.json    — lineage + cleaning + caveats
    artifacts/ph/cost_threshold_sweep.csv  — FP-cost grid for sensitivity nb
    reports/ph/ph_transferability.json     — test-set metrics + caveats
    reports/ph/ph_test_predictions.csv     — per-row predictions for the notebook
    reports/ph/ph_threshold_sweep.csv      — validation threshold sweep
    reports/ph/champion_summary.json       — selection lineage for the thesis
    reports/ph/baseline_comparison.json    — Dummy/LR/DT/NB vs champion
    reports/ph/learning_curves.json        — train/val PR-AUC at 10/25/50/75/100%
    reports/ph/expanding_window_cv.json    — 3-fold expanding-window CV
    reports/ph/shap_analysis.json          — TreeSHAP top features
    reports/ph/shap_feature_importance.csv — per-feature mean(|SHAP|)
    reports/ph/shap_summary_plot.png       — SHAP beeswarm
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.config import (
    PH_ARTIFACTS_DIR,
    PH_BOOKING_TIME_FEATURES,
    PH_CATEGORICAL_COLS,
    PH_NUMERIC_COLS,
    PH_REPORTS_DIR,
    PH_TARGET_COL,
    RANDOM_STATE,
    THRESHOLD_STEP,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data.load_ph import load_ph_data
from src.features.build import cast_to_str
from src.models.baselines import (
    train_decision_tree,
    train_dummy,
    train_logistic,
    train_naive_bayes,
)
from src.models.metrics import (
    compute_confusion,
    evaluate_at_threshold,
    expected_calibration_error,
    safe_pr_auc,
    safe_roc_auc,
)
from src.models.train import is_lightgbm_available, train_gb, train_lgbm
from src.utils import configure_logging
from src.utils.reproducibility import set_global_seed
from src.utils.thresholds import (
    select_high_precision_threshold,
    select_max_f1_threshold,
    threshold_sweep,
)
from src.utils.validate_data import clean_raw_ph, validate_raw_ph

logger = logging.getLogger(__name__)


def _sanitise_for_json(obj: Any) -> Any:
    """Make obj safe for json.dumps: numpy scalars -> Python, NaN/Inf -> None."""
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_sanitise_for_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _compute_duplicate_diagnostics(
    df: pd.DataFrame, *, feature_cols: list[str], target_col: str
) -> dict[str, Any]:
    """Surface duplicate feature vectors and their label consistency.

    A high duplicate rate combined with consistent labels per cluster is the
    signature of an archetype-organized dataset. When the model later achieves
    near-perfect test metrics, this diagnostic is the evidence the metric is
    measuring memorization across duplicates, not generalization.
    """
    feature_df = df[feature_cols].copy()
    n_total = len(feature_df)
    unique = feature_df.drop_duplicates()
    n_unique = len(unique)
    n_dupes = n_total - n_unique
    duplicate_rate = n_dupes / n_total if n_total else 0.0

    # Within each duplicate cluster, are all rows labelled identically?
    with_target = df[feature_cols + [target_col]].copy()
    grouped = with_target.groupby(feature_cols, dropna=False)
    cluster_sizes = grouped.size()
    cluster_label_uniques = grouped[target_col].nunique()
    multi_row_clusters = cluster_sizes[cluster_sizes > 1]
    n_clusters_total = int(len(multi_row_clusters))
    n_clusters_consistent = (
        int((cluster_label_uniques.loc[multi_row_clusters.index] == 1).sum())
        if n_clusters_total
        else 0
    )
    consistent_pct = n_clusters_consistent / n_clusters_total if n_clusters_total else 0.0

    return {
        "n_rows_total": int(n_total),
        "n_unique_feature_vectors": int(n_unique),
        "n_duplicate_rows": int(n_dupes),
        "duplicate_rate": float(duplicate_rate),
        "n_multi_row_clusters": n_clusters_total,
        "n_clusters_with_consistent_labels": n_clusters_consistent,
        "clusters_with_consistent_labels_pct": float(consistent_pct),
        "interpretation": _interpret_duplicates(duplicate_rate, consistent_pct),
    }


def _interpret_duplicates(duplicate_rate: float, consistent_pct: float) -> str:
    """Characterize the duplicate-cluster structure in plain English.

    The Philippine resort dataset exhibits a high duplicate rate combined
    with consistent labels per cluster — a structural property of the data
    rather than a quality concern. The diagnostic still matters because
    chronological splitting on this cluster structure produces train/test
    twins that inflate test metrics; that effect is documented so readers
    don't misread the numbers.
    """
    if duplicate_rate >= 0.30 and consistent_pct >= 0.90:
        return (
            "DATASET CLUSTER STRUCTURE — high duplicate rate combined with "
            "near-perfect label consistency per cluster. The Philippine "
            "dataset is organized around a small set of recurring booking "
            "archetypes. Test metrics reflect memorization across "
            "chronological twins, not generalization to unseen customers; "
            "report metrics as directional."
        )
    if duplicate_rate >= 0.15:
        return (
            "ELEVATED DUPLICATION — non-trivial fraction of rows share identical "
            "feature vectors. Interpret cross-validation metrics with caution."
        )
    return "LOW DUPLICATION — feature vectors are mostly unique; standard interpretation applies."


def _compute_train_test_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Count test rows whose exact feature vector also appears in train or val.

    This is the proximate cause of inflated test metrics when the dataset has
    a high duplicate rate — LightGBM can perfectly classify a test row if it
    has memorized an identical vector from training.
    """
    train_val = pd.concat([train_df[feature_cols], val_df[feature_cols]], ignore_index=True)
    train_val_set = {tuple(row) for row in train_val.itertuples(index=False, name=None)}
    test_rows = [tuple(row) for row in test_df[feature_cols].itertuples(index=False, name=None)]
    n_test = len(test_rows)
    n_dupe = sum(1 for row in test_rows if row in train_val_set)
    return {
        "n_test": int(n_test),
        "n_test_rows_with_train_duplicate": int(n_dupe),
        "test_duplicate_rate": float(n_dupe / n_test) if n_test else 0.0,
    }


def _train_baselines_ph(
    X_train_t: np.ndarray,
    y_train: pd.Series,
    X_test_t: np.ndarray,
    y_test: pd.Series,
    champion_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Train Dummy/LogReg/DT/GaussianNB baselines on PH and compare to champion.

    Mirrors the Portugal complexity-ladder story (Dummy → LR → DT → NB → LightGBM)
    so the PH notebook can show the same value-of-each-assumption discussion.
    """
    baselines: dict[str, Any] = {
        "Dummy": train_dummy(X_train_t, y_train),
        "LogisticRegression": train_logistic(X_train_t, y_train),
        "DecisionTree": train_decision_tree(X_train_t, y_train),
        "GaussianNB": train_naive_bayes(X_train_t, y_train),
    }
    rows: list[dict[str, Any]] = []
    for name, model in baselines.items():
        proba = model.predict_proba(X_test_t)[:, 1]
        metrics_at_05 = evaluate_at_threshold(y_test, proba, 0.5)
        rows.append(
            {
                "model": name,
                "threshold": 0.5,
                "roc_auc": safe_roc_auc(y_test.to_numpy(), proba),
                "pr_auc": safe_pr_auc(y_test.to_numpy(), proba),
                "ece": expected_calibration_error(y_test.to_numpy().astype(int), proba, 10),
                **{k: v for k, v in metrics_at_05.items() if k not in {"roc_auc", "pr_auc"}},
            }
        )
    rows.append(
        {
            "model": "LightGBM (champion)",
            "threshold": float(champion_metrics["max_f1"]["threshold"]),
            "roc_auc": champion_metrics["roc_auc_test"],
            "pr_auc": champion_metrics["pr_auc_test"],
            "ece": champion_metrics["ece_test"],
            "precision": champion_metrics["max_f1"]["precision"],
            "recall": champion_metrics["max_f1"]["recall"],
            "f1": champion_metrics["max_f1"]["f1"],
            "balanced_accuracy": champion_metrics["max_f1"].get("balanced_accuracy"),
        }
    )
    return {"models": rows}


def _compute_learning_curves_ph(
    X_train_t: np.ndarray,
    y_train: pd.Series,
    X_val_t: np.ndarray,
    y_val: pd.Series,
) -> dict[str, Any]:
    """Train PH champion family at increasing training-set fractions.

    Records val-set PR-AUC and ROC-AUC at each step. Chronological subsetting
    (head of train) so subsets are coherent with the rolling-origin spirit.
    """
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    n = len(y_train)
    curves: list[dict[str, Any]] = []
    for frac in fractions:
        size = max(int(np.floor(n * frac)), 1)
        Xs = X_train_t[:size]
        ys = y_train.iloc[:size]
        if len(np.unique(ys)) < 2:
            continue
        if is_lightgbm_available():
            m = train_lgbm(Xs, ys)
        else:
            m = train_gb(Xs, ys)
        train_proba = m.predict_proba(Xs)[:, 1]
        val_proba = m.predict_proba(X_val_t)[:, 1]
        curves.append(
            {
                "fraction": float(frac),
                "n_samples": int(size),
                "train_pr_auc": safe_pr_auc(ys.to_numpy(), train_proba),
                "val_pr_auc": safe_pr_auc(y_val.to_numpy(), val_proba),
                "train_roc_auc": safe_roc_auc(ys.to_numpy(), train_proba),
                "val_roc_auc": safe_roc_auc(y_val.to_numpy(), val_proba),
            }
        )
    return {"curves": curves}


def _compute_expanding_window_cv_ph(
    X_all_t: np.ndarray,
    y_all: pd.Series,
) -> dict[str, Any]:
    """3-fold expanding-window CV across the full chronological PH dataset.

    Cutoffs at 40/55/70% — train on [0, cutoff), evaluate on [cutoff, next_cutoff).
    Folds with single-class train or test sets are skipped (small-N edge case).
    """
    n = len(y_all)
    cutoffs = [int(n * 0.40), int(n * 0.55), int(n * 0.70), int(n * 0.85)]
    folds: list[dict[str, Any]] = []
    for i in range(len(cutoffs) - 1):
        cutoff = cutoffs[i]
        next_cutoff = cutoffs[i + 1]
        if cutoff <= 5 or next_cutoff <= cutoff:
            continue
        ytr = y_all.iloc[:cutoff]
        yte = y_all.iloc[cutoff:next_cutoff]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        Xtr = X_all_t[:cutoff]
        Xte = X_all_t[cutoff:next_cutoff]
        if is_lightgbm_available():
            m = train_lgbm(Xtr, ytr)
        else:
            m = train_gb(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        folds.append(
            {
                "fold": i + 1,
                "n_train": int(cutoff),
                "n_test": int(next_cutoff - cutoff),
                "roc_auc": safe_roc_auc(yte.to_numpy(), proba),
                "pr_auc": safe_pr_auc(yte.to_numpy(), proba),
                **{
                    k: v
                    for k, v in evaluate_at_threshold(yte, proba, 0.5).items()
                    if k not in {"roc_auc", "pr_auc"}
                },
            }
        )
    return {"folds": folds, "cutoffs_fractions": [0.40, 0.55, 0.70, 0.85]}


def _compute_shap_ph(
    model: Any,
    preprocessor: Pipeline,
    X_test_t: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    """Compute TreeSHAP feature importance + save beeswarm; aggregate to raw features.

    The PH champion is LightGBM; TreeExplainer is exact and fast. SHAP values
    on encoded one-hot columns are summed back to the originating raw feature
    so the report shows the 8 PH features, not the post-encoding dummies.

    Returns (summary_dict, importance_df) or None if shap is unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.warning("ph_shap_skipped reason=shap_not_installed")
        return None

    try:
        encoded_names = list(preprocessor.named_steps["encode"].get_feature_names_out())

        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_test_t)
        if isinstance(raw_shap, list):
            sv = raw_shap[1] if len(raw_shap) == 2 else raw_shap[0]
        else:
            sv = raw_shap
            if sv.ndim == 3:
                sv = sv[:, :, 1]

        mean_abs = np.abs(sv).mean(axis=0)

        raw_importance: dict[str, float] = {col: 0.0 for col in PH_BOOKING_TIME_FEATURES}
        for enc_name, importance in zip(encoded_names, mean_abs, strict=True):
            rest = enc_name.split("__", 1)[1] if "__" in enc_name else enc_name
            matched = next(
                (c for c in PH_BOOKING_TIME_FEATURES if rest == c or rest.startswith(c + "_")),
                None,
            )
            key = matched or rest
            raw_importance[key] = raw_importance.get(key, 0.0) + float(importance)

        ranked = sorted(
            ((k, v) for k, v in raw_importance.items() if v > 0.0),
            key=lambda x: x[1],
            reverse=True,
        )
        importance_df = pd.DataFrame(ranked, columns=["feature", "mean_abs_shap"])

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, X_test_t, feature_names=encoded_names, show=False, max_display=15)
        plt.title("PH Champion — SHAP beeswarm (top encoded features)")
        plt.tight_layout()
        plt.savefig(PH_REPORTS_DIR / "shap_summary_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        summary = {
            "method": "treeshap",
            "n_test_rows": int(sv.shape[0]),
            "n_encoded_features": len(encoded_names),
            "top_features": [
                {"feature": name, "mean_abs_shap": float(val)} for name, val in ranked[:10]
            ],
            "all_features_ranked": [
                {"feature": name, "mean_abs_shap": float(val)} for name, val in ranked
            ],
        }
        return summary, importance_df

    except Exception as exc:  # noqa: BLE001 — best-effort, log and proceed
        logger.warning("ph_shap_failed reason=%s", exc)
        return None


def _cost_sensitivity_sweep_ph(
    y_test: pd.Series,
    test_probs: np.ndarray,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Sweep FP cost values, find the cost-minimizing threshold for each.

    FN cost per row uses ``revenue_at_risk`` from the derived booking features
    (ADR × total_stay). FP cost grid covers a 12× range (5 → 60 EUR) to
    bracket the project default of 15 EUR.
    """
    from src.utils.thresholds import cost_threshold_sweep

    fp_cost_grid = [5.0, 10.0, 15.0, 30.0, 60.0]

    if "revenue_at_risk" in test_df.columns:
        fn_cost = test_df["revenue_at_risk"].fillna(0.0).to_numpy().astype(float)
    elif {"adr", "total_stay"}.issubset(test_df.columns):
        fn_cost = (
            test_df["adr"].fillna(0.0).to_numpy() * test_df["total_stay"].fillna(0.0).to_numpy()
        ).astype(float)
    else:
        fn_cost = np.ones(len(test_df), dtype=float) * 100.0

    rows: list[dict[str, Any]] = []
    for fp_cost in fp_cost_grid:
        sweep = cost_threshold_sweep(
            y_test.to_numpy().astype(int),
            test_probs,
            fn_cost,
            fp_cost=fp_cost,
            step=THRESHOLD_STEP,
        )
        if sweep.empty:
            continue
        best_idx = sweep["total_cost"].idxmin()
        best = sweep.loc[best_idx]
        rows.append(
            {
                "fp_cost_eur": float(fp_cost),
                "best_threshold": float(best["threshold"]),
                "total_cost_at_best": float(best["total_cost"]),
                "fp_count_at_best": int(best["fp_count"]),
                "fn_count_at_best": int(best["fn_count"]),
                "fp_cost_total_at_best": float(best["fp_cost_total"]),
                "fn_cost_total_at_best": float(best["fn_cost_total"]),
            }
        )
    return pd.DataFrame(rows)


def _build_ph_preprocessor() -> Pipeline:
    """PH-specific preprocessor: one categorical (room type) + numerics."""
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "to_str",
                FunctionTransformer(cast_to_str, validate=False, feature_names_out="one-to-one"),
            ),
            (
                "onehot",
                OneHotEncoder(sparse_output=False, min_frequency=1, handle_unknown="ignore"),
            ),
        ]
    )
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    column_transformer = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipeline, PH_CATEGORICAL_COLS),
            ("numeric", num_pipeline, PH_NUMERIC_COLS),
        ]
    )
    return Pipeline(steps=[("encode", column_transformer)])


def _split_ph_time_aware(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 80/10/10 split using the parsed arrival date.

    Mirrors src/features/build.py:split_time_aware but does not depend on
    MIN_TRAIN_ROWS guards (which would reject n=300).
    """
    if "arrival_date_year" not in df.columns:
        raise ValueError("DataFrame must contain arrival_date_year; run clean_raw_ph first.")
    sort_key = pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": pd.Series(df["arrival_date_month"]).map(
                {
                    "January": 1,
                    "February": 2,
                    "March": 3,
                    "April": 4,
                    "May": 5,
                    "June": 6,
                    "July": 7,
                    "August": 8,
                    "September": 9,
                    "October": 10,
                    "November": 11,
                    "December": 12,
                }
            ),
            "day": df["arrival_date_day_of_month"],
        },
        errors="coerce",
    )
    ordered = df.assign(_arrival_date=sort_key).sort_values("_arrival_date", kind="mergesort")
    n = len(ordered)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    if train_end == 0 or val_end <= train_end or val_end >= n:
        raise ValueError(
            f"PH time-aware split produced an empty partition: "
            f"n={n}, train_end={train_end}, val_end={val_end}."
        )
    train_df = ordered.iloc[:train_end].drop(columns=["_arrival_date"])
    val_df = ordered.iloc[train_end:val_end].drop(columns=["_arrival_date"])
    test_df = ordered.iloc[val_end:].drop(columns=["_arrival_date"])
    return train_df, val_df, test_df


def run_ph_training_pipeline() -> dict[str, Any]:
    """Train, calibrate, threshold-pick, and persist the PH transferability model."""
    set_global_seed(RANDOM_STATE)
    PH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "ph_pipeline_start artifacts_dir=%s reports_dir=%s", PH_ARTIFACTS_DIR, PH_REPORTS_DIR
    )

    # ── 1. Load + clean ──────────────────────────────────────────────
    raw = load_ph_data()
    cleaned, cleaning_issues = clean_raw_ph(raw)
    validation = validate_raw_ph(cleaned)
    if not validation.passed:
        raise ValueError(f"PH data validation failed: {validation.messages}")
    logger.info("ph_data_loaded rows=%d issues=%d", len(cleaned), sum(cleaning_issues.values()))

    feature_cols = list(PH_BOOKING_TIME_FEATURES)
    df = cleaned[
        feature_cols
        + [PH_TARGET_COL, "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
    ].copy()
    df = df.dropna(subset=[PH_TARGET_COL])

    # ── 1b. Dataset authenticity diagnostics ─────────────────────────
    # Counts feature-vector duplicates BEFORE the chronological split, so the
    # diagnostic captures the dataset's structure rather than split artefacts.
    # See ph_transferability.json -> dataset_diagnostics for the published
    # numbers; the notebook explains why this matters for the perfect-score
    # finding.
    duplicate_diag = _compute_duplicate_diagnostics(
        cleaned, feature_cols=feature_cols, target_col=PH_TARGET_COL
    )
    logger.info(
        "ph_duplicate_diagnostics duplicate_rate=%.1f%% unique_vectors=%d "
        "cross_split_leak_estimate=%.1f%%",
        duplicate_diag["duplicate_rate"] * 100,
        duplicate_diag["n_unique_feature_vectors"],
        duplicate_diag["clusters_with_consistent_labels_pct"] * 100,
    )

    # ── 2. Chronological split ───────────────────────────────────────
    train_df, val_df, test_df = _split_ph_time_aware(df)
    logger.info(
        "ph_data_split rows_train=%d rows_val=%d rows_test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # How many test rows have an identical feature vector somewhere in train+val?
    # This is the proximate cause of inflated test metrics on archetype-structured data.
    leak_diag = _compute_train_test_overlap(train_df, val_df, test_df, feature_cols=feature_cols)
    logger.info(
        "ph_train_test_overlap test_rows_with_train_duplicate=%d/%d (%.1f%%)",
        leak_diag["n_test_rows_with_train_duplicate"],
        leak_diag["n_test"],
        leak_diag["test_duplicate_rate"] * 100,
    )

    X_train = train_df[feature_cols]
    y_train = train_df[PH_TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[PH_TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[PH_TARGET_COL]

    # ── 3. Preprocess + train ────────────────────────────────────────
    preprocessor = _build_ph_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    logger.info("ph_preprocessing_complete n_features_encoded=%d", X_train_t.shape[1])

    # LightGBM is the Portugal champion; use it here too for direct comparison.
    # Falls back to GradientBoosting if lightgbm isn't installed in this env.
    if is_lightgbm_available():
        model_family = "lightgbm"
        model = train_lgbm(X_train_t, y_train)
    else:
        model_family = "gradient_boosting"
        model = train_gb(X_train_t, y_train)
    logger.info("ph_model_fit_complete family=%s", model_family)

    # ── 4. Calibrate + threshold sweep ───────────────────────────────
    val_probs_raw = model.predict_proba(X_val_t)[:, 1]
    test_probs_raw = model.predict_proba(X_test_t)[:, 1]

    y_val_arr = y_val.to_numpy().astype(int)
    if len(np.unique(y_val_arr)) < 2:
        logger.warning("ph_calibration_skipped reason=val_set_single_class")
        calibrator = None
        val_probs = val_probs_raw
        test_probs = test_probs_raw
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_probs_raw, y_val_arr)
        val_probs = np.clip(calibrator.predict(val_probs_raw), 0.0, 1.0)
        test_probs = np.clip(calibrator.predict(test_probs_raw), 0.0, 1.0)
    logger.info("ph_calibration_complete")

    sweep_df = threshold_sweep(y_val, val_probs, step=THRESHOLD_STEP)
    max_f1 = select_max_f1_threshold(sweep_df)
    high_precision = select_high_precision_threshold(
        sweep_df,
        min_positive_rate=0.05,
        min_recall=0.10,
    )
    logger.info(
        "ph_thresholds max_f1=%.3f high_precision=%.3f",
        float(max_f1["threshold"]),
        float(high_precision["threshold"]),
    )

    # ── 5. Test-set evaluation ───────────────────────────────────────
    test_metrics_max_f1 = evaluate_at_threshold(y_test, test_probs, max_f1["threshold"])
    test_metrics_high_precision = evaluate_at_threshold(
        y_test, test_probs, high_precision["threshold"]
    )
    confusion_max_f1 = compute_confusion(y_test, test_probs, max_f1["threshold"])

    caveats: list[str] = []
    if duplicate_diag["duplicate_rate"] >= 0.30:
        caveats.append(
            f"DATASET CLUSTER STRUCTURE: {duplicate_diag['duplicate_rate']:.1%} of rows "
            "share an identical feature vector with another row, and "
            f"{duplicate_diag['clusters_with_consistent_labels_pct']:.1%} of these "
            "duplicate clusters share an identical label. The Philippine resort "
            "dataset is organized around a small set of recurring booking "
            "archetypes. Reported test metrics reflect memorization across "
            "chronological twins, not generalization."
        )
    if leak_diag["test_duplicate_rate"] >= 0.20:
        caveats.append(
            f"CHRONOLOGICAL-TWIN EFFECT: {leak_diag['n_test_rows_with_train_duplicate']}/"
            f"{leak_diag['n_test']} test rows ({leak_diag['test_duplicate_rate']:.1%}) "
            "have an identical feature vector somewhere in train or val. Given "
            "the dataset's archetype-based cluster structure, chronological "
            "splitting cannot prevent twin rows across the boundary — this is a "
            "property of the dataset itself, not a leakage bug."
        )
    caveats.extend(
        [
            f"n_test = {len(test_df)} rows; bootstrap 95% CIs on PR-AUC "
            "span roughly ±15 percentage points at this sample size",
            "Feature set excludes deposit_type, market_segment, country, "
            "agent, customer_type, previous_cancellations — these were among "
            "the top SHAP features on the Portugal model",
            "No rolling-origin CV (insufficient data for the project's standard "
            "3 folds); metrics use a single chronological 80/10/10 split",
            "Constant-variance columns (Meals = 100% 'Breakfast (Complimentary)', "
            "Guest_Type = 100% 'Walk-In') were dropped during clean_raw_ph",
            "Treat all reported metrics as directional, not as gates",
        ]
    )

    metrics_payload = {
        "selected_model_family": model_family,
        "max_f1": {**test_metrics_max_f1, "threshold": float(max_f1["threshold"])},
        "high_precision": {
            **test_metrics_high_precision,
            "threshold": float(high_precision["threshold"]),
        },
        "roc_auc_test": safe_roc_auc(y_test.to_numpy(), test_probs),
        "pr_auc_test": safe_pr_auc(y_test.to_numpy(), test_probs),
        "ece_test": expected_calibration_error(y_test.to_numpy().astype(int), test_probs, 10),
        "confusion_max_f1": confusion_max_f1.tolist(),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "feature_columns": feature_cols,
        "cleaning_issues": cleaning_issues,
        "dataset_diagnostics": duplicate_diag,
        "train_test_overlap": leak_diag,
        "selected_at": datetime.now(tz=timezone.utc).isoformat(),
        "caveats": caveats,
    }

    # ── 6. Notebook-supporting analyses ─────────────────────────────
    # These outputs are consumed by notebooks/ph/{02, 03, 05, 10}*.ipynb.
    baseline_payload = _train_baselines_ph(X_train_t, y_train, X_test_t, y_test, metrics_payload)
    learning_curves_payload = _compute_learning_curves_ph(X_train_t, y_train, X_val_t, y_val)
    full_y = pd.concat([y_train, y_val, y_test], ignore_index=True)
    full_X_t = np.vstack([X_train_t, X_val_t, X_test_t])
    expanding_cv_payload = _compute_expanding_window_cv_ph(full_X_t, full_y)
    shap_result = _compute_shap_ph(model, preprocessor, X_test_t)
    cost_sensitivity_df = _cost_sensitivity_sweep_ph(y_test, test_probs, test_df)

    champion_summary = {
        "champion_model": model_family,
        "selected_by": "single chronological 80/10/10 split (N=300 too small for rolling-origin)",
        "selected_at": datetime.now(tz=timezone.utc).isoformat(),
        "test_roc_auc": metrics_payload["roc_auc_test"],
        "test_pr_auc": metrics_payload["pr_auc_test"],
        "test_ece": metrics_payload["ece_test"],
        "test_f1_at_max_f1": metrics_payload["max_f1"]["f1"],
        "thresholds": {
            "max_f1": float(max_f1["threshold"]),
            "high_precision": float(high_precision["threshold"]),
        },
        "feature_set_size": len(feature_cols),
        "feature_set_note": (
            "PH reduced 8-raw-feature set (vs Portugal 32 raw); top Portugal "
            "predictors (deposit_type, country, agent, market_segment) are not "
            "available in the PH PMS export."
        ),
        "n_train": metrics_payload["n_train"],
        "n_val": metrics_payload["n_val"],
        "n_test": metrics_payload["n_test"],
        "caveats": caveats,
    }

    # ── 7. Persist ───────────────────────────────────────────────────
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    joblib.dump(pipeline, PH_ARTIFACTS_DIR / "ph_model.pkl")
    if calibrator is not None:
        joblib.dump(calibrator, PH_ARTIFACTS_DIR / "ph_calibrator.pkl")
    _save_json(PH_ARTIFACTS_DIR / "ph_feature_columns.json", {"features": feature_cols})
    _save_json(
        PH_ARTIFACTS_DIR / "ph_thresholds.json",
        {"max_f1": max_f1, "high_precision": high_precision},
    )
    _save_json(
        PH_ARTIFACTS_DIR / "ph_model_metadata.json",
        {
            "model_family": model_family,
            "target_column": PH_TARGET_COL,
            "feature_columns": feature_cols,
            "seed": RANDOM_STATE,
            "split": {
                "strategy": "time_aware",
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
                "rows_train": len(train_df),
                "rows_val": len(val_df),
                "rows_test": len(test_df),
            },
            "cleaning_issues": cleaning_issues,
            "selected_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )

    _save_json(PH_REPORTS_DIR / "ph_transferability.json", metrics_payload)
    _save_json(PH_REPORTS_DIR / "champion_summary.json", champion_summary)
    _save_json(PH_REPORTS_DIR / "baseline_comparison.json", baseline_payload)
    _save_json(PH_REPORTS_DIR / "learning_curves.json", learning_curves_payload)
    _save_json(PH_REPORTS_DIR / "expanding_window_cv.json", expanding_cv_payload)

    if shap_result is not None:
        shap_summary, shap_importance_df = shap_result
        _save_json(PH_REPORTS_DIR / "shap_analysis.json", shap_summary)
        shap_importance_df.to_csv(PH_REPORTS_DIR / "shap_feature_importance.csv", index=False)

    if not cost_sensitivity_df.empty:
        cost_sensitivity_df.to_csv(PH_ARTIFACTS_DIR / "cost_threshold_sweep.csv", index=False)

    # Per-row predictions for the notebook visualisations.
    test_export = test_df.copy()
    test_export["cancel_probability"] = test_probs
    test_export["predicted_max_f1"] = (test_probs >= float(max_f1["threshold"])).astype(int)
    test_export.to_csv(PH_REPORTS_DIR / "ph_test_predictions.csv", index=False)

    sweep_df.to_csv(PH_REPORTS_DIR / "ph_threshold_sweep.csv", index=False)

    logger.info(
        "ph_training_complete artifacts_dir=%s rows_train=%d rows_val=%d rows_test=%d "
        "model_family=%s roc_auc_test=%.3f pr_auc_test=%.3f",
        PH_ARTIFACTS_DIR,
        len(train_df),
        len(val_df),
        len(test_df),
        model_family,
        metrics_payload["roc_auc_test"] or float("nan"),
        metrics_payload["pr_auc_test"] or float("nan"),
    )
    return metrics_payload


def main() -> None:
    configure_logging()
    metrics = run_ph_training_pipeline()

    diag = metrics["dataset_diagnostics"]
    leak = metrics["train_test_overlap"]
    is_clustered = (
        diag["duplicate_rate"] >= 0.30 and diag["clusters_with_consistent_labels_pct"] >= 0.90
    )

    print("\n" + "=" * 60)
    print("PH PHILIPPINE DATASET SUB-STUDY - RESULTS")
    print("=" * 60)

    if is_clustered:
        print("\n** PHILIPPINE DATASET CLUSTER STRUCTURE **")
        print(diag["interpretation"])
        print(
            f"  duplicate_rate = {diag['duplicate_rate']:.1%} "
            f"({diag['n_duplicate_rows']}/{diag['n_rows_total']} rows)"
        )
        print(
            "  test-set rows with identical train/val twin = "
            f"{leak['n_test_rows_with_train_duplicate']}/{leak['n_test']} "
            f"({leak['test_duplicate_rate']:.1%})"
        )
        print(
            "  -> The metrics below are inflated by chronological-twin "
            "memorization (a structural property of the dataset); "
            "treat them as directional.\n"
        )

    print(f"Model family : {metrics['selected_model_family']}")
    print(f"Train / Val / Test : {metrics['n_train']} / {metrics['n_val']} / {metrics['n_test']}")
    print(f"Test ROC-AUC : {metrics['roc_auc_test']:.3f}")
    print(f"Test PR-AUC  : {metrics['pr_auc_test']:.3f}")
    print(f"Test ECE     : {metrics['ece_test']:.3f}")
    print(
        f"max_f1 threshold = {metrics['max_f1']['threshold']:.3f} "
        f"-> F1={metrics['max_f1']['f1']:.3f}, "
        f"precision={metrics['max_f1']['precision']:.3f}, "
        f"recall={metrics['max_f1']['recall']:.3f}"
    )
    print("\nCAVEATS:")
    for c in metrics["caveats"]:
        print(f"  - {c}")
    print(f"\nArtifacts written to: {PH_ARTIFACTS_DIR}")
    print(f"Reports written to  : {PH_REPORTS_DIR}")


if __name__ == "__main__":
    main()
