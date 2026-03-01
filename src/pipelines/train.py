"""Reusable training pipeline for cancellation modeling."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess  # nosec B404 – used only for `git rev-parse HEAD` with a hardcoded list
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from src.config import (
    ARTIFACTS_DIR,
    BOOKING_TIME_FEATURES,
    CALIBRATION_ECE_BINS,
    CALIBRATION_METHOD,
    DATA_PATH,
    FN_RECOVERY_NIGHTS,
    FP_INTERVENTION_COST,
    LATE_WINDOW_MAX_LEAD_DAYS,
    LEAKAGE_COLS,
    MIN_POSITIVE_RATE,
    MIN_RECALL_FOR_HIGH_PRECISION,
    MODEL_SELECTION_POLICY,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
    ROLLING_SELECTION_CUTOFF_FRACS,
    ROLLING_SELECTION_MIN_TRAIN_ROWS,
    ROLLING_SELECTION_MIN_VAL_ROWS,
    ROLLING_SELECTION_VAL_RATIO,
    SEGMENT_METRIC_GATES,
    TARGET_COL,
    THRESHOLD_STEP,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data.load import load_raw_data
from src.features.build import build_preprocessor, split_time_aware
from src.models.metrics import (
    compute_confusion,
    evaluate_at_threshold,
    expected_calibration_error,
    safe_pr_auc,
    safe_roc_auc,
)
from src.models.train import is_lightgbm_available, train_gb, train_lgbm, train_xgb
from src.utils.business import (
    assign_risk_tiers,
    compute_cost_threshold_policy,
    compute_fn_cost_vector,
    safe_threshold_metrics,
)
from src.utils.reproducibility import set_global_seed
from src.utils.thresholds import (
    select_high_precision_threshold,
    select_max_f1_threshold,
    threshold_sweep,
)
from src.utils.validate_data import (
    assert_no_leakage_columns,
    clean_raw,
    validate_raw,
)

logger = logging.getLogger(__name__)

CANDIDATE_MODEL_FAMILIES = (
    ("gradient_boosting", "xgboost", "lightgbm")
    if is_lightgbm_available()
    else ("gradient_boosting", "xgboost")
)
FALLBACK_MODEL_FAMILY = "xgboost"


@dataclass(frozen=True)
class TrainingOutputs:
    """Structured summary of persisted training outputs."""

    artifacts_dir: Path
    reports_dir: Path
    model_path: Path
    metrics: dict[str, Any]
    thresholds: dict[str, Any]


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _sanitise_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN / ±Inf) with None.

    Python's ``json.dumps`` serialises ``float('nan')`` as the literal token
    ``NaN``, which is not valid JSON (RFC 8259).  Replacing with ``null``
    (Python ``None``) preserves the key while producing valid output.
    """
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_for_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_sanitise_for_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_data_path(data_path: str | None) -> Path:
    raw_path = Path(data_path) if data_path is not None else DATA_PATH
    if raw_path.is_absolute():
        return raw_path
    return raw_path.resolve()


def _source_tree_hash() -> str:
    tracked: list[Path] = []
    tracked.extend((PROJECT_ROOT / "src").rglob("*.py"))
    tracked.extend((PROJECT_ROOT / "scripts").rglob("*.py"))
    tracked.extend(
        path
        for path in (
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / "requirements.txt",
        )
        if path.exists()
    )
    hasher = hashlib.sha256()
    for path in sorted(set(tracked)):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _git_commit_hash() -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(  # nosec B603 B607 – hardcoded list, no user input
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip(), None
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, str(exc)


def _fit_probability_calibrator(
    val_probs_raw: np.ndarray,
    y_val: pd.Series,
) -> IsotonicRegression:
    if CALIBRATION_METHOD != "isotonic":
        raise ValueError(f"Unsupported calibration method: {CALIBRATION_METHOD}")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_raw, y_val.to_numpy().astype(int))
    return calibrator


def _calibration_metrics(
    y_val: pd.Series,
    y_test: pd.Series,
    val_raw: np.ndarray,
    val_cal: np.ndarray,
    test_raw: np.ndarray,
    test_cal: np.ndarray,
) -> dict[str, Any]:
    y_val_np = y_val.to_numpy().astype(int)
    y_test_np = y_test.to_numpy().astype(int)
    return {
        "method": CALIBRATION_METHOD,
        "ece_bins": CALIBRATION_ECE_BINS,
        "validation": {
            "brier_raw": float(brier_score_loss(y_val_np, val_raw)),
            "brier_calibrated": float(brier_score_loss(y_val_np, val_cal)),
            "ece_raw": expected_calibration_error(y_val_np, val_raw, CALIBRATION_ECE_BINS),
            "ece_calibrated": expected_calibration_error(y_val_np, val_cal, CALIBRATION_ECE_BINS),
        },
        "test": {
            "brier_raw": float(brier_score_loss(y_test_np, test_raw)),
            "brier_calibrated": float(brier_score_loss(y_test_np, test_cal)),
            "ece_raw": expected_calibration_error(y_test_np, test_raw, CALIBRATION_ECE_BINS),
            "ece_calibrated": expected_calibration_error(y_test_np, test_cal, CALIBRATION_ECE_BINS),
        },
    }


def _segment_metrics(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    segment_gate_cfg = cast(dict[str, Any], SEGMENT_METRIC_GATES)
    min_rows = int(segment_gate_cfg["min_rows"])
    policy = str(segment_gate_cfg["policy"])
    dimensions = cast(dict[str, int], segment_gate_cfg["dimensions"])
    rows: list[dict[str, Any]] = []

    y_full = y_test.to_numpy().astype(int)
    for dimension, top_n in dimensions.items():
        if dimension not in test_df.columns:
            continue
        segment_values = test_df[dimension].fillna("UNKNOWN").astype(str)
        top_values = segment_values.value_counts().head(int(top_n)).index.tolist()
        for value in top_values:
            mask = segment_values == value
            idx = np.where(mask.to_numpy())[0]
            n_rows = int(len(idx))
            y_seg = y_full[idx]
            p_seg = probs[idx]
            gated = n_rows >= min_rows and len(np.unique(y_seg)) >= 2
            row: dict[str, Any] = {
                "policy": policy,
                "dimension": dimension,
                "segment": value,
                "n_rows": n_rows,
                "positive_rate": float(np.mean(y_seg)) if n_rows else 0.0,
                "gated": gated,
                "threshold": float(threshold),
            }
            if gated:
                metrics = evaluate_at_threshold(y_seg, p_seg, threshold)
                row.update(metrics)
            else:
                row.update(
                    {
                        "roc_auc": None,
                        "pr_auc": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "balanced_accuracy": None,
                    }
                )
                if n_rows < min_rows:
                    row["skip_reason"] = "insufficient_rows"
                elif len(np.unique(y_seg)) < 2:
                    row["skip_reason"] = "single_class"
            rows.append(row)

    return {
        "policy": policy,
        "min_rows": min_rows,
        "dimensions": dimensions,
        "rows": rows,
    }


def _late_window_report(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    probs: np.ndarray,
    thresholds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if "lead_time" not in test_df.columns:
        return {"skipped": True, "reason": "lead_time column unavailable"}

    mask = pd.to_numeric(test_df["lead_time"], errors="coerce").fillna(np.inf) <= float(
        LATE_WINDOW_MAX_LEAD_DAYS
    )
    idx = np.where(mask.to_numpy())[0]
    y_all = y_test.to_numpy().astype(int)
    late_y = y_all[idx]
    late_probs = probs[idx]
    overall_cancel_rate = float(np.mean(y_all)) if len(y_all) else 0.0
    late_cancel_rate = float(np.mean(late_y)) if len(late_y) else 0.0

    policy_metrics: dict[str, Any] = {}
    for policy_name, payload in thresholds.items():
        threshold = float(payload["threshold"])
        if len(idx) == 0:
            policy_metrics[policy_name] = {
                "threshold": threshold,
                "metrics": None,
            }
            continue
        policy_metrics[policy_name] = {
            "threshold": threshold,
            "metrics": safe_threshold_metrics(late_y, late_probs, threshold),
        }

    return {
        "lead_time_days_max": int(LATE_WINDOW_MAX_LEAD_DAYS),
        "n_rows_late_window": int(len(idx)),
        "n_rows_test_total": int(len(test_df)),
        "late_window_share": float(len(idx) / len(test_df)) if len(test_df) else 0.0,
        "cancel_rate_overall_test": overall_cancel_rate,
        "cancel_rate_late_window": late_cancel_rate,
        "policy_metrics": policy_metrics,
    }


def _hypothesis_summary(
    *,
    selected_model_family: str,
    selection_report: dict[str, Any],
    cost_summary: dict[str, Any],
    late_window_report: dict[str, Any],
) -> dict[str, Any]:
    model_scores = {
        row["model_family"]: {
            "rolling_pr_auc_mean": row["rolling_pr_auc_mean"],
            "rolling_roc_auc_mean": row["rolling_roc_auc_mean"],
        }
        for row in selection_report.get("candidates", [])
    }
    h2_supported = selected_model_family in {"xgboost", "lightgbm"}
    h4_supported = float(cost_summary.get("savings_vs_050", 0.0)) > 0

    return {
        "H1": {
            "statement": "Lead time, deposit type, and previous cancellations are significant predictors.",
            "status": "deferred_to_thesis_shap",
            "evidence": "Use reports/thesis/shap_analysis.json for feature attribution ranking.",
        },
        "H2": {
            "statement": "Gradient-boosted trees outperform simpler baselines.",
            "status": "supported" if h2_supported else "not_supported",
            "selected_model_family": selected_model_family,
            "rolling_selection_scores": model_scores,
        },
        "H3": {
            "statement": "Lead time has highest SHAP importance, then deposit type and previous cancellations.",
            "status": "deferred_to_thesis_shap",
            "evidence": "Use reports/thesis/shap_feature_importance.csv for ranking verification.",
        },
        "H4": {
            "statement": "Cost-sensitive thresholding reduces expected revenue loss.",
            "status": "supported" if h4_supported else "not_supported",
            "savings_vs_threshold_050": float(cost_summary.get("savings_vs_050", 0.0)),
            "savings_vs_no_model": float(cost_summary.get("savings_vs_no_model", 0.0)),
            "late_window_cancel_rate": float(
                late_window_report.get("cancel_rate_late_window", 0.0)
            ),
        },
    }


def _artifact_lineage(
    *,
    data_source_path: Path,
    artifacts_dir: Path,
    artifact_files: list[str],
) -> dict[str, Any]:
    commit_hash, commit_error = _git_commit_hash()
    artifact_hashes = {
        name: _sha256_file(artifacts_dir / name)
        for name in artifact_files
        if (artifacts_dir / name).exists()
    }
    bundle_hasher = hashlib.sha256()
    for name in sorted(artifact_hashes):
        bundle_hasher.update(name.encode("utf-8"))
        bundle_hasher.update(b":")
        bundle_hasher.update(artifact_hashes[name].encode("utf-8"))
        bundle_hasher.update(b"\n")

    return {
        "data": {
            "source_path": str(data_source_path),
            "sha256": _sha256_file(data_source_path) if data_source_path.exists() else None,
        },
        "code": {
            "git_commit": commit_hash,
            "git_commit_error": commit_error,
            "source_tree_sha256": _source_tree_hash(),
        },
        "artifacts": {
            "files": artifact_hashes,
            "bundle_sha256": bundle_hasher.hexdigest(),
        },
    }


def _compute_scale_pos_weight(y_train: pd.Series) -> float:
    y_arr = np.asarray(y_train)
    pos = float(np.sum(y_arr == 1))
    neg = float(np.sum(y_arr == 0))
    return (neg / pos) if pos > 0 else 1.0


def _fit_model_family(
    model_family: str,
    X_train,
    y_train,
    *,
    X_val=None,
    y_val=None,
    scale_pos_weight: float | None = None,
    use_early_stopping: bool = False,
):
    if model_family == "gradient_boosting":
        # sklearn GradientBoostingClassifier has no scale_pos_weight parameter;
        # class imbalance is handled via the deviance loss function.
        return train_gb(X_train, y_train)
    if model_family == "xgboost":
        if use_early_stopping and X_val is not None and y_val is not None:
            return train_xgb(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                scale_pos_weight=scale_pos_weight,
            )
        return train_xgb(X_train, y_train, scale_pos_weight=scale_pos_weight)
    if model_family == "lightgbm":
        if use_early_stopping and X_val is not None and y_val is not None:
            return train_lgbm(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
            )
        return train_lgbm(X_train, y_train)
    raise ValueError(f"Unsupported model family: {model_family}")


def _rolling_selection_windows(total_rows: int) -> list[tuple[float, int, int]]:
    val_rows = max(int(total_rows * ROLLING_SELECTION_VAL_RATIO), ROLLING_SELECTION_MIN_VAL_ROWS)
    windows: list[tuple[float, int, int]] = []
    for cutoff_frac in ROLLING_SELECTION_CUTOFF_FRACS:
        train_end = int(total_rows * cutoff_frac)
        val_end = train_end + val_rows
        if train_end < ROLLING_SELECTION_MIN_TRAIN_ROWS:
            continue
        if val_end > total_rows:
            continue
        windows.append((cutoff_frac, train_end, val_end))
    return windows


def _rank_metric_value(value: float) -> float:
    if np.isnan(value):
        return -1.0
    return float(value)


def _select_model_family(selection_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    windows = _rolling_selection_windows(len(selection_df))
    candidates: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    if not windows:
        return {
            "policy": MODEL_SELECTION_POLICY,
            "primary_metric": "rolling_pr_auc_mean",
            "secondary_metric": "rolling_roc_auc_mean",
            "winner": FALLBACK_MODEL_FAMILY,
            "fallback_reason": "Insufficient rows for rolling selection windows.",
            "window": {
                "cutoff_fracs": ROLLING_SELECTION_CUTOFF_FRACS,
                "val_ratio": ROLLING_SELECTION_VAL_RATIO,
                "min_train_rows": ROLLING_SELECTION_MIN_TRAIN_ROWS,
                "min_val_rows": ROLLING_SELECTION_MIN_VAL_ROWS,
                "computed_windows": [],
            },
            "candidates": [],
            "folds": [],
        }

    for model_family in CANDIDATE_MODEL_FAMILIES:
        pr_values: list[float] = []
        roc_values: list[float] = []

        for fold_id, (cutoff_frac, train_end, val_end) in enumerate(windows, start=1):
            fold_train = selection_df.iloc[:train_end]
            fold_val = selection_df.iloc[train_end:val_end]
            X_train_fold = fold_train[feature_cols]
            y_train_fold = fold_train[TARGET_COL].astype(int)
            X_val_fold = fold_val[feature_cols]
            y_val_fold = fold_val[TARGET_COL].astype(int)

            preprocessor = build_preprocessor()
            X_train_t = preprocessor.fit_transform(X_train_fold)
            X_val_t = preprocessor.transform(X_val_fold)

            model = _fit_model_family(
                model_family,
                X_train_t,
                y_train_fold,
                scale_pos_weight=_compute_scale_pos_weight(y_train_fold),
                use_early_stopping=False,
            )
            probs = model.predict_proba(X_val_t)[:, 1]
            roc_auc = safe_roc_auc(y_val_fold.to_numpy(), probs)
            pr_auc = safe_pr_auc(y_val_fold.to_numpy(), probs)

            fold_rows.append(
                {
                    "model_family": model_family,
                    "fold": fold_id,
                    "cutoff_frac": float(cutoff_frac),
                    "n_train": int(train_end),
                    "n_val": int(val_end - train_end),
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                }
            )
            if not np.isnan(roc_auc) and not np.isnan(pr_auc):
                roc_values.append(float(roc_auc))
                pr_values.append(float(pr_auc))

        candidate_row = {
            "model_family": model_family,
            "folds_evaluated": len(pr_values),
            "rolling_pr_auc_mean": float(np.mean(pr_values)) if pr_values else float("nan"),
            "rolling_roc_auc_mean": float(np.mean(roc_values)) if roc_values else float("nan"),
        }
        candidates.append(candidate_row)

    eligible = [row for row in candidates if row["folds_evaluated"] > 0]
    if eligible:
        ranked = sorted(
            eligible,
            key=lambda row: (
                -_rank_metric_value(row["rolling_pr_auc_mean"]),
                -_rank_metric_value(row["rolling_roc_auc_mean"]),
                row["model_family"],
            ),
        )
        winner = str(ranked[0]["model_family"])
        fallback_reason = None
    else:
        winner = FALLBACK_MODEL_FAMILY
        fallback_reason = "No candidate produced valid rolling metrics."

    return {
        "policy": MODEL_SELECTION_POLICY,
        "primary_metric": "rolling_pr_auc_mean",
        "secondary_metric": "rolling_roc_auc_mean",
        "winner": winner,
        "fallback_reason": fallback_reason,
        "window": {
            "cutoff_fracs": ROLLING_SELECTION_CUTOFF_FRACS,
            "val_ratio": ROLLING_SELECTION_VAL_RATIO,
            "min_train_rows": ROLLING_SELECTION_MIN_TRAIN_ROWS,
            "min_val_rows": ROLLING_SELECTION_MIN_VAL_ROWS,
            "computed_windows": [
                {"cutoff_frac": frac, "train_end": train_end, "val_end": val_end}
                for frac, train_end, val_end in windows
            ],
        },
        "candidates": candidates,
        "folds": fold_rows,
    }


def _model_metadata(
    feature_cols: list[str],
    cleaning_issues: dict[str, int],
    train_rows: int,
    val_rows: int,
    test_rows: int,
    scale_pos_weight: float,
    model_family: str,
    selection_summary: dict[str, Any],
    calibration_summary: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_selection_policy": MODEL_SELECTION_POLICY,
        "model_type": model_family,
        "feature_columns": feature_cols,
        "leakage_columns_excluded": LEAKAGE_COLS,
        "target_column": TARGET_COL,
        "seed": RANDOM_STATE,
        "split": {
            "strategy": "time_aware",
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
        },
        "training": {
            "rows_train": train_rows,
            "rows_val": val_rows,
            "rows_test": test_rows,
            "scale_pos_weight": scale_pos_weight if model_family == "xgboost" else None,
        },
        "model_selection": selection_summary,
        "calibration": calibration_summary,
        "lineage": lineage,
        "cleaning_issues": cleaning_issues,
    }


def run_training_pipeline(
    *,
    artifacts_dir: Path = ARTIFACTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    data_path: str | None = None,
    max_rows: int | None = None,
) -> TrainingOutputs:
    """Train model, evaluate thresholds, and persist aligned artifacts/reports."""
    set_global_seed(RANDOM_STATE)
    _ensure_dirs(artifacts_dir, reports_dir)

    logger.info("pipeline_start artifacts_dir=%s reports_dir=%s", artifacts_dir, reports_dir)

    df = load_raw_data(path=data_path)
    df, cleaning_issues = clean_raw(df)
    logger.info(
        "data_loaded rows=%d cleaning_issues=%d",
        len(df),
        sum(cleaning_issues.values()),
    )
    validation = validate_raw(df)
    if not validation.passed:
        raise ValueError(f"Data validation failed: {validation.messages}")

    feature_cols = BOOKING_TIME_FEATURES.copy()
    assert_no_leakage_columns(feature_cols)

    df = df.drop(columns=[col for col in LEAKAGE_COLS if col in df.columns])
    df = df[feature_cols + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL])
    if max_rows is not None:
        df = df.head(max_rows).copy()

    train_df, val_df, test_df = split_time_aware(df)
    # split_time_aware raises ValueError on empty partitions; this guard is a
    # belt-and-suspenders check for unexpected empty DataFrames after splitting.
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Time-aware split produced an empty partition.")
    logger.info(
        "data_split rows_train=%d rows_val=%d rows_test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    selection_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    logger.info("model_selection_start n_candidates=%d", len(CANDIDATE_MODEL_FAMILIES))
    selection_report = _select_model_family(selection_df, feature_cols)
    selected_model_family = str(selection_report["winner"])
    logger.info(
        "model_selection_complete winner=%s fallback=%s",
        selected_model_family,
        selection_report.get("fallback_reason"),
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    logger.info(
        "preprocessing_complete n_features_raw=%d n_features_encoded=%d",
        len(feature_cols),
        X_train_t.shape[1],
    )

    scale_pos_weight = _compute_scale_pos_weight(y_train)
    logger.info(
        "model_fit_start family=%s scale_pos_weight=%.3f", selected_model_family, scale_pos_weight
    )
    model = _fit_model_family(
        selected_model_family,
        X_train_t,
        y_train,
        scale_pos_weight=scale_pos_weight,
        use_early_stopping=False,
    )

    val_probs_raw = model.predict_proba(X_val_t)[:, 1]
    test_probs_raw = model.predict_proba(X_test_t)[:, 1]
    # NOTE: Calibrator is fit on val data; val calibration metrics are optimistic.
    # Test-set metrics (computed on held-out test data below) are unbiased.
    calibrator = _fit_probability_calibrator(val_probs_raw, y_val)
    val_probs = np.clip(calibrator.predict(val_probs_raw), 0.0, 1.0)
    test_probs = np.clip(calibrator.predict(test_probs_raw), 0.0, 1.0)
    logger.info("calibration_complete method=%s", CALIBRATION_METHOD)
    calibration_report = _calibration_metrics(
        y_val=y_val,
        y_test=y_test,
        val_raw=val_probs_raw,
        val_cal=val_probs,
        test_raw=test_probs_raw,
        test_cal=test_probs,
    )

    sweep_df = threshold_sweep(y_val, val_probs, step=THRESHOLD_STEP)
    high_precision = select_high_precision_threshold(
        sweep_df,
        MIN_POSITIVE_RATE,
        MIN_RECALL_FOR_HIGH_PRECISION,
    )
    max_f1 = select_max_f1_threshold(sweep_df)
    logger.info(
        "threshold_selection_complete max_f1=%.3f high_precision=%.3f",
        float(max_f1["threshold"]),
        float(high_precision["threshold"]),
    )
    cost_summary, cost_sweep_df = compute_cost_threshold_policy(
        y_val.to_numpy().astype(int),
        val_probs,
        compute_fn_cost_vector(val_df, fn_recovery_nights=FN_RECOVERY_NIGHTS),
        fp_cost=FP_INTERVENTION_COST,
        step=THRESHOLD_STEP,
    )
    cost_sensitive = {
        "threshold": float(cost_summary["threshold"]),
        "fp_cost_assumption": float(FP_INTERVENTION_COST),
        "fn_recovery_nights": float(FN_RECOVERY_NIGHTS),
        "validation_total_cost": float(cost_summary["total_cost"]),
        "validation_savings_vs_050": float(cost_summary["savings_vs_050"]),
        "validation_savings_vs_no_model": float(cost_summary["savings_vs_no_model"]),
    }
    thresholds = {
        "high_precision": high_precision,
        "max_f1": max_f1,
        "cost_sensitive": cost_sensitive,
    }

    metrics = {
        "high_precision": evaluate_at_threshold(y_test, test_probs, high_precision["threshold"]),
        "max_f1": evaluate_at_threshold(y_test, test_probs, max_f1["threshold"]),
        "cost_sensitive": evaluate_at_threshold(y_test, test_probs, cost_sensitive["threshold"]),
        "selected_model_family": selected_model_family,
        "scale_pos_weight": scale_pos_weight if selected_model_family == "xgboost" else None,
        "model_selection": {
            "policy": selection_report["policy"],
            "winner": selected_model_family,
            "candidates": selection_report["candidates"],
            "fallback_reason": selection_report.get("fallback_reason"),
        },
        "calibration": calibration_report,
        "data_cleaning": cleaning_issues,
        "cost_thresholding": {
            **cost_summary,
            "risk_tier_medium_threshold": float(RISK_TIER_MEDIUM_THRESHOLD),
            "risk_tier_high_threshold": float(RISK_TIER_HIGH_THRESHOLD),
        },
    }

    confusion_high_precision = compute_confusion(y_test, test_probs, high_precision["threshold"])
    confusion_max_f1 = compute_confusion(y_test, test_probs, max_f1["threshold"])
    confusion_cost_sensitive = compute_confusion(y_test, test_probs, cost_sensitive["threshold"])
    segment_report = _segment_metrics(test_df, y_test, test_probs, float(max_f1["threshold"]))
    late_window = _late_window_report(test_df, y_test, test_probs, thresholds)
    hypothesis = _hypothesis_summary(
        selected_model_family=selected_model_family,
        selection_report=selection_report,
        cost_summary=cost_summary,
        late_window_report=late_window,
    )

    risk_tiers = assign_risk_tiers(
        test_probs,
        medium_threshold=RISK_TIER_MEDIUM_THRESHOLD,
        high_threshold=RISK_TIER_HIGH_THRESHOLD,
    )
    powerbi_export = test_df.copy()
    powerbi_export["cancel_probability"] = test_probs
    powerbi_export["risk_tier"] = risk_tiers
    powerbi_export["predicted_cancel_max_f1"] = (
        powerbi_export["cancel_probability"] >= float(max_f1["threshold"])
    ).astype(int)
    powerbi_export["predicted_cancel_high_precision"] = (
        powerbi_export["cancel_probability"] >= float(high_precision["threshold"])
    ).astype(int)
    powerbi_export["predicted_cancel_cost_sensitive"] = (
        powerbi_export["cancel_probability"] >= float(cost_sensitive["threshold"])
    ).astype(int)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    model_path = artifacts_dir / "best_model.pkl"
    calibrator_path = artifacts_dir / "probability_calibrator.pkl"
    tracked_artifact_files = [
        "best_model.pkl",
        "probability_calibrator.pkl",
        "feature_columns.json",
        "thresholds.json",
        "threshold_sweep.csv",
        "cost_threshold_sweep.csv",
    ]
    joblib.dump(pipeline, model_path)
    joblib.dump(calibrator, calibrator_path)
    _save_json(artifacts_dir / "feature_columns.json", {"features": feature_cols})
    _save_json(artifacts_dir / "thresholds.json", thresholds)
    sweep_df.to_csv(artifacts_dir / "threshold_sweep.csv", index=False)
    cost_sweep_df.to_csv(artifacts_dir / "cost_threshold_sweep.csv", index=False)
    lineage = _artifact_lineage(
        data_source_path=_resolve_data_path(data_path),
        artifacts_dir=artifacts_dir,
        artifact_files=tracked_artifact_files,
    )
    _save_json(artifacts_dir / "hashes.json", lineage["artifacts"])
    _save_json(
        artifacts_dir / "model_metadata.json",
        _model_metadata(
            feature_cols=feature_cols,
            cleaning_issues=cleaning_issues,
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            scale_pos_weight=scale_pos_weight,
            model_family=selected_model_family,
            selection_summary={
                "policy": selection_report["policy"],
                "winner": selected_model_family,
                "primary_metric": selection_report["primary_metric"],
                "secondary_metric": selection_report["secondary_metric"],
                "candidates": selection_report["candidates"],
                "fallback_reason": selection_report.get("fallback_reason"),
            },
            calibration_summary={
                "method": CALIBRATION_METHOD,
                "artifact": "probability_calibrator.pkl",
                "validation_brier_raw": calibration_report["validation"]["brier_raw"],
                "validation_brier_calibrated": calibration_report["validation"]["brier_calibrated"],
            },
            lineage=lineage,
        ),
    )

    _save_json(reports_dir / "metrics.json", metrics)
    _save_json(reports_dir / "model_selection_summary.json", selection_report)
    _save_json(reports_dir / "calibration_metrics.json", calibration_report)
    _save_json(reports_dir / "segment_metrics.json", segment_report)
    _save_json(reports_dir / "cost_threshold_summary.json", cost_summary)
    _save_json(reports_dir / "late_window_metrics.json", late_window)
    _save_json(reports_dir / "hypothesis_summary.json", hypothesis)
    powerbi_export.to_csv(reports_dir / "test_predictions_for_powerbi.csv", index=False)

    pd.DataFrame(selection_report["folds"]).to_csv(
        reports_dir / "model_selection_rolling.csv",
        index=False,
    )
    pd.DataFrame(segment_report["rows"]).to_csv(
        reports_dir / "segment_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        confusion_high_precision,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(reports_dir / "confusion_matrix_high_precision.csv")

    pd.DataFrame(
        confusion_max_f1,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(reports_dir / "confusion_matrix_max_f1.csv")

    pd.DataFrame(
        confusion_cost_sensitive,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(reports_dir / "confusion_matrix_cost_sensitive.csv")

    _save_json(reports_dir / "threshold_summary.json", thresholds)

    logger.info(
        "training_complete artifacts_dir=%s reports_dir=%s rows_train=%s rows_val=%s rows_test=%s selected_model=%s",
        artifacts_dir,
        reports_dir,
        len(train_df),
        len(val_df),
        len(test_df),
        selected_model_family,
    )
    return TrainingOutputs(
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        model_path=model_path,
        metrics=metrics,
        thresholds=thresholds,
    )
