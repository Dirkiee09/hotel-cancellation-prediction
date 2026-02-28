"""Deterministic multi-model benchmark for cancellation classification."""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    BOOKING_TIME_FEATURES,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_N_ITERATIONS,
    CALIBRATION_ECE_BINS,
    FN_RECOVERY_NIGHTS,
    FP_INTERVENTION_COST,
    LATE_WINDOW_MAX_LEAD_DAYS,
    LEAKAGE_COLS,
    MIN_POSITIVE_RATE,
    MIN_RECALL_FOR_HIGH_PRECISION,
    RANDOM_STATE,
    REPORTS_DIR,
    ROLLING_SELECTION_CUTOFF_FRACS,
    ROLLING_SELECTION_MIN_TRAIN_ROWS,
    ROLLING_SELECTION_MIN_VAL_ROWS,
    ROLLING_SELECTION_VAL_RATIO,
    TARGET_COL,
    TEMPORAL_STABILITY_BUCKETS,
    THRESHOLD_STEP,
)
from src.data.load import load_raw_data
from src.features.build import add_arrival_date, build_preprocessor, split_time_aware
from src.models.metrics import (
    compute_confusion,
    evaluate_at_threshold,
    expected_calibration_error,
    safe_pr_auc,
    safe_roc_auc,
)
from src.models.train import is_lightgbm_available
from src.utils.business import (
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
from src.utils.validate_data import assert_no_leakage_columns, clean_raw, validate_raw

lgb: Any
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBenchmarkOutputs:
    """Summary of persisted benchmark outputs."""

    reports_dir: Path
    table_paths: dict[str, Path]
    champion_model: str


@dataclass(frozen=True)
class ModelSpec:
    """Single benchmark model specification."""

    name: str
    factory: Callable[[], Any]
    params_for_report: dict[str, Any]


@dataclass
class ModelResult:
    """Runtime outputs for one model."""

    name: str
    model: Any
    calibrator: IsotonicRegression
    val_probs: np.ndarray
    test_probs: np.ndarray
    threshold_default: float
    threshold_max_f1: float
    threshold_high_precision: float
    threshold_cost_sensitive: float
    threshold_high_precision_constraints_met: bool
    cost_summary: dict[str, float]
    prob_metrics: dict[str, float]
    metrics_050: dict[str, float]
    metrics_max_f1: dict[str, float]
    metrics_high_precision: dict[str, float]
    confusion_counts: dict[str, int]
    confusion_rates: dict[str, float]
    fit_seconds: float
    predict_seconds: float
    bundle_size_mb: float


def _bundle_size_mb(preprocessor: Any, model: Any, calibrator: Any) -> float:
    buffer = io.BytesIO()
    joblib.dump({"preprocessor": preprocessor, "model": model, "calibrator": calibrator}, buffer)
    return float(buffer.tell() / (1024.0 * 1024.0))


def _write_table(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    df.to_csv(csv_path, index=False)
    try:
        markdown = df.to_markdown(index=False)
    except Exception:
        markdown = df.to_string(index=False)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    return csv_path


def _model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="logistic_regression",
            factory=lambda: LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            params_for_report={"max_iter": 2000, "solver": "lbfgs", "random_state": RANDOM_STATE},
        ),
        ModelSpec(
            name="decision_tree",
            factory=lambda: DecisionTreeClassifier(
                random_state=RANDOM_STATE,
            ),
            params_for_report={"random_state": RANDOM_STATE},
        ),
        ModelSpec(
            name="random_forest",
            factory=lambda: RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            params_for_report={
                "n_estimators": 300,
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            },
        ),
        ModelSpec(
            name="gradient_boosting",
            factory=lambda: GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            params_for_report={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": RANDOM_STATE,
            },
        ),
        ModelSpec(
            name="xgboost",
            factory=lambda: XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=1,
                eval_metric="logloss",
            ),
            params_for_report={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
                "eval_metric": "logloss",
            },
        ),
    ]
    if is_lightgbm_available() and lgb is not None:
        specs.append(
            ModelSpec(
                name="lightgbm",
                factory=lambda: lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                params_for_report={
                    "n_estimators": 300,
                    "max_depth": 7,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": RANDOM_STATE,
                    "n_jobs": 1,
                },
            )
        )
    return specs


def _prepare_data(
    data_path: str | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
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
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Time-aware split produced an empty partition.")
    return train_df, val_df, test_df, feature_cols


def _evaluate_model(
    *,
    spec: ModelSpec,
    preprocessor: Any,
    X_train_t: np.ndarray,
    y_train_np: np.ndarray,
    X_val_t: np.ndarray,
    y_val_np: np.ndarray,
    fn_cost_val: np.ndarray,
    X_test_t: np.ndarray,
    y_test_np: np.ndarray,
) -> ModelResult:
    model = spec.factory()

    fit_start = time.perf_counter()
    model.fit(X_train_t, y_train_np)
    fit_seconds = float(time.perf_counter() - fit_start)

    pred_start = time.perf_counter()
    val_probs_raw = model.predict_proba(X_val_t)[:, 1]
    test_probs_raw = model.predict_proba(X_test_t)[:, 1]
    predict_seconds = float(time.perf_counter() - pred_start)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_raw, y_val_np)
    val_probs = np.clip(calibrator.predict(val_probs_raw), 0.0, 1.0)
    test_probs = np.clip(calibrator.predict(test_probs_raw), 0.0, 1.0)

    sweep = threshold_sweep(y_val_np, val_probs, step=THRESHOLD_STEP)
    max_f1_info = select_max_f1_threshold(sweep)
    high_precision_info = select_high_precision_threshold(
        sweep,
        MIN_POSITIVE_RATE,
        MIN_RECALL_FOR_HIGH_PRECISION,
    )
    threshold_max_f1 = float(max_f1_info["threshold"])
    threshold_high_precision = float(high_precision_info["threshold"])
    cost_summary, _ = compute_cost_threshold_policy(
        y_val_np,
        val_probs,
        fn_cost_val,
        fp_cost=FP_INTERVENTION_COST,
        step=THRESHOLD_STEP,
    )
    threshold_cost_sensitive = float(cost_summary["threshold"])

    metrics_050 = evaluate_at_threshold(y_test_np, test_probs, 0.50)
    metrics_max_f1 = evaluate_at_threshold(y_test_np, test_probs, threshold_max_f1)
    metrics_high_precision = evaluate_at_threshold(y_test_np, test_probs, threshold_high_precision)

    cm = compute_confusion(y_test_np, test_probs, threshold_max_f1)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    tpr = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    tnr = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    fpr = float(fp / (fp + tn)) if (fp + tn) else float("nan")
    fnr = float(fn / (fn + tp)) if (fn + tp) else float("nan")

    prob_metrics = {
        "roc_auc": safe_roc_auc(y_test_np, test_probs),
        "pr_auc": safe_pr_auc(y_test_np, test_probs),
        "brier_score": float(brier_score_loss(y_test_np, test_probs)),
        "ece": expected_calibration_error(y_test_np, test_probs, CALIBRATION_ECE_BINS),
    }

    return ModelResult(
        name=spec.name,
        model=model,
        calibrator=calibrator,
        val_probs=val_probs,
        test_probs=test_probs,
        threshold_default=0.50,
        threshold_max_f1=threshold_max_f1,
        threshold_high_precision=threshold_high_precision,
        threshold_cost_sensitive=threshold_cost_sensitive,
        threshold_high_precision_constraints_met=bool(high_precision_info["constraint_met"]),
        cost_summary=cost_summary,
        prob_metrics=prob_metrics,
        metrics_050=metrics_050,
        metrics_max_f1=metrics_max_f1,
        metrics_high_precision=metrics_high_precision,
        confusion_counts={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        confusion_rates={"tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr},
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        bundle_size_mb=_bundle_size_mb(preprocessor, model, calibrator),
    )


def _rolling_windows(total_rows: int) -> list[tuple[float, int, int]]:
    val_rows = max(int(total_rows * ROLLING_SELECTION_VAL_RATIO), ROLLING_SELECTION_MIN_VAL_ROWS)
    windows: list[tuple[float, int, int]] = []
    for cutoff_frac in ROLLING_SELECTION_CUTOFF_FRACS:
        train_end = int(total_rows * cutoff_frac)
        val_end = train_end + val_rows
        if train_end < ROLLING_SELECTION_MIN_TRAIN_ROWS or val_end > total_rows:
            continue
        windows.append((cutoff_frac, train_end, val_end))
    return windows


def _paired_bootstrap_delta(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric_fn_a: Callable[[np.ndarray, np.ndarray], float],
    metric_fn_b: Callable[[np.ndarray, np.ndarray], float] | None = None,
    *,
    n_bootstraps: int,
    seed: int,
) -> tuple[float, float, float, float, int]:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)
    n = len(y_true)
    fn_b = metric_fn_b or metric_fn_a
    observed = float(metric_fn_a(y_true, probs_a) - fn_b(y_true, probs_b))

    deltas = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        try:
            deltas[i] = metric_fn_a(yt, probs_a[idx]) - fn_b(yt, probs_b[idx])
        except ValueError:
            deltas[i] = np.nan

    valid = deltas[~np.isnan(deltas)]
    if len(valid) == 0:
        return observed, float("nan"), float("nan"), float("nan"), 0

    ci_lower = float(np.percentile(valid, 2.5))
    ci_upper = float(np.percentile(valid, 97.5))
    p_left = float(np.mean(valid <= 0.0))
    p_right = float(np.mean(valid >= 0.0))
    p_value = float(min(1.0, 2.0 * min(p_left, p_right)))
    return observed, ci_lower, ci_upper, p_value, int(len(valid))


def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    n_bootstraps: int,
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    from sklearn.metrics import average_precision_score

    metric_map: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "roc_auc": lambda yt, yp: float(roc_auc_score(yt, yp)),
        "pr_auc": lambda yt, yp: float(average_precision_score(yt, yp)),
        "f1_max_f1_threshold": lambda yt, yp: float(
            f1_score(yt, (yp >= threshold).astype(int), zero_division=0)
        ),
    }

    rows: list[dict[str, Any]] = []
    for metric_name, metric_fn in metric_map.items():
        try:
            point_estimate = float(metric_fn(y_true, y_prob))
        except ValueError:
            point_estimate = float("nan")

        values = np.empty(n_bootstraps)
        for i in range(n_bootstraps):
            idx = rng.randint(0, n, size=n)
            try:
                values[i] = metric_fn(y_true[idx], y_prob[idx])
            except ValueError:
                values[i] = np.nan

        valid = values[~np.isnan(values)]
        if len(valid):
            ci_lower = float(np.percentile(valid, 100 * alpha / 2))
            ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))
        else:
            ci_lower = float("nan")
            ci_upper = float("nan")

        rows.append(
            {
                "metric": metric_name,
                "point_estimate": point_estimate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_bootstraps": int(len(valid)),
                "alpha": alpha,
            }
        )
    return rows


def run_model_benchmark(
    *,
    reports_dir: Path = REPORTS_DIR,
    data_path: str | None = None,
    max_rows: int | None = None,
    n_bootstraps: int = BOOTSTRAP_N_ITERATIONS,
    temporal_buckets: int = TEMPORAL_STABILITY_BUCKETS,
) -> ModelBenchmarkOutputs:
    """Run deterministic multi-model benchmark and persist 16 benchmark tables."""
    set_global_seed(RANDOM_STATE)
    benchmark_dir = reports_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    logger.info("benchmark_start output_dir=%s", benchmark_dir)

    train_df, val_df, test_df, feature_cols = _prepare_data(data_path, max_rows)
    selection_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    y_test_np = test_df[TARGET_COL].astype(int).to_numpy()

    X_train = train_df[feature_cols]
    y_train_np = train_df[TARGET_COL].astype(int).to_numpy()
    X_val = val_df[feature_cols]
    y_val_np = val_df[TARGET_COL].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    fn_cost_val = compute_fn_cost_vector(val_df, fn_recovery_nights=FN_RECOVERY_NIGHTS)
    fn_cost_test = compute_fn_cost_vector(test_df, fn_recovery_nights=FN_RECOVERY_NIGHTS)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    specs = _model_specs()
    results: dict[str, ModelResult] = {}
    for spec in specs:
        logger.info("benchmark_fit model=%s", spec.name)
        results[spec.name] = _evaluate_model(
            spec=spec,
            preprocessor=preprocessor,
            X_train_t=X_train_t,
            y_train_np=y_train_np,
            X_val_t=X_val_t,
            y_val_np=y_val_np,
            fn_cost_val=fn_cost_val,
            X_test_t=X_test_t,
            y_test_np=y_test_np,
        )

    table_paths: dict[str, Path] = {}

    # 01: dataset split summary
    split_rows: list[dict[str, Any]] = []
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        dates = add_arrival_date(split_df)
        split_rows.append(
            {
                "split": split_name,
                "n_rows": int(len(split_df)),
                "date_min": str(dates.min()),
                "date_max": str(dates.max()),
                "target_rate": float(split_df[TARGET_COL].astype(int).mean()),
                "n_features": int(len(feature_cols)),
            }
        )
    df_01 = pd.DataFrame(split_rows)
    table_paths["01_dataset_split_summary"] = _write_table(
        df_01, benchmark_dir, "01_dataset_split_summary"
    )

    # 02: model specs
    df_02 = pd.DataFrame(
        [
            {
                "model": spec.name,
                "estimator_class": results[spec.name].model.__class__.__name__,
                "random_state": RANDOM_STATE,
                "calibration": "isotonic_on_validation",
                "threshold_policy_default": 0.50,
                "threshold_policy_max_f1": "from_validation_sweep",
                "threshold_policy_high_precision": "precision_max under recall>=0.20 and positive_rate>=0.05",
                "params_json": json.dumps(spec.params_for_report, sort_keys=True),
            }
            for spec in specs
        ]
    )
    table_paths["02_model_specs"] = _write_table(df_02, benchmark_dir, "02_model_specs")

    # 03: holdout probability metrics
    df_03 = pd.DataFrame(
        [
            {
                "model": name,
                **res.prob_metrics,
            }
            for name, res in results.items()
        ]
    ).sort_values(["pr_auc", "roc_auc", "model"], ascending=[False, False, True])
    table_paths["03_holdout_probability_metrics"] = _write_table(
        df_03, benchmark_dir, "03_holdout_probability_metrics"
    )

    # 04/05/06 threshold metrics
    def _threshold_table(
        name: str, accessor: str, threshold_accessor: str
    ) -> tuple[str, pd.DataFrame]:
        rows = []
        for model_name, res in results.items():
            metrics_dict = getattr(res, accessor)
            rows.append(
                {
                    "model": model_name,
                    "threshold": float(getattr(res, threshold_accessor)),
                    "precision": float(metrics_dict["precision"]),
                    "recall": float(metrics_dict["recall"]),
                    "f1": float(metrics_dict["f1"]),
                    "balanced_accuracy": float(metrics_dict["balanced_accuracy"]),
                    "roc_auc": float(metrics_dict["roc_auc"]),
                    "pr_auc": float(metrics_dict["pr_auc"]),
                }
            )
        return name, pd.DataFrame(rows).sort_values("model")

    for table_name, accessor, threshold_accessor in [
        ("04_holdout_threshold_metrics_050", "metrics_050", "threshold_default"),
        ("05_holdout_threshold_metrics_max_f1", "metrics_max_f1", "threshold_max_f1"),
        (
            "06_holdout_threshold_metrics_high_precision",
            "metrics_high_precision",
            "threshold_high_precision",
        ),
    ]:
        _, table_df = _threshold_table(table_name, accessor, threshold_accessor)
        table_paths[table_name] = _write_table(table_df, benchmark_dir, table_name)

    # 07 thresholds per model
    df_07 = pd.DataFrame(
        [
            {
                "model": name,
                "threshold_default": float(res.threshold_default),
                "threshold_max_f1": float(res.threshold_max_f1),
                "threshold_high_precision": float(res.threshold_high_precision),
                "threshold_cost_sensitive": float(res.threshold_cost_sensitive),
                "high_precision_constraints_met": bool(
                    res.threshold_high_precision_constraints_met
                ),
                "min_positive_rate": MIN_POSITIVE_RATE,
                "min_recall": MIN_RECALL_FOR_HIGH_PRECISION,
            }
            for name, res in results.items()
        ]
    ).sort_values("model")
    table_paths["07_thresholds_per_model"] = _write_table(
        df_07, benchmark_dir, "07_thresholds_per_model"
    )

    # 08 confusion counts
    df_08 = pd.DataFrame(
        [
            {"model": name, **res.confusion_counts, "policy": "max_f1"}
            for name, res in results.items()
        ]
    ).sort_values("model")
    table_paths["08_confusion_matrix_counts_per_model"] = _write_table(
        df_08, benchmark_dir, "08_confusion_matrix_counts_per_model"
    )

    # 09 confusion rates
    df_09 = pd.DataFrame(
        [
            {"model": name, **res.confusion_rates, "policy": "max_f1"}
            for name, res in results.items()
        ]
    ).sort_values("model")
    table_paths["09_confusion_matrix_rates_per_model"] = _write_table(
        df_09, benchmark_dir, "09_confusion_matrix_rates_per_model"
    )

    # 10 rolling-origin fold metrics
    rolling_rows: list[dict[str, Any]] = []
    windows = _rolling_windows(len(selection_df))
    for spec in specs:
        for fold_id, (cutoff_frac, train_end, val_end) in enumerate(windows, start=1):
            fold_train = selection_df.iloc[:train_end]
            fold_val = selection_df.iloc[train_end:val_end]
            X_tr = fold_train[feature_cols]
            y_tr = fold_train[TARGET_COL].astype(int).to_numpy()
            X_va = fold_val[feature_cols]
            y_va = fold_val[TARGET_COL].astype(int).to_numpy()

            fold_preprocessor = build_preprocessor()
            X_tr_t = fold_preprocessor.fit_transform(X_tr)
            X_va_t = fold_preprocessor.transform(X_va)
            model = spec.factory()
            model.fit(X_tr_t, y_tr)
            probs = model.predict_proba(X_va_t)[:, 1]
            rolling_rows.append(
                {
                    "model": spec.name,
                    "fold": fold_id,
                    "cutoff_frac": float(cutoff_frac),
                    "n_train": int(train_end),
                    "n_val": int(val_end - train_end),
                    "roc_auc": safe_roc_auc(y_va, probs),
                    "pr_auc": safe_pr_auc(y_va, probs),
                }
            )
    df_10 = pd.DataFrame(rolling_rows)
    table_paths["10_rolling_origin_fold_metrics"] = _write_table(
        df_10, benchmark_dir, "10_rolling_origin_fold_metrics"
    )

    # 11 rolling-origin summary
    if df_10.empty:
        df_11 = pd.DataFrame(
            columns=[
                "model",
                "folds",
                "roc_auc_mean",
                "roc_auc_std",
                "roc_auc_min",
                "roc_auc_max",
                "pr_auc_mean",
                "pr_auc_std",
                "pr_auc_min",
                "pr_auc_max",
            ]
        )
    else:
        grouped = df_10.groupby("model", as_index=False)
        df_11 = grouped.agg(
            folds=("fold", "count"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            pr_auc_min=("pr_auc", "min"),
            pr_auc_max=("pr_auc", "max"),
        )
    table_paths["11_rolling_origin_summary"] = _write_table(
        df_11, benchmark_dir, "11_rolling_origin_summary"
    )

    # 12 temporal stability by bucket
    test_with_date = test_df.copy()
    test_with_date["_arrival_date"] = add_arrival_date(test_with_date)
    test_with_date = test_with_date.sort_values("_arrival_date").reset_index(drop=True)
    bucket_indices = np.array_split(np.arange(len(test_with_date)), temporal_buckets)
    rows_12: list[dict[str, Any]] = []
    for model_name, res in results.items():
        threshold = res.threshold_max_f1
        probs_sorted = res.test_probs[test_with_date.index.to_numpy()]
        for bucket_id, idx in enumerate(bucket_indices, start=1):
            sub = test_with_date.iloc[idx]
            y_sub = sub[TARGET_COL].astype(int).to_numpy()
            p_sub = probs_sorted[idx]
            y_pred = (p_sub >= threshold).astype(int)
            rows_12.append(
                {
                    "model": model_name,
                    "bucket": bucket_id,
                    "n_rows": int(len(idx)),
                    "date_min": str(sub["_arrival_date"].min()),
                    "date_max": str(sub["_arrival_date"].max()),
                    "cancel_rate": float(np.mean(y_sub)),
                    "roc_auc": safe_roc_auc(y_sub, p_sub),
                    "pr_auc": safe_pr_auc(y_sub, p_sub),
                    "f1_at_max_f1_threshold": float(f1_score(y_sub, y_pred, zero_division=0)),
                }
            )
    df_12 = pd.DataFrame(rows_12).sort_values(["model", "bucket"])
    table_paths["12_temporal_stability_by_bucket"] = _write_table(
        df_12, benchmark_dir, "12_temporal_stability_by_bucket"
    )

    # 13 bootstrap confidence intervals
    rows_13: list[dict[str, Any]] = []
    for model_name, res in results.items():
        ci_rows = _bootstrap_metric_ci(
            y_test_np,
            res.test_probs,
            res.threshold_max_f1,
            n_bootstraps=n_bootstraps,
            alpha=BOOTSTRAP_ALPHA,
            seed=RANDOM_STATE,
        )
        for row in ci_rows:
            rows_13.append({"model": model_name, **row})
    df_13 = pd.DataFrame(rows_13).sort_values(["model", "metric"])
    table_paths["13_bootstrap_confidence_intervals"] = _write_table(
        df_13, benchmark_dir, "13_bootstrap_confidence_intervals"
    )

    # 14 paired significance vs champion
    ranking_source = df_03.sort_values(
        ["pr_auc", "roc_auc", "model"], ascending=[False, False, True]
    )
    champion_model = str(ranking_source.iloc[0]["model"])
    champion = results[champion_model]

    def _metric_f1_with_threshold(threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda yt, yp: float(f1_score(yt, (yp >= threshold).astype(int), zero_division=0))

    rows_14: list[dict[str, Any]] = []
    for model_name, res in results.items():
        if model_name == champion_model:
            continue
        metric_defs: list[
            tuple[
                str,
                Callable[[np.ndarray, np.ndarray], float],
                Callable[[np.ndarray, np.ndarray], float],
            ]
        ] = [
            ("roc_auc", safe_roc_auc, safe_roc_auc),
            ("pr_auc", safe_pr_auc, safe_pr_auc),
            (
                "f1_max_f1_threshold",
                _metric_f1_with_threshold(champion.threshold_max_f1),
                _metric_f1_with_threshold(res.threshold_max_f1),
            ),
        ]
        for metric_name, metric_a, metric_b in metric_defs:
            observed, ci_lo, ci_hi, p_value, n_valid = _paired_bootstrap_delta(
                y_test_np,
                champion.test_probs,
                res.test_probs,
                metric_a,
                metric_b,
                n_bootstraps=n_bootstraps,
                seed=RANDOM_STATE,
            )

            rows_14.append(
                {
                    "champion_model": champion_model,
                    "challenger_model": model_name,
                    "metric": metric_name,
                    "observed_delta": observed,
                    "delta_ci_lower": ci_lo,
                    "delta_ci_upper": ci_hi,
                    "p_value_two_sided": p_value,
                    "significant_at_05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
                    "n_bootstraps": n_valid,
                }
            )
    df_14 = pd.DataFrame(rows_14).sort_values(["challenger_model", "metric"])
    table_paths["14_paired_significance_vs_champion"] = _write_table(
        df_14, benchmark_dir, "14_paired_significance_vs_champion"
    )

    # 15 training/inference cost
    late_idx = np.where(
        pd.to_numeric(test_df["lead_time"], errors="coerce").fillna(np.inf).to_numpy()
        <= float(LATE_WINDOW_MAX_LEAD_DAYS)
    )[0]
    late_y = y_test_np[late_idx]
    df_15 = pd.DataFrame(
        [
            {
                "model": name,
                "fit_seconds": res.fit_seconds,
                "predict_seconds": res.predict_seconds,
                "bundle_size_mb": res.bundle_size_mb,
                "n_rows_train": int(len(train_df)),
                "n_rows_test": int(len(test_df)),
                "threshold_cost_sensitive": float(res.threshold_cost_sensitive),
                "test_total_cost_cost_sensitive": float(
                    (
                        ((res.test_probs >= res.threshold_cost_sensitive) & (y_test_np == 0)).sum()
                        * FP_INTERVENTION_COST
                    )
                    + fn_cost_test[
                        ((res.test_probs < res.threshold_cost_sensitive) & (y_test_np == 1))
                    ].sum()
                ),
                "test_total_cost_threshold_050": float(
                    (((res.test_probs >= 0.5) & (y_test_np == 0)).sum() * FP_INTERVENTION_COST)
                    + fn_cost_test[((res.test_probs < 0.5) & (y_test_np == 1))].sum()
                ),
                "test_late_window_f1_cost_sensitive": safe_threshold_metrics(
                    late_y,
                    res.test_probs[late_idx],
                    res.threshold_cost_sensitive,
                )["f1"]
                if len(late_idx)
                else None,
                "test_late_window_recall_cost_sensitive": safe_threshold_metrics(
                    late_y,
                    res.test_probs[late_idx],
                    res.threshold_cost_sensitive,
                )["recall"]
                if len(late_idx)
                else None,
            }
            for name, res in results.items()
        ]
    ).sort_values("model")
    table_paths["15_training_inference_cost"] = _write_table(
        df_15, benchmark_dir, "15_training_inference_cost"
    )

    # 16 rankings
    df_16 = (
        df_03.sort_values(["pr_auc", "roc_auc", "model"], ascending=[False, False, True])
        .reset_index(drop=True)
        .copy()
    )
    df_16.insert(0, "rank", np.arange(1, len(df_16) + 1))
    table_paths["16_rankings"] = _write_table(df_16, benchmark_dir, "16_rankings")

    logger.info(
        "benchmark_complete output_dir=%s champion_model=%s tables=%d",
        benchmark_dir,
        champion_model,
        len(table_paths),
    )
    return ModelBenchmarkOutputs(
        reports_dir=benchmark_dir,
        table_paths=table_paths,
        champion_model=champion_model,
    )
