"""Stratified 10-fold cross-validation across multiple algorithms.

Runs the same algorithm benchmark on both datasets so the thesis can report
a standard ML-textbook number alongside the project's chronological
rolling-origin estimate.

**Why a second CV scheme?**
The headline pipeline uses *chronological* splits (rolling-origin selection +
80/10/10 by ``arrival_date``) because the deployment story is "predict
tomorrow's bookings using yesterday's model". That's the right number to
report for deployment. **Stratified k-fold ignores time** — it shuffles rows
and preserves the positive-class ratio in every fold — which is the
standard algorithm-comparison protocol in published ML work. The two
numbers measure different things; both belong in the thesis.

Algorithms benchmarked
----------------------
- Dummy (most-frequent baseline)
- LogisticRegression
- DecisionTree (max_depth=5, balanced)
- GaussianNB
- GradientBoosting (sklearn)
- XGBoost
- LightGBM (skipped with a warning if not installed)

Outputs
-------
``reports/cv/<dataset>_stratified_10fold_folds.csv``
    One row per (model, fold) with discrimination + threshold-0.5 metrics.

``reports/cv/<dataset>_stratified_10fold_summary.json``
    Aggregated mean/std/95 % CI per model, plus dataset-level metadata
    (n_rows, positive_rate, seed, fold sizes).

Usage
-----
.. code-block:: bash

    python scripts/stratified_cv.py --dataset both
    python scripts/stratified_cv.py --dataset portugal --max-rows 20000
    python scripts/stratified_cv.py --dataset philippine --n-splits 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.config import (
    BOOKING_TIME_FEATURES,
    LEAKAGE_COLS,
    PH_BOOKING_TIME_FEATURES,
    PH_CATEGORICAL_COLS,
    PH_NUMERIC_COLS,
    PH_TARGET_COL,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COL,
)
from src.data.load import load_raw_data
from src.data.load_ph import load_ph_data
from src.features.build import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    add_derived_booking_features,
    cast_to_str,
)
from src.models.baselines import (
    train_decision_tree,
    train_dummy,
    train_logistic,
    train_naive_bayes,
)
from src.models.metrics import (
    evaluate_at_threshold,
    expected_calibration_error,
    safe_pr_auc,
    safe_roc_auc,
)
from src.models.train import is_lightgbm_available, train_gb, train_lgbm, train_xgb
from src.utils import configure_logging
from src.utils.reproducibility import set_global_seed
from src.utils.validate_data import clean_raw, clean_raw_ph

logger = logging.getLogger(__name__)

CV_REPORTS_DIR = PROJECT_ROOT / "reports" / "cv"

# Algorithms run on both datasets. Order matters: it becomes the table order
# in the summary and the legend order in the notebook plots.
ALGORITHM_ORDER: tuple[str, ...] = (
    "Dummy",
    "LogisticRegression",
    "DecisionTree",
    "GaussianNB",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
)


@dataclass
class DatasetConfig:
    """Everything needed to run k-fold on one dataset."""

    name: str
    feature_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    target_col: str
    loader_label: str  # human-readable for logs
    one_hot_min_frequency: float | int  # 0.01 (Portugal) or 1 (PH small-N)


PORTUGAL_CONFIG = DatasetConfig(
    name="portugal",
    feature_cols=list(BOOKING_TIME_FEATURES),
    categorical_cols=list(CATEGORICAL_COLS),
    numeric_cols=list(NUMERIC_COLS),
    target_col=TARGET_COL,
    loader_label="Portugal (data/hotel_bookings.csv)",
    one_hot_min_frequency=0.01,
)

PHILIPPINE_CONFIG = DatasetConfig(
    name="philippine",
    feature_cols=list(PH_BOOKING_TIME_FEATURES),
    categorical_cols=list(PH_CATEGORICAL_COLS),
    numeric_cols=list(PH_NUMERIC_COLS),
    target_col=PH_TARGET_COL,
    loader_label="Philippine (data/Punta_Villa_Resort_PH_Dataset.csv)",
    one_hot_min_frequency=1,
)


def _sanitise_for_json(obj: Any) -> Any:
    """Mirror src/pipelines/train.py — replace NaN/Inf with None for JSON."""
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


def _build_preprocessor(cfg: DatasetConfig) -> Pipeline:
    """Build a fresh preprocessor per fold (no fit state leaks across folds)."""
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "to_str",
                FunctionTransformer(cast_to_str, validate=False, feature_names_out="one-to-one"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    sparse_output=False,
                    min_frequency=cfg.one_hot_min_frequency,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    column_transformer = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipeline, cfg.categorical_cols),
            ("numeric", num_pipeline, cfg.numeric_cols),
        ]
    )
    return Pipeline(steps=[("encode", column_transformer)])


def _load_and_prepare(cfg: DatasetConfig, max_rows: int | None) -> pd.DataFrame:
    """Load → clean → derive features → return DataFrame restricted to model columns + target."""
    if cfg.name == "portugal":
        raw = load_raw_data()
        cleaned, _ = clean_raw(raw)
        cleaned = cleaned.drop(columns=[col for col in LEAKAGE_COLS if col in cleaned.columns])
        df = cleaned[cfg.feature_cols + [cfg.target_col]].copy()
    elif cfg.name == "philippine":
        raw = load_ph_data()
        cleaned, _ = clean_raw_ph(raw)
        # PH cleaning already calls add_derived_booking_features internally and
        # produces every PH_BOOKING_TIME_FEATURES column, so we just slice.
        df = cleaned[cfg.feature_cols + [cfg.target_col]].copy()
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    # Portugal: derive features here (clean_raw doesn't).
    if cfg.name == "portugal":
        derived = add_derived_booking_features(df)
        # add_derived_booking_features re-orders; re-slice + reattach target.
        df = derived[cfg.feature_cols].copy()
        df[cfg.target_col] = cleaned[cfg.target_col].to_numpy()

    df = df.dropna(subset=[cfg.target_col])
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()

    return df.reset_index(drop=True)


@dataclass
class AlgorithmSpec:
    """How to fit one algorithm; abstracts away the per-family quirks."""

    name: str
    requires_scale_pos_weight: bool = False
    skip_if: str = ""  # populated at runtime ("lightgbm not installed", etc.)
    fitter: Any = field(default=None)


def _build_algorithm_specs() -> list[AlgorithmSpec]:
    specs = [
        AlgorithmSpec("Dummy", fitter=lambda X, y, **_: train_dummy(X, y)),
        AlgorithmSpec("LogisticRegression", fitter=lambda X, y, **_: train_logistic(X, y)),
        AlgorithmSpec("DecisionTree", fitter=lambda X, y, **_: train_decision_tree(X, y)),
        AlgorithmSpec("GaussianNB", fitter=lambda X, y, **_: train_naive_bayes(X, y)),
        AlgorithmSpec("GradientBoosting", fitter=lambda X, y, **_: train_gb(X, y)),
        AlgorithmSpec(
            "XGBoost",
            requires_scale_pos_weight=True,
            fitter=lambda X, y, scale_pos_weight=None, **_: train_xgb(
                X, y, scale_pos_weight=scale_pos_weight
            ),
        ),
    ]
    if is_lightgbm_available():
        specs.append(AlgorithmSpec("LightGBM", fitter=lambda X, y, **_: train_lgbm(X, y)))
    else:
        # Still record it so the summary documents the skip.
        specs.append(AlgorithmSpec("LightGBM", skip_if="lightgbm not installed"))
    return specs


def _scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    arr = np.asarray(y).astype(int)
    pos = float(np.sum(arr == 1))
    neg = float(np.sum(arr == 0))
    return (neg / pos) if pos > 0 else 1.0


def _fold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Per-fold metric bundle: discrimination + calibration + @threshold."""
    y_true_int = y_true.astype(int)
    at_thr = evaluate_at_threshold(y_true_int, y_prob, threshold)
    # Brier requires probabilities; ECE needs the bin count from config (we use 10).
    return {
        "roc_auc": safe_roc_auc(y_true_int, y_prob),
        "pr_auc": safe_pr_auc(y_true_int, y_prob),
        "brier": float(brier_score_loss(y_true_int, y_prob))
        if len(np.unique(y_true_int)) >= 2
        else float("nan"),
        "ece": expected_calibration_error(y_true_int, y_prob, bins=10),
        "precision_at_05": at_thr["precision"],
        "recall_at_05": at_thr["recall"],
        "f1_at_05": at_thr["f1"],
        "balanced_accuracy_at_05": at_thr["balanced_accuracy"],
    }


def _run_dataset(
    cfg: DatasetConfig,
    n_splits: int,
    max_rows: int | None,
) -> dict[str, Any]:
    """Run stratified k-fold on one dataset across all algorithms."""
    logger.info("cv_start dataset=%s n_splits=%d", cfg.name, n_splits)
    df = _load_and_prepare(cfg, max_rows=max_rows)
    n_total = len(df)
    n_pos = int(df[cfg.target_col].sum())
    pos_rate = n_pos / n_total if n_total else 0.0
    logger.info(
        "cv_dataset_ready name=%s n_rows=%d positive_rate=%.4f",
        cfg.name,
        n_total,
        pos_rate,
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(df[cfg.feature_cols], df[cfg.target_col]))

    specs = _build_algorithm_specs()
    fold_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for spec in specs:
        if spec.skip_if:
            logger.warning("cv_algorithm_skipped name=%s reason=%s", spec.name, spec.skip_if)
            skipped.append({"algorithm": spec.name, "reason": spec.skip_if})
            continue
        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            train_slice = df.iloc[train_idx]
            test_slice = df.iloc[test_idx]
            X_train = train_slice[cfg.feature_cols]
            y_train = train_slice[cfg.target_col]
            X_test = test_slice[cfg.feature_cols]
            y_test = test_slice[cfg.target_col]

            preprocessor = _build_preprocessor(cfg)
            try:
                X_train_t = preprocessor.fit_transform(X_train)
                X_test_t = preprocessor.transform(X_test)
            except Exception as exc:  # noqa: BLE001 — keep loop going on bad fold
                logger.warning(
                    "cv_preprocess_failed algo=%s fold=%d err=%s", spec.name, fold_id, exc
                )
                continue

            t0 = time.perf_counter()
            kwargs: dict[str, Any] = {}
            if spec.requires_scale_pos_weight:
                kwargs["scale_pos_weight"] = _scale_pos_weight(y_train)
            try:
                model = spec.fitter(X_train_t, y_train, **kwargs)
                proba = model.predict_proba(X_test_t)[:, 1]
            except Exception as exc:  # noqa: BLE001 — log and continue
                logger.warning("cv_fit_failed algo=%s fold=%d err=%s", spec.name, fold_id, exc)
                continue
            fit_seconds = time.perf_counter() - t0

            metrics = _fold_metrics(y_test.to_numpy(), proba)
            fold_rows.append(
                {
                    "dataset": cfg.name,
                    "algorithm": spec.name,
                    "fold": fold_id,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_pos_rate": float(np.mean(y_train.to_numpy())),
                    "test_pos_rate": float(np.mean(y_test.to_numpy())),
                    "fit_seconds": float(fit_seconds),
                    **metrics,
                }
            )
            logger.info(
                "cv_fold_complete dataset=%s algo=%s fold=%d/%d pr_auc=%.4f roc_auc=%.4f time=%.1fs",
                cfg.name,
                spec.name,
                fold_id,
                n_splits,
                metrics["pr_auc"],
                metrics["roc_auc"],
                fit_seconds,
            )

    fold_df = pd.DataFrame(fold_rows)
    summary = _summarise(fold_df, dataset_name=cfg.name)
    summary.update(
        {
            "dataset_meta": {
                "name": cfg.name,
                "loader": cfg.loader_label,
                "n_rows_used": int(n_total),
                "n_positive": n_pos,
                "positive_rate": float(pos_rate),
                "feature_cols": cfg.feature_cols,
                "n_splits": int(n_splits),
                "stratified": True,
                "shuffle": True,
                "random_state": int(RANDOM_STATE),
            },
            "skipped_algorithms": skipped,
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "calibration_note": (
                "Probabilities are raw model outputs (no isotonic calibration per fold). "
                "Brier/ECE reflect the raw scores; discrimination metrics (ROC-AUC, "
                "PR-AUC) are calibration-invariant."
            ),
        }
    )
    return {"fold_df": fold_df, "summary": summary}


def _ci_95(values: np.ndarray) -> tuple[float, float]:
    """Normal-approx 95 % CI on the mean (n_splits = 10 is small enough that
    bootstrap would barely move the bounds, so the closed-form ±1.96·sem
    keeps the JSON deterministic)."""
    arr = values[~np.isnan(values)]
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1)) / float(np.sqrt(len(arr)))
    return (mean - 1.96 * sem, mean + 1.96 * sem)


def _summarise(fold_df: pd.DataFrame, *, dataset_name: str) -> dict[str, Any]:
    """Aggregate per-fold metrics into mean/std/CI per algorithm."""
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "brier",
        "ece",
        "precision_at_05",
        "recall_at_05",
        "f1_at_05",
        "balanced_accuracy_at_05",
    ]
    per_algo: list[dict[str, Any]] = []
    for algo in ALGORITHM_ORDER:
        sub = fold_df[fold_df["algorithm"] == algo]
        if sub.empty:
            continue
        algo_block: dict[str, Any] = {
            "algorithm": algo,
            "folds_evaluated": int(len(sub)),
            "fit_seconds_mean": float(sub["fit_seconds"].mean()),
        }
        for metric in metric_cols:
            values = sub[metric].to_numpy()
            ci_low, ci_high = _ci_95(values)
            algo_block[metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        per_algo.append(algo_block)

    # Rank by mean PR-AUC (primary), then ROC-AUC (secondary).
    ranking = sorted(
        per_algo,
        key=lambda b: (-b["pr_auc"]["mean"], -b["roc_auc"]["mean"]),
    )
    return {
        "dataset": dataset_name,
        "primary_metric": "pr_auc",
        "ranking": [
            {
                "rank": idx + 1,
                "algorithm": entry["algorithm"],
                "pr_auc_mean": entry["pr_auc"]["mean"],
                "pr_auc_std": entry["pr_auc"]["std"],
                "roc_auc_mean": entry["roc_auc"]["mean"],
                "roc_auc_std": entry["roc_auc"]["std"],
            }
            for idx, entry in enumerate(ranking)
        ],
        "per_algorithm": per_algo,
    }


def _persist(dataset_name: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    CV_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = CV_REPORTS_DIR / f"{dataset_name}_stratified_10fold_folds.csv"
    summary_path = CV_REPORTS_DIR / f"{dataset_name}_stratified_10fold_summary.json"
    payload["fold_df"].to_csv(fold_path, index=False)
    _save_json(summary_path, payload["summary"])
    return fold_path, summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("portugal", "philippine", "both"),
        default="both",
        help="Which dataset(s) to evaluate. Default: both.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of stratified folds (default: 10).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows (useful for smoke tests). Default: use all rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    set_global_seed(RANDOM_STATE)

    targets: list[DatasetConfig] = []
    if args.dataset in ("portugal", "both"):
        targets.append(PORTUGAL_CONFIG)
    if args.dataset in ("philippine", "both"):
        targets.append(PHILIPPINE_CONFIG)

    for cfg in targets:
        try:
            result = _run_dataset(cfg, n_splits=args.n_splits, max_rows=args.max_rows)
        except FileNotFoundError as exc:
            logger.warning("cv_dataset_skipped name=%s reason=%s", cfg.name, exc)
            continue
        fold_path, summary_path = _persist(cfg.name, result)
        logger.info(
            "cv_persisted dataset=%s folds=%s summary=%s",
            cfg.name,
            fold_path,
            summary_path,
        )
        # Console-friendly recap of the leaderboard.
        ranking = result["summary"]["ranking"]
        logger.info("cv_ranking dataset=%s primary_metric=pr_auc", cfg.name)
        for entry in ranking:
            logger.info(
                "  %d. %-18s pr_auc=%.4f ± %.4f   roc_auc=%.4f ± %.4f",
                entry["rank"],
                entry["algorithm"],
                entry["pr_auc_mean"],
                entry["pr_auc_std"],
                entry["roc_auc_mean"],
                entry["roc_auc_std"],
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
