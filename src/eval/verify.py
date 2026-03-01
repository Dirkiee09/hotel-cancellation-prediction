"""Deterministic verification of Gradient Boosting vs XGBoost for cancellation modeling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR,
    BOOKING_TIME_FEATURES,
    CALIBRATION_METHOD,
    LEAKAGE_COLS,
    TARGET_COL,
    THRESHOLD_STEP,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data.load import load_raw_data
from src.features.build import add_arrival_date, build_preprocessor
from src.models.metrics import safe_pr_auc, safe_roc_auc
from src.utils.reproducibility import set_global_seed
from src.utils.thresholds import select_max_f1_threshold, threshold_sweep
from src.utils.validate_data import clean_raw, validate_raw

SEED = 42

NOTEBOOK_LOCATION_ROWS = [
    {
        "Topic": "Holdout split construction (index/date cutoff) definition",
        "main.ipynb": "cell 6; notebooks/main.ipynb:256,258,262",
        "analysis.ipynb": "cell 8; notebooks/analysis.ipynb:354,356,360",
    },
    {
        "Topic": "Holdout train/test usage (feature matrices from split frames)",
        "main.ipynb": "cell 7; notebooks/main.ipynb:305,307",
        "analysis.ipynb": "cell 11; notebooks/analysis.ipynb:460,462",
    },
    {
        "Topic": "TimeSeriesSplit configuration definition",
        "main.ipynb": "not present as explicit setup cell",
        "analysis.ipynb": "cell 37; notebooks/analysis.ipynb:2320,2322",
    },
    {
        "Topic": "TimeSeriesSplit usage for CV",
        "main.ipynb": "optional only if cv_results exists in globals() (cell 17)",
        "analysis.ipynb": "cell 38; notebooks/analysis.ipynb:2429",
    },
    {
        "Topic": "Preprocessing pipeline definition",
        "main.ipynb": "cell 10; notebooks/main.ipynb:385,386",
        "analysis.ipynb": "cell 14; notebooks/analysis.ipynb:540,541",
    },
    {
        "Topic": "Preprocessing pipeline usage in train/eval",
        "main.ipynb": "cells 13/16/21; notebooks/main.ipynb:444,527,678",
        "analysis.ipynb": "cells 17/21/38/44; notebooks/analysis.ipynb:599,682,889,2421,2719",
    },
    {
        "Topic": "Gradient Boosting & XGBoost hyperparameters (fixed, not searched)",
        "main.ipynb": "cell 15; notebooks/main.ipynb:595,597",
        "analysis.ipynb": "cell 20; notebooks/analysis.ipynb:806,808",
    },
    {
        "Topic": "Thresholding logic definition (compute_thresholds)",
        "main.ipynb": "cell 17; notebooks/main.ipynb:811-862",
        "analysis.ipynb": "cell 22; notebooks/analysis.ipynb:994-1045",
    },
    {
        "Topic": "Thresholding logic usage (policy/reliability/report)",
        "main.ipynb": "cells 19/21; notebooks/main.ipynb:1267,1472",
        "analysis.ipynb": "cells 27/29/51; notebooks/analysis.ipynb:1150,1214,3252",
    },
    {
        "Topic": "best_cv_tscv / best_cv_skf definition and usage",
        "main.ipynb": "defined cell 17 (793-800), used cell 21 (1442-1508)",
        "analysis.ipynb": "defined+used cell 49 (2934-2964)",
    },
]


@dataclass
class PreparedData:
    df_clean: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    split_idx: int
    split_date: pd.Timestamp


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def _format_table(df: pd.DataFrame, round_to: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(float).round(round_to)
    return out


def _to_markdown(df: pd.DataFrame, round_to: int = 4) -> str:
    formatted = _format_table(df, round_to=round_to)
    try:
        return formatted.to_markdown(index=False)
    except Exception:
        return "```text\n" + formatted.to_string(index=False) + "\n```"


def _prepare_data() -> PreparedData:
    raw = load_raw_data()
    cleaned, _ = clean_raw(raw)
    validation = validate_raw(cleaned)
    if not validation.passed:
        raise ValueError(f"Data validation failed: {validation.messages}")

    cleaned["arrival_date"] = add_arrival_date(cleaned)
    if cleaned["arrival_date"].isna().any():
        cleaned = cleaned.dropna(subset=["arrival_date"]).copy()
    cleaned = cleaned.sort_values("arrival_date").reset_index(drop=True)

    leakage_cols = [c for c in LEAKAGE_COLS if c in cleaned.columns]
    df_clean = cleaned.drop(columns=leakage_cols).copy()

    required_cols = BOOKING_TIME_FEATURES + [TARGET_COL, "arrival_date"]
    missing_cols = [c for c in required_cols if c not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after cleanup: {missing_cols}")
    df_clean = df_clean[required_cols].copy()

    train_end = int(len(df_clean) * TRAIN_RATIO)
    val_end = train_end + int(len(df_clean) * VAL_RATIO)
    train_df = df_clean.iloc[:val_end].copy()
    test_df = df_clean.iloc[val_end:].copy()
    split_date = pd.Timestamp(train_df["arrival_date"].max())

    X_train = train_df[BOOKING_TIME_FEATURES].copy()
    y_train = train_df[TARGET_COL].astype(int).copy()
    X_test = test_df[BOOKING_TIME_FEATURES].copy()
    y_test = test_df[TARGET_COL].astype(int).copy()

    return PreparedData(
        df_clean=df_clean,
        train_df=train_df,
        test_df=test_df,
        feature_cols=BOOKING_TIME_FEATURES.copy(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        split_idx=val_end,
        split_date=split_date,
    )


def _model_factories(seed: int = SEED) -> dict[str, Callable[..., object]]:
    def gb(**overrides):
        cfg = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": seed,
        }
        cfg.update(overrides)
        return GradientBoostingClassifier(**cfg)

    def xgb(**overrides):
        cfg = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": seed,
            "n_jobs": 1,
            "eval_metric": "logloss",
        }
        cfg.update(overrides)
        return XGBClassifier(**cfg)

    return {
        "Gradient Boosting": gb,
        "XGBoost": xgb,
    }


def _evaluate_holdout(prepared: PreparedData, seed: int = SEED):
    factories = _model_factories(seed)
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(prepared.X_train)
    X_test_t = preprocessor.transform(prepared.X_test)

    y_train_np = prepared.y_train.to_numpy()
    y_test_np = prepared.y_test.to_numpy()

    rows: list[dict] = []
    probabilities: dict[str, dict[str, np.ndarray]] = {}

    for model_name, model_fn in factories.items():
        model = model_fn()
        model.fit(X_train_t, y_train_np)
        p_train = model.predict_proba(X_train_t)[:, 1]
        p_test = model.predict_proba(X_test_t)[:, 1]
        m = _metrics(y_test_np, p_test, threshold=0.5)

        cm = confusion_matrix(y_test_np, (p_test >= 0.5).astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "Model": model_name,
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
                "F1@0.50": m["f1"],
                "Precision@0.50": m["precision"],
                "Recall@0.50": m["recall"],
                "Brier": m["brier_score"],
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )
        probabilities[model_name] = {"train": p_train, "test": p_test}

    holdout_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return holdout_df, probabilities


def _evaluate_rolling_origin(
    prepared: PreparedData,
    cutoff_fracs: list[float] | None = None,
    test_window_frac: float = 0.10,
    seed: int = SEED,
):
    if cutoff_fracs is None:
        cutoff_fracs = [0.50, 0.58, 0.66, 0.74, 0.82]

    factories = _model_factories(seed)
    df = prepared.df_clean
    n = len(df)
    test_window = int(n * test_window_frac)
    test_window = max(test_window, 2000)

    rows: list[dict] = []
    for frac in cutoff_fracs:
        train_end = int(n * frac)
        test_end = min(n, train_end + test_window)
        if train_end < 5000 or (test_end - train_end) < 1000:
            continue

        train_slice = df.iloc[:train_end].copy()
        test_slice = df.iloc[train_end:test_end].copy()
        X_train = train_slice[prepared.feature_cols]
        y_train = train_slice[TARGET_COL].astype(int).to_numpy()
        X_test = test_slice[prepared.feature_cols]
        y_test = test_slice[TARGET_COL].astype(int).to_numpy()

        preprocessor = build_preprocessor()
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        cutoff_date = pd.Timestamp(train_slice["arrival_date"].max())
        test_start = pd.Timestamp(test_slice["arrival_date"].min())
        test_end_date = pd.Timestamp(test_slice["arrival_date"].max())

        for model_name, model_fn in factories.items():
            model = model_fn()
            model.fit(X_train_t, y_train)
            p_test = model.predict_proba(X_test_t)[:, 1]
            m = _metrics(y_test, p_test, threshold=0.5)
            rows.append(
                {
                    "Model": model_name,
                    "cutoff_frac": frac,
                    "cutoff_date": str(cutoff_date.date()),
                    "test_start_date": str(test_start.date()),
                    "test_end_date": str(test_end_date.date()),
                    "n_train": int(len(train_slice)),
                    "n_test": int(len(test_slice)),
                    "ROC-AUC": m["roc_auc"],
                    "PR-AUC": m["pr_auc"],
                    "F1@0.50": m["f1"],
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("Model")[["ROC-AUC", "PR-AUC", "F1@0.50"]].agg(["mean", "std"]).reset_index()
    )
    summary.columns = [
        "Model",
        "ROC-AUC mean",
        "ROC-AUC std",
        "PR-AUC mean",
        "PR-AUC std",
        "F1@0.50 mean",
        "F1@0.50 std",
    ]
    summary = summary.sort_values("ROC-AUC mean", ascending=False).reset_index(drop=True)
    return detail, summary


def _evaluate_tscv(prepared: PreparedData, n_splits: int = 5, seed: int = SEED):
    factories = _model_factories(seed)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = prepared.X_train.reset_index(drop=True)
    y = prepared.y_train.reset_index(drop=True).astype(int)
    dates = prepared.train_df["arrival_date"].reset_index(drop=True)

    rows: list[dict] = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx].to_numpy()
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx].to_numpy()

        preprocessor = build_preprocessor()
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_val_t = preprocessor.transform(X_val)

        for model_name, model_fn in factories.items():
            model = model_fn()
            model.fit(X_tr_t, y_tr)
            p_val = model.predict_proba(X_val_t)[:, 1]
            m = _metrics(y_val, p_val, threshold=0.5)
            rows.append(
                {
                    "Model": model_name,
                    "fold": fold,
                    "train_end_date": str(pd.Timestamp(dates.iloc[tr_idx].max()).date()),
                    "val_start_date": str(pd.Timestamp(dates.iloc[val_idx].min()).date()),
                    "val_end_date": str(pd.Timestamp(dates.iloc[val_idx].max()).date()),
                    "ROC-AUC": m["roc_auc"],
                    "PR-AUC": m["pr_auc"],
                    "Precision@0.50": m["precision"],
                    "Recall@0.50": m["recall"],
                    "F1@0.50": m["f1"],
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("Model")[["ROC-AUC", "PR-AUC", "Precision@0.50", "Recall@0.50", "F1@0.50"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "Model",
        "ROC-AUC mean",
        "ROC-AUC std",
        "PR-AUC mean",
        "PR-AUC std",
        "Precision mean",
        "Precision std",
        "Recall mean",
        "Recall std",
        "F1 mean",
        "F1 std",
    ]
    summary = summary.sort_values("ROC-AUC mean", ascending=False).reset_index(drop=True)
    return detail, summary, tscv


def _evaluate_grid(
    prepared: PreparedData,
    model_name: str,
    model_builder: Callable[..., object],
    grid: list[dict],
    n_splits: int = 5,
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = prepared.X_train.reset_index(drop=True)
    y = prepared.y_train.reset_index(drop=True).astype(int)

    rows: list[dict] = []
    for params in grid:
        fold_metrics: list[tuple[float, float, float]] = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx].to_numpy()
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx].to_numpy()

            preprocessor = build_preprocessor()
            X_tr_t = preprocessor.fit_transform(X_tr)
            X_val_t = preprocessor.transform(X_val)

            model = model_builder(**params)
            model.fit(X_tr_t, y_tr)
            p_val = model.predict_proba(X_val_t)[:, 1]
            m = _metrics(y_val, p_val, threshold=0.5)
            fold_metrics.append((m["roc_auc"], m["pr_auc"], m["f1"]))

        arr = np.asarray(fold_metrics, dtype=float)
        rows.append(
            {
                "Model": model_name,
                "params": json.dumps(params, sort_keys=True),
                "Mean ROC-AUC": float(np.nanmean(arr[:, 0])),
                "Std ROC-AUC": float(np.nanstd(arr[:, 0])),
                "Mean PR-AUC": float(np.nanmean(arr[:, 1])),
                "Mean F1": float(np.nanmean(arr[:, 2])),
            }
        )

    grid_df = pd.DataFrame(rows).sort_values("Mean ROC-AUC", ascending=False).reset_index(drop=True)
    return grid_df


def _evaluate_curves(
    prepared: PreparedData,
    n_splits: int = 3,
    seed: int = SEED,
):
    factories = _model_factories(seed)

    def evaluate_param_curve(model_name: str, varying: str, values: list[int], fixed: dict):
        builder = factories[model_name]
        rows: list[dict] = []
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = prepared.X_train.reset_index(drop=True)
        y = prepared.y_train.reset_index(drop=True).astype(int)

        for value in values:
            params = dict(fixed)
            params[varying] = value
            fold_roc: list[float] = []
            for tr_idx, val_idx in tscv.split(X):
                X_tr = X.iloc[tr_idx]
                y_tr = y.iloc[tr_idx].to_numpy()
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx].to_numpy()
                preprocessor = build_preprocessor()
                X_tr_t = preprocessor.fit_transform(X_tr)
                X_val_t = preprocessor.transform(X_val)
                model = builder(**params)
                model.fit(X_tr_t, y_tr)
                p_val = model.predict_proba(X_val_t)[:, 1]
                fold_roc.append(safe_roc_auc(y_val, p_val))
            rows.append(
                {
                    "Model": model_name,
                    "varying": varying,
                    "value": value,
                    "Mean ROC-AUC": float(np.nanmean(fold_roc)),
                }
            )
        return pd.DataFrame(rows)

    n_estimators_values = [50, 100, 200, 300]
    max_depth_values = [2, 3, 5, 7]

    curves = []
    curves.append(
        evaluate_param_curve(
            "Gradient Boosting",
            "n_estimators",
            n_estimators_values,
            fixed={"max_depth": 5},
        )
    )
    curves.append(
        evaluate_param_curve(
            "XGBoost",
            "n_estimators",
            n_estimators_values,
            fixed={"max_depth": 5, "learning_rate": 0.1},
        )
    )
    curves.append(
        evaluate_param_curve(
            "Gradient Boosting",
            "max_depth",
            max_depth_values,
            fixed={"n_estimators": 100},
        )
    )
    curves.append(
        evaluate_param_curve(
            "XGBoost",
            "max_depth",
            max_depth_values,
            fixed={"n_estimators": 100, "learning_rate": 0.1},
        )
    )

    return pd.concat(curves, ignore_index=True)


def _calibration_and_threshold_experiment(
    prepared: PreparedData,
    holdout_probs: dict[str, dict[str, np.ndarray]],
    seed: int = SEED,
):
    factories = _model_factories(seed)
    y_test = prepared.y_test.to_numpy()
    X_test = prepared.X_test

    # Validation window for threshold selection: last 20% of train timeline.
    val_start = int(len(prepared.train_df) * 0.8)
    train_sub = prepared.train_df.iloc[:val_start].copy()
    val_sub = prepared.train_df.iloc[val_start:].copy()

    X_subtrain = train_sub[prepared.feature_cols]
    y_subtrain = train_sub[TARGET_COL].astype(int).to_numpy()
    X_val = val_sub[prepared.feature_cols]
    y_val = val_sub[TARGET_COL].astype(int).to_numpy()

    rows: list[dict] = []
    calib_points: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for model_name, model_fn in factories.items():
        # Uncalibrated baseline
        preprocessor = build_preprocessor()
        X_sub_t = preprocessor.fit_transform(X_subtrain)
        X_val_t = preprocessor.transform(X_val)
        X_test_t = preprocessor.transform(X_test)
        model = model_fn()
        model.fit(X_sub_t, y_subtrain)
        p_val_uncal = model.predict_proba(X_val_t)[:, 1]
        p_test_uncal = model.predict_proba(X_test_t)[:, 1]

        sweep_uncal = threshold_sweep(y_val, p_val_uncal, step=THRESHOLD_STEP)
        best_uncal = select_max_f1_threshold(sweep_uncal)
        thr_uncal = float(best_uncal["threshold"])
        f1_uncal = float(f1_score(y_test, (p_test_uncal >= thr_uncal).astype(int), zero_division=0))

        # CalibratedClassifierCV experiment (calibration via CV on subtrain only).
        calibrated = CalibratedClassifierCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("classifier", model_fn()),
                ]
            ),
            method=CALIBRATION_METHOD,
            cv=TimeSeriesSplit(n_splits=3),
        )
        calibrated.fit(X_subtrain, y_subtrain)
        p_val_cal = calibrated.predict_proba(X_val)[:, 1]
        p_test_cal = calibrated.predict_proba(X_test)[:, 1]
        sweep_cal = threshold_sweep(y_val, p_val_cal, step=THRESHOLD_STEP)
        best_cal = select_max_f1_threshold(sweep_cal)
        thr_cal = float(best_cal["threshold"])
        f1_cal = float(f1_score(y_test, (p_test_cal >= thr_cal).astype(int), zero_division=0))

        rows.append(
            {
                "Model": model_name,
                "Brier (uncalibrated)": float(brier_score_loss(y_test, p_test_uncal)),
                "Brier (calibrated-CV)": float(brier_score_loss(y_test, p_test_cal)),
                "Threshold@maxF1 on validation (uncalibrated)": thr_uncal,
                "Threshold@maxF1 on validation (calibrated-CV)": thr_cal,
                "Test F1 using validation-selected threshold (uncalibrated)": f1_uncal,
                "Test F1 using validation-selected threshold (calibrated-CV)": f1_cal,
            }
        )

        frac_uncal, mean_uncal = calibration_curve(
            y_test, holdout_probs[model_name]["test"], n_bins=10, strategy="quantile"
        )
        frac_cal, mean_cal = calibration_curve(y_test, p_test_cal, n_bins=10, strategy="quantile")
        calib_points[model_name] = {
            "uncalibrated": (mean_uncal, frac_uncal),
            "calibrated_cv": (mean_cal, frac_cal),
        }

    return pd.DataFrame(rows), calib_points


def _evaluate_time_bucket_drift(
    prepared: PreparedData,
    holdout_probs: dict[str, dict[str, np.ndarray]],
    n_buckets: int = 5,
):
    test_df = prepared.test_df.reset_index(drop=True).copy()
    y_test = prepared.y_test.reset_index(drop=True).to_numpy()
    bucket_indices = np.array_split(np.arange(len(test_df)), n_buckets)

    rows: list[dict] = []
    for bucket_id, idx in enumerate(bucket_indices, start=1):
        bucket_df = test_df.iloc[idx]
        y_bucket = y_test[idx]
        start_date = pd.Timestamp(bucket_df["arrival_date"].min())
        end_date = pd.Timestamp(bucket_df["arrival_date"].max())
        for model_name, model_probs in holdout_probs.items():
            p_bucket = model_probs["test"][idx]
            m = _metrics(y_bucket, p_bucket, threshold=0.5)
            rows.append(
                {
                    "bucket": bucket_id,
                    "bucket_start": str(start_date.date()),
                    "bucket_end": str(end_date.date()),
                    "Model": model_name,
                    "n_rows": int(len(idx)),
                    "cancel_rate": float(np.mean(y_bucket)),
                    "ROC-AUC": m["roc_auc"],
                    "PR-AUC": m["pr_auc"],
                    "F1@0.50": m["f1"],
                }
            )
    return pd.DataFrame(rows)


def _plot_holdout_curves(
    prepared: PreparedData,
    holdout_probs: dict[str, dict[str, np.ndarray]],
    fig_dir: Path,
):
    y_test = prepared.y_test.to_numpy()

    roc_path = fig_dir / "verification_holdout_roc_curve.png"
    pr_path = fig_dir / "verification_holdout_pr_curve.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model_probs in holdout_probs.items():
        fpr, tpr, _ = roc_curve(y_test, model_probs["test"])
        auc_val = safe_roc_auc(y_test, model_probs["test"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Holdout ROC Curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, model_probs in holdout_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, model_probs["test"])
        pr_auc = safe_pr_auc(y_test, model_probs["test"])
        ax.plot(recall, precision, linewidth=2, label=f"{model_name} (PR-AUC={pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Holdout Precision-Recall Curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=180)
    plt.close(fig)

    return roc_path, pr_path


def _plot_metric_vs_cutoff(rolling_detail: pd.DataFrame, fig_dir: Path) -> Path:
    path = fig_dir / "verification_metric_vs_cutoff.png"
    tmp = rolling_detail.copy()
    tmp["cutoff_date"] = pd.to_datetime(tmp["cutoff_date"])
    tmp = tmp.sort_values("cutoff_date")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for model_name in tmp["Model"].unique():
        sub = tmp[tmp["Model"] == model_name]
        axes[0].plot(sub["cutoff_date"], sub["ROC-AUC"], marker="o", linewidth=2, label=model_name)
        axes[1].plot(sub["cutoff_date"], sub["F1@0.50"], marker="o", linewidth=2, label=model_name)

    axes[0].set_title("Rolling-Origin ROC-AUC by Cutoff")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Rolling-Origin F1@0.50 by Cutoff")
    axes[1].set_ylabel("F1@0.50")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.set_xlabel("Cutoff date")
    axes[1].legend(frameon=False)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_calibration(
    calib_points: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], fig_dir: Path
) -> Path:
    path = fig_dir / "verification_calibration_curve.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, points in calib_points.items():
        mean_uncal, frac_uncal = points["uncalibrated"]
        mean_cal, frac_cal = points["calibrated_cv"]
        ax.plot(mean_uncal, frac_uncal, marker="o", linewidth=2, label=f"{model_name} uncal")
        ax.plot(
            mean_cal,
            frac_cal,
            marker="x",
            linewidth=1.8,
            linestyle="--",
            label=f"{model_name} calibrated-CV",
        )
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Perfect calibration"
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration Curves (Holdout)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_learning_curves(curve_df: pd.DataFrame, fig_dir: Path) -> Path:
    path = fig_dir / "verification_learning_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for varying, ax in zip(["n_estimators", "max_depth"], axes):
        sub = curve_df[curve_df["varying"] == varying]
        for model_name in sub["Model"].unique():
            model_sub = sub[sub["Model"] == model_name].sort_values("value")
            ax.plot(
                model_sub["value"],
                model_sub["Mean ROC-AUC"],
                marker="o",
                linewidth=2,
                label=model_name,
            )
        ax.set_xlabel(varying)
        ax.set_ylabel("Mean TimeSeriesSplit ROC-AUC")
        ax.set_title(f"Learning Curve vs {varying}")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_drift(drift_df: pd.DataFrame, fig_dir: Path) -> Path:
    path = fig_dir / "verification_metric_by_time_bucket.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name in drift_df["Model"].unique():
        sub = drift_df[drift_df["Model"] == model_name].sort_values("bucket")
        axes[0].plot(sub["bucket"], sub["ROC-AUC"], marker="o", linewidth=2, label=model_name)
        axes[1].plot(sub["bucket"], sub["F1@0.50"], marker="o", linewidth=2, label=model_name)
    axes[0].set_title("ROC-AUC by Time Bucket (latest holdout split)")
    axes[0].set_xlabel("Bucket (chronological)")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("F1@0.50 by Time Bucket")
    axes[1].set_xlabel("Bucket (chronological)")
    axes[1].set_ylabel("F1@0.50")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _deterministic_policy_summary(
    holdout_df: pd.DataFrame, rolling_summary: pd.DataFrame
) -> tuple[str, str, str]:
    policy_sorted = rolling_summary.sort_values(
        ["ROC-AUC mean", "PR-AUC mean", "Model"], ascending=[False, False, True]
    )
    policy_top = str(policy_sorted.iloc[0]["Model"])

    holdout_sorted = holdout_df.sort_values(
        ["ROC-AUC", "PR-AUC", "Model"], ascending=[False, False, True]
    )
    holdout_top = str(holdout_sorted.iloc[0]["Model"])

    if policy_top == holdout_top:
        text = (
            f"Deterministic policy selects `{policy_top}` and holdout confirmation agrees "
            f"(holdout top is also `{holdout_top}`)."
        )
    else:
        text = (
            f"Deterministic policy selects `{policy_top}` by rolling-origin metrics, while holdout "
            f"confirmation top is `{holdout_top}`."
        )
    return policy_top, holdout_top, text


def _under_tuning_takeaway(grid_df: pd.DataFrame, baseline_params: dict) -> str:
    best = grid_df.iloc[0]
    baseline_key = json.dumps(baseline_params, sort_keys=True)
    baseline_row = grid_df[grid_df["params"] == baseline_key]
    if baseline_row.empty:
        return "Baseline configuration was not part of the diagnostic grid."
    baseline_score = float(baseline_row.iloc[0]["Mean ROC-AUC"])
    best_score = float(best["Mean ROC-AUC"])
    delta = best_score - baseline_score
    if delta > 0.003:
        return (
            f"Potential under-tuning: best grid ROC-AUC ({best_score:.4f}) exceeds notebook baseline "
            f"({baseline_score:.4f}) by {delta:.4f}."
        )
    return (
        f"No material under-tuning signal in this grid: best ROC-AUC ({best_score:.4f}) vs baseline "
        f"({baseline_score:.4f}), delta {delta:.4f}."
    )


def _build_report(
    output_path: Path,
    prepared: PreparedData,
    holdout_df: pd.DataFrame,
    rolling_detail: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    tscv_detail: pd.DataFrame,
    tscv_summary: pd.DataFrame,
    gb_grid_df: pd.DataFrame,
    xgb_grid_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    notebook_locations = pd.DataFrame(NOTEBOOK_LOCATION_ROWS)
    policy_winner, holdout_top, policy_text = _deterministic_policy_summary(
        holdout_df, rolling_summary
    )

    gb_baseline = {"max_depth": 5, "n_estimators": 100}
    xgb_baseline = {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100}
    gb_tuning_text = _under_tuning_takeaway(gb_grid_df, gb_baseline)
    xgb_tuning_text = _under_tuning_takeaway(xgb_grid_df, xgb_baseline)

    _models_needed = {"Gradient Boosting", "XGBoost"}
    for _df_name, _df in [
        ("holdout_df", holdout_df),
        ("rolling_summary", rolling_summary),
        ("tscv_summary", tscv_summary),
    ]:
        _missing = _models_needed - set(_df["Model"])
        if _missing:
            raise ValueError(
                f"GB vs XGBoost comparison requires both models in {_df_name}; "
                f"missing: {sorted(_missing)}"
            )

    holdout_delta = float(
        holdout_df.set_index("Model").loc["Gradient Boosting", "ROC-AUC"]
        - holdout_df.set_index("Model").loc["XGBoost", "ROC-AUC"]
    )
    rolling_delta = float(
        rolling_summary.set_index("Model").loc["Gradient Boosting", "ROC-AUC mean"]
        - rolling_summary.set_index("Model").loc["XGBoost", "ROC-AUC mean"]
    )
    cv_delta = float(
        tscv_summary.set_index("Model").loc["Gradient Boosting", "ROC-AUC mean"]
        - tscv_summary.set_index("Model").loc["XGBoost", "ROC-AUC mean"]
    )

    if abs(holdout_delta) < 0.003 and abs(rolling_delta) < 0.003 and abs(cv_delta) < 0.003:
        conclusion = (
            "No meaningful difference between Gradient Boosting and XGBoost under the current setup; "
            "fold/time variance likely explains the rank switching."
        )
    elif holdout_delta > 0 and rolling_delta > 0 and cv_delta >= -0.003:
        conclusion = (
            "Gradient Boosting is more robust on future windows (holdout + rolling-origin), while "
            "TimeSeriesSplit CV is close. This suggests GB generalizes slightly better to the late timeline."
        )
    elif holdout_delta < 0 and rolling_delta < 0 and cv_delta < 0:
        conclusion = (
            "XGBoost consistently outperforms Gradient Boosting on holdout, rolling-origin, and CV."
        )
    else:
        conclusion = (
            "Mixed signal: holdout and CV do not agree on a winner. Selection policy should be explicit "
            "and deterministic to avoid model flip-flops."
        )

    md: list[str] = []
    md.append("# Model Verification Report")
    md.append("")
    md.append(f"- Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(
        "- Seed policy: `random=42`, `numpy=42`, model seeds set to `42`, single-thread model fitting where supported"
    )
    md.append("")
    md.append("## 1) Exact Notebook Definitions and Usages")
    md.append(_to_markdown(notebook_locations, round_to=4))
    md.append("")
    md.append("## 2) Exact Split and Pipeline Parameters Used")
    md.append(
        "- Data cleaned with `clean_raw`, validated with `validate_raw`, then sorted by `arrival_date`."
    )
    md.append(
        f"- Holdout split: train+val ends at index `{prepared.split_idx}` of `{len(prepared.df_clean)}` rows (TRAIN_RATIO={TRAIN_RATIO}, VAL_RATIO={VAL_RATIO})."
    )
    md.append(f"- Holdout cutoff date: `{prepared.split_date.date()}`.")
    md.append(
        f"- Holdout train rows: `{len(prepared.train_df)}`, holdout test rows: `{len(prepared.test_df)}`."
    )
    md.append(
        "- TimeSeriesSplit config for CV: `TimeSeriesSplit(n_splits=5)` with default `gap=0`, default fold test size."
    )
    md.append("- Preprocessing pipeline (identical for both compared models):")
    md.append(
        "  - Categorical: `SimpleImputer(constant='UNKNOWN')` -> cast to string -> `OneHotEncoder(handle_unknown='ignore', min_frequency=0.01)`"
    )
    md.append("  - Numeric: `SimpleImputer(strategy='median')`")
    md.append("")
    md.append("## 3) Notebook Model Hyperparameters (as actually used)")
    notebook_params_df = pd.DataFrame(
        [
            {
                "Model": "Gradient Boosting",
                "Notebook params": json.dumps(
                    {"n_estimators": 100, "max_depth": 5, "random_state": 42}, sort_keys=True
                ),
            },
            {
                "Model": "XGBoost",
                "Notebook params": json.dumps(
                    {
                        "n_estimators": 100,
                        "max_depth": 5,
                        "learning_rate": 0.1,
                        "random_state": 42,
                        "n_jobs": 1,
                        "eval_metric": "logloss",
                    },
                    sort_keys=True,
                ),
            },
        ]
    )
    md.append(_to_markdown(notebook_params_df))
    md.append("")
    md.append("### Diagnostic Search Spaces (for under-tuning check)")
    diagnostic_grid_df = pd.DataFrame(
        [
            {
                "Model": "Gradient Boosting",
                "Grid tested": str([json.loads(p) for p in gb_grid_df["params"].tolist()]),
            },
            {
                "Model": "XGBoost",
                "Grid tested": str([json.loads(p) for p in xgb_grid_df["params"].tolist()]),
            },
        ]
    )
    md.append(_to_markdown(diagnostic_grid_df))
    md.append("")
    md.append("## 4) Comparison A: Single Holdout (current split)")
    md.append(_to_markdown(holdout_df))
    md.append("")
    md.append("## 5) Comparison B: Rolling-Origin / Backtest (5 cutoffs)")
    md.append("### Per-cutoff detail")
    md.append(_to_markdown(rolling_detail))
    md.append("")
    md.append("### Distribution summary (mean/std across cutoffs)")
    md.append(_to_markdown(rolling_summary))
    md.append("")
    md.append("## 6) Comparison C: TimeSeriesSplit CV (n_splits=5)")
    md.append("### Fold-level detail")
    md.append(_to_markdown(tscv_detail))
    md.append("")
    md.append("### Mean/std summary")
    md.append(_to_markdown(tscv_summary))
    md.append("")
    md.append("## 7) Diagnosis")
    md.append("### 7.1 Under-tuning check")
    md.append(f"- Gradient Boosting: {gb_tuning_text}")
    md.append(f"- XGBoost: {xgb_tuning_text}")
    md.append("")
    md.append("Gradient Boosting grid results:")
    md.append(_to_markdown(gb_grid_df))
    md.append("")
    md.append("XGBoost grid results:")
    md.append(_to_markdown(xgb_grid_df))
    md.append("")
    md.append("### 7.2 Calibration + threshold-on-validation experiment")
    md.append(
        f"- Calibration experiment used `CalibratedClassifierCV(method='{CALIBRATION_METHOD}', cv=TimeSeriesSplit(n_splits=3))` on subtrain only."
    )
    md.append(_to_markdown(calibration_df))
    md.append("")
    md.append("### 7.3 Feature drift check (metric by late-time bucket on holdout)")
    md.append(_to_markdown(drift_df))
    md.append("")
    md.append("### 7.4 Learning curves vs `n_estimators` / `max_depth`")
    md.append(_to_markdown(curve_df))
    md.append("")
    md.append("## 8) Deterministic Selection Policy Status")
    md.append(
        "- Policy implemented: `rolling_origin_mean_roc_auc_then_pr_auc_v1` "
        "(rank by rolling-origin mean ROC-AUC, tie-break by rolling-origin mean PR-AUC)."
    )
    md.append(f"- {policy_text}")
    md.append("- Policy code path references:")
    md.append("  - `main.ipynb` uses `select_best_model` in cell 17.")
    md.append("  - `analysis.ipynb` uses `select_best_model` in cell 22.")
    md.append("- Holdout is retained as a confirmation signal only (not the primary selector).")
    md.append("")
    md.append("## 9) Figures")
    for name, p in figures.items():
        md.append(f"- {name}: `{p.as_posix()}`")
    md.append("")
    md.append("## 10) Conclusion")
    md.append(f"- Policy winner: `{policy_winner}`")
    md.append(f"- Holdout top (confirmation): `{holdout_top}`")
    md.append(f"- {conclusion}")

    output_path.write_text("\n".join(md), encoding="utf-8")


def run_model_verification(
    output_report_path: Path | None = None,
    figures_dir: Path | None = None,
    seed: int = SEED,
) -> dict[str, str]:
    """Run deterministic GB vs XGB verification and write markdown report + figures."""
    set_global_seed(seed)

    if output_report_path is None:
        output_report_path = ARTIFACTS_DIR / "model_verification_report.md"
    if figures_dir is None:
        figures_dir = ARTIFACTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_data()
    holdout_df, holdout_probs = _evaluate_holdout(prepared, seed=seed)
    rolling_detail, rolling_summary = _evaluate_rolling_origin(prepared, seed=seed)
    tscv_detail, tscv_summary, _ = _evaluate_tscv(prepared, n_splits=5, seed=seed)

    gb_grid = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 5},
        {"n_estimators": 300, "max_depth": 5},
    ]
    xgb_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    ]

    factories = _model_factories(seed)
    gb_grid_df = _evaluate_grid(
        prepared, "Gradient Boosting", factories["Gradient Boosting"], gb_grid, n_splits=5
    )
    xgb_grid_df = _evaluate_grid(prepared, "XGBoost", factories["XGBoost"], xgb_grid, n_splits=5)
    curve_df = _evaluate_curves(prepared, n_splits=3, seed=seed)
    calibration_df, calib_points = _calibration_and_threshold_experiment(
        prepared, holdout_probs, seed=seed
    )
    drift_df = _evaluate_time_bucket_drift(prepared, holdout_probs, n_buckets=5)

    figures: dict[str, Path] = {}
    roc_path, pr_path = _plot_holdout_curves(prepared, holdout_probs, figures_dir)
    figures["Holdout ROC curve"] = roc_path
    figures["Holdout PR curve"] = pr_path
    figures["Rolling metric vs cutoff"] = _plot_metric_vs_cutoff(rolling_detail, figures_dir)
    figures["Calibration curves"] = _plot_calibration(calib_points, figures_dir)
    figures["Learning curves"] = _plot_learning_curves(curve_df, figures_dir)
    figures["Metric by time bucket"] = _plot_drift(drift_df, figures_dir)

    _build_report(
        output_path=output_report_path,
        prepared=prepared,
        holdout_df=holdout_df,
        rolling_detail=rolling_detail,
        rolling_summary=rolling_summary,
        tscv_detail=tscv_detail,
        tscv_summary=tscv_summary,
        gb_grid_df=gb_grid_df,
        xgb_grid_df=xgb_grid_df,
        calibration_df=calibration_df,
        drift_df=drift_df,
        curve_df=curve_df,
        figures=figures,
    )

    return {
        "report_path": str(output_report_path),
        "figures_dir": str(figures_dir),
    }
