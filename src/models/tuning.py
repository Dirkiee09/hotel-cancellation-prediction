"""Optuna-based hyperparameter tuning with rolling-origin objective."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import (
    RANDOM_STATE,
    ROLLING_SELECTION_CUTOFF_FRACS,
    ROLLING_SELECTION_MIN_TRAIN_ROWS,
    ROLLING_SELECTION_MIN_VAL_ROWS,
    ROLLING_SELECTION_VAL_RATIO,
    TARGET_COL,
)
from src.features.build import build_preprocessor
from src.models.train import train_gb, train_xgb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuningResult:
    """Structured output from an Optuna tuning run."""

    best_params: dict[str, Any]
    best_score: float
    study_summary: dict[str, Any]
    all_trials: list[dict[str, Any]]


def _suggest_xgb_params(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _suggest_gb_params(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _rolling_origin_objective(
    trial: Any,
    selection_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
) -> float:
    """Objective: mean PR-AUC across rolling-origin windows."""
    if model_family == "xgboost":
        params = _suggest_xgb_params(trial)
    elif model_family == "gradient_boosting":
        params = _suggest_gb_params(trial)
    else:
        raise ValueError(f"Unsupported model family for tuning: {model_family}")

    total_rows = len(selection_df)
    val_rows = max(
        int(total_rows * ROLLING_SELECTION_VAL_RATIO),
        ROLLING_SELECTION_MIN_VAL_ROWS,
    )
    pr_values: list[float] = []

    for cutoff_frac in ROLLING_SELECTION_CUTOFF_FRACS:
        train_end = int(total_rows * cutoff_frac)
        val_end = train_end + val_rows
        if train_end < ROLLING_SELECTION_MIN_TRAIN_ROWS or val_end > total_rows:
            continue

        fold_train = selection_df.iloc[:train_end]
        fold_val = selection_df.iloc[train_end:val_end]
        X_tr = fold_train[feature_cols]
        y_tr = fold_train[TARGET_COL].astype(int)
        X_val = fold_val[feature_cols]
        y_val = fold_val[TARGET_COL].astype(int)

        preprocessor = build_preprocessor()
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_val_t = preprocessor.transform(X_val)

        if model_family == "xgboost":
            model = train_xgb(X_tr_t, y_tr, params=params)
        else:
            model = train_gb(X_tr_t, y_tr, params=params)

        probs = model.predict_proba(X_val_t)[:, 1]
        try:
            pr_auc = float(average_precision_score(y_val.to_numpy(), probs))
            pr_values.append(pr_auc)
        except ValueError:
            continue

    if not pr_values:
        return 0.0
    return float(np.mean(pr_values))


def tune_model(
    selection_df: pd.DataFrame,
    feature_cols: list[str],
    model_family: str,
    *,
    n_trials: int = 50,
    timeout: int | None = 600,
    seed: int = RANDOM_STATE,
) -> TuningResult:
    """Run Optuna study for a model family using rolling-origin CV."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _rolling_origin_objective(trial, selection_df, feature_cols, model_family),
        n_trials=n_trials,
        timeout=timeout,
    )

    all_trials = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]

    logger.info(
        "tuning_complete model_family=%s best_score=%.4f n_trials=%d",
        model_family,
        study.best_value,
        len(study.trials),
    )

    return TuningResult(
        best_params=study.best_params,
        best_score=study.best_value,
        study_summary={
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "model_family": model_family,
        },
        all_trials=all_trials,
    )
