"""Model training utilities."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from src.config import EARLY_STOPPING_ROUNDS, RANDOM_STATE

lgb: Any
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def get_default_xgb_params() -> dict[str, Any]:
    # Capacity (trees/depth/lr/subsampling) is intentionally identical to
    # get_default_lgbm_params() so champion/challenger selection compares
    # algorithms, not hyperparameter budgets.
    return {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "eval_metric": "logloss",
    }


def get_default_gb_params() -> dict[str, Any]:
    return {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
    }


def get_default_lgbm_params() -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "objective": "binary",
    }


def is_lightgbm_available() -> bool:
    return lgb is not None


def train_gb(
    X_train: ArrayLike,
    y_train: ArrayLike,
    params: dict[str, Any] | None = None,
    scale_pos_weight: float | None = None,
) -> GradientBoostingClassifier:
    config = get_default_gb_params()
    if params:
        config.update(params)
    model = GradientBoostingClassifier(**config)
    # sklearn GB has no scale_pos_weight parameter; emulate it with per-sample
    # weights so class weighting is symmetric across candidate families.
    sample_weight = None
    if scale_pos_weight is not None:
        y_arr = np.asarray(y_train)
        sample_weight = np.where(y_arr == 1, float(scale_pos_weight), 1.0)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_xgb(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_val: ArrayLike | None = None,
    y_val: ArrayLike | None = None,
    scale_pos_weight: float | None = None,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    config = get_default_xgb_params()
    if params:
        config.update(params)
    if scale_pos_weight is not None:
        config["scale_pos_weight"] = scale_pos_weight

    if X_val is not None and y_val is not None:
        # Without early_stopping_rounds the eval_set is only monitored, never
        # acted on — boosting would silently run to n_estimators regardless.
        config.setdefault("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
        model = XGBClassifier(**config)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model = XGBClassifier(**config)
        model.fit(X_train, y_train)
    return model


def train_lgbm(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_val: ArrayLike | None = None,
    y_val: ArrayLike | None = None,
    scale_pos_weight: float | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    if lgb is None:  # pragma: no cover - dependency dependent
        raise ImportError("lightgbm is not installed")

    config = get_default_lgbm_params()
    if params:
        config.update(params)
    if scale_pos_weight is not None:
        config["scale_pos_weight"] = scale_pos_weight
    model = lgb.LGBMClassifier(**config)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    else:
        model.fit(X_train, y_train)
    return model
