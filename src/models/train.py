"""Model training utilities."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

lgb: Any
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def get_default_xgb_params() -> dict[str, Any]:
    return {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
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
) -> GradientBoostingClassifier:
    config = get_default_gb_params()
    if params:
        config.update(params)
    model = GradientBoostingClassifier(**config)
    model.fit(X_train, y_train)
    return model


def train_xgb(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_val: ArrayLike | None = None,
    y_val: ArrayLike | None = None,
    scale_pos_weight: float | None = None,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int | None = None,
) -> XGBClassifier:
    config = get_default_xgb_params()
    if params:
        config.update(params)
    if scale_pos_weight is not None:
        config["scale_pos_weight"] = scale_pos_weight
    if early_stopping_rounds is not None:
        config["early_stopping_rounds"] = early_stopping_rounds

    model = XGBClassifier(**config)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def train_lgbm(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_val: ArrayLike | None = None,
    y_val: ArrayLike | None = None,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int | None = None,
) -> Any:
    if lgb is None:  # pragma: no cover - dependency dependent
        raise ImportError("lightgbm is not installed")

    config = get_default_lgbm_params()
    if params:
        config.update(params)

    callbacks = []
    if early_stopping_rounds is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

    model = lgb.LGBMClassifier(**config)

    if X_val is not None and y_val is not None:
        callbacks.append(lgb.log_evaluation(period=0))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    else:
        model.fit(X_train, y_train)
    return model
