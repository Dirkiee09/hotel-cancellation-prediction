"""Baseline classifiers for thesis comparison."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


def train_dummy(
    X_train,
    y_train,
    strategy: str = "most_frequent",
) -> DummyClassifier:
    """Train a dummy classifier as a naive baseline."""
    model = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_logistic(
    X_train,
    y_train,
    params: dict[str, Any] | None = None,
) -> LogisticRegression:
    """Train a logistic regression as an interpretable baseline."""
    config: dict[str, Any] = {
        "max_iter": 2000,
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
    }
    if params:
        config.update(params)
    model = LogisticRegression(**config)
    model.fit(X_train, y_train)
    return model
