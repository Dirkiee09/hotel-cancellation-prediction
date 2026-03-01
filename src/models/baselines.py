"""Baseline classifiers for thesis comparison."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE


def train_dummy(
    X_train: Any,
    y_train: Any,
    strategy: str = "most_frequent",
) -> DummyClassifier:
    """Train a dummy classifier as a naive baseline."""
    model = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_logistic(
    X_train: Any,
    y_train: Any,
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


def train_decision_tree(
    X_train: Any,
    y_train: Any,
    params: dict[str, Any] | None = None,
) -> DecisionTreeClassifier:
    """Train a pruned Decision Tree as an interpretable thesis baseline.

    Deliberately shallow (max_depth=5) so the tree can be visualised in full
    and each path explained to a non-technical reader.  class_weight='balanced'
    compensates for the ~63/37 class imbalance without requiring a manual weight
    calculation, keeping the baseline simple and reproducible.
    """
    config: dict[str, Any] = {
        "max_depth": 5,
        "min_samples_leaf": 50,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    if params:
        config.update(params)
    model = DecisionTreeClassifier(**config)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(
    X_train: Any,
    y_train: Any,
    params: dict[str, Any] | None = None,
) -> GaussianNB:
    """Train a Gaussian Naive Bayes as a probabilistic zero-assumption baseline.

    Assumes feature independence — intentionally wrong for structured hotel data.
    Its performance gap vs LightGBM quantifies the value of capturing feature
    interactions, which is a useful thesis discussion point.
    """
    config: dict[str, Any] = {}
    if params:
        config.update(params)
    model = GaussianNB(**config)
    model.fit(X_train, y_train)
    return model
