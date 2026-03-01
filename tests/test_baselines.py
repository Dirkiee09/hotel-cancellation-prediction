"""Tests for baseline model trainers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.baselines import train_decision_tree, train_dummy, train_logistic, train_naive_bayes


@pytest.fixture()
def synthetic_data():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    return X, y


def test_train_dummy_returns_predictions(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_dummy(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (500, 2)


def test_train_dummy_stratified(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_dummy(X, y, strategy="stratified")
    probs = model.predict_proba(X)
    assert probs.shape == (500, 2)


def test_train_logistic_returns_predictions(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_logistic(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (500, 2)
    # Should be a non-trivial classifier
    preds = (probs[:, 1] >= 0.5).astype(int)
    accuracy = float(np.mean(preds == y))
    assert accuracy > 0.6


def test_train_logistic_accepts_custom_params(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_logistic(X, y, params={"C": 0.01})
    probs = model.predict_proba(X)
    assert probs.shape == (500, 2)


def test_train_decision_tree_returns_predictions(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_decision_tree(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (500, 2)
    # Pruned tree should still be better than chance
    preds = (probs[:, 1] >= 0.5).astype(int)
    accuracy = float(np.mean(preds == y))
    assert accuracy > 0.6


def test_train_decision_tree_respects_max_depth(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_decision_tree(X, y)
    assert model.get_depth() <= 5


def test_train_decision_tree_accepts_custom_params(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_decision_tree(X, y, params={"max_depth": 3})
    assert model.get_depth() <= 3
    probs = model.predict_proba(X)
    assert probs.shape == (500, 2)


def test_train_naive_bayes_returns_predictions(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_naive_bayes(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (500, 2)
    # Probabilities should sum to 1 for each sample
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_train_naive_bayes_accepts_custom_params(synthetic_data) -> None:
    X, y = synthetic_data
    model = train_naive_bayes(X, y, params={"var_smoothing": 1e-8})
    probs = model.predict_proba(X)
    assert probs.shape == (500, 2)
