"""Tests for statistical testing utilities."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.eval.statistical import (
    bootstrap_all_metrics,
    bootstrap_metric,
    paired_bootstrap_test,
)


@pytest.fixture()
def binary_data():
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=500)
    probs = np.clip(y * 0.7 + rng.normal(0, 0.2, size=500), 0, 1)
    return y, probs


def test_bootstrap_metric_returns_valid_ci(binary_data) -> None:
    y, probs = binary_data
    ci = bootstrap_metric(y, probs, roc_auc_score, n_bootstraps=200)
    assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper
    assert ci.ci_lower > 0.5  # decent model


def test_bootstrap_all_metrics_keys(binary_data) -> None:
    y, probs = binary_data
    results = bootstrap_all_metrics(y, probs, threshold=0.5, n_bootstraps=200)
    assert set(results.keys()) == {"roc_auc", "pr_auc", "f1"}
    for ci in results.values():
        assert ci.ci_lower <= ci.ci_upper


def test_paired_bootstrap_identical_models(binary_data) -> None:
    y, probs = binary_data
    result = paired_bootstrap_test(y, probs, probs, roc_auc_score, n_bootstraps=200)
    assert result["observed_delta"] == 0.0
    assert result["p_value"] == 1.0


def test_paired_bootstrap_different_models(binary_data) -> None:
    y, probs = binary_data
    worse_probs = np.clip(probs + np.random.RandomState(99).normal(0, 0.3, size=len(probs)), 0, 1)
    result = paired_bootstrap_test(y, probs, worse_probs, roc_auc_score, n_bootstraps=500)
    assert "p_value" in result
    assert "observed_delta" in result
    assert isinstance(result["significant_at_05"], bool)
