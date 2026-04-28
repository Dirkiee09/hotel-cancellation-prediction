"""Tests for src/models/metrics.py — metric computation functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.metrics import (
    compute_confusion,
    evaluate_at_threshold,
    expected_calibration_error,
    safe_pr_auc,
    safe_roc_auc,
)


class TestEvaluateAtThreshold:
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        result = evaluate_at_threshold(y_true, y_prob, threshold=0.5)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.9, 0.8, 0.1, 0.2]
        result = evaluate_at_threshold(y_true, y_prob, threshold=0.5)
        assert result["recall"] == pytest.approx(0.0)

    def test_single_class_returns_nan_auc(self):
        y_true = [0, 0, 0, 0]
        y_prob = [0.1, 0.2, 0.3, 0.4]
        result = evaluate_at_threshold(y_true, y_prob, threshold=0.5)
        assert math.isnan(result["roc_auc"])
        assert math.isnan(result["pr_auc"])

    def test_threshold_boundary(self):
        y_true = [0, 1]
        y_prob = [0.5, 0.5]
        result = evaluate_at_threshold(y_true, y_prob, threshold=0.5)
        # Both at exactly 0.5 → both predicted positive
        assert result["recall"] == pytest.approx(1.0)


class TestComputeConfusion:
    def test_shape(self):
        cm = compute_confusion([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
        assert cm.shape == (2, 2)

    def test_perfect_matrix(self):
        cm = compute_confusion([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
        assert cm[0, 0] == 2  # TN
        assert cm[1, 1] == 2  # TP
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN


class TestSafeAuc:
    def test_roc_auc_normal(self):
        val = safe_roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        assert 0.0 <= val <= 1.0

    def test_roc_auc_single_class(self):
        val = safe_roc_auc(np.array([1, 1, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        assert math.isnan(val)

    def test_pr_auc_normal(self):
        val = safe_pr_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        assert 0.0 <= val <= 1.0

    def test_pr_auc_single_class(self):
        # sklearn PR-AUC doesn't raise for all-negative; returns a value with a warning.
        # safe_pr_auc should still return a finite float (sklearn handles this internally).
        val = safe_pr_auc(np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4]))
        assert isinstance(val, float)


class TestECE:
    def test_empty_input(self):
        assert math.isnan(expected_calibration_error(np.array([]), np.array([]), bins=10))

    def test_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = expected_calibration_error(y_true, y_prob, bins=10)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_poor_calibration_is_positive(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        ece = expected_calibration_error(y_true, y_prob, bins=10)
        assert ece > 0.5

    def test_ece_bounded(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_prob = rng.rand(100)
        ece = expected_calibration_error(y_true, y_prob, bins=10)
        assert 0.0 <= ece <= 1.0
