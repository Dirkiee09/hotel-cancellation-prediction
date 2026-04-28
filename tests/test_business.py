"""Tests for src/utils/business.py — cost calculations and risk tiers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.business import assign_risk_tiers, compute_fn_cost_vector, safe_threshold_metrics


class TestAssignRiskTiers:
    def test_low_medium_high(self):
        probs = [0.1, 0.5, 0.9]
        tiers = assign_risk_tiers(probs, medium_threshold=0.4, high_threshold=0.7)
        assert list(tiers) == ["low", "medium", "high"]

    def test_boundary_values(self):
        tiers = assign_risk_tiers([0.4, 0.7], medium_threshold=0.4, high_threshold=0.7)
        assert tiers[0] == "medium"
        assert tiers[1] == "high"

    def test_all_low(self):
        tiers = assign_risk_tiers([0.0, 0.1, 0.2], medium_threshold=0.4, high_threshold=0.7)
        assert all(t == "low" for t in tiers)

    def test_empty_input(self):
        tiers = assign_risk_tiers([], medium_threshold=0.4, high_threshold=0.7)
        assert len(tiers) == 0


class TestComputeFnCostVector:
    def test_basic_cost(self):
        df = pd.DataFrame({"adr": [100.0, 200.0], "total_stay": [3.0, 5.0]})
        costs = compute_fn_cost_vector(df, fn_recovery_nights=1.0)
        # cost = adr * max(total_stay - recovery, 0)
        assert costs[0] == pytest.approx(200.0)  # 100 * (3-1)
        assert costs[1] == pytest.approx(800.0)  # 200 * (5-1)

    def test_short_stay_zero_cost(self):
        df = pd.DataFrame({"adr": [100.0], "total_stay": [1.0]})
        costs = compute_fn_cost_vector(df, fn_recovery_nights=1.0)
        assert costs[0] == pytest.approx(0.0)

    def test_fallback_to_weekend_week(self):
        df = pd.DataFrame(
            {
                "adr": [100.0],
                "stays_in_weekend_nights": [1.0],
                "stays_in_week_nights": [2.0],
            }
        )
        costs = compute_fn_cost_vector(df, fn_recovery_nights=1.0)
        assert costs[0] == pytest.approx(200.0)  # 100 * (3-1)


class TestSafeThresholdMetrics:
    def test_normal_two_class(self):
        result = safe_threshold_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
        assert result["roc_auc"] is not None
        assert result["pr_auc"] is not None
        assert result["f1"] == pytest.approx(1.0)

    def test_single_class_returns_none_auc(self):
        result = safe_threshold_metrics([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], threshold=0.5)
        assert result["roc_auc"] is None
        assert result["pr_auc"] is None
