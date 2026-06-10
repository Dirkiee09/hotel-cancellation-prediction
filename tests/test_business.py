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


def test_cost_policy_reports_intervene_all_baseline() -> None:
    """The summary must include the trivial flag-everyone policy as a baseline.

    Comparing only against "no model" overstates model value: under an
    asymmetric cost model the cheapest trivial policy is intervening on every
    booking, and savings must be reported against it.
    """
    import numpy as np
    import pytest

    from src.utils.business import compute_cost_threshold_policy

    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    fn_cost = np.full(5, 100.0)
    summary, _sweep = compute_cost_threshold_policy(y, p, fn_cost, fp_cost=15.0, step=0.01)

    # threshold 0.0 flags everyone: 3 FPs x 15.0, zero FN cost
    assert summary["intervene_all_cost"] == pytest.approx(45.0)
    assert summary["savings_vs_intervene_all"] == pytest.approx(
        summary["intervene_all_cost"] - summary["total_cost"]
    )


def test_evaluate_cost_at_threshold_single_point() -> None:
    """Cost evaluation at a fixed threshold (no re-optimisation) for test-set reporting."""
    import numpy as np
    import pytest

    from src.utils.business import evaluate_cost_at_threshold

    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.6, 0.4, 0.9])
    fn_cost = np.array([50.0, 50.0, 80.0, 80.0])
    result = evaluate_cost_at_threshold(y, p, fn_cost, fp_cost=15.0, threshold=0.5)

    # pred = [0,1,0,1] -> 1 FP (15.0), 1 FN (80.0)
    assert result["fp_count"] == 1
    assert result["fn_count"] == 1
    assert result["total_cost"] == pytest.approx(95.0)
    assert result["threshold"] == pytest.approx(0.5)
