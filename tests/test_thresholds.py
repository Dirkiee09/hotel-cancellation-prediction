"""Regression tests for threshold utilities.

Includes regression tests for the resolve_thresholds falsy-zero bug
(thr = value or 0.5 treated 0.0 as missing).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.thresholds import (
    _make_threshold_grid,
    resolve_thresholds,
    select_max_f1_threshold,
    select_min_cost_threshold,
    threshold_sweep,
)

# ---------------------------------------------------------------------------
# resolve_thresholds — regression tests for the falsy-zero bug
# ---------------------------------------------------------------------------


def _make_raw(hp: float, f1: float, cost: float) -> dict:
    return {
        "high_precision": {"threshold": hp},
        "max_f1": {"threshold": f1},
        "cost_sensitive": {"threshold": cost},
    }


def test_resolve_thresholds_zero_hp_not_corrupted() -> None:
    """threshold=0.0 must not be silently replaced by the 0.5 fallback."""
    raw = _make_raw(hp=0.0, f1=0.35, cost=0.03)
    thresholds, sources, cost_fallback, alerts = resolve_thresholds(raw)
    assert thresholds["high_precision"] == pytest.approx(0.0)
    assert not cost_fallback
    assert alerts == []


def test_resolve_thresholds_zero_f1_not_corrupted() -> None:
    """max_f1 threshold=0.0 must survive."""
    raw = _make_raw(hp=0.9, f1=0.0, cost=0.03)
    thresholds, _, _, _ = resolve_thresholds(raw)
    assert thresholds["max_f1"] == pytest.approx(0.0)


def test_resolve_thresholds_zero_cost_not_corrupted() -> None:
    """cost_sensitive threshold=0.0 must survive."""
    raw = _make_raw(hp=0.9, f1=0.35, cost=0.0)
    thresholds, _, _, _ = resolve_thresholds(raw)
    assert thresholds["cost_sensitive"] == pytest.approx(0.0)


def test_resolve_thresholds_clamps_above_one() -> None:
    """Out-of-range threshold >1 is clamped to 1.0."""
    raw = _make_raw(hp=1.5, f1=2.0, cost=99.0)
    thresholds, _, _, _ = resolve_thresholds(raw)
    assert thresholds["high_precision"] == pytest.approx(1.0)
    assert thresholds["max_f1"] == pytest.approx(1.0)
    assert thresholds["cost_sensitive"] == pytest.approx(1.0)


def test_resolve_thresholds_clamps_below_zero() -> None:
    """Out-of-range threshold <0 is clamped to 0.0."""
    raw = _make_raw(hp=-0.5, f1=-1.0, cost=-0.1)
    thresholds, _, _, _ = resolve_thresholds(raw)
    assert thresholds["high_precision"] == pytest.approx(0.0)
    assert thresholds["max_f1"] == pytest.approx(0.0)
    assert thresholds["cost_sensitive"] == pytest.approx(0.0)


def test_resolve_thresholds_missing_max_f1_falls_back() -> None:
    """Missing max_f1 key falls back to 0.5."""
    raw: dict[str, object] = {"high_precision": {"threshold": 0.9}, "cost_sensitive": {"threshold": 0.03}}
    thresholds, sources, _, _ = resolve_thresholds(raw)
    assert thresholds["max_f1"] == pytest.approx(0.5)
    assert sources["max_f1"] == "artifact"  # source is artifact (key present as None-returning)


def test_resolve_thresholds_missing_cost_uses_f1_fallback() -> None:
    """Missing cost_sensitive key falls back to max_f1 threshold."""
    raw: dict[str, object] = {"high_precision": {"threshold": 0.9}, "max_f1": {"threshold": 0.35}}
    thresholds, sources, cost_fallback, alerts = resolve_thresholds(raw)
    assert cost_fallback
    assert thresholds["cost_sensitive"] == pytest.approx(0.35)
    assert sources["cost_sensitive"] == "max_f1_fallback"
    assert len(alerts) == 1


def test_resolve_thresholds_normal_values() -> None:
    """Typical real-world thresholds round-trip correctly."""
    raw = _make_raw(hp=0.87, f1=0.35, cost=0.03)
    thresholds, sources, cost_fallback, alerts = resolve_thresholds(raw)
    assert thresholds["high_precision"] == pytest.approx(0.87)
    assert thresholds["max_f1"] == pytest.approx(0.35)
    assert thresholds["cost_sensitive"] == pytest.approx(0.03)
    assert not cost_fallback
    assert alerts == []


# ---------------------------------------------------------------------------
# threshold_sweep and selector helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def sweep_data():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_prob = rng.uniform(0, 1, size=500)
    return y_true, y_prob


def test_threshold_sweep_shape(sweep_data) -> None:
    y_true, y_prob = sweep_data
    df = threshold_sweep(y_true, y_prob, step=0.01)
    assert len(df) == 100
    assert set(df.columns) >= {"threshold", "precision", "recall", "f1", "positive_rate"}


def test_threshold_sweep_thresholds_in_range(sweep_data) -> None:
    y_true, y_prob = sweep_data
    df = threshold_sweep(y_true, y_prob, step=0.05)
    assert df["threshold"].min() >= 0.0
    assert df["threshold"].max() < 1.0


def test_select_max_f1_returns_best(sweep_data) -> None:
    y_true, y_prob = sweep_data
    df = threshold_sweep(y_true, y_prob)
    result = select_max_f1_threshold(df)
    assert "threshold" in result
    assert result["f1"] == pytest.approx(df["f1"].max(), abs=1e-6)


def test_select_min_cost_threshold(sweep_data) -> None:
    y_true, y_prob = sweep_data
    from src.utils.thresholds import cost_threshold_sweep

    fn_cost = np.ones(len(y_true)) * 140.0
    df = cost_threshold_sweep(y_true, y_prob, fn_cost, fp_cost=15.0)
    result = select_min_cost_threshold(df)
    assert "threshold" in result
    assert result["total_cost"] == pytest.approx(df["total_cost"].min(), abs=1e-6)


def test_make_threshold_grid_no_accumulation() -> None:
    """Grid must not overshoot 1.0 due to float accumulation."""
    grid = _make_threshold_grid(0.01)
    assert len(grid) == 100
    assert float(grid[-1]) < 1.0
    assert float(grid[0]) == pytest.approx(0.0)
