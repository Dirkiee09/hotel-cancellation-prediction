"""Tests for src/utils/drift.py — PSI computation + zone classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.drift import cat_psi, compute_psi, psi_zone


def test_compute_psi_identical_distributions_is_near_zero() -> None:
    """PSI(X, X) should be ~0 (small epsilon noise allowed)."""
    rng = np.random.default_rng(42)
    arr = rng.normal(size=1000)
    psi = compute_psi(arr, arr)
    assert psi == 0.0 or abs(psi) < 1e-6


def test_compute_psi_large_shift_exceeds_retrain_threshold() -> None:
    """A 2-sigma mean shift on a normal distribution should produce PSI > 0.25."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=0.0, scale=1.0, size=2000)
    shifted = rng.normal(loc=2.0, scale=1.0, size=2000)
    psi = compute_psi(baseline, shifted)
    assert psi > 0.25, f"Expected PSI > 0.25 for 2-sigma shift, got {psi}"


def test_compute_psi_empty_input_returns_zero() -> None:
    """Empty arrays should return 0.0, not raise."""
    assert compute_psi([], [1.0, 2.0, 3.0]) == 0.0
    assert compute_psi([1.0, 2.0, 3.0], []) == 0.0


def test_compute_psi_is_nonnegative_for_random_pairs() -> None:
    """PSI is bounded below by zero by construction."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        a = rng.normal(size=500)
        b = rng.normal(size=500)
        assert compute_psi(a, b) >= 0.0


def test_cat_psi_identical_categories_is_near_zero() -> None:
    """PSI(X, X) for categoricals is near 0."""
    s = pd.Series(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    psi = cat_psi(s, s)
    assert abs(psi) < 1e-9


def test_cat_psi_complete_category_swap_is_large() -> None:
    """If baseline has 'a' and current has 'b', PSI is large."""
    base = pd.Series(["a"] * 100)
    curr = pd.Series(["b"] * 100)
    psi = cat_psi(base, curr)
    assert psi > 0.25


def test_cat_psi_handles_new_categories() -> None:
    """A category present in current but not baseline must not raise."""
    base = pd.Series(["a", "a", "b", "b"])
    curr = pd.Series(["a", "b", "c", "c"])  # 'c' is new
    psi = cat_psi(base, curr)
    assert psi > 0


def test_cat_psi_handles_empty_series() -> None:
    """Empty series returns 0.0 instead of raising."""
    assert cat_psi(pd.Series([], dtype=str), pd.Series(["a", "b"])) == 0.0
    assert cat_psi(pd.Series(["a", "b"]), pd.Series([], dtype=str)) == 0.0


def test_psi_zone_threshold_boundaries() -> None:
    """Zone labels at the canonical 0.10 / 0.25 boundaries."""
    assert psi_zone(0.0) == "stable"
    assert psi_zone(0.09) == "stable"
    assert psi_zone(0.10) == "monitor"  # boundary belongs to monitor zone
    assert psi_zone(0.24) == "monitor"
    assert psi_zone(0.25) == "retrain"  # boundary belongs to retrain zone
    assert psi_zone(1.0) == "retrain"
