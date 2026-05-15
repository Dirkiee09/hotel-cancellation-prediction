"""Population Stability Index (PSI) and zone classification for drift monitoring.

Lifted from notebooks/08_model_monitoring.ipynb so the same logic can be
imported by:
    * The notebook itself (back-compat)
    * scripts/compute_live_drift.py (live drift export for Power BI)
    * Any future drift tooling (CI alerts, scheduled drift jobs, etc.)

PSI interpretation (industry standard):
    * < 0.10  -> stable, no action needed
    * 0.10 - 0.25 -> monitor, investigate the cause
    * >= 0.25 -> retrain or recalibrate
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_EPSILON = 1e-6  # avoids log(0) when a bin or category is empty in one set
_BIN_EPSILON = 1e-4  # smoothing for the numeric histogram counts

PSI_STABLE_THRESHOLD: float = 0.10
PSI_MONITOR_THRESHOLD: float = 0.25


def compute_psi(
    baseline_arr: np.ndarray | pd.Series | Iterable[float],
    current_arr: np.ndarray | pd.Series | Iterable[float],
    n_bins: int = 10,
) -> float:
    """Population Stability Index for a numeric distribution.

    Bins are quantile-based on the baseline so each baseline bin gets ~10% of
    the mass. The outer bin edges are set to +/-inf so the current distribution
    can extend beyond the baseline range without losing rows.
    """
    base = np.asarray(list(baseline_arr), dtype=float)
    curr = np.asarray(list(current_arr), dtype=float)
    if base.size == 0 or curr.size == 0:
        return 0.0

    bins = np.percentile(base, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    b_pct = (np.histogram(base, bins)[0] + _BIN_EPSILON) / base.size
    c_pct = (np.histogram(curr, bins)[0] + _BIN_EPSILON) / curr.size
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def cat_psi(
    base_series: pd.Series,
    curr_series: pd.Series,
) -> float:
    """PSI for a categorical column.

    Iterates over the union of categories observed in either distribution.
    Adds a small epsilon to avoid log(0) for categories present in one but
    not the other.
    """
    base = pd.Series(base_series).dropna()
    curr = pd.Series(curr_series).dropna()
    if base.empty or curr.empty:
        return 0.0

    categories = set(base.unique()) | set(curr.unique())
    psi_val = 0.0
    for cat in categories:
        exp = float((base == cat).mean()) + _EPSILON
        act = float((curr == cat).mean()) + _EPSILON
        psi_val += (act - exp) * np.log(act / exp)
    return float(psi_val)


def psi_zone(psi_val: float) -> str:
    """Industry-standard PSI zone label.

    Returns one of 'stable' (<0.10), 'monitor' (0.10-0.25), 'retrain' (>=0.25).
    """
    if psi_val < PSI_STABLE_THRESHOLD:
        return "stable"
    if psi_val < PSI_MONITOR_THRESHOLD:
        return "monitor"
    return "retrain"
