"""Statistical testing utilities for thesis analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.config import RANDOM_STATE


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for a single metric."""

    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_bootstraps: int


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    *,
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for a metric function.

    Parameters
    ----------
    metric_fn : callable(y_true, y_prob) -> float
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    point = float(metric_fn(y_true, y_prob))

    scores = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        if len(np.unique(y_t)) < 2:
            scores[i] = np.nan
            continue
        scores[i] = metric_fn(y_t, y_p)

    valid = scores[~np.isnan(scores)]
    if len(valid) == 0:
        # All bootstrap samples produced single-class resamples; CI is undefined.
        return BootstrapCI(
            metric="custom",
            point_estimate=point,
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            alpha=alpha,
            n_bootstraps=0,
        )
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return BootstrapCI(
        metric="custom",
        point_estimate=point,
        ci_lower=lo,
        ci_upper=hi,
        alpha=alpha,
        n_bootstraps=int(len(valid)),
    )


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_STATE,
) -> dict[str, BootstrapCI]:
    """Bootstrap CIs for ROC-AUC, PR-AUC, and F1 at a given threshold."""
    metric_fns: list[tuple[str, object]] = [
        ("roc_auc", lambda yt, yp: float(roc_auc_score(yt, yp))),
        ("pr_auc", lambda yt, yp: float(average_precision_score(yt, yp))),
        (
            "f1",
            lambda yt, yp, _t=threshold: float(
                f1_score(yt, (yp >= _t).astype(int), zero_division=0)
            ),
        ),
    ]
    results: dict[str, BootstrapCI] = {}
    for name, fn in metric_fns:
        ci = bootstrap_metric(y_true, y_prob, fn, n_bootstraps=n_bootstraps, alpha=alpha, seed=seed)
        results[name] = BootstrapCI(
            metric=name,
            point_estimate=ci.point_estimate,
            ci_lower=ci.ci_lower,
            ci_upper=ci.ci_upper,
            alpha=alpha,
            n_bootstraps=n_bootstraps,
        )
    return results


def paired_bootstrap_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric_fn,
    *,
    n_bootstraps: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict[str, object]:
    """Two-sided paired bootstrap test: H0 is metric(A) == metric(B).

    Returns dict with observed_delta, p_value, n_bootstraps, significant_at_05.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)
    n = len(y_true)

    observed_delta = float(metric_fn(y_true, probs_a) - metric_fn(y_true, probs_b))

    deltas = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            deltas[i] = np.nan
            continue
        deltas[i] = metric_fn(yt, probs_a[idx]) - metric_fn(yt, probs_b[idx])

    valid = deltas[~np.isnan(deltas)]
    if len(valid) == 0:
        return {
            "observed_delta": observed_delta,
            "p_value": float("nan"),
            "n_bootstraps": 0,
            "significant_at_05": False,
        }
    # Standard two-sided bootstrap p-value: 2 * min tail probability.
    # Equivalent implementation is used in benchmark.py for consistency.
    p_left = float(np.mean(valid <= 0.0))
    p_right = float(np.mean(valid >= 0.0))
    p_value = float(min(1.0, 2.0 * min(p_left, p_right)))

    return {
        "observed_delta": observed_delta,
        "p_value": p_value,
        "n_bootstraps": int(len(valid)),
        "significant_at_05": p_value < 0.05,
    }
