"""Threshold sweep utilities."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _make_threshold_grid(step: float) -> np.ndarray:
    """Return a deterministic threshold grid without floating-point accumulation.

    Uses ``np.linspace`` instead of ``np.arange`` so the number of points is
    always exactly ``round(1.0 / step)`` regardless of floating-point rounding.
    """
    n = round(1.0 / step)
    return np.round(np.linspace(0.0, 1.0 - step, n), 4)


def threshold_sweep(y_true: Any, y_prob: Any, step: float = 0.01) -> pd.DataFrame:
    """Sweep classification thresholds and record precision/recall/F1/positive-rate.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1).
    y_prob:
        Predicted probabilities in [0, 1].
    step:
        Threshold step size (default 0.01 → 100 thresholds from 0.00 to 0.99).
    """
    thresholds = _make_threshold_grid(step)
    rows = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        positive_rate = float(y_pred.mean())
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "positive_rate": positive_rate,
            }
        )

    return pd.DataFrame(rows)


def select_high_precision_threshold(
    df: pd.DataFrame,
    min_positive_rate: float,
    min_recall: float,
) -> dict[str, Any]:
    if df.empty:
        return {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "positive_rate": 0.0,
            "constraint_met": False,
            "positive_rate_constraint_met": False,
            "recall_constraint_met": False,
            "min_positive_rate": float(min_positive_rate),
            "min_recall": float(min_recall),
        }
    positive_rate_ok = df["positive_rate"] >= min_positive_rate
    recall_ok = df["recall"] >= min_recall
    eligible = df[positive_rate_ok & recall_ok]
    all_constraints_met = not eligible.empty

    if all_constraints_met:
        best = eligible.sort_values(
            ["precision", "recall", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        recall_only = df[recall_ok]
        if not recall_only.empty:
            best = recall_only.sort_values(
                ["precision", "recall", "threshold"],
                ascending=[False, False, True],
            ).iloc[0]
        else:
            best = df.sort_values(
                ["precision", "recall", "threshold"],
                ascending=[False, False, True],
            ).iloc[0]

    result = cast(dict[str, Any], best.to_dict())
    result["constraint_met"] = all_constraints_met
    result["positive_rate_constraint_met"] = bool(positive_rate_ok.any())
    result["recall_constraint_met"] = bool(recall_ok.any())
    result["min_positive_rate"] = float(min_positive_rate)
    result["min_recall"] = float(min_recall)
    return result


def select_max_f1_threshold(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "positive_rate": 0.0}
    # Drop rows with NaN f1 before sorting to avoid undefined ordering.
    valid = df.dropna(subset=["f1"])
    if valid.empty:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "positive_rate": 0.0}
    best = valid.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
    return cast(dict[str, Any], best.to_dict())


def cost_threshold_sweep(
    y_true: Any,
    y_prob: Any,
    fn_cost: Any,
    *,
    fp_cost: float,
    step: float = 0.01,
) -> pd.DataFrame:
    thresholds = _make_threshold_grid(step)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    fn_cost = np.asarray(fn_cost, dtype=float)
    if len(y_true) != len(y_prob) or len(y_true) != len(fn_cost):
        raise ValueError("y_true, y_prob, and fn_cost must have matching lengths")

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)
        fp_count = int(np.sum(fp_mask))
        fn_count = int(np.sum(fn_mask))
        fp_total = float(fp_count * fp_cost)
        fn_total = float(fn_cost[fn_mask].sum())
        total_cost = fp_total + fn_total
        rows.append(
            {
                "threshold": float(t),
                "fp_count": fp_count,
                "fn_count": fn_count,
                "fp_cost_total": fp_total,
                "fn_cost_total": fn_total,
                "total_cost": total_cost,
            }
        )
    return pd.DataFrame(rows)


def select_min_cost_threshold(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "threshold": 0.5,
            "total_cost": 0.0,
            "fp_count": 0,
            "fn_count": 0,
            "fp_cost_total": 0.0,
            "fn_cost_total": 0.0,
        }
    best = df.sort_values(["total_cost", "threshold"], ascending=[True, True]).iloc[0]
    return cast(dict[str, Any], best.to_dict())


def resolve_thresholds(
    raw_thresholds: dict[str, object],
) -> tuple[dict[str, float], dict[str, str], bool, list[str]]:
    """Extract validated threshold values from artifact payload.

    Returns (thresholds_dict, sources_dict, cost_fallback_used, alerts).
    """
    alerts: list[str] = []
    threshold_sources: dict[str, str] = {
        "high_precision": "artifact",
        "max_f1": "artifact",
        "cost_sensitive": "artifact",
    }

    def _threshold_or_none(payload: object) -> float | None:
        if isinstance(payload, dict):
            value = payload.get("threshold")
            if isinstance(value, int | float):
                return float(value)
        return None

    _hp = _threshold_or_none(raw_thresholds.get("high_precision"))
    thr_hp = 0.5 if _hp is None else float(np.clip(_hp, 0.0, 1.0))
    _f1 = _threshold_or_none(raw_thresholds.get("max_f1"))
    thr_f1 = 0.5 if _f1 is None else float(np.clip(_f1, 0.0, 1.0))

    cost_payload = raw_thresholds.get("cost_sensitive")
    cost_threshold = _threshold_or_none(cost_payload)
    cost_missing = cost_threshold is None
    if cost_missing:
        threshold_sources["cost_sensitive"] = "max_f1_fallback"
        alerts.append("cost_sensitive threshold missing; using max_f1 fallback.")
        thr_cost = thr_f1
    else:
        if cost_threshold is None:
            raise ValueError("cost_threshold resolved to None despite cost_missing=False")
        thr_cost = float(np.clip(cost_threshold, 0.0, 1.0))

    thresholds = {
        "high_precision": thr_hp,
        "max_f1": thr_f1,
        "cost_sensitive": thr_cost,
    }
    return thresholds, threshold_sources, cost_missing, alerts
