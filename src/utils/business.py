"""Business-facing thresholding and segmentation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.thresholds import cost_threshold_sweep, select_min_cost_threshold


def _numeric_column(df: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def safe_threshold_metrics(y_true, y_prob, threshold: float) -> dict[str, float | None]:
    """Compute threshold metrics, tolerating single-class slices."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    roc_auc: float | None = None
    pr_auc: float | None = None
    if len(np.unique(y_true_arr)) >= 2:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
    }


def compute_fn_cost_vector(df: pd.DataFrame, fn_recovery_nights: float) -> np.ndarray:
    """Booking-specific false-negative cost proxy from ADR and stay duration."""
    adr = _numeric_column(df, "adr", default=0.0)

    if "total_stay" in df.columns:
        total_stay = _numeric_column(df, "total_stay", default=0.0)
    else:
        weekend = _numeric_column(df, "stays_in_weekend_nights", default=0.0)
        week = _numeric_column(df, "stays_in_week_nights", default=0.0)
        total_stay = weekend + week

    lost_nights = (total_stay - float(fn_recovery_nights)).clip(lower=0.0)
    return (adr * lost_nights).to_numpy(dtype=float)


def compute_cost_threshold_policy(
    y_true,
    y_prob,
    fn_cost,
    *,
    fp_cost: float,
    step: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Find the minimum-cost decision threshold and baseline comparisons."""
    y_true_arr = np.asarray(y_true).astype(int)
    sweep_df = cost_threshold_sweep(y_true_arr, y_prob, fn_cost, fp_cost=fp_cost, step=step)
    best = select_min_cost_threshold(sweep_df)

    idx_050 = int((sweep_df["threshold"] - 0.50).abs().idxmin())
    baseline_050 = sweep_df.iloc[idx_050].to_dict()
    no_model_cost = float(np.asarray(fn_cost, dtype=float)[y_true_arr == 1].sum())

    summary = {
        "threshold": float(best["threshold"]),
        "total_cost": float(best["total_cost"]),
        "fp_count": int(best["fp_count"]),
        "fn_count": int(best["fn_count"]),
        "fp_cost_total": float(best["fp_cost_total"]),
        "fn_cost_total": float(best["fn_cost_total"]),
        "fp_cost_assumption": float(fp_cost),
        "no_model_cost": no_model_cost,
        "baseline_050_cost": float(baseline_050["total_cost"]),
        "savings_vs_050": float(baseline_050["total_cost"] - best["total_cost"]),
        "savings_vs_no_model": float(no_model_cost - best["total_cost"]),
    }
    return summary, sweep_df


def assign_risk_tiers(
    probabilities,
    *,
    medium_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Assign Low/Medium/High risk tier labels based on probability thresholds."""
    probs = np.asarray(probabilities, dtype=float)
    tiers = np.full(len(probs), "low", dtype=object)
    tiers[probs >= medium_threshold] = "medium"
    tiers[probs >= high_threshold] = "high"
    return tiers
