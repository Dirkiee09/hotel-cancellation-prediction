"""Threshold sweep utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def threshold_sweep(y_true, y_prob, step: float = 0.01) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.0, 1.0, step), 4)
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
) -> dict:
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
                ["precision", "recall", "positive_rate"],
                ascending=[False, False, False],
            ).iloc[0]

    result = best.to_dict()
    result["constraint_met"] = all_constraints_met
    result["positive_rate_constraint_met"] = bool(positive_rate_ok.any())
    result["recall_constraint_met"] = bool(recall_ok.any())
    result["min_positive_rate"] = float(min_positive_rate)
    result["min_recall"] = float(min_recall)
    return result


def select_max_f1_threshold(df: pd.DataFrame) -> dict:
    best = df.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
    return best.to_dict()


def cost_threshold_sweep(
    y_true,
    y_prob,
    fn_cost,
    *,
    fp_cost: float,
    step: float = 0.01,
) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.0, 1.0, step), 4)
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


def select_min_cost_threshold(df: pd.DataFrame) -> dict:
    best = df.sort_values(["total_cost", "threshold"], ascending=[True, True]).iloc[0]
    return best.to_dict()
