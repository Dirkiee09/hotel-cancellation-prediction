"""Model evaluation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    if len(np.unique(y_true_arr)) >= 2:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
    }


def compute_confusion(y_true, y_prob, threshold: float):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC that returns NaN on single-class inputs instead of crashing."""
    try:
        return float(roc_auc_score(np.asarray(y_true), np.asarray(y_prob)))
    except ValueError:
        return float("nan")


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR-AUC that returns NaN on single-class inputs instead of crashing."""
    try:
        return float(average_precision_score(np.asarray(y_true), np.asarray(y_prob)))
    except ValueError:
        return float("nan")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int) -> float:
    """Weighted expected calibration error across probability bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    total = len(y_true)
    if total == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        avg_conf = float(np.mean(y_prob[mask]))
        avg_acc = float(np.mean(y_true[mask]))
        weight = float(np.sum(mask)) / float(total)
        ece += weight * abs(avg_acc - avg_conf)
    return float(ece)
