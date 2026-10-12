"""Post-defense revision diagnostics (panel points 5, 6, 7).

Computes, for BOTH evaluation streams (per-record primary-robustness pair):

1. Extended bootstrap CIs — recall, precision, ECE and the cost-saving
   estimate, alongside the existing ROC-AUC / PR-AUC / F1 (panel point 5).
2. Calibration diagnostics — Brier score, log loss, calibration slope and
   intercept, plus calibration-by-segment tables across hotel, market
   segment, distribution channel, arrival month and risk tier (point 6).
3. Cost decomposition scenario grid — sweeps resale probability,
   intervention cost and guest-friction cost, re-optimising the threshold
   per scenario on the de-duplicated stream (point 7).

Streams:
    per_record    reports/test_predictions_for_powerbi.csv  (+ reports/metrics.json)
    deduplicated  reports/dedup/test_predictions_for_powerbi.csv (+ reports/dedup/metrics.json)

Outputs (this script is the sole writer):
    reports/thesis/confidence_intervals_extended.json
    reports/thesis/calibration_diagnostics.json
    reports/thesis/calibration_by_segment.csv
    reports/thesis/cost_scenarios.csv

Fast (no retraining): reads saved test predictions only. Usage:
    python scripts/training/revision_diagnostics.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score

from src.config import (
    ARTIFACTS_DIR,
    FN_RECOVERY_NIGHTS,
    FP_INTERVENTION_COST,
    RANDOM_STATE,
    REPORTS_DIR,
    THRESHOLD_STEP,
)
from src.eval.statistical import bootstrap_metric
from src.models.metrics import expected_calibration_error
from src.utils import configure_logging
from src.utils.business import (
    compute_cost_threshold_policy,
    compute_fn_cost_vector,
    evaluate_cost_at_threshold,
)

logger = logging.getLogger(__name__)

N_BOOTSTRAPS = 2000
ECE_BINS = 10
SMALL_SEGMENT_N = 200
SEGMENT_DIMS = (
    "hotel",
    "market_segment",
    "distribution_channel",
    "arrival_date_month",
    "risk_tier",
)
RESALE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
INTERVENTION_GRID = (5.0, 10.0, 15.0, 25.0, 50.0)
FRICTION_GRID = (0.0, 5.0, 10.0)


def _stream_paths(stream: str) -> tuple[Path, Path, Path]:
    if stream == "per_record":
        return (
            REPORTS_DIR / "test_predictions_for_powerbi.csv",
            REPORTS_DIR / "metrics.json",
            ARTIFACTS_DIR / "thresholds.json",
        )
    base = REPORTS_DIR / "dedup"
    return (
        base / "test_predictions_for_powerbi.csv",
        base / "metrics.json",
        base / "thresholds.json",
    )


def _load_stream(stream: str) -> dict | None:
    pred_path, metrics_path, thresholds_path = _stream_paths(stream)
    if not (pred_path.exists() and metrics_path.exists() and thresholds_path.exists()):
        logger.warning("stream_missing stream=%s (run dedup_sensitivity.py first?)", stream)
        return None
    df = pd.read_csv(pred_path)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    return {
        "df": df,
        "y": df["is_canceled"].to_numpy(dtype=int),
        "p": df["cancel_probability"].to_numpy(dtype=float),
        "thr_max_f1": float(thresholds["max_f1"]["threshold"]),
        "thr_cost": float(thresholds["cost_sensitive"]["threshold"]),
    }


def _ci_dict(ci) -> dict[str, float | int]:
    return {
        "point_estimate": round(float(ci.point_estimate), 6),
        "ci_lower": round(float(ci.ci_lower), 6),
        "ci_upper": round(float(ci.ci_upper), 6),
        "alpha": ci.alpha,
        "n_bootstraps": ci.n_bootstraps,
    }


def _extended_cis(data: dict) -> dict:
    y, p, thr = data["y"], data["p"], data["thr_max_f1"]
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    fns = {
        "roc_auc": lambda yt, yp: float(roc_auc_score(yt, yp)),
        "pr_auc": lambda yt, yp: float(average_precision_score(yt, yp)),
        "f1": lambda yt, yp: float(f1_score(yt, (yp >= thr).astype(int), zero_division=0)),
        "recall": lambda yt, yp: float(recall_score(yt, (yp >= thr).astype(int), zero_division=0)),
        "precision": lambda yt, yp: float(
            precision_score(yt, (yp >= thr).astype(int), zero_division=0)
        ),
        "ece": lambda yt, yp: float(expected_calibration_error(yt, yp, ECE_BINS)),
    }
    out = {
        name: _ci_dict(bootstrap_metric(y, p, fn, n_bootstraps=N_BOOTSTRAPS))
        for name, fn in fns.items()
    }
    out["cost_savings_vs_050"] = _cost_savings_ci(data)
    out["threshold_max_f1"] = data["thr_max_f1"]
    out["threshold_cost_sensitive"] = data["thr_cost"]
    return out


def _cost_savings_ci(data: dict) -> dict[str, float | int]:
    """Bootstrap CI for (cost at naive 0.50) - (cost at the selected cost threshold).

    A local row-bootstrap is required because the per-booking FN cost vector
    must be resampled together with (y, p) — bootstrap_metric only carries
    the label/probability pair.
    """
    df, y, p = data["df"], data["y"], data["p"]
    fn_cost = compute_fn_cost_vector(df, FN_RECOVERY_NIGHTS)
    thr_cost = data["thr_cost"]

    def savings(idx: np.ndarray) -> float:
        c_naive = evaluate_cost_at_threshold(
            y[idx], p[idx], fn_cost[idx], fp_cost=FP_INTERVENTION_COST, threshold=0.50
        )["total_cost"]
        c_model = evaluate_cost_at_threshold(
            y[idx], p[idx], fn_cost[idx], fp_cost=FP_INTERVENTION_COST, threshold=thr_cost
        )["total_cost"]
        return c_naive - c_model

    n = len(y)
    point = savings(np.arange(n))
    rng = np.random.RandomState(RANDOM_STATE)
    scores = np.array([savings(rng.randint(0, n, size=n)) for _ in range(N_BOOTSTRAPS)])
    return {
        "point_estimate": round(float(point), 2),
        "ci_lower": round(float(np.percentile(scores, 2.5)), 2),
        "ci_upper": round(float(np.percentile(scores, 97.5)), 2),
        "alpha": 0.05,
        "n_bootstraps": N_BOOTSTRAPS,
    }


def _calibration_diagnostics(data: dict) -> dict:
    y, p = data["y"], data["p"]
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    # Effectively unregularised logistic recalibration fit: slope 1 / intercept 0 = perfect.
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y)
    return {
        "n_test": int(len(y)),
        "brier_score": round(float(brier_score_loss(y, p)), 6),
        "log_loss": round(float(log_loss(y, p_clip, labels=[0, 1])), 6),
        "ece": round(float(expected_calibration_error(y, p, ECE_BINS)), 6),
        "ece_bins": ECE_BINS,
        "calibration_slope": round(float(fit.coef_[0][0]), 4),
        "calibration_intercept": round(float(fit.intercept_[0]), 4),
    }


def _calibration_by_segment(stream: str, data: dict) -> list[dict]:
    df = data["df"]
    rows: list[dict] = []
    for dim in SEGMENT_DIMS:
        if dim not in df.columns:
            logger.warning("segment_dim_missing stream=%s dim=%s", stream, dim)
            continue
        for value, grp in df.groupby(dim, observed=True):
            y_g = grp["is_canceled"].to_numpy(dtype=int)
            p_g = grp["cancel_probability"].to_numpy(dtype=float)
            n = len(grp)
            ece_g = (
                float(expected_calibration_error(y_g, p_g, ECE_BINS))
                if n >= SMALL_SEGMENT_N
                else None
            )
            rows.append(
                {
                    "Stream": stream,
                    "Dimension": dim,
                    "Segment": value,
                    "N Bookings": n,
                    "Mean Predicted": round(float(p_g.mean()), 4),
                    "Observed Rate": round(float(y_g.mean()), 4),
                    "Calibration Gap": round(float(p_g.mean() - y_g.mean()), 4),
                    "ECE": round(ece_g, 4) if ece_g is not None else None,
                    "Small Sample Flag": bool(n < SMALL_SEGMENT_N),
                }
            )
    return rows


def _cost_scenarios(data: dict) -> list[dict]:
    """Scenario grid on the de-duplicated stream.

    Cost model per scenario:  C(t) = FP(t) * (c_int + c_friction)
                                     + sum_FN (1 - p_resale) * adr * max(stay - recovery, 0)
    The threshold is re-optimised per scenario (that is the sensitivity being tested).
    """
    df, y, p = data["df"], data["y"], data["p"]
    fn_base = compute_fn_cost_vector(df, FN_RECOVERY_NIGHTS)
    rows: list[dict] = []
    for p_resale in RESALE_GRID:
        fn_cost = (1.0 - p_resale) * fn_base
        for c_int in INTERVENTION_GRID:
            for c_frict in FRICTION_GRID:
                summary, _ = compute_cost_threshold_policy(
                    y, p, fn_cost, fp_cost=c_int + c_frict, step=THRESHOLD_STEP
                )
                rows.append(
                    {
                        "Resale Probability": p_resale,
                        "Intervention Cost": c_int,
                        "Friction Cost": c_frict,
                        "Optimal Threshold": round(summary["threshold"], 2),
                        "Total Cost": round(summary["total_cost"], 2),
                        "Savings Vs 050": round(summary["savings_vs_050"], 2),
                        "Savings Vs Intervene All": round(summary["savings_vs_intervene_all"], 2),
                        "Savings Vs No Model": round(summary["savings_vs_no_model"], 2),
                    }
                )
    return rows


def main() -> None:
    configure_logging()
    thesis_dir = REPORTS_DIR / "thesis"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    cis: dict = {
        "generated_at_utc": stamp,
        "n_bootstraps": N_BOOTSTRAPS,
        "method": "percentile bootstrap, resampling test rows with replacement",
        "note": (
            "Model-comparison p-values reported elsewhere are exploratory and not "
            "corrected for multiple comparisons; these intervals quantify sampling "
            "uncertainty of single-model metrics."
        ),
    }
    calib: dict = {"generated_at_utc": stamp, "ece_bins": ECE_BINS}
    seg_rows: list[dict] = []
    scenario_rows: list[dict] = []

    for stream in ("per_record", "deduplicated"):
        data = _load_stream(stream)
        if data is None:
            continue
        logger.info("diagnostics_stream stream=%s n_test=%d", stream, len(data["y"]))
        cis[stream] = _extended_cis(data)
        calib[stream] = _calibration_diagnostics(data)
        seg_rows.extend(_calibration_by_segment(stream, data))
        if stream == "deduplicated":
            scenario_rows = _cost_scenarios(data)

    (thesis_dir / "confidence_intervals_extended.json").write_text(
        json.dumps(cis, indent=2, sort_keys=True), encoding="utf-8"
    )
    (thesis_dir / "calibration_diagnostics.json").write_text(
        json.dumps(calib, indent=2, sort_keys=True), encoding="utf-8"
    )
    if seg_rows:
        pd.DataFrame(seg_rows).to_csv(thesis_dir / "calibration_by_segment.csv", index=False)
    if scenario_rows:
        pd.DataFrame(scenario_rows).to_csv(thesis_dir / "cost_scenarios.csv", index=False)
    logger.info(
        "revision_diagnostics_written dir=%s segments=%d scenarios=%d",
        thesis_dir,
        len(seg_rows),
        len(scenario_rows),
    )


if __name__ == "__main__":
    main()
