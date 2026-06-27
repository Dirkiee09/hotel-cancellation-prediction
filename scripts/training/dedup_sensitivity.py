"""Duplicate-row sensitivity experiment for the thesis limitations section.

The raw dataset contains ~32k exact duplicate rows (likely tour-operator
block bookings). The 2026-06 audit verified they do NOT leak across the
chronological train/test boundary, but within-split duplication still
(a) weights repeated bookings more heavily during training and
(b) violates the i.i.d. assumption behind bootstrap confidence intervals.

This experiment retrains the full pipeline on a fully de-duplicated copy of
the dataset and reports the headline test metrics next to the production
run, giving the thesis a *number* for "how much do duplicates matter"
instead of a hand-wave.

Writes reports/dedup_sensitivity.json. Run time ~10-20 min (full retrain).

Usage:
    python scripts/dedup_sensitivity.py
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.config import REPORTS_DIR
from src.data.load import load_raw_data
from src.pipelines.train import run_training_pipeline
from src.utils import configure_logging

logger = logging.getLogger(__name__)

HEADLINE_KEYS = ("roc_auc", "pr_auc", "f1", "precision", "recall")


def main() -> None:
    configure_logging()

    raw = load_raw_data()
    n_before = len(raw)
    deduped = raw.drop_duplicates()
    n_after = len(deduped)
    logger.info(
        "dedup_sensitivity rows_before=%d rows_after=%d removed=%d (%.1f%%)",
        n_before,
        n_after,
        n_before - n_after,
        100.0 * (n_before - n_after) / n_before,
    )

    with tempfile.TemporaryDirectory(prefix="dedup-sensitivity-") as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "hotel_bookings_dedup.csv"
        deduped.to_csv(csv_path, index=False)

        outputs = run_training_pipeline(
            artifacts_dir=tmp_path / "artifacts",
            reports_dir=tmp_path / "reports",
            data_path=str(csv_path),
        )
        dedup_metrics = outputs.metrics

    baseline = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))

    def _headline(metrics: dict) -> dict:
        return {k: round(float(metrics["max_f1"][k]), 4) for k in HEADLINE_KEYS}

    base_h = _headline(baseline)
    dedup_h = _headline(dedup_metrics)
    report = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "rows_raw": n_before,
        "rows_after_dedup": n_after,
        "rows_removed": n_before - n_after,
        "duplicate_share": round((n_before - n_after) / n_before, 4),
        "baseline_with_duplicates": {
            **base_h,
            "selected_model_family": baseline["selected_model_family"],
        },
        "deduplicated": {
            **dedup_h,
            "selected_model_family": dedup_metrics["selected_model_family"],
        },
        "delta_dedup_minus_baseline": {k: round(dedup_h[k] - base_h[k], 4) for k in HEADLINE_KEYS},
        "interpretation": (
            "Metrics at the max_f1 policy on each run's own chronological test split. "
            "Duplicates never cross the train/test boundary (verified separately), so "
            "deltas here reflect re-weighting of repeated bookings, not leakage. Small "
            "deltas support reporting the with-duplicates results as primary; the "
            "remaining caveat is that bootstrap CIs assume independent rows and are "
            "therefore somewhat narrow under within-split duplication."
        ),
    }

    out_path = REPORTS_DIR / "dedup_sensitivity.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("dedup_sensitivity_written path=%s", out_path)
    print(json.dumps(report["delta_dedup_minus_baseline"], indent=2))


if __name__ == "__main__":
    main()
