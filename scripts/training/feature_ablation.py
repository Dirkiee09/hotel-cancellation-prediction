"""Deposit-type robustness ablation for the post-defense revisions (panel point 11).

The panel noted that ``deposit_type`` — the #1 SHAP driver — may partly reflect
how block bookings were recorded ("Non-Refund" tracking artifact), so a
near-deterministic feature could flatter the model. This experiment answers
"does the model remain useful without that feature?" with a number:

1. Re-runs the full training pipeline on the de-duplicated stream (identical
   to ``dedup_sensitivity.py``) with ``deposit_type`` neutralized to a single
   constant value. A constant column carries zero information, so this is a
   retrain-without-signal ablation that leaves the raw schema intact (no
   change to ``BOOKING_TIME_FEATURES`` or validation).
2. Writes cancellation-rate cross-tabs of deposit_type x market_segment and
   deposit_type x distribution_channel (the interaction evidence the panel
   asked for) from the same de-duplicated stream.

Outputs:
    reports/deposit_ablation.json           (deltas vs the de-dup baseline)
    reports/thesis/deposit_interactions.csv (interaction cross-tabs)

Run time ~10-20 min (full retrain). Usage:
    python scripts/training/feature_ablation.py
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR, TARGET_COL
from src.data.load import load_raw_data
from src.pipelines.train import run_training_pipeline
from src.utils import configure_logging

logger = logging.getLogger(__name__)

HEADLINE_KEYS = ("roc_auc", "pr_auc", "f1", "precision", "recall")
ABLATED_FEATURE = "deposit_type"
NEUTRAL_VALUE = "No Deposit"
SMALL_CELL_N = 100


def _write_interaction_crosstabs(deduped: pd.DataFrame) -> Path:
    """Cancellation-rate cross-tabs showing how the deposit effect varies by segment."""
    rows: list[dict[str, object]] = []
    for dim in ("market_segment", "distribution_channel"):
        grouped = deduped.groupby([ABLATED_FEATURE, dim], observed=True)[TARGET_COL].agg(
            ["count", "mean"]
        )
        for (deposit, segment), row in grouped.iterrows():
            rows.append(
                {
                    "Dimension": dim,
                    "Deposit Type": deposit,
                    "Segment": segment,
                    "N Bookings": int(row["count"]),
                    "Cancellation Rate": round(float(row["mean"]), 4),
                    "Small Sample Flag": bool(row["count"] < SMALL_CELL_N),
                }
            )
    out = REPORTS_DIR / "thesis" / "deposit_interactions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("deposit_interactions_written path=%s cells=%d", out, len(rows))
    return out


def _dedup_baseline() -> dict:
    """Headline metrics of the de-duplicated baseline run (preferred: persisted metrics)."""
    persisted = REPORTS_DIR / "dedup" / "metrics.json"
    if persisted.exists():
        metrics = json.loads(persisted.read_text(encoding="utf-8"))
        return {k: round(float(metrics["max_f1"][k]), 4) for k in HEADLINE_KEYS}
    summary = json.loads((REPORTS_DIR / "dedup_sensitivity.json").read_text(encoding="utf-8"))
    return {k: float(summary["deduplicated"][k]) for k in HEADLINE_KEYS}


def main() -> None:
    configure_logging()

    raw = load_raw_data()
    deduped = raw.drop_duplicates()
    logger.info("feature_ablation stream=dedup rows=%d", len(deduped))

    _write_interaction_crosstabs(deduped)

    ablated = deduped.copy()
    ablated[ABLATED_FEATURE] = NEUTRAL_VALUE

    with tempfile.TemporaryDirectory(prefix="deposit-ablation-") as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "hotel_bookings_dedup_no_deposit_signal.csv"
        ablated.to_csv(csv_path, index=False)

        outputs = run_training_pipeline(
            artifacts_dir=tmp_path / "artifacts",
            reports_dir=tmp_path / "reports",
            data_path=str(csv_path),
        )
        ablation_metrics = outputs.metrics

    baseline_h = _dedup_baseline()
    ablated_h = {k: round(float(ablation_metrics["max_f1"][k]), 4) for k in HEADLINE_KEYS}

    report = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "stream": "deduplicated",
        "ablated_feature": ABLATED_FEATURE,
        "neutralization": f"column set to constant '{NEUTRAL_VALUE}' (zero information)",
        "baseline_dedup": baseline_h,
        "ablated_dedup": ablated_h,
        "delta_ablated_minus_baseline": {
            k: round(ablated_h[k] - baseline_h[k], 4) for k in HEADLINE_KEYS
        },
        "selected_model_family_ablated": ablation_metrics["selected_model_family"],
        "interpretation": (
            "Retraining the identical pipeline on the de-duplicated stream with "
            "deposit_type neutralized quantifies how much the headline performance "
            "depends on that single (partly artifactual) feature. Small deltas mean "
            "the model does not hinge on the recording artifact; the remaining "
            "signal is carried by correlated structural features."
        ),
    }
    out_path = REPORTS_DIR / "deposit_ablation.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("deposit_ablation_written path=%s", out_path)
    print(json.dumps(report["delta_ablated_minus_baseline"], indent=2))


if __name__ == "__main__":
    main()
