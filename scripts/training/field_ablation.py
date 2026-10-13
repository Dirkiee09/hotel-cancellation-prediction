"""Field ablation on the de-duplicated stream: what the headline depends on.

Retrains the full pipeline with one field (or a pair) neutralised to a single
constant, which carries zero information while leaving the raw schema intact,
in the same way ``feature_ablation.py`` does for ``deposit_type``. Variants:

  no_parking               required_car_parking_spaces = 0. Every booking that
                           requested parking arrived (0.0% of 7,416 cancelled),
                           which suggests the field may be filled in at check-in.
  no_parking_no_requests   ...and total_of_special_requests = 0, which guests can
                           also add after booking.
  no_country               country = "UNKNOWN". Country is the second-strongest
                           SHAP group; this replaces the country ablation that
                           notebook 03 used to read from files no script produced.

Output (this script is the sole writer):
    reports/field_ablation.json

Run time ~10 min (three retrains). Usage:
    python scripts/training/field_ablation.py
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import REPORTS_DIR
from src.data.load import load_raw_data
from src.pipelines.train import run_training_pipeline
from src.utils import configure_logging

logger = logging.getLogger(__name__)

HEADLINE_KEYS = ("roc_auc", "pr_auc", "f1", "precision", "recall")
VARIANTS: dict[str, dict[str, Any]] = {
    "no_parking": {"required_car_parking_spaces": 0},
    "no_parking_no_requests": {"required_car_parking_spaces": 0, "total_of_special_requests": 0},
    "no_country": {"country": "UNKNOWN"},
}


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    cost = metrics["cost_thresholding_test"]
    return {
        "selected_model_family": metrics["selected_model_family"],
        "max_f1": {k: round(float(metrics["max_f1"][k]), 4) for k in HEADLINE_KEYS},
        "ece_calibrated": round(float(metrics["calibration"]["test"]["ece_calibrated"]), 4),
        "cost_threshold": float(cost["threshold"]),
        "cost_cut_vs_contact_everyone_pct": round(
            100.0 * (cost["intervene_all_cost"] - cost["total_cost"]) / cost["intervene_all_cost"],
            2,
        ),
    }


def main() -> None:
    configure_logging()
    deduped = load_raw_data().drop_duplicates()
    logger.info("field_ablation stream=dedup rows=%d", len(deduped))

    baseline_path = REPORTS_DIR / "dedup" / "metrics.json"
    baseline = _summary(json.loads(baseline_path.read_text(encoding="utf-8")))

    variants: dict[str, Any] = {}
    for name, overrides in VARIANTS.items():
        ablated = deduped.copy()
        for column, value in overrides.items():
            ablated[column] = value
        with tempfile.TemporaryDirectory(prefix=f"field-ablation-{name}-") as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "data.csv"
            ablated.to_csv(csv_path, index=False)
            outputs = run_training_pipeline(
                artifacts_dir=tmp_path / "artifacts",
                reports_dir=tmp_path / "reports",
                data_path=str(csv_path),
            )
        result = _summary(outputs.metrics)
        result["neutralised"] = overrides
        result["delta_max_f1_vs_baseline"] = {
            k: round(result["max_f1"][k] - baseline["max_f1"][k], 4) for k in HEADLINE_KEYS
        }
        variants[name] = result
        logger.info("field_ablation variant=%s pr_auc=%.4f", name, result["max_f1"]["pr_auc"])

    report = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "stream": "deduplicated",
        "baseline": baseline,
        "variants": variants,
        "interpretation": (
            "Each variant retrains the identical pipeline with the listed fields set to a "
            "constant. A large drop means the headline depends on that field. Parking is "
            "the one to watch: if it is recorded at arrival rather than at booking, the "
            "no_parking row is the stricter estimate."
        ),
    }
    out_path = REPORTS_DIR / "field_ablation.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("field_ablation_written path=%s", out_path)


if __name__ == "__main__":
    main()
