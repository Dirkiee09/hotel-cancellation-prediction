"""Reproducibility check for cancellation training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Any

from src.config import REPRO_TOLERANCE
from src.pipelines import run_training_pipeline
from src.utils import configure_logging

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_numeric_deltas(
    left: Any,
    right: Any,
    *,
    prefix: str = "",
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left).intersection(right))
        for key in keys:
            nested_prefix = f"{prefix}.{key}" if prefix else key
            deltas.update(_collect_numeric_deltas(left[key], right[key], prefix=nested_prefix))
        return deltas

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if math.isnan(float(left)) and math.isnan(float(right)):
            deltas[prefix] = 0.0
        else:
            deltas[prefix] = abs(float(left) - float(right))
    return deltas


def run_repro_check(
    *, tolerance: float = REPRO_TOLERANCE, max_rows: int | None = None
) -> dict[str, Any]:
    """Train twice and verify key metrics/thresholds remain stable."""
    with tempfile.TemporaryDirectory(prefix="repro-check-") as tmp:
        root = Path(tmp)
        run1 = root / "run1"
        run2 = root / "run2"
        out1 = run_training_pipeline(
            artifacts_dir=run1 / "artifacts",
            reports_dir=run1 / "reports",
            max_rows=max_rows,
        )
        out2 = run_training_pipeline(
            artifacts_dir=run2 / "artifacts",
            reports_dir=run2 / "reports",
            max_rows=max_rows,
        )

        thr1 = _load_json(out1.artifacts_dir / "thresholds.json")
        thr2 = _load_json(out2.artifacts_dir / "thresholds.json")
        feat1 = _load_json(out1.artifacts_dir / "feature_columns.json")
        feat2 = _load_json(out2.artifacts_dir / "feature_columns.json")

        if feat1 != feat2:
            raise AssertionError("Feature columns differ across deterministic reruns.")

        metric_deltas = _collect_numeric_deltas(out1.metrics, out2.metrics)
        threshold_deltas = _collect_numeric_deltas(thr1, thr2)
        all_deltas = {**metric_deltas, **threshold_deltas}
        max_delta = max(all_deltas.values()) if all_deltas else 0.0
        unstable = {name: delta for name, delta in all_deltas.items() if delta > tolerance}
        if unstable:
            sample = dict(sorted(unstable.items(), key=lambda item: item[1], reverse=True)[:10])
            raise AssertionError(
                f"Reproducibility check failed: max_delta={max_delta} tolerance={tolerance} sample={sample}"
            )

        logger.info(
            "repro_check_passed tolerance=%s max_delta=%s rows_mode=%s",
            tolerance,
            max_delta,
            max_rows if max_rows is not None else "full",
        )
        return {
            "tolerance": tolerance,
            "max_delta": max_delta,
            "checked_values": len(all_deltas),
            "max_rows": max_rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic training reproducibility check."
    )
    parser.add_argument("--tolerance", type=float, default=REPRO_TOLERANCE)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    run_repro_check(tolerance=args.tolerance, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
