"""Verify that threshold values are consistent across all artifact and report files.

Checks three sources of truth against each other:
  1. artifacts/thresholds.json          — canonical serving thresholds
  2. reports/thesis/model_family_summary.json  — thesis analysis snapshot
  3. reports/benchmarks/07_thresholds_per_model.csv — benchmark table

Fails with a non-zero exit code and a clear diff if any value is inconsistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import ARTIFACTS_DIR, REPORTS_DIR
from src.utils import configure_logging

logger = logging.getLogger(__name__)

_TOLERANCE = 1e-6


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_row(path: Path, model_name: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("model", "").strip().lower() == model_name.lower():
                return row
    raise ValueError(f"Model '{model_name}' not found in {path}")


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= _TOLERANCE


def run_sync_check(
    artifacts_dir: Path = ARTIFACTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> list[str]:
    """Return a list of mismatch error strings (empty = all consistent)."""
    errors: list[str] = []

    # ── 1. Load canonical thresholds from artifacts ────────────────────
    thresholds_path = artifacts_dir / "thresholds.json"
    if not thresholds_path.exists():
        return [f"MISSING: {thresholds_path} — run `make train` first"]
    raw = _load_json(thresholds_path)

    def _get_thr(policy: str) -> float | None:
        payload = raw.get(policy)
        if isinstance(payload, dict):
            val = payload.get("threshold")
            if isinstance(val, int | float):
                return float(val)
        return None

    canon_f1 = _get_thr("max_f1")
    canon_hp = _get_thr("high_precision")
    canon_cost = _get_thr("cost_sensitive")

    if canon_f1 is None or canon_hp is None:
        errors.append("artifacts/thresholds.json: missing max_f1 or high_precision threshold")
        return errors  # can't compare further

    # ── 2. Cross-check reports/thesis/model_family_summary.json ──────────
    summary_path = reports_dir / "thesis" / "model_family_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        thesis_f1 = summary.get("max_f1_threshold")
        thesis_cost = summary.get("cost_sensitive_threshold")

        if thesis_f1 is not None and not _near(float(thesis_f1), canon_f1):
            errors.append(
                f"max_f1 threshold mismatch: "
                f"artifacts={canon_f1} vs thesis/model_family_summary={thesis_f1}"
            )
        if canon_cost is not None and thesis_cost is not None:
            if not _near(float(thesis_cost), canon_cost):
                errors.append(
                    f"cost_sensitive threshold mismatch: "
                    f"artifacts={canon_cost} vs thesis/model_family_summary={thesis_cost}"
                )
    else:
        logger.warning("sync_check: %s not found — skipping thesis snapshot check", summary_path)

    # ── 3. Cross-check reports/benchmarks/07_thresholds_per_model.csv ──────
    bench_path = reports_dir / "benchmarks" / "07_thresholds_per_model.csv"
    if bench_path.exists():
        # champion model is always lightgbm per MODEL_SELECTION_POLICY
        try:
            row = _load_csv_row(bench_path, "lightgbm")
        except ValueError as exc:
            errors.append(f"benchmarks/07_thresholds_per_model.csv: {exc}")
            row = {}

        if row:
            bench_f1 = row.get("threshold_max_f1")
            bench_hp = row.get("threshold_high_precision")
            bench_cost = row.get("threshold_cost_sensitive")

            if bench_f1 is not None and not _near(float(bench_f1), canon_f1):
                errors.append(
                    f"max_f1 threshold mismatch: "
                    f"artifacts={canon_f1} vs benchmarks/07={bench_f1}"
                )
            if bench_hp is not None and not _near(float(bench_hp), canon_hp):
                errors.append(
                    f"high_precision threshold mismatch: "
                    f"artifacts={canon_hp} vs benchmarks/07={bench_hp}"
                )
            if canon_cost is not None and bench_cost is not None:
                if not _near(float(bench_cost), canon_cost):
                    errors.append(
                        f"cost_sensitive threshold mismatch: "
                        f"artifacts={canon_cost} vs benchmarks/07={bench_cost}"
                    )
    else:
        logger.warning("sync_check: %s not found — skipping benchmark check", bench_path)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that thresholds are consistent across artifacts and reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Artifacts directory to validate against.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Reports directory containing thesis/ and benchmarks/ subdirectories.",
    )
    args = parser.parse_args()

    configure_logging()
    errors = run_sync_check(
        artifacts_dir=args.artifacts_dir.resolve(),
        reports_dir=args.reports_dir.resolve(),
    )

    if errors:
        logger.error("sync_check FAILED — %d mismatch(es):", len(errors))
        for msg in errors:
            logger.error("  ✗ %s", msg)
        logger.error(
            "Run `make train && make benchmark && make thesis-analysis-fast` "
            "to regenerate reports from current artifacts."
        )
        sys.exit(1)

    logger.info(
        "sync_check OK — thresholds consistent across artifacts, thesis snapshot, and benchmarks"
    )


if __name__ == "__main__":
    main()
