"""Compute drift metrics for the Power BI monitoring dashboard page.

Compares the live prediction stream (data/predictions/predictions_live.csv)
against the training-set holdout baseline (reports/test_predictions_for_powerbi.csv)
and writes a small CSV that Power BI Page 8 ("Is the model staying healthy?")
loads.

Usage:
    python scripts/compute_live_drift.py
    python scripts/compute_live_drift.py --min-rows 50   # require at least
                                                         # this many live rows

Output: data/predictions/drift_metrics.csv with columns
    feature, kind, psi, zone, n_live, n_baseline, computed_at_utc

Where:
    * `kind`   is "numeric" or "categorical"
    * `psi`    is the Population Stability Index
    * `zone`   is "stable" (PSI < 0.10) / "monitor" (0.10-0.25) / "retrain" (>=0.25)
    * `computed_at_utc` lets Power BI show "last updated X ago" on the page
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow `python scripts/compute_live_drift.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PREDICTION_LOG_CSV, PROJECT_ROOT, REPORTS_DIR  # noqa: E402
from src.utils.drift import cat_psi, compute_psi, psi_zone  # noqa: E402

BASELINE_CSV = REPORTS_DIR / "test_predictions_for_powerbi.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "predictions" / "drift_metrics.csv"

# Features to monitor. The probability column has different names in the live
# CSV vs the training-set CSV (probability vs cancel_probability) so we map
# both to a unified label in the output.
NUMERIC_FEATURES: list[str] = ["lead_time", "adr"]
CATEGORICAL_FEATURES: list[str] = [
    "country",
    "market_segment",
    "deposit_type",
    "customer_type",
    "agent",
]


def compute_drift(
    live: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Compute PSI per feature + a unified 'predicted_probability' drift row."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_live = len(live)
    n_base = len(baseline)
    rows: list[dict[str, object]] = []

    # Probability drift: probability (live) vs cancel_probability (baseline)
    if "probability" in live.columns and "cancel_probability" in baseline.columns:
        psi = compute_psi(live["probability"].dropna(), baseline["cancel_probability"].dropna())
        rows.append(
            {
                "feature": "predicted_probability",
                "kind": "numeric",
                "psi": round(psi, 6),
                "zone": psi_zone(psi),
                "n_live": n_live,
                "n_baseline": n_base,
                "computed_at_utc": now,
            }
        )

    for feat in NUMERIC_FEATURES:
        if feat not in live.columns or feat not in baseline.columns:
            continue
        psi = compute_psi(live[feat].dropna(), baseline[feat].dropna())
        rows.append(
            {
                "feature": feat,
                "kind": "numeric",
                "psi": round(psi, 6),
                "zone": psi_zone(psi),
                "n_live": n_live,
                "n_baseline": n_base,
                "computed_at_utc": now,
            }
        )

    for feat in CATEGORICAL_FEATURES:
        if feat not in live.columns or feat not in baseline.columns:
            continue
        psi = cat_psi(live[feat].astype(str), baseline[feat].astype(str))
        rows.append(
            {
                "feature": feat,
                "kind": "categorical",
                "psi": round(psi, 6),
                "zone": psi_zone(psi),
                "n_live": n_live,
                "n_baseline": n_base,
                "computed_at_utc": now,
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--live",
        type=Path,
        default=PREDICTION_LOG_CSV,
        help=f"Live predictions CSV (default: {PREDICTION_LOG_CSV})",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_CSV,
        help=f"Training-set baseline CSV (default: {BASELINE_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Drift metrics CSV (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10,
        help="Minimum number of live rows required to compute drift (default: 10)",
    )
    args = parser.parse_args()

    if not args.live.exists():
        print(f"No live predictions CSV at {args.live}. Run a few /predict calls first.")
        return 1
    if not args.baseline.exists():
        print(
            f"No baseline CSV at {args.baseline}. Run `make train` to "
            f"produce reports/test_predictions_for_powerbi.csv."
        )
        return 1

    live = pd.read_csv(args.live)
    if len(live) < args.min_rows:
        print(
            f"Only {len(live)} live predictions (minimum {args.min_rows} required). "
            f"Make more predictions through the Gradio UI first."
        )
        return 1

    baseline = pd.read_csv(args.baseline)

    drift = compute_drift(live, baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    drift.to_csv(args.output, index=False)

    print(
        f"Computed drift for {len(drift)} features (live n={len(live)}, baseline n={len(baseline)})"
    )
    print()
    print(drift.to_string(index=False))
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
