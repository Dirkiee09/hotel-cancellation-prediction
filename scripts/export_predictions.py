"""Export the prediction-log SQLite database to a Power BI-friendly CSV.

Usage:
    python scripts/export_predictions.py                  # default paths
    python scripts/export_predictions.py --since 2026-05-01  # filter window
    python scripts/export_predictions.py --reset          # drop the DB (cleanup)

Default paths (both git-ignored, configured in src/config.py):
    DB:  data/predictions/predictions.sqlite (source of truth)
    CSV: data/predictions/predictions_live.csv (Power BI consumes this)

Never edit the CSV directly; it is regenerated on every export.

Power BI connection recipe:
    1. Open Power BI Desktop
    2. Home > Get Data > Text/CSV
    3. Browse to <repo>/data/predictions/predictions_live.csv
    4. Load
    5. After new predictions arrive, run this script and click Refresh in Power BI

The CSV has one row per /predict call with the full BookingRequest fields,
the PredictionResponse fields (probability, threshold labels, risk_tier),
and the top SHAP-contributing features as a JSON column.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/export_predictions.py` from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PREDICTION_LOG_CSV, PREDICTION_LOG_DB  # noqa: E402


def export(
    db_path: Path = PREDICTION_LOG_DB,
    csv_path: Path = PREDICTION_LOG_CSV,
    since: str | None = None,
) -> int:
    """Write the predictions table to CSV. Returns the number of rows exported."""
    if not db_path.exists():
        print(f"No prediction log found at {db_path}. Nothing to export.")
        return 0

    where = ""
    params: tuple = ()
    if since is not None:
        where = "WHERE timestamp_utc >= ?"
        params = (since,)

    with sqlite3.connect(db_path) as conn:
        # `where` is a literal string from this module (either "" or
        # "WHERE timestamp_utc >= ?"); the only user-controlled value (`since`)
        # is bound via the params tuple, not interpolated into the SQL.
        df = pd.read_sql(
            f"SELECT * FROM predictions {where} ORDER BY timestamp_utc",  # nosec B608
            conn,
            params=params,
        )

    if df.empty:
        print(f"Predictions table is empty (since={since!r}). Nothing to export.")
        return 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Exported {len(df):,} predictions to {csv_path}")
    return len(df)


def reset(db_path: Path = PREDICTION_LOG_DB) -> None:
    """Delete the SQLite DB. Use to wipe demo / test data."""
    if db_path.exists():
        db_path.unlink()
        print(f"Deleted {db_path}")
    else:
        print(f"No prediction log at {db_path} — nothing to delete.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO 8601 timestamp; only export predictions logged on or after this.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the SQLite DB instead of exporting (use to wipe demo data).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PREDICTION_LOG_DB,
        help=f"Path to the SQLite DB (default: {PREDICTION_LOG_DB})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PREDICTION_LOG_CSV,
        help=f"Path to write the CSV (default: {PREDICTION_LOG_CSV})",
    )
    args = parser.parse_args()

    if args.reset:
        reset(args.db)
        return 0

    export(db_path=args.db, csv_path=args.csv, since=args.since)
    return 0


if __name__ == "__main__":
    sys.exit(main())
