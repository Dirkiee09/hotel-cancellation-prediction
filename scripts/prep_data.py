"""Prepare processed data for exploration or reuse."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BOOKING_TIME_FEATURES, LEAKAGE_COLS, TARGET_COL
from src.data.load import load_raw_data
from src.utils.validate_data import clean_raw


def main() -> None:
    df = load_raw_data()
    df, _ = clean_raw(df)
    df = df.drop(columns=[col for col in LEAKAGE_COLS if col in df.columns])
    df = df[BOOKING_TIME_FEATURES + [TARGET_COL]].copy()

    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hotel_bookings_processed.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
