"""Adapt the Kaggle Hotel-Reservations dataset to the project schema.

The project expects the Antonio/Almeida 2019 hotel_bookings.csv schema (32 columns).
The Hotel-Reservations-1.csv dataset (Kaggle/Bansal) uses a different schema (19 columns,
``no_of_*`` naming). This script renames columns, maps target labels, fills missing
booking-time fields with sensible defaults, and writes a project-compatible CSV.

Usage::

    python scripts/adapt_dataset.py \\
        --input data/Hotel-Reservations-1.csv \\
        --output data/hotel_bookings.csv

After running, the standard pipeline works unchanged::

    make train
    make eval
"""

from __future__ import annotations

import argparse
import calendar
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Column renames: source → project schema ──────────────────────────────
COLUMN_RENAME = {
    "no_of_adults": "adults",
    "no_of_children": "children",
    "no_of_weekend_nights": "stays_in_weekend_nights",
    "no_of_week_nights": "stays_in_week_nights",
    "type_of_meal_plan": "meal",
    "required_car_parking_space": "required_car_parking_spaces",
    "room_type_reserved": "reserved_room_type",
    "arrival_year": "arrival_date_year",
    "arrival_date": "arrival_date_day_of_month",
    "market_segment_type": "market_segment",
    "repeated_guest": "is_repeated_guest",
    "no_of_previous_cancellations": "previous_cancellations",
    "no_of_previous_bookings_not_canceled": "previous_bookings_not_canceled",
    "avg_price_per_room": "adr",
    "no_of_special_requests": "total_of_special_requests",
}

# ── Categorical value normalization (source → project conventions) ───────
MEAL_MAP = {
    "Meal Plan 1": "BB",
    "Meal Plan 2": "HB",
    "Meal Plan 3": "FB",
    "Not Selected": "Undefined",
}

ROOM_TYPE_MAP = {f"Room_Type {i}": chr(ord("A") + i - 1) for i in range(1, 8)}

# Market segment → distribution channel (project expects both columns)
SEGMENT_TO_CHANNEL = {
    "Online": "TA/TO",
    "Offline": "TA/TO",
    "Corporate": "Corporate",
    "Aviation": "Direct",
    "Complementary": "Direct",
}


def _adapt(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Hotel-Reservations rows into the project schema."""
    out = df.rename(columns=COLUMN_RENAME).copy()

    # Target: "Not_Canceled"/"Canceled" → 0/1
    if "booking_status" not in out.columns:
        raise ValueError("Source CSV missing 'booking_status' column.")
    out["is_canceled"] = (out["booking_status"] == "Canceled").astype(int)
    out = out.drop(columns=["booking_status", "Booking_ID"], errors="ignore")

    # arrival_date_month: numeric (1-12) → string month name ("January", ...)
    month_num = pd.to_numeric(out["arrival_month"], errors="coerce").fillna(1).astype(int)
    out["arrival_date_month"] = month_num.map(lambda m: calendar.month_name[max(1, min(12, m))])

    # arrival_date_week_number: derived from full date
    arrival_dt = pd.to_datetime(
        {
            "year": out["arrival_date_year"],
            "month": month_num,
            "day": out["arrival_date_day_of_month"],
        },
        errors="coerce",
    )
    out["arrival_date_week_number"] = arrival_dt.dt.isocalendar().week.astype("Int64")
    out = out.drop(columns=["arrival_month"], errors="ignore")

    # Drop rows with invalid arrival dates (Feb 30 etc.)
    invalid = arrival_dt.isna()
    if invalid.any():
        logger.info("Dropping %d rows with invalid arrival dates.", int(invalid.sum()))
        out = out.loc[~invalid].copy()

    # Categorical normalizations
    out["meal"] = out["meal"].map(MEAL_MAP).fillna(out["meal"])
    out["reserved_room_type"] = (
        out["reserved_room_type"].map(ROOM_TYPE_MAP).fillna(out["reserved_room_type"])
    )

    # distribution_channel inferred from market_segment
    out["distribution_channel"] = out["market_segment"].map(SEGMENT_TO_CHANNEL).fillna("TA/TO")

    # Default-fill missing booking-time fields the project requires.
    # These are dataset-agnostic placeholders — the model treats them as constants
    # for this dataset, but predictions remain valid because the OneHotEncoder uses
    # handle_unknown="ignore" and the median imputer handles numerics.
    defaults = {
        "hotel": "City Hotel",
        "country": "UNKNOWN",
        "babies": 0,
        "deposit_type": "No Deposit",
        "agent": "NULL",
        "customer_type": "Transient",
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value

    # Sort chronologically — required by split_time_aware()
    out = out.sort_values(
        ["arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Hotel-Reservations-1.csv"),
        help="Source CSV (Kaggle Hotel-Reservations schema).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hotel_bookings.csv"),
        help="Destination CSV (project schema). Existing file is backed up to .bak.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up an existing destination CSV.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    logger.info("Reading %s", args.input)
    src = pd.read_csv(args.input)
    logger.info("Source shape: %s rows × %s cols", *src.shape)

    adapted = _adapt(src)
    logger.info(
        "Adapted shape: %s rows × %s cols  (cancel rate: %.1f%%)",
        len(adapted),
        adapted.shape[1],
        100 * adapted["is_canceled"].mean(),
    )

    if args.output.exists() and not args.no_backup:
        backup = args.output.with_suffix(args.output.suffix + ".bak")
        logger.info("Backing up existing %s → %s", args.output, backup)
        shutil.copy2(args.output, backup)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_csv(args.output, index=False)
    logger.info("Wrote %s", args.output)
    logger.info("Next steps:  make train  →  make eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
