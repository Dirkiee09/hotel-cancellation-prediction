"""Basic data validation checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import (
    ADR_MAX_VALID,
    BOOKING_TIME_FEATURES,
    LEAKAGE_COLS,
    PH_BOOKING_TIME_FEATURES,
    PH_COLUMN_RENAMES,
    PH_TARGET_COL,
    TARGET_COL,
)
from src.features.build import add_derived_booking_features

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    checks: dict[str, bool]
    messages: list[str]


NON_NEGATIVE_COLS = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_stay",
    "total_guests",
    "adr_per_person",
    "revenue_at_risk",
]


def leakage_columns_present(columns: list[str]) -> list[str]:
    """Return leakage/post-outcome columns if present in a feature set."""
    return sorted(set(columns).intersection(LEAKAGE_COLS))


def assert_no_leakage_columns(columns: list[str]) -> None:
    """Raise on leakage/post-outcome features to prevent silent data leakage."""
    leaking = leakage_columns_present(columns)
    if leaking:
        raise ValueError(f"Leakage columns found in feature set: {leaking}")


def clean_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean known data issues and build engineered booking-time features."""
    issues: dict[str, int] = {}

    # Count nulls before normalization fills them
    if "children" in df.columns:
        issues["children_null_filled_zero"] = int(df["children"].isna().sum())
    else:
        issues["children_null_filled_zero"] = 0

    df = add_derived_booking_features(df.copy())

    if "agent" in df.columns:
        issues["agent_filled_as_direct"] = int((df["agent"] == 0).sum())
    if "country" in df.columns:
        issues["country_unknown_rows"] = int(
            df["country"].astype(str).str.strip().str.lower().eq("unknown").sum()
        )

    if "adr" in df.columns and "total_guests" in df.columns:
        adr = pd.to_numeric(df["adr"], errors="coerce")
        neg_mask = adr < 0
        high_mask = adr >= ADR_MAX_VALID
        total_guests_num = pd.to_numeric(df["total_guests"], errors="coerce")
        zero_guest_mask = total_guests_num.fillna(0) <= 0

        issues["rows_dropped_negative_adr"] = int(neg_mask.sum())
        issues["rows_dropped_adr_outlier"] = int(high_mask.sum())
        issues["rows_dropped_zero_guests"] = int((total_guests_num == 0).sum())
        issues["rows_dropped_null_guests"] = int(total_guests_num.isna().sum())

        valid_mask = ~(neg_mask | high_mask | zero_guest_mask)
        df = df.loc[valid_mask].copy()
        issues["rows_dropped_total"] = int((~valid_mask).sum())
    else:
        issues["rows_dropped_negative_adr"] = 0
        issues["rows_dropped_adr_outlier"] = 0
        issues["rows_dropped_zero_guests"] = 0
        issues["rows_dropped_total"] = 0

    if "company" in df.columns:
        df = df.drop(columns=["company"])

    total_dropped = issues.get("rows_dropped_total", 0)
    if total_dropped > 0:
        logger.info(
            "clean_raw: dropped %d rows (neg_adr=%d, adr_outlier=%d, zero_guests=%d)",
            total_dropped,
            issues.get("rows_dropped_negative_adr", 0),
            issues.get("rows_dropped_adr_outlier", 0),
            issues.get("rows_dropped_zero_guests", 0),
        )

    return df, issues


def validate_raw(df: pd.DataFrame) -> ValidationResult:
    """Run basic validation checks on raw data."""
    checks: dict[str, bool] = {}
    messages: list[str] = []

    required_cols = set(BOOKING_TIME_FEATURES + [TARGET_COL])
    missing = sorted(required_cols - set(df.columns))
    checks["required_columns_present"] = len(missing) == 0
    if missing:
        messages.append(f"Missing required columns: {missing}")

    target_non_empty = TARGET_COL in df.columns and df[TARGET_COL].notna().any()
    checks["target_non_empty"] = bool(target_non_empty)
    if not target_non_empty:
        messages.append("Target column has no non-null values.")

    target_ok = False
    if TARGET_COL in df.columns and target_non_empty:
        target_ok = bool(df[TARGET_COL].dropna().isin([0, 1]).all())
    checks["target_binary"] = bool(target_ok)
    if not target_ok:
        messages.append("Target column must be binary 0/1.")

    nonneg_ok = True
    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            if (df[col].dropna() < 0).any():
                nonneg_ok = False
                messages.append(f"Negative values found in {col}.")
    checks["non_negative_numeric"] = nonneg_ok

    passed = all(checks.values())
    return ValidationResult(passed=passed, checks=checks, messages=messages)


# ── Philippine transferability probe ─────────────────────────────────
# These helpers normalise the PH raw export (Punta_Villa_Resort_2022_2024.csv)
# to the project's canonical column shape, so downstream feature engineering
# (add_derived_booking_features) can be reused. The PH pipeline lives in
# scripts/train_ph.py; this module just provides the data layer.


def clean_raw_ph(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalise PH raw data to the project's canonical column shape.

    Steps:
        1. Rename PH columns to project-canonical names (Lead_Time_Days
           -> lead_time, etc.) via ``PH_COLUMN_RENAMES``.
        2. Binarise ``Booking_Status`` ("Cancelled" -> 1, "Checked-in" -> 0)
           into ``is_canceled``.
        3. Parse ``Arrival_Date`` into year/month/day_of_month columns and
           a month-name string column, so ``add_derived_booking_features``
           can compute ``month_sin``/``month_cos``.
        4. Drop constant-variance columns (``Meals``, ``Guest_Type``),
           the original status/date/ID columns, and the redundant
           ``Nights_Stayed`` (which equals ``Weekend_Nights + Week_Nights``).
        5. Apply the Portugal-shared feature derivation helper so the
           output exposes ``total_stay``, ``total_guests``, ``month_sin``,
           ``revenue_at_risk``, etc.

    Returns
    -------
    (df, issues): cleaned DataFrame and a dict of per-step row counters
    for traceability (mirrors the Portugal ``clean_raw`` contract).
    """
    issues: dict[str, int] = {}
    out = df.copy()

    out = out.rename(columns=PH_COLUMN_RENAMES)

    if "Booking_Status" in out.columns:
        valid_mask = out["Booking_Status"].isin(["Cancelled", "Checked-in"])
        issues["rows_dropped_unknown_status"] = int((~valid_mask).sum())
        out = out.loc[valid_mask].copy()
        out[PH_TARGET_COL] = (out["Booking_Status"] == "Cancelled").astype(int)
    else:
        raise ValueError(
            "PH dataset must contain a 'Booking_Status' column "
            "with values 'Cancelled' / 'Checked-in'."
        )

    if "Arrival_Date" not in out.columns:
        raise ValueError("PH dataset must contain an 'Arrival_Date' column (YYYY-MM-DD).")
    arrival = pd.to_datetime(out["Arrival_Date"], errors="coerce")
    nat_count = int(arrival.isna().sum())
    issues["rows_dropped_unparseable_arrival_date"] = nat_count
    if nat_count > 0:
        out = out.loc[arrival.notna()].copy()
        arrival = arrival.loc[arrival.notna()]
    out["arrival_date_year"] = arrival.dt.year.astype(int)
    out["arrival_date_day_of_month"] = arrival.dt.day.astype(int)
    out["arrival_date_month"] = arrival.dt.month_name()

    drop_cols = [
        "Booking_Status",
        "Booking_ID",
        "Meals",
        "Guest_Type",
        "Nights_Stayed",
        "Booking_Date",
        "Arrival_Date",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    out = add_derived_booking_features(out)

    return out, issues


def validate_raw_ph(df: pd.DataFrame) -> ValidationResult:
    """Validate the cleaned PH DataFrame against the reduced feature schema."""
    checks: dict[str, bool] = {}
    messages: list[str] = []

    required_cols = set(PH_BOOKING_TIME_FEATURES + [PH_TARGET_COL])
    missing = sorted(required_cols - set(df.columns))
    checks["required_columns_present"] = len(missing) == 0
    if missing:
        messages.append(f"Missing required columns: {missing}")

    target_non_empty = PH_TARGET_COL in df.columns and df[PH_TARGET_COL].notna().any()
    checks["target_non_empty"] = bool(target_non_empty)
    if not target_non_empty:
        messages.append("Target column has no non-null values.")

    target_ok = False
    if PH_TARGET_COL in df.columns and target_non_empty:
        target_ok = bool(df[PH_TARGET_COL].dropna().isin([0, 1]).all())
    checks["target_binary"] = bool(target_ok)
    if not target_ok:
        messages.append("Target column must be binary 0/1.")

    passed = all(checks.values())
    return ValidationResult(passed=passed, checks=checks, messages=messages)
