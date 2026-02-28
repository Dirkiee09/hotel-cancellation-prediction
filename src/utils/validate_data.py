"""Basic data validation checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ADR_MAX_VALID, BOOKING_TIME_FEATURES, LEAKAGE_COLS, TARGET_COL
from src.features.build import add_derived_booking_features


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
        zero_guest_mask = pd.to_numeric(df["total_guests"], errors="coerce").fillna(0) <= 0

        issues["rows_dropped_negative_adr"] = int(neg_mask.sum())
        issues["rows_dropped_adr_outlier"] = int(high_mask.sum())
        issues["rows_dropped_zero_guests"] = int(zero_guest_mask.sum())

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
