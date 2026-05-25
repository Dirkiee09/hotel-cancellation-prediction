"""Smoke tests for the Philippine transferability probe data layer.

These tests verify that ``load_ph_data`` and ``clean_raw_ph`` produce a
DataFrame that the PH training pipeline can consume. They do NOT verify
model quality (the dataset is too small for meaningful regression
detection), they only verify the data-cleaning contract.

The Punta Villa Resort CSV is a proprietary PMS export and is gitignored
(see .gitignore line `data/*.csv`); it ships locally but is never pushed
to remote. CI runners therefore have no file to load. This entire test
module is skipped when the file is absent so the PH data-layer contract
is still verified locally (where the file exists) without breaking CI
(where it does not). See CLAUDE.md "PH Sub-Study" for the parallel
intent that the PH model is not part of the Portugal CI pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import PH_BOOKING_TIME_FEATURES, PH_DATA_PATH, PH_TARGET_COL
from src.data.load_ph import load_ph_data
from src.utils.validate_data import clean_raw_ph, validate_raw_ph

pytestmark = pytest.mark.skipif(
    not Path(PH_DATA_PATH).exists(),
    reason=(
        f"PH dataset not found at {PH_DATA_PATH}; skipping PH data-layer tests. "
        "The Punta Villa Resort CSV is gitignored (proprietary PMS export) and "
        "is expected to be absent in CI environments."
    ),
)


def test_load_ph_returns_expected_rows() -> None:
    """The real PH CSV ships with 193 rows; alert if the file gets swapped."""
    df = load_ph_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 193
    assert "Booking_Status" in df.columns


def test_clean_raw_ph_binarises_target() -> None:
    """Target must become {0, 1}; 29 cancellations expected in the real data."""
    df = load_ph_data()
    cleaned, _ = clean_raw_ph(df)
    assert PH_TARGET_COL in cleaned.columns
    assert set(cleaned[PH_TARGET_COL].unique()) <= {0, 1}
    assert int(cleaned[PH_TARGET_COL].sum()) == 29


def test_clean_raw_ph_renames_to_canonical_columns() -> None:
    """Source columns are renamed to project-canonical Portugal-style names."""
    df = load_ph_data()
    cleaned, _ = clean_raw_ph(df)
    for col in (
        "lead_time",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "adr",
        "reserved_room_type",
        "deposit_type",
        "total_of_special_requests",
    ):
        assert col in cleaned.columns, f"missing canonical column: {col}"


def test_clean_raw_ph_produces_all_required_features() -> None:
    """Every column in PH_BOOKING_TIME_FEATURES must exist after cleaning."""
    df = load_ph_data()
    cleaned, _ = clean_raw_ph(df)
    missing = [c for c in PH_BOOKING_TIME_FEATURES if c not in cleaned.columns]
    assert missing == [], f"missing features after clean: {missing}"


def test_clean_raw_ph_drops_constant_variance_columns() -> None:
    """Meals and Guest_Type are constant; they must not survive cleaning."""
    df = load_ph_data()
    cleaned, _ = clean_raw_ph(df)
    assert "Meals" not in cleaned.columns
    assert "Guest_Type" not in cleaned.columns
    assert "Booking_ID" not in cleaned.columns


def test_clean_raw_ph_parses_arrival_date_components() -> None:
    """Arrival_Date string is split into year/month/day for downstream split."""
    df = load_ph_data()
    cleaned, _ = clean_raw_ph(df)
    assert "arrival_date_year" in cleaned.columns
    assert "arrival_date_month" in cleaned.columns
    assert "arrival_date_day_of_month" in cleaned.columns
    assert cleaned["arrival_date_year"].between(2022, 2025).all()
    assert cleaned["arrival_date_day_of_month"].between(1, 31).all()


def test_clean_raw_ph_derives_cyclic_features() -> None:
    """month_sin/month_cos must be present and within the unit circle."""
    df = load_ph_data()
    cleaned, _ = clean_raw_ph(df)
    assert "month_sin" in cleaned.columns
    assert "month_cos" in cleaned.columns
    # sin^2 + cos^2 should be ~1 for every row (cyclic encoding sanity)
    radii_sq = cleaned["month_sin"] ** 2 + cleaned["month_cos"] ** 2
    assert (radii_sq.between(0.999, 1.001)).all()


def test_clean_raw_ph_is_idempotent_on_validation() -> None:
    """validate_raw_ph must pass on freshly cleaned data — no schema drift."""
    df = load_ph_data()
    cleaned, _ = clean_raw_ph(df)
    result = validate_raw_ph(cleaned)
    assert result.passed, f"validation failed: {result.messages}"


def test_clean_raw_ph_does_not_mutate_input() -> None:
    """clean_raw_ph must operate on a copy — input DF stays untouched."""
    df = load_ph_data()
    before_columns = list(df.columns)
    before_len = len(df)
    _ = clean_raw_ph(df)
    assert list(df.columns) == before_columns
    assert len(df) == before_len
