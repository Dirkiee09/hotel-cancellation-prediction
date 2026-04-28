"""Tests for src/utils/validate_data.py — clean_raw and validate_raw."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import ADR_MAX_VALID, TARGET_COL
from src.utils.validate_data import (
    assert_no_leakage_columns,
    clean_raw,
    leakage_columns_present,
    validate_raw,
)


def _make_raw_df(n: int = 20, **overrides) -> pd.DataFrame:
    """Minimal DataFrame that resembles hotel_bookings.csv."""
    data = {
        "hotel": ["Resort Hotel"] * n,
        "lead_time": [100] * n,
        "arrival_date_year": [2017] * n,
        "arrival_date_month": ["July"] * n,
        "arrival_date_week_number": [27] * n,
        "arrival_date_day_of_month": [1] * n,
        "stays_in_weekend_nights": [1] * n,
        "stays_in_week_nights": [2] * n,
        "adults": [2] * n,
        "children": [0.0] * n,
        "babies": [0] * n,
        "meal": ["BB"] * n,
        "country": ["PRT"] * n,
        "market_segment": ["Direct"] * n,
        "distribution_channel": ["Direct"] * n,
        "is_repeated_guest": [0] * n,
        "previous_cancellations": [0] * n,
        "previous_bookings_not_canceled": [0] * n,
        "reserved_room_type": ["C"] * n,
        "deposit_type": ["No Deposit"] * n,
        "agent": [0] * n,
        "company": [0] * n,
        "customer_type": ["Transient"] * n,
        "adr": [100.0] * n,
        "required_car_parking_spaces": [0] * n,
        "total_of_special_requests": [1] * n,
        TARGET_COL: [0, 1] * (n // 2),
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ── clean_raw ────────────────────────────────────────────────────────────────


class TestCleanRaw:
    def test_drops_negative_adr(self):
        df = _make_raw_df(n=4, adr=[-10.0, 50.0, 80.0, 120.0])
        cleaned, issues = clean_raw(df)
        assert issues["rows_dropped_negative_adr"] == 1
        assert (cleaned["adr"] >= 0).all()

    def test_drops_adr_above_max(self):
        df = _make_raw_df(n=4, adr=[50.0, 80.0, ADR_MAX_VALID + 1, 120.0])
        cleaned, issues = clean_raw(df)
        assert issues["rows_dropped_adr_outlier"] >= 1
        assert len(cleaned) < 4

    def test_drops_zero_guest_rows(self):
        df = _make_raw_df(n=4, adults=[0, 2, 2, 2], children=[0, 0, 0, 0], babies=[0, 0, 0, 0])
        cleaned, issues = clean_raw(df)
        assert issues["rows_dropped_zero_guests"] >= 1

    def test_drops_company_column(self):
        df = _make_raw_df()
        assert "company" in df.columns
        cleaned, _ = clean_raw(df)
        assert "company" not in cleaned.columns

    def test_adds_derived_features(self):
        df = _make_raw_df()
        cleaned, _ = clean_raw(df)
        for col in ("total_stay", "total_guests", "adr_per_person", "revenue_at_risk"):
            assert col in cleaned.columns

    def test_fills_children_null(self):
        df = _make_raw_df(n=4, children=[None, None, 1.0, 2.0])
        _, issues = clean_raw(df)
        assert issues["children_null_filled_zero"] == 2


# ── validate_raw ─────────────────────────────────────────────────────────────


class TestValidateRaw:
    def test_passes_on_valid_data(self):
        df = _make_raw_df()
        cleaned, _ = clean_raw(df)
        result = validate_raw(cleaned)
        assert result.passed
        assert len(result.messages) == 0

    def test_fails_on_missing_columns(self):
        df = _make_raw_df().drop(columns=["lead_time", "adr"])
        result = validate_raw(df)
        assert not result.checks["required_columns_present"]
        assert not result.passed

    def test_fails_on_non_binary_target(self):
        df = _make_raw_df(is_canceled=[0, 1, 2, 3] * 5)
        cleaned, _ = clean_raw(df)
        result = validate_raw(cleaned)
        assert not result.checks["target_binary"]

    def test_fails_on_negative_numeric(self):
        df = _make_raw_df()
        cleaned, _ = clean_raw(df)
        cleaned.loc[cleaned.index[0], "lead_time"] = -5
        result = validate_raw(cleaned)
        assert not result.checks["non_negative_numeric"]


# ── leakage helpers ──────────────────────────────────────────────────────────


class TestLeakage:
    def test_detects_leakage_columns(self):
        found = leakage_columns_present(["lead_time", "reservation_status", "adr"])
        assert "reservation_status" in found

    def test_no_leakage(self):
        assert leakage_columns_present(["lead_time", "adr"]) == []

    def test_assert_raises_on_leakage(self):
        with pytest.raises(ValueError, match="Leakage"):
            assert_no_leakage_columns(["reservation_status"])
