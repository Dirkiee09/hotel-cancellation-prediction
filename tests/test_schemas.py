"""Tests for src/app/schemas.py — BookingRequest validation."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from src.app.schemas import BookingRequest
from src.config import ADR_MAX_VALID

_BASE: dict[str, Any] = {
    "hotel": "Resort Hotel",
    "lead_time": 100,
    "arrival_date_year": 2017,
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "adults": 2,
    "adr": 100.0,
}


class TestBookingRequestValid:
    def test_minimal_valid(self):
        req = BookingRequest(**_BASE)
        assert req.hotel == "Resort Hotel"
        assert req.adults == 2

    def test_arrival_date_derives_fields(self):
        req = BookingRequest(
            hotel="City Hotel",
            lead_time=50,
            arrival_date=date(2024, 3, 15),
            adults=1,
            adr=80.0,
        )
        assert req.arrival_date_year == 2024
        assert req.arrival_date_month == "March"
        assert req.arrival_date_day_of_month == 15

    def test_month_abbreviation_normalized(self):
        data = {**_BASE, "arrival_date_month": "Jul"}
        req = BookingRequest(**data)
        assert req.arrival_date_month == "July"

    def test_agent_coerced_to_str(self):
        data = {**_BASE, "agent": 123}
        req = BookingRequest(**data)
        assert req.agent == "123"


class TestBookingRequestInvalid:
    def test_adults_zero_rejected(self):
        with pytest.raises(ValidationError):
            BookingRequest(**{**_BASE, "adults": 0})

    def test_adr_above_max_rejected(self):
        with pytest.raises(ValidationError):
            BookingRequest(**{**_BASE, "adr": ADR_MAX_VALID + 1})

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            BookingRequest(**{**_BASE, "unknown_field": "foo"})

    def test_missing_arrival_fields(self):
        with pytest.raises(ValidationError):
            BookingRequest(hotel="Resort Hotel", lead_time=100, adults=2, adr=100.0)

    def test_arrival_date_conflict(self):
        with pytest.raises(ValidationError, match="conflicts"):
            BookingRequest(
                hotel="Resort Hotel",
                lead_time=100,
                arrival_date=date(2024, 3, 15),
                arrival_date_year=2025,  # conflicts
                adults=2,
                adr=100.0,
            )

    def test_invalid_month_name(self):
        with pytest.raises(ValidationError):
            BookingRequest(**{**_BASE, "arrival_date_month": "NotAMonth"})

    def test_negative_lead_time_rejected(self):
        with pytest.raises(ValidationError):
            BookingRequest(**{**_BASE, "lead_time": -1})
