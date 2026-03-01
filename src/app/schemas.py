"""Pydantic schemas for API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import ADR_MAX_VALID

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTH_LOOKUP = {month.lower(): month for month in MONTHS} | {
    month[:3].lower(): month for month in MONTHS
}


class BookingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotel: str = Field(..., min_length=1)
    lead_time: int = Field(..., ge=0)
    arrival_date: datetime | date | None = Field(
        default=None,
        description="ISO date or datetime; if provided, arrival_date_* fields are derived.",
    )
    arrival_date_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    arrival_date_month: Optional[str] = None
    arrival_date_week_number: Optional[int] = Field(default=None, ge=1, le=53)
    arrival_date_day_of_month: Optional[int] = Field(default=None, ge=1, le=31)
    stays_in_weekend_nights: Optional[int] = Field(default=None, ge=0)
    stays_in_week_nights: Optional[int] = Field(default=None, ge=0)
    adults: int = Field(..., ge=1, le=20)
    children: Optional[int] = Field(default=None, ge=0, le=20)
    babies: Optional[int] = Field(default=None, ge=0, le=20)
    meal: Optional[str] = None
    country: Optional[str] = None
    market_segment: Optional[str] = None
    distribution_channel: Optional[str] = None
    is_repeated_guest: Optional[int] = Field(default=None, ge=0, le=1)
    previous_cancellations: Optional[int] = Field(default=None, ge=0, le=20)
    previous_bookings_not_canceled: Optional[int] = Field(default=None, ge=0, le=50)
    reserved_room_type: Optional[str] = None
    deposit_type: Optional[str] = None
    agent: Optional[str] = None
    company: Optional[str] = None
    customer_type: Optional[str] = None
    adr: Optional[float] = Field(default=None, ge=0, le=ADR_MAX_VALID)
    required_car_parking_spaces: Optional[int] = Field(default=None, ge=0, le=10)
    total_of_special_requests: Optional[int] = Field(default=None, ge=0, le=10)

    @field_validator("agent", "company", mode="before")
    @classmethod
    def _coerce_to_str(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    @field_validator("arrival_date_month", mode="before")
    @classmethod
    def _normalize_month(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            normalized = _MONTH_LOOKUP.get(cleaned.lower())
            if normalized:
                return normalized
        raise ValueError("arrival_date_month must be a full month name (e.g., January)")

    @model_validator(mode="after")
    def _enforce_arrival_date(self) -> "BookingRequest":
        if self.arrival_date is not None:
            arrival = (
                self.arrival_date.date()
                if isinstance(self.arrival_date, datetime)
                else self.arrival_date
            )
            year = arrival.year
            month = MONTHS[arrival.month - 1]
            week = int(arrival.isocalendar().week)
            day = arrival.day

            if self.arrival_date_year is not None and self.arrival_date_year != year:
                raise ValueError("arrival_date_year conflicts with arrival_date")
            if self.arrival_date_month is not None and self.arrival_date_month != month:
                raise ValueError("arrival_date_month conflicts with arrival_date")
            if self.arrival_date_week_number is not None and self.arrival_date_week_number != week:
                raise ValueError("arrival_date_week_number conflicts with arrival_date")
            if self.arrival_date_day_of_month is not None and self.arrival_date_day_of_month != day:
                raise ValueError("arrival_date_day_of_month conflicts with arrival_date")

            self.arrival_date_year = year
            self.arrival_date_month = month
            self.arrival_date_week_number = week
            self.arrival_date_day_of_month = day

        missing = [
            name
            for name in (
                "arrival_date_year",
                "arrival_date_month",
                "arrival_date_week_number",
                "arrival_date_day_of_month",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError("arrival_date or all arrival_date_* fields are required")
        return self


class PredictionResponse(BaseModel):
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Predicted probability of cancellation",
    )
    label_high_precision: int = Field(
        ...,
        ge=0,
        le=1,
        description="Label using high-precision threshold",
    )
    label_max_f1: int = Field(
        ...,
        ge=0,
        le=1,
        description="Label using max-F1 threshold",
    )
    label_cost_sensitive: int = Field(
        ...,
        ge=0,
        le=1,
        description="Label using cost-sensitive threshold",
    )
    risk_tier: str = Field(
        ...,
        description="Operational tier derived from cancellation probability",
    )
    threshold_high_precision: float = Field(..., ge=0, le=1)
    threshold_max_f1: float = Field(..., ge=0, le=1)
    threshold_cost_sensitive: float = Field(..., ge=0, le=1)
    cost_threshold_source: str = Field(
        ...,
        description="Source for cost-sensitive threshold (artifact or fallback policy).",
    )
    cost_threshold_fallback_used: bool = Field(
        ...,
        description="True when cost-sensitive threshold is missing and max-F1 fallback is used.",
    )
    alerts: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings relevant to this prediction response.",
    )


class ModelInfoResponse(BaseModel):
    model_selection_policy: str
    model_type: str
    feature_count: int = Field(..., ge=0)
    has_calibrator: bool
    thresholds: dict[str, float]
    threshold_sources: dict[str, str]
    risk_tier_thresholds: dict[str, float]
    lineage_bundle_sha256: Optional[str] = None
    alerts: list[str] = Field(default_factory=list)
