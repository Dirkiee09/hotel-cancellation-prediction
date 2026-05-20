"""Pydantic schemas for the PH (Philippine dataset) sub-study API.

Reduced PH feature set (8 raw + arrival date) vs Portugal's 32 fields. Field
validation mirrors src/app/schemas.py where applicable; the rest of the
fields the Portugal schema requires (country, agent, deposit_type, etc.) are
NOT present in the PH PMS export, so they are deliberately omitted here.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PHBookingRequest(BaseModel):
    """A single booking submitted to /predict on the PH server.

    Fields correspond to columns in the PH PMS export (renamed to project-
    canonical names via PH_COLUMN_RENAMES). All fields except adults and the
    arrival date are optional with sensible defaults.
    """

    model_config = ConfigDict(extra="forbid")

    lead_time: int = Field(..., ge=0, le=730, description="Days from booking to arrival")
    arrival_date: datetime | date | None = Field(
        default=None,
        description="ISO date or datetime; if provided, arrival_date_* fields are derived.",
    )
    arrival_date_year: Optional[int] = Field(default=None, ge=2020, le=2030)
    arrival_date_month: Optional[int] = Field(default=None, ge=1, le=12)
    arrival_date_day_of_month: Optional[int] = Field(default=None, ge=1, le=31)
    weekend_nights: int = Field(default=0, ge=0, le=20)
    week_nights: int = Field(default=0, ge=0, le=30)
    adults: int = Field(..., ge=1, le=10, description="Number of adult guests")
    children: int = Field(default=0, ge=0, le=10)
    babies: int = Field(default=0, ge=0, le=10)
    adr: float = Field(..., ge=0, le=100_000, description="Average daily rate in PHP")
    reserved_room_type: str = Field(
        ...,
        min_length=1,
        description="Room type code (e.g., 'Standard', 'Deluxe')",
    )

    @field_validator("reserved_room_type", mode="before")
    @classmethod
    def _coerce_room_type(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @model_validator(mode="after")
    def _enforce_arrival_components(self) -> "PHBookingRequest":
        # If arrival_date is provided, derive the components; conflict with
        # explicit components → error.
        if self.arrival_date is not None:
            arrival = (
                self.arrival_date.date()
                if isinstance(self.arrival_date, datetime)
                else self.arrival_date
            )
            year, month, day = arrival.year, arrival.month, arrival.day
            if self.arrival_date_year is not None and self.arrival_date_year != year:
                raise ValueError("arrival_date_year conflicts with arrival_date")
            if self.arrival_date_month is not None and self.arrival_date_month != month:
                raise ValueError("arrival_date_month conflicts with arrival_date")
            if self.arrival_date_day_of_month is not None and self.arrival_date_day_of_month != day:
                raise ValueError("arrival_date_day_of_month conflicts with arrival_date")
            self.arrival_date_year = year
            self.arrival_date_month = month
            self.arrival_date_day_of_month = day

        missing = [
            name
            for name in (
                "arrival_date_year",
                "arrival_date_month",
                "arrival_date_day_of_month",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError("arrival_date or all arrival_date_* fields are required")
        return self

    def to_inference_dict(self) -> dict[str, object]:
        """Convert to the dict shape expected by ``predict_ph``."""
        arrival = self._iso_arrival_date()
        return {
            "lead_time": self.lead_time,
            "weekend_nights": self.weekend_nights,
            "week_nights": self.week_nights,
            "adults": self.adults,
            "children": self.children,
            "babies": self.babies,
            "adr": self.adr,
            "reserved_room_type": self.reserved_room_type,
            "arrival_date": arrival,
        }

    def _iso_arrival_date(self) -> str:
        y = self.arrival_date_year
        m = self.arrival_date_month
        d = self.arrival_date_day_of_month
        assert y is not None and m is not None and d is not None  # enforced by validator
        return f"{y:04d}-{m:02d}-{d:02d}"


class PHPredictionResponse(BaseModel):
    """Response from POST /predict on the PH server."""

    probability: float = Field(..., ge=0, le=1, description="Calibrated P(cancel)")
    label_max_f1: int = Field(..., ge=0, le=1)
    label_high_precision: int = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., description="low | medium | high")
    threshold_max_f1: float = Field(..., ge=0, le=1)
    threshold_high_precision: float = Field(..., ge=0, le=1)
    alerts: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings; always contains the Philippine-dataset caveat.",
    )
    top_features: list[dict[str, object]] = Field(
        default_factory=list,
        description="Top SHAP contributors with 'feature', 'value', 'contribution' keys.",
    )


class PHModelInfoResponse(BaseModel):
    """Response from GET /model-info on the PH server."""

    server_kind: str = Field(default="ph-philippine-substudy")
    model_family: str
    feature_count: int = Field(..., ge=0)
    has_calibrator: bool
    n_train: int = Field(..., ge=0)
    n_test: int = Field(..., ge=0)
    test_roc_auc: Optional[float] = None
    test_pr_auc: Optional[float] = None
    thresholds: dict[str, float]
    risk_tier_thresholds: dict[str, float]
    dataset_caveat: str
    dataset_diagnostics: dict[str, object] = Field(default_factory=dict)
