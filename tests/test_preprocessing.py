from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BOOKING_TIME_FEATURES
from src.features.build import build_preprocessor, ensure_model_features


def test_preprocessor_handles_missing_values_and_schema() -> None:
    rows = [
        {
            "hotel": "City Hotel",
            "lead_time": 14,
            "arrival_date_year": 2017,
            "arrival_date_month": "July",
            "arrival_date_week_number": 30,
            "arrival_date_day_of_month": 10,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "agent": "9",
            "company": None,
            "customer_type": "Transient",
            "adr": 120.5,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 1,
        },
        {
            "hotel": "Resort Hotel",
            "lead_time": 90,
            "arrival_date_year": 2017,
            "arrival_date_month": "August",
            "arrival_date_week_number": 31,
            "arrival_date_day_of_month": 5,
            "stays_in_weekend_nights": 2,
            "stays_in_week_nights": 5,
            "adults": 2,
            "children": None,
            "babies": 0,
            "meal": None,
            "country": None,
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "is_repeated_guest": 1,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 2,
            "reserved_room_type": "C",
            "deposit_type": "No Deposit",
            "agent": None,
            "company": None,
            "customer_type": "Transient-Party",
            "adr": None,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 2,
        },
    ]
    # ensure_model_features derives computed columns (total_stay, month_sin, etc.)
    # from raw booking fields so the numeric imputer has observed values to work with.
    frame = ensure_model_features(pd.DataFrame(rows))
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape[0] == frame.shape[0]
    assert transformed.shape[1] >= len(BOOKING_TIME_FEATURES)
    assert np.isfinite(np.asarray(transformed)).all()


def test_week_number_is_not_a_model_feature() -> None:
    """arrival_date_week_number must stay out of the model feature set.

    The raw dataset's week numbering disagrees with ISO-8601 for ~54% of dates,
    so any serving-side derivation produces a different distribution than the
    training data (training/serving skew). Seasonality is already captured by
    month one-hot + month_sin/month_cos.
    """
    from src.config import BOOKING_TIME_FEATURES

    assert "arrival_date_week_number" not in BOOKING_TIME_FEATURES
