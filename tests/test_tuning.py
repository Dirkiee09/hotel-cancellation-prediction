"""Tests for Optuna hyperparameter tuning (skipped if optuna not installed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from src.config import BOOKING_TIME_FEATURES  # noqa: E402
from src.models.tuning import tune_model  # noqa: E402
from src.utils.validate_data import clean_raw  # noqa: E402


@pytest.fixture()
def small_selection_df():
    """Minimal synthetic data matching the expected schema."""
    rng = np.random.RandomState(42)
    n = 3000
    raw = pd.DataFrame(
        {
            "hotel": rng.choice(["Resort Hotel", "City Hotel"], n),
            "lead_time": rng.randint(0, 400, n),
            "arrival_date_year": rng.choice([2015, 2016, 2017], n),
            "arrival_date_month": rng.choice(["July", "August", "January"], n),
            "arrival_date_week_number": rng.randint(1, 53, n),
            "arrival_date_day_of_month": rng.randint(1, 28, n),
            "stays_in_weekend_nights": rng.randint(0, 5, n),
            "stays_in_week_nights": rng.randint(0, 10, n),
            "adults": rng.randint(1, 4, n),
            "children": rng.choice([0.0, 1.0, 2.0], n),
            "babies": rng.choice([0, 1], n),
            "meal": rng.choice(["BB", "HB", "SC"], n),
            "country": rng.choice(["PRT", "GBR", "FRA", "ESP"], n),
            "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Direct"], n),
            "distribution_channel": rng.choice(["TA/TO", "Direct"], n),
            "is_repeated_guest": rng.choice([0, 1], n),
            "previous_cancellations": rng.choice([0, 1, 2], n),
            "previous_bookings_not_canceled": rng.choice([0, 1], n),
            "reserved_room_type": rng.choice(["A", "D", "E"], n),
            "deposit_type": rng.choice(["No Deposit", "Non Refund"], n),
            "agent": rng.choice([9.0, 14.0, 240.0, 0.0], n),
            "company": rng.choice([0.0, 40.0, 223.0], n),
            "customer_type": rng.choice(["Transient", "Contract"], n),
            "adr": rng.uniform(30, 300, n),
            "required_car_parking_spaces": rng.choice([0, 1], n),
            "total_of_special_requests": rng.choice([0, 1, 2, 3], n),
            "is_canceled": rng.choice([0, 1], n),
        }
    )
    cleaned, _ = clean_raw(raw)
    return cleaned


def test_tune_model_returns_result(small_selection_df) -> None:
    result = tune_model(
        small_selection_df,
        BOOKING_TIME_FEATURES,
        "xgboost",
        n_trials=3,
        timeout=120,
    )
    assert result.best_params is not None
    assert result.best_score >= 0
    assert len(result.all_trials) == 3
    assert result.study_summary["model_family"] == "xgboost"


def test_tune_model_gradient_boosting(small_selection_df) -> None:
    result = tune_model(
        small_selection_df,
        BOOKING_TIME_FEATURES,
        "gradient_boosting",
        n_trials=2,
        timeout=120,
    )
    assert result.best_params is not None
    assert len(result.all_trials) == 2


def test_tune_model_lightgbm(small_selection_df) -> None:
    """The champion family must be tunable with the same Optuna machinery.

    Regression guard: tuning previously covered only XGBoost and GradientBoosting,
    leaving the deployed champion's hyperparameters outside the documented search.
    """
    pytest.importorskip("lightgbm")
    result = tune_model(
        small_selection_df,
        BOOKING_TIME_FEATURES,
        "lightgbm",
        n_trials=2,
        timeout=120,
    )
    assert result.best_params is not None
    assert len(result.all_trials) == 2
    assert result.study_summary["model_family"] == "lightgbm"
