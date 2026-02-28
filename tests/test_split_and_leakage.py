from __future__ import annotations

import pytest

from src.config import BOOKING_TIME_FEATURES, TARGET_COL
from src.data.load import load_raw_data
from src.features.build import add_arrival_date, split_time_aware
from src.utils.validate_data import assert_no_leakage_columns, clean_raw


def test_split_time_aware_is_chronological() -> None:
    df = load_raw_data()
    df, _ = clean_raw(df)
    subset = df.sample(n=600, random_state=42).copy()
    subset = subset[BOOKING_TIME_FEATURES + [TARGET_COL]]

    train_df, val_df, test_df = split_time_aware(subset)
    train_dates = add_arrival_date(train_df)
    val_dates = add_arrival_date(val_df)
    test_dates = add_arrival_date(test_df)

    assert train_df.shape[0] > 0
    assert val_df.shape[0] > 0
    assert test_df.shape[0] > 0
    assert train_dates.max() <= val_dates.min()
    assert val_dates.max() <= test_dates.min()


def test_no_leakage_columns_guard() -> None:
    assert_no_leakage_columns(BOOKING_TIME_FEATURES)
    with pytest.raises(ValueError):
        assert_no_leakage_columns(BOOKING_TIME_FEATURES + ["reservation_status"])
