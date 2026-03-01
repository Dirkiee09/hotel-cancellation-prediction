"""Feature selection and preprocessing pipeline."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.config import BOOKING_TIME_FEATURES, LATE_WINDOW_MAX_LEAD_DAYS, TRAIN_RATIO, VAL_RATIO

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

CATEGORICAL_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
]

NUMERIC_COLS = sorted([col for col in BOOKING_TIME_FEATURES if col not in CATEGORICAL_COLS])
NULL_LIKE_REPLACEMENTS: dict[str, Any] = {"NULL": pd.NA, "null": pd.NA, "": pd.NA, " ": pd.NA}


def cast_to_str(values: Any) -> Any:
    return values.astype(str)


def _replace_null_like(series: pd.Series) -> pd.Series:
    return series.replace(cast(dict[Any, Any], NULL_LIKE_REPLACEMENTS))


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _column_or_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _month_num_from_name(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.title()
    return normalized.map(MONTH_MAP)


def normalize_booking_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known raw-input quirks before feature derivation."""
    out = df.copy()

    if "agent" in out.columns:
        out["agent"] = _to_numeric(_replace_null_like(out["agent"])).fillna(0).astype(int)
    if "company" in out.columns:
        out["company"] = _to_numeric(_replace_null_like(out["company"]))
    if "children" in out.columns:
        out["children"] = _to_numeric(out["children"]).fillna(0)
    if "country" in out.columns:
        out["country"] = _replace_null_like(out["country"]).fillna("Unknown")
    if "arrival_date_month" in out.columns:
        out["arrival_date_month"] = out["arrival_date_month"].astype(str).str.strip().str.title()

    return out


def add_derived_booking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create thesis-roadmap engineered features from booking-time fields."""
    out = normalize_booking_inputs(df)

    weekend = _to_numeric(_column_or_default(out, "stays_in_weekend_nights")).fillna(0)
    week = _to_numeric(_column_or_default(out, "stays_in_week_nights")).fillna(0)
    adults = _to_numeric(_column_or_default(out, "adults")).fillna(0)
    children = _to_numeric(_column_or_default(out, "children")).fillna(0)
    babies = _to_numeric(_column_or_default(out, "babies")).fillna(0)
    lead_time = _to_numeric(_column_or_default(out, "lead_time", np.nan))
    adr = _to_numeric(_column_or_default(out, "adr", np.nan))

    out["total_stay"] = weekend + week
    out["total_guests"] = adults + children + babies
    # When total_guests == 0 (data quality issue; schema enforces adults >= 1),
    # denominator defaults to 1.0, so adr_per_person == adr.
    guests_denom = out["total_guests"].where(out["total_guests"] > 0, 1.0)
    out["adr_per_person"] = adr / guests_denom
    out["is_weekend_heavy"] = (weekend > week).astype(int)
    out["revenue_at_risk"] = adr * out["total_stay"]
    # NaN lead_time -> fillna(False): unknown lead time is conservatively treated as "not late window"
    out["is_late_window"] = (lead_time <= LATE_WINDOW_MAX_LEAD_DAYS).fillna(False).astype(int)

    if "company" in out.columns:
        out["had_company"] = out["company"].notna().astype(int)
    elif "had_company" not in out.columns:
        out["had_company"] = 0

    if "arrival_date_month" in out.columns:
        month_num = _month_num_from_name(out["arrival_date_month"])
        angle = 2.0 * np.pi * (month_num / 12.0)
        out["month_sin"] = np.sin(angle)
        out["month_cos"] = np.cos(angle)
        out.loc[month_num.isna(), ["month_sin", "month_cos"]] = np.nan
    else:
        out["month_sin"] = np.nan
        out["month_cos"] = np.nan

    return out


def ensure_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive, add, and order model feature columns."""
    out = add_derived_booking_features(df)
    for col in BOOKING_TIME_FEATURES:
        if col not in out.columns:
            out[col] = pd.NA
    return out[BOOKING_TIME_FEATURES]


def add_arrival_date(df: pd.DataFrame) -> pd.Series:
    month_num = _month_num_from_name(df["arrival_date_month"])
    return pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": month_num,
            "day": df["arrival_date_day_of_month"],
        },
        errors="coerce",
    )


def sort_by_arrival_date(df: pd.DataFrame) -> pd.DataFrame:
    """Add and sort by arrival date using a stable ordering strategy."""
    out = df.copy()
    out["_arrival_date"] = add_arrival_date(out)
    return out.sort_values("_arrival_date", kind="mergesort")


def split_time_ordered(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/validation/test partitions.

    Data is sorted by arrival date (ascending) before splitting.  The splits
    are non-overlapping and contiguous: train comes first, then validation,
    then test.

    Raises
    ------
    ValueError
        If any resulting partition is empty.  This can happen when the dataset
        is too small for the configured split ratios.
    """
    ordered = sort_by_arrival_date(df)
    n = len(ordered)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if train_end == 0 or val_end <= train_end or val_end >= n:
        raise ValueError(
            f"Time-aware split produced an empty partition: "
            f"n={n}, train_end={train_end}, val_end={val_end}. "
            f"Dataset is too small for train_ratio={train_ratio}, val_ratio={val_ratio}."
        )

    train_df = ordered.iloc[:train_end].drop(columns=["_arrival_date"])
    val_df = ordered.iloc[train_end:val_end].drop(columns=["_arrival_date"])
    test_df = ordered.iloc[val_end:].drop(columns=["_arrival_date"])
    return train_df, val_df, test_df


def split_time_aware(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Project-default chronological split used by training pipeline."""
    return split_time_ordered(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)


def make_onehot_encoder() -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, min_frequency=0.01, handle_unknown="ignore")


def build_preprocessor() -> Pipeline:
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "to_str",
                FunctionTransformer(cast_to_str, validate=False, feature_names_out="one-to-one"),
            ),
            ("onehot", make_onehot_encoder()),
        ]
    )

    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    column_transformer = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipeline, CATEGORICAL_COLS),
            ("numeric", num_pipeline, NUMERIC_COLS),
        ]
    )
    return Pipeline(
        steps=[
            ("encode", column_transformer),
        ]
    )
