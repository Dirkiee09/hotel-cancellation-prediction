"""Export the ADR regressor's test-set predictions for the Power BI dashboard.

The ADR regression model (artifacts/adr_regressor.pkl) is the project's second
trained ML model — alongside the cancellation classifier. Its test-set
predictions aren't currently saved as a CSV, so this script materialises them
for Power BI Page 5 ("ADR Forecasting"):

    reports/adr_test_predictions.csv
        One row per test booking with actual ADR, predicted ADR, residual,
        and context columns (hotel, market_segment, room type, country,
        arrival month) for slicing in Power BI.

    reports/adr_segment_performance.csv
        Aggregated RMSE / MAE / row-count per (hotel × reserved_room_type)
        segment, filtered to segments with at least 50 rows. Powers the
        "where does the model struggle" heatmap.

Usage:
    python scripts/export_adr_predictions.py
    make export-adr

The ADR regressor uses a different chronological split than the cancellation
pipeline (split_date 2017-04-23, ~23k test rows). The split parameters are
read from artifacts/adr_regressor_metadata.pkl so this script always lines
up with whatever the latest train produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow `python scripts/export_adr_predictions.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS_DIR, REPORTS_DIR  # noqa: E402
from src.data.load import load_raw_data  # noqa: E402
from src.features.build import add_arrival_date  # noqa: E402

ADR_REGRESSOR_PATH = ARTIFACTS_DIR / "adr_regressor.pkl"
ADR_METADATA_PATH = ARTIFACTS_DIR / "adr_regressor_metadata.pkl"
OUTPUT_PREDS = REPORTS_DIR / "adr_test_predictions.csv"
OUTPUT_SEGMENTS = REPORTS_DIR / "adr_segment_performance.csv"

MIN_SEGMENT_ROWS = 50  # below this, segment metrics are noisy — drop


def _load_artifacts() -> tuple[object, dict]:
    if not ADR_REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            f"ADR regressor not found at {ADR_REGRESSOR_PATH}. " f"Run `make train` to generate it."
        )
    if not ADR_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"ADR regressor metadata not found at {ADR_METADATA_PATH}. "
            f"Run `make train` to regenerate."
        )
    regressor = joblib.load(ADR_REGRESSOR_PATH)
    metadata = joblib.load(ADR_METADATA_PATH)
    return regressor, metadata


def _build_test_set(metadata: dict) -> pd.DataFrame:
    """Recreate the test set the regressor was evaluated on.

    Mirrors the split logic: filter adr > 0, sort by arrival_date,
    take rows on or after metadata['split_date'].
    """
    df = load_raw_data()
    df = df[df["adr"] > 0].copy()  # matches metadata['adr_filtered']
    df["_arrival_date"] = add_arrival_date(df)  # add_arrival_date returns a Series
    df = df.sort_values("_arrival_date").reset_index(drop=True)

    split_date = pd.Timestamp(metadata["split_date"])
    test_df = df[df["_arrival_date"] >= split_date].reset_index(drop=True)
    return test_df


def _compute_predictions(regressor, test_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Run the regressor on the test set and build a tidy output DataFrame."""
    X_test = test_df[features]
    y_actual = test_df["adr"].to_numpy()
    y_predicted = regressor.predict(X_test)
    residual = y_actual - y_predicted

    out = pd.DataFrame(
        {
            "hotel": test_df["hotel"].astype(str),
            "market_segment": test_df["market_segment"].astype(str),
            "reserved_room_type": test_df["reserved_room_type"].astype(str),
            "country": test_df["country"].fillna("UNK").astype(str),
            "lead_time": test_df["lead_time"].astype(int),
            "arrival_date_month": test_df["arrival_date_month"].astype(str),
            "arrival_date_year": test_df["arrival_date_year"].astype(int),
            "adr_actual": np.round(y_actual, 2),
            "adr_predicted": np.round(y_predicted, 2),
            "residual": np.round(residual, 2),
            "abs_residual": np.round(np.abs(residual), 2),
        }
    )
    return out


def _compute_segment_performance(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-segment RMSE/MAE for the Power BI segment heatmap."""
    grouped = preds.groupby(["hotel", "reserved_room_type"], as_index=False).agg(
        n_rows=("adr_actual", "size"),
        rmse=("residual", lambda s: float(np.sqrt(np.mean(s.to_numpy() ** 2)))),
        mae=("abs_residual", "mean"),
        mean_actual_adr=("adr_actual", "mean"),
        mean_predicted_adr=("adr_predicted", "mean"),
    )
    grouped = grouped[grouped["n_rows"] >= MIN_SEGMENT_ROWS].reset_index(drop=True)
    grouped["rmse"] = grouped["rmse"].round(2)
    grouped["mae"] = grouped["mae"].round(2)
    grouped["mean_actual_adr"] = grouped["mean_actual_adr"].round(2)
    grouped["mean_predicted_adr"] = grouped["mean_predicted_adr"].round(2)
    return grouped.sort_values("rmse", ascending=False).reset_index(drop=True)


def main() -> int:
    regressor, metadata = _load_artifacts()
    split_date = pd.Timestamp(metadata["split_date"])
    print(f"Loaded ADR regressor: {metadata['model_name']}")
    print(f"  Reported test RMSE = {metadata['test_rmse']:.2f}")
    print(f"  Reported test MAE  = {metadata['test_mae']:.2f}")
    print(f"  Reported test R^2  = {metadata['test_r2']:.4f}")
    print(f"  Split date         = {split_date.date()}")
    print()

    test_df = _build_test_set(metadata)
    print(f"Reconstructed test set: {len(test_df):,} rows")

    preds = _compute_predictions(regressor, test_df, metadata["features"])
    OUTPUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUTPUT_PREDS, index=False)
    print(f"Wrote {OUTPUT_PREDS} ({len(preds):,} rows, {len(preds.columns)} columns)")

    realised_rmse = float(np.sqrt(np.mean(preds["residual"].to_numpy() ** 2)))
    realised_mae = float(preds["abs_residual"].mean())
    print(f"  Realised RMSE      = {realised_rmse:.2f}")
    print(f"  Realised MAE       = {realised_mae:.2f}")

    segments = _compute_segment_performance(preds)
    segments.to_csv(OUTPUT_SEGMENTS, index=False)
    print(f"Wrote {OUTPUT_SEGMENTS} ({len(segments)} segments with >= {MIN_SEGMENT_ROWS} rows)")
    print()
    print("Per-segment RMSE (top 6 worst-performing):")
    print(segments.head(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
