"""Data loading utilities."""

from __future__ import annotations

import pandas as pd

from src.config import DATA_PATH


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw hotel bookings dataset."""
    csv_path = path or str(DATA_PATH)
    return pd.read_csv(csv_path, low_memory=False)
