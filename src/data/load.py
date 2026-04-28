"""Data loading utilities."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import DATA_PATH

logger = logging.getLogger(__name__)


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw hotel bookings dataset."""
    csv_path = path or str(DATA_PATH)
    try:
        df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Failed to decode {csv_path} as UTF-8. "
            f"The file may use a different encoding: {exc}"
        ) from exc
    if df.empty:
        raise ValueError(f"CSV file {csv_path} is empty (0 rows).")
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), csv_path)
    return df
