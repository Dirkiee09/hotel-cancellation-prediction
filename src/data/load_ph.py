"""Loader for the Philippine resort booking dataset.

This sits parallel to ``src.data.load`` (which loads the Portugal data) and is
intentionally isolated — the Portugal pipeline does not import this module.
See CLAUDE.md "PH Sub-Study" for context.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import PH_DATA_PATH

logger = logging.getLogger(__name__)


def load_ph_data(path: str | None = None) -> pd.DataFrame:
    """Load the Punta Villa Resort (Philippines) bookings CSV."""
    csv_path = path or str(PH_DATA_PATH)
    try:
        df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Failed to decode {csv_path} as UTF-8: {exc}. The PH dataset is "
            f"expected to be UTF-8 (or UTF-8 with BOM)."
        ) from exc
    if df.empty:
        raise ValueError(f"CSV file {csv_path} is empty (0 rows).")
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), csv_path)
    return df
