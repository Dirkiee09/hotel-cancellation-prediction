"""SQLite-backed audit log for the PH /predict endpoint.

Parallel to src/serving/prediction_log.py but with the reduced PH schema:
8 booking-time input fields (no country / deposit_type / market_segment) +
PH-specific output fields (no cost_sensitive policy, no ADR forecast).

Power BI integration story:
    PH FastAPI /predict
        -> log_ph_prediction() appends to ph_predictions.sqlite
        -> export_ph_to_csv() materialises ph_predictions_live.csv
        -> Power BI Desktop reads the CSV (parallel to Portugal's dashboard)

Design contract (same as Portugal logger):
    * Logging is a side-effect — MUST NOT crash the /predict response.
    * Connection-per-call. Idempotent CREATE TABLE.
    * Failures log at WARNING level, never raise.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PH_PREDICTION_LOG_CSV, PH_PREDICTION_LOG_DB

logger = logging.getLogger(__name__)


_INPUT_COLUMNS = (
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "weekend_nights",
    "week_nights",
    "adults",
    "children",
    "babies",
    "adr",
    "reserved_room_type",
)

_OUTPUT_COLUMNS = (
    "probability",
    "label_max_f1",
    "label_high_precision",
    "risk_tier",
    "threshold_max_f1",
    "threshold_high_precision",
)

_JSON_COLUMNS = ("top_features",)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ph_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    lead_time INTEGER,
    arrival_date_year INTEGER,
    arrival_date_month INTEGER,
    arrival_date_day_of_month INTEGER,
    weekend_nights INTEGER,
    week_nights INTEGER,
    adults INTEGER,
    children INTEGER,
    babies INTEGER,
    adr REAL,
    reserved_room_type TEXT,
    probability REAL,
    label_max_f1 INTEGER,
    label_high_precision INTEGER,
    risk_tier TEXT,
    threshold_max_f1 REAL,
    threshold_high_precision REAL,
    top_features TEXT
)
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ph_timestamp ON ph_predictions (timestamp_utc);",
    "CREATE INDEX IF NOT EXISTS idx_ph_risk_tier ON ph_predictions (risk_tier);",
)


def init_db(db_path: Path = PH_PREDICTION_LOG_DB) -> None:
    """Create the ph_predictions table + indexes if missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        for stmt in _CREATE_INDEX_SQL:
            conn.execute(stmt)
        conn.commit()


def _coerce_for_sqlite(value: Any) -> Any:
    """Convert Pydantic-dumped values to SQLite-storable scalars."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def log_ph_prediction(
    request: dict[str, Any],
    response: dict[str, Any],
    db_path: Path = PH_PREDICTION_LOG_DB,
) -> int | None:
    """Append one (PH request, PH response) pair to the ph_predictions table.

    Non-raising. Wired into a FastAPI BackgroundTask — failures here must
    never break the /predict response. On failure returns None and logs a
    WARNING.
    """
    try:
        init_db(db_path)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        row: dict[str, Any] = {"timestamp_utc": timestamp}
        for col in _INPUT_COLUMNS:
            row[col] = _coerce_for_sqlite(request.get(col))
        for col in _OUTPUT_COLUMNS:
            row[col] = _coerce_for_sqlite(response.get(col))
        for col in _JSON_COLUMNS:
            payload = response.get(col)
            row[col] = json.dumps(payload, default=str) if payload is not None else None

        columns = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        sql = f"INSERT INTO ph_predictions ({columns}) VALUES ({placeholders})"  # nosec B608

        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(sql, row)
            conn.commit()
            return int(cur.lastrowid) if cur.lastrowid is not None else None
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        logger.warning("ph_prediction_log_write_failed error=%s", exc)
        return None


def count_ph_predictions(db_path: Path = PH_PREDICTION_LOG_DB) -> int:
    """Return the number of rows in the ph_predictions table (0 if absent)."""
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM ph_predictions")
        return int(cur.fetchone()[0])


def export_ph_to_csv(
    db_path: Path | None = None,
    csv_path: Path | None = None,
) -> int:
    """Materialise the PH predictions table to CSV for Power BI.

    Returns the number of rows exported. Returns 0 if the DB is missing or
    empty. Imports pandas lazily so /predict-path callers don't pay the cost.
    """
    db = db_path if db_path is not None else PH_PREDICTION_LOG_DB
    csv = csv_path if csv_path is not None else PH_PREDICTION_LOG_CSV

    if not db.exists():
        return 0

    import pandas as pd  # local import: keeps log_ph_prediction lightweight

    with sqlite3.connect(db) as conn:
        df = pd.read_sql("SELECT * FROM ph_predictions ORDER BY timestamp_utc", conn)

    if df.empty:
        return 0

    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    return int(len(df))
