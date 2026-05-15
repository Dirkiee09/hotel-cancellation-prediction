"""SQLite-backed audit log for every /predict call.

Power BI integration story:
    FastAPI /predict
        -> log_prediction() appends one row to PREDICTION_LOG_DB (SQLite)
        -> scripts/export_predictions.py materialises a CSV
        -> Power BI Desktop reads the CSV (no ODBC driver setup required)

Design contract:
    * Logging is a side-effect. It MUST NOT crash the /predict response.
      Wrap calls in try/except at the caller and run via FastAPI
      BackgroundTasks so failures here are non-fatal.
    * Connection-per-call. sqlite3 is thread-safe in this mode and the
      file lock is acquired only during the brief INSERT, which is fine
      for the thesis-demo write volume (< 1 Hz).
    * Schema is idempotent (CREATE TABLE IF NOT EXISTS) so the first call
      after a fresh checkout creates the DB.
    * No PII concerns for this dataset: the BookingRequest fields are
      operational (lead time, ADR, country code, market segment) not
      personally identifying. In a real production deployment with
      guest_id or email this module would need a redaction step.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PREDICTION_LOG_DB

logger = logging.getLogger(__name__)

# Columns persisted from BookingRequest (input) — names match the Pydantic schema
# exactly so the CSV column order matches what notebook 08 already expects.
_INPUT_COLUMNS = (
    "hotel",
    "lead_time",
    "arrival_date",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
)

# Columns persisted from PredictionResponse (output)
_OUTPUT_COLUMNS = (
    "probability",
    "label_high_precision",
    "label_max_f1",
    "label_cost_sensitive",
    "risk_tier",
    "threshold_high_precision",
    "threshold_max_f1",
    "threshold_cost_sensitive",
    "cost_threshold_source",
    "cost_threshold_fallback_used",
)

# JSON-serialized complex columns (lists / nested dicts)
_JSON_COLUMNS = ("alerts", "top_features")

ALL_COLUMNS = ("prediction_id", "timestamp_utc", *_INPUT_COLUMNS, *_OUTPUT_COLUMNS, *_JSON_COLUMNS)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    hotel TEXT,
    lead_time INTEGER,
    arrival_date TEXT,
    arrival_date_year INTEGER,
    arrival_date_month TEXT,
    arrival_date_week_number INTEGER,
    arrival_date_day_of_month INTEGER,
    stays_in_weekend_nights INTEGER,
    stays_in_week_nights INTEGER,
    adults INTEGER,
    children INTEGER,
    babies INTEGER,
    meal TEXT,
    country TEXT,
    market_segment TEXT,
    distribution_channel TEXT,
    is_repeated_guest INTEGER,
    previous_cancellations INTEGER,
    previous_bookings_not_canceled INTEGER,
    reserved_room_type TEXT,
    deposit_type TEXT,
    agent TEXT,
    company TEXT,
    customer_type TEXT,
    adr REAL,
    required_car_parking_spaces INTEGER,
    total_of_special_requests INTEGER,
    probability REAL,
    label_high_precision INTEGER,
    label_max_f1 INTEGER,
    label_cost_sensitive INTEGER,
    risk_tier TEXT,
    threshold_high_precision REAL,
    threshold_max_f1 REAL,
    threshold_cost_sensitive REAL,
    cost_threshold_source TEXT,
    cost_threshold_fallback_used INTEGER,
    alerts TEXT,
    top_features TEXT
);
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions (timestamp_utc);",
    "CREATE INDEX IF NOT EXISTS idx_risk_tier ON predictions (risk_tier);",
)


def init_db(db_path: Path = PREDICTION_LOG_DB) -> None:
    """Create the predictions table and indexes if they don't exist."""
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
        # SQLite has no native bool — store as 0/1 to match downstream Power BI
        return int(value)
    if isinstance(value, (str, int, float)):
        return value
    # date/datetime/anything else → ISO string
    return str(value)


def log_prediction(
    request: dict[str, Any],
    response: dict[str, Any],
    db_path: Path = PREDICTION_LOG_DB,
) -> int | None:
    """Append one (request, response) pair to the predictions table.

    Returns the new row's prediction_id, or None if logging failed.
    Failures are logged at WARNING level but never raised — this function
    is wired into a FastAPI BackgroundTask and must not crash the API.
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
        # The column names are literals from _INPUT_COLUMNS / _OUTPUT_COLUMNS /
        # _JSON_COLUMNS tuples defined in this module — no user input touches
        # the SQL string itself. All values are bound via :name placeholders.
        sql = f"INSERT INTO predictions ({columns}) VALUES ({placeholders})"  # nosec B608

        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(sql, row)
            conn.commit()
            return int(cur.lastrowid) if cur.lastrowid is not None else None
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        logger.warning("prediction_log_write_failed error=%s", exc)
        return None


def count_predictions(db_path: Path = PREDICTION_LOG_DB) -> int:
    """Return the number of rows in the predictions table (0 if absent)."""
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM predictions")
        return int(cur.fetchone()[0])


def export_to_csv(
    db_path: Path | None = None,
    csv_path: Path | None = None,
    since: str | None = None,
) -> int:
    """Materialise the predictions table to CSV for Power BI.

    Returns the number of rows exported. Returns 0 if the DB is missing
    or the (filtered) table is empty. Imports pandas lazily so callers
    that don't need export aren't forced to pay the import cost.
    """
    from src.config import PREDICTION_LOG_CSV  # local import: avoids cycle

    db = db_path if db_path is not None else PREDICTION_LOG_DB
    csv = csv_path if csv_path is not None else PREDICTION_LOG_CSV

    if not db.exists():
        return 0

    import pandas as pd  # local import: keeps log_prediction lightweight

    where = ""
    params: tuple = ()
    if since is not None:
        where = "WHERE timestamp_utc >= ?"
        params = (since,)

    with sqlite3.connect(db) as conn:
        # `where` is a literal string from this module; `since` is bound
        # via params, never interpolated.
        df = pd.read_sql(
            f"SELECT * FROM predictions {where} ORDER BY timestamp_utc",  # nosec B608
            conn,
            params=params,
        )

    if df.empty:
        return 0

    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    return len(df)
