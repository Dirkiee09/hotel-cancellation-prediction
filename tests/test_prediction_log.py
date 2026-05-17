"""Tests for the SQLite prediction-log persistence layer.

Covers:
- log_prediction creates the DB + table on first call (idempotent schema)
- multiple log_prediction calls append rather than overwrite
- JSON columns (alerts, top_features) round-trip correctly
- log_prediction never raises even when handed garbage input
- /predict end-to-end persists a row via FastAPI BackgroundTasks
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

import src.app.main as app_main
import src.serving.inference as inference_mod
from src.app.schemas import BookingRequest
from src.serving.inference import load_artifacts
from src.serving.prediction_log import (
    ALL_COLUMNS,
    count_predictions,
    init_db,
    log_prediction,
)

_REQUEST_FIXTURE: dict = {
    "hotel": "City Hotel",
    "lead_time": 60,
    "arrival_date": "2026-08-15",
    "arrival_date_year": 2026,
    "arrival_date_month": "August",
    "arrival_date_week_number": 33,
    "arrival_date_day_of_month": 15,
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
    "adr": 105.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
}

_RESPONSE_FIXTURE: dict = {
    "probability": 0.4321,
    "label_high_precision": 0,
    "label_max_f1": 1,
    "label_cost_sensitive": 1,
    "risk_tier": "medium",
    "threshold_high_precision": 0.98,
    "threshold_max_f1": 0.40,
    "threshold_cost_sensitive": 0.04,
    "cost_threshold_source": "artifact",
    "cost_threshold_fallback_used": False,
    "alerts": ["calibrator not loaded"],
    "top_features": [
        {"feature": "lead_time", "value": 60, "contribution": 0.21},
        {"feature": "agent", "value": "9", "contribution": 0.13},
    ],
    "predicted_adr": 92.5,
    "adr_residual": 12.5,
}


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    init_db(db)  # second call must not raise
    assert db.exists()
    with sqlite3.connect(db) as conn:
        # Table and at least one index exist
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "predictions" in tables
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert any("timestamp" in i for i in indexes)


def test_log_prediction_inserts_row(tmp_path: Path) -> None:
    db = tmp_path / "predictions.sqlite"
    rowid = log_prediction(_REQUEST_FIXTURE, _RESPONSE_FIXTURE, db)
    assert rowid is not None and rowid > 0
    assert count_predictions(db) == 1

    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT * FROM predictions").fetchone()
    assert row is not None
    # column 1 is timestamp_utc (column 0 is auto-incrementing prediction_id)
    timestamp_str = row[1]
    datetime.fromisoformat(timestamp_str)  # raises if not ISO 8601


def test_log_prediction_appends_multiple_rows(tmp_path: Path) -> None:
    db = tmp_path / "predictions.sqlite"
    for _ in range(3):
        log_prediction(_REQUEST_FIXTURE, _RESPONSE_FIXTURE, db)
    assert count_predictions(db) == 3


def test_log_prediction_serialises_json_columns(tmp_path: Path) -> None:
    db = tmp_path / "predictions.sqlite"
    log_prediction(_REQUEST_FIXTURE, _RESPONSE_FIXTURE, db)
    with sqlite3.connect(db) as conn:
        alerts_json, top_features_json = conn.execute(
            "SELECT alerts, top_features FROM predictions"
        ).fetchone()
    assert json.loads(alerts_json) == ["calibrator not loaded"]
    top_features = json.loads(top_features_json)
    assert top_features[0]["feature"] == "lead_time"
    assert top_features[1]["contribution"] == 0.13


def test_log_prediction_coerces_bool_to_int(tmp_path: Path) -> None:
    """SQLite has no native bool. Storing as 0/1 keeps Power BI columns numeric."""
    db = tmp_path / "predictions.sqlite"
    log_prediction(_REQUEST_FIXTURE, _RESPONSE_FIXTURE, db)
    with sqlite3.connect(db) as conn:
        fb = conn.execute("SELECT cost_threshold_fallback_used FROM predictions").fetchone()[0]
    assert fb == 0
    assert isinstance(fb, int)


def test_log_prediction_swallows_errors(tmp_path: Path, caplog) -> None:
    """A bad db_path must not crash the caller — logging is best-effort."""
    bad_path = tmp_path / "no_such_dir" / "x" / "y" / "z" / "predictions.sqlite"
    bad_path.parent.parent.parent.parent.mkdir(parents=True)  # leave one level missing
    # Even if init_db succeeds (it'll create the missing dirs), a corrupt request
    # shouldn't crash. Pass an unserialisable value to force a coercion path.
    weird_request = dict(_REQUEST_FIXTURE)
    weird_request["arrival_date"] = object()  # not serialisable as scalar

    result = log_prediction(weird_request, _RESPONSE_FIXTURE, bad_path)
    # Either succeeds (object str-coerced) or returns None — but never raises
    assert result is None or isinstance(result, int)


def test_all_columns_schema_matches_table(tmp_path: Path) -> None:
    """ALL_COLUMNS tuple must mirror the actual SQL schema column order."""
    db = tmp_path / "predictions.sqlite"
    init_db(db)
    with sqlite3.connect(db) as conn:
        actual = [r[1] for r in conn.execute("PRAGMA table_info(predictions)")]
    assert tuple(actual) == ALL_COLUMNS


def test_count_predictions_returns_zero_when_db_missing(tmp_path: Path) -> None:
    assert count_predictions(tmp_path / "does_not_exist.sqlite") == 0


def test_log_prediction_persists_adr_columns(tmp_path: Path) -> None:
    """predicted_adr and adr_residual must round-trip through SQLite."""
    db = tmp_path / "predictions.sqlite"
    log_prediction(_REQUEST_FIXTURE, _RESPONSE_FIXTURE, db)
    with sqlite3.connect(db) as conn:
        predicted_adr, adr_residual = conn.execute(
            "SELECT predicted_adr, adr_residual FROM predictions"
        ).fetchone()
    assert predicted_adr == 92.5
    assert adr_residual == 12.5


def test_log_prediction_handles_none_adr(tmp_path: Path) -> None:
    """When the ADR regressor isn't loaded, the columns persist as SQL NULL."""
    db = tmp_path / "predictions.sqlite"
    response_no_adr = dict(_RESPONSE_FIXTURE)
    response_no_adr["predicted_adr"] = None
    response_no_adr["adr_residual"] = None
    log_prediction(_REQUEST_FIXTURE, response_no_adr, db)
    with sqlite3.connect(db) as conn:
        predicted_adr, adr_residual = conn.execute(
            "SELECT predicted_adr, adr_residual FROM predictions"
        ).fetchone()
    assert predicted_adr is None
    assert adr_residual is None


def test_migrate_schema_adds_missing_columns(tmp_path: Path) -> None:
    """Older DBs (without predicted_adr / adr_residual) auto-migrate on next init_db.

    Regression guard: if init_db only created tables but didn't ALTER existing
    ones, deployments that upgraded across this commit would silently drop
    ADR data on every log_prediction call.
    """
    db = tmp_path / "predictions.sqlite"

    # Create a DB with the OLD schema (no predicted_adr / adr_residual).
    legacy_sql = """
    CREATE TABLE predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        probability REAL,
        risk_tier TEXT,
        alerts TEXT,
        top_features TEXT
    );
    """
    with sqlite3.connect(db) as conn:
        conn.execute(legacy_sql)
        conn.commit()
        # Sanity: the new columns do NOT exist yet
        cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
        assert "predicted_adr" not in cols
        assert "adr_residual" not in cols

    # Run init_db — should add both columns without dropping any data.
    init_db(db)
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
        assert "predicted_adr" in cols
        assert "adr_residual" in cols

    # Calling init_db again must be idempotent.
    init_db(db)
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
        assert "predicted_adr" in cols
        assert "adr_residual" in cols


def test_predict_endpoint_persists_row(
    monkeypatch,
    trained_artifacts_dir,
    sample_record,
    tmp_path: Path,
) -> None:
    """End-to-end: hit /predict via TestClient, verify a row landed in SQLite."""
    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)

    # Redirect the log to a tmp DB so the production reports/ directory isn't touched.
    db_path = tmp_path / "predictions.sqlite"
    monkeypatch.setattr(app_main, "PREDICTION_LOG_DB", db_path)

    client = TestClient(app_main.app)
    payload = {k: v for k, v in sample_record.items() if k in BookingRequest.model_fields}
    year = int(payload["arrival_date_year"])
    month_name = str(payload["arrival_date_month"])
    day = int(payload["arrival_date_day_of_month"])
    month_num = datetime.strptime(month_name, "%B").month
    payload["arrival_date"] = datetime(year, month_num, day).isoformat()
    for key in (
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
    ):
        payload.pop(key, None)

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text

    # TestClient runs background tasks synchronously after the response
    # returns, so the row should be present by the time the call completes.
    assert count_predictions(db_path) == 1

    with sqlite3.connect(db_path) as conn:
        row = dict(
            zip(
                [r[1] for r in conn.execute("PRAGMA table_info(predictions)")],
                conn.execute("SELECT * FROM predictions").fetchone(),
            )
        )
    assert row["probability"] == response.json()["probability"]
    assert row["risk_tier"] == response.json()["risk_tier"]
    assert json.loads(row["top_features"]) == response.json()["top_features"]
