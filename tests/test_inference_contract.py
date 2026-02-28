from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

import src.app.main as app_main
from src.app.schemas import BookingRequest
from src.serving.inference import load_artifacts


def test_predict_endpoint_contract_with_arrival_date(
    monkeypatch,
    trained_artifacts_dir,
    sample_record,
) -> None:
    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(app_main, "_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)

    payload = dict(sample_record)
    payload = {k: v for k, v in payload.items() if k in BookingRequest.model_fields}
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
    body = response.json()

    assert isinstance(body["label_high_precision"], int)
    assert isinstance(body["label_max_f1"], int)
    assert isinstance(body["label_cost_sensitive"], int)
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert isinstance(body["cost_threshold_source"], str)
    assert isinstance(body["cost_threshold_fallback_used"], bool)
    assert isinstance(body["alerts"], list)
    assert 0.0 <= float(body["probability"]) <= 1.0
    assert 0.0 <= float(body["threshold_high_precision"]) <= 1.0
    assert 0.0 <= float(body["threshold_max_f1"]) <= 1.0
    assert 0.0 <= float(body["threshold_cost_sensitive"]) <= 1.0


def test_predict_endpoint_rejects_missing_arrival_fields(
    monkeypatch, trained_artifacts_dir, sample_record
) -> None:
    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(app_main, "_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)

    payload = dict(sample_record)
    payload = {k: v for k, v in payload.items() if k in BookingRequest.model_fields}
    payload.pop("arrival_date_year", None)
    payload.pop("arrival_date_month", None)
    payload.pop("arrival_date_week_number", None)
    payload.pop("arrival_date_day_of_month", None)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, response.text


def test_model_info_endpoint_contract(monkeypatch, trained_artifacts_dir) -> None:
    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(app_main, "_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)

    response = client.get("/model-info")
    assert response.status_code == 200, response.text
    body = response.json()

    assert isinstance(body["model_selection_policy"], str)
    assert isinstance(body["model_type"], str)
    assert int(body["feature_count"]) > 0
    assert isinstance(body["has_calibrator"], bool)
    assert isinstance(body["thresholds"], dict)
    assert set(body["thresholds"].keys()) == {"high_precision", "max_f1", "cost_sensitive"}
    assert isinstance(body["threshold_sources"], dict)
    assert "cost_sensitive" in body["threshold_sources"]
    assert isinstance(body["risk_tier_thresholds"], dict)
    assert "medium" in body["risk_tier_thresholds"]
    assert "high" in body["risk_tier_thresholds"]
    assert isinstance(body["alerts"], list)


def test_predict_cost_threshold_fallback_alert(
    monkeypatch,
    trained_artifacts_dir,
    sample_record,
) -> None:
    artifacts = load_artifacts(trained_artifacts_dir)
    artifacts.thresholds = {
        "high_precision": {"threshold": 0.8},
        "max_f1": {"threshold": 0.4},
    }
    monkeypatch.setattr(app_main, "_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)

    payload = dict(sample_record)
    payload = {k: v for k, v in payload.items() if k in BookingRequest.model_fields}
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
    body = response.json()

    assert body["cost_threshold_fallback_used"] is True
    assert body["cost_threshold_source"] == "max_f1_fallback"
    assert len(body["alerts"]) >= 1
    assert float(body["threshold_cost_sensitive"]) == float(body["threshold_max_f1"])
