from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import src.app.main as app_main
import src.serving.inference as inference_mod
from src.app.schemas import BookingRequest
from src.serving.inference import load_artifacts, predict_adr, predict_proba


def test_predict_endpoint_contract_with_arrival_date(
    monkeypatch,
    trained_artifacts_dir,
    sample_record,
) -> None:
    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
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
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
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
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
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
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
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


def test_predict_endpoint_rejects_invalid_types(monkeypatch, trained_artifacts_dir) -> None:
    """Malformed payload (wrong types) should return 422."""
    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)

    payload = {"hotel": 12345, "lead_time": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, response.text


def test_healthz_endpoint(monkeypatch, trained_artifacts_dir) -> None:
    """Health check endpoint should return 200 when artifacts are loaded."""
    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_proba_handles_nan_feature_values(
    trained_artifacts_dir,
    sample_record,
) -> None:
    """Inference gracefully handles records with NaN feature values."""
    artifacts = load_artifacts(trained_artifacts_dir)
    record = dict(sample_record)
    # Set numeric features to NaN — preprocessor should impute them
    record["previous_cancellations"] = float("nan")
    record["previous_bookings_not_canceled"] = float("nan")
    record["lead_time"] = float("nan")
    probs, feat_df = predict_proba(record, artifacts)
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0


def test_calibrator_output_bounded_under_extreme_inputs(
    trained_artifacts_dir, sample_record
) -> None:
    """Isotonic calibrator output must always remain in [0, 1] regardless of input.

    Regression guard: the inference path applies np.clip after calibrator.predict.
    """
    artifacts = load_artifacts(trained_artifacts_dir)
    extreme_records = []
    for lead_time, adr, prev_cancels in [
        (0, 0.01, 0),
        (999, 999.0, 50),
        (1, 250.0, 0),
        (180, 50.0, 10),
    ]:
        rec = dict(sample_record)
        rec["lead_time"] = lead_time
        rec["adr"] = adr
        rec["previous_cancellations"] = prev_cancels
        extreme_records.append(rec)
    probs, _ = predict_proba(extreme_records, artifacts)
    assert len(probs) == len(extreme_records)
    for p in probs:
        assert 0.0 <= p <= 1.0, f"calibrator produced out-of-bounds probability: {p}"


def test_predict_proba_invariant_to_input_column_order(
    trained_artifacts_dir, sample_record
) -> None:
    """predict_proba must produce identical output when input columns are reordered.

    Internally _prepare_features selects feature_columns in the canonical order,
    so a shuffled input dict should yield the same probability.
    """
    import random

    artifacts = load_artifacts(trained_artifacts_dir)
    keys = list(sample_record.keys())
    rng = random.Random(42)
    shuffled_keys = keys.copy()
    rng.shuffle(shuffled_keys)
    canonical = dict(sample_record)
    shuffled = {k: sample_record[k] for k in shuffled_keys}
    probs_a, _ = predict_proba(canonical, artifacts)
    probs_b, _ = predict_proba(shuffled, artifacts)
    assert probs_a == probs_b


def test_explain_prediction_returns_top_n_dicts(trained_artifacts_dir, sample_record) -> None:
    """explain_prediction returns a list of dicts with the documented contract."""
    from src.serving.inference import explain_prediction

    artifacts = load_artifacts(trained_artifacts_dir)
    _probs, feature_df = predict_proba(sample_record, artifacts)
    top = explain_prediction(feature_df, artifacts, top_n=3)
    # Empty list is a valid graceful-fallback when shap is unavailable
    if not top:
        return
    assert len(top) <= 3
    for entry in top:
        assert isinstance(entry, dict)
        assert set(entry.keys()) >= {"feature", "value", "contribution"}
        assert isinstance(entry["feature"], str)
        assert isinstance(entry["contribution"], float)


def test_aggregate_encoded_contributions_no_prefix_collision() -> None:
    """A raw feature that is a prefix of another must not absorb its contributions.

    Regression guard: substring matching attributed `num__adr_per_person` to `adr`
    because "adr" appears earlier in the raw feature list.
    """
    from src.serving.inference import _aggregate_encoded_contributions

    encoded_names = ["num__adr", "num__adr_per_person", "cat__hotel_City Hotel"]
    contributions = [0.1, 0.5, 0.2]
    raw_features = ["hotel", "adr", "adr_per_person"]

    agg = _aggregate_encoded_contributions(contributions, encoded_names, raw_features)

    assert agg["adr"] == pytest.approx(0.1)
    assert agg["adr_per_person"] == pytest.approx(0.5)
    assert agg["hotel"] == pytest.approx(0.2)


def test_aggregate_encoded_contributions_onehot_columns_sum() -> None:
    """One-hot columns of the same categorical feature are summed together."""
    from src.serving.inference import _aggregate_encoded_contributions

    encoded_names = [
        "cat__deposit_type_No Deposit",
        "cat__deposit_type_Non Refund",
        "num__lead_time",
    ]
    contributions = [0.3, -0.1, 0.7]
    raw_features = ["deposit_type", "lead_time"]

    agg = _aggregate_encoded_contributions(contributions, encoded_names, raw_features)

    assert agg["deposit_type"] == pytest.approx(0.2)
    assert agg["lead_time"] == pytest.approx(0.7)


def test_tree_explainer_is_cached_per_artifacts(trained_artifacts_dir) -> None:
    """The SHAP TreeExplainer must be built once and reused across predictions."""
    pytest.importorskip("shap")
    from src.serving.inference import _get_tree_explainer

    artifacts = load_artifacts(trained_artifacts_dir)
    explainer_a = _get_tree_explainer(artifacts)
    explainer_b = _get_tree_explainer(artifacts)
    assert explainer_a is not None
    assert explainer_a is explainer_b


def test_healthz_returns_503_when_calibrator_missing(monkeypatch, trained_artifacts_dir) -> None:
    """Readiness fails cleanly when an essential artifact is missing.

    Regression guard: previously /healthz only caught FileNotFoundError on best_model.pkl,
    so missing thresholds/calibrator/feature_columns silently degraded /predict to a
    0.5 fallback while /healthz reported 200.
    """
    artifacts = load_artifacts(trained_artifacts_dir)
    artifacts.calibrator = None  # simulate missing probability_calibrator.pkl
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)
    response = client.get("/healthz")
    assert response.status_code == 503
    assert "probability_calibrator.pkl" in response.json()["detail"]


def test_healthz_returns_503_when_thresholds_missing(monkeypatch, trained_artifacts_dir) -> None:
    """Readiness fails cleanly when thresholds.json is missing/empty."""
    artifacts = load_artifacts(trained_artifacts_dir)
    artifacts.thresholds = {}  # simulate missing thresholds.json
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)
    response = client.get("/healthz")
    assert response.status_code == 503
    assert "thresholds.json" in response.json()["detail"]


def test_predict_adr_returns_none_without_regressor(trained_artifacts_dir, sample_record) -> None:
    """predict_adr is best-effort: returns None when the ADR regressor is absent.

    The training-pipeline fixture doesn't build the ADR regressor, so this is
    the production-realistic path for clean checkouts.
    """
    artifacts = load_artifacts(trained_artifacts_dir)
    assert artifacts.adr_regressor is None  # fixture invariant
    assert predict_adr(sample_record, artifacts) is None


def test_predict_adr_returns_none_when_regressor_explicitly_unset(
    trained_artifacts_dir, sample_record
) -> None:
    """Even if metadata exists but regressor is None, the function must not crash."""
    artifacts = load_artifacts(trained_artifacts_dir)
    artifacts.adr_regressor = None
    artifacts.adr_metadata = {"features": ["lead_time"]}
    assert predict_adr(sample_record, artifacts) is None


def test_predict_endpoint_exposes_adr_fields(
    monkeypatch, trained_artifacts_dir, sample_record
) -> None:
    """The response schema must always include predicted_adr / adr_residual keys.

    With no regressor loaded, both are None (not absent). This stable shape
    makes the Power BI CSV column set deterministic regardless of which
    artifacts are present.
    """
    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
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
    body = response.json()
    assert "predicted_adr" in body
    assert "adr_residual" in body
    # Without an ADR regressor in this fixture, both fields are None.
    assert body["predicted_adr"] is None
    assert body["adr_residual"] is None


def test_predict_does_not_write_production_log(
    monkeypatch, trained_artifacts_dir, sample_record
) -> None:
    """/predict during tests must never touch the production prediction store.

    Regression guard: the endpoint's background task wrote to the real
    data/predictions/predictions.sqlite during test runs, polluting live
    monitoring data with synthetic test bookings.
    """
    import src.config as config

    prod_db = config.PROJECT_ROOT / "data" / "predictions" / "predictions.sqlite"
    before = prod_db.stat().st_mtime_ns if prod_db.exists() else None

    artifacts = load_artifacts(trained_artifacts_dir)
    monkeypatch.setattr(inference_mod, "_CACHED_ARTIFACTS", artifacts)
    client = TestClient(app_main.app)
    payload = {k: v for k, v in sample_record.items() if k in BookingRequest.model_fields}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text

    after = prod_db.stat().st_mtime_ns if prod_db.exists() else None
    assert before == after, "test /predict call modified the production prediction log"
