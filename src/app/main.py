"""FastAPI app entry point."""

from __future__ import annotations

import threading

import gradio as gr
from fastapi import FastAPI, HTTPException, status

from src.app.schemas import BookingRequest, ModelInfoResponse, PredictionResponse
from src.config import MODEL_SELECTION_POLICY, RISK_TIER_HIGH_THRESHOLD, RISK_TIER_MEDIUM_THRESHOLD
from src.serving.inference import ModelArtifacts, load_artifacts, predict_proba

from .ui import BACKGROUND_CSS, build_ui

app = FastAPI(title="Hotel Booking Cancellation")

_ARTIFACTS: ModelArtifacts | None = None
_ARTIFACTS_LOCK = threading.Lock()


def get_artifacts() -> ModelArtifacts:
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS
    with _ARTIFACTS_LOCK:
        if _ARTIFACTS is None:
            _ARTIFACTS = load_artifacts()
    return _ARTIFACTS


@app.get("/")
def health():
    return {"status": "ok", "service": "alive"}


@app.get("/healthz")
def readiness():
    try:
        get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Artifacts not ready: {exc}",
        ) from exc
    return {"status": "ok", "service": "ready"}


def _resolve_thresholds(
    raw_thresholds: dict[str, object],
) -> tuple[dict[str, float], dict[str, str], bool, list[str]]:
    alerts: list[str] = []
    threshold_sources: dict[str, str] = {
        "high_precision": "artifact",
        "max_f1": "artifact",
        "cost_sensitive": "artifact",
    }

    def _threshold_or_none(payload: object) -> float | None:
        if isinstance(payload, dict):
            value = payload.get("threshold")
            if isinstance(value, int | float):
                return float(value)
        return None

    thr_hp = _threshold_or_none(raw_thresholds.get("high_precision")) or 0.5
    thr_f1 = _threshold_or_none(raw_thresholds.get("max_f1")) or 0.5

    cost_payload = raw_thresholds.get("cost_sensitive")
    cost_threshold = _threshold_or_none(cost_payload)
    cost_missing = cost_threshold is None
    if cost_missing:
        threshold_sources["cost_sensitive"] = "max_f1_fallback"
        alerts.append("cost_sensitive threshold missing; using max_f1 fallback.")
        thr_cost = thr_f1
    else:
        assert cost_threshold is not None
        thr_cost = cost_threshold

    thresholds = {
        "high_precision": thr_hp,
        "max_f1": thr_f1,
        "cost_sensitive": thr_cost,
    }
    return thresholds, threshold_sources, cost_missing, alerts


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    try:
        artifacts = get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    thresholds, threshold_sources, _, alerts = _resolve_thresholds(artifacts.thresholds or {})
    metadata = artifacts.metadata or {}
    lineage_bundle_sha256 = metadata.get("lineage", {}).get("artifacts", {}).get("bundle_sha256")
    model_type = str(metadata.get("model_type") or "unknown")
    artifact_policy = metadata.get("model_selection_policy")
    if artifact_policy and artifact_policy != MODEL_SELECTION_POLICY:
        alerts.append(
            f"artifact policy '{artifact_policy}' differs from current policy '{MODEL_SELECTION_POLICY}'."
        )

    return ModelInfoResponse(
        model_selection_policy=str(artifact_policy or MODEL_SELECTION_POLICY),
        model_type=model_type,
        feature_count=len(artifacts.feature_columns or []),
        has_calibrator=artifacts.calibrator is not None,
        thresholds=thresholds,
        threshold_sources=threshold_sources,
        risk_tier_thresholds={
            "medium": float(RISK_TIER_MEDIUM_THRESHOLD),
            "high": float(RISK_TIER_HIGH_THRESHOLD),
        },
        lineage_bundle_sha256=(
            str(lineage_bundle_sha256) if lineage_bundle_sha256 is not None else None
        ),
        alerts=alerts,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: BookingRequest):
    try:
        artifacts = get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    probs, _ = predict_proba(payload.model_dump(exclude={"arrival_date"}), artifacts)
    prob = float(probs[0])
    thresholds, threshold_sources, cost_fallback, alerts = _resolve_thresholds(
        artifacts.thresholds or {}
    )
    thr_hp = thresholds["high_precision"]
    thr_f1 = thresholds["max_f1"]
    thr_cost = thresholds["cost_sensitive"]

    if prob >= RISK_TIER_HIGH_THRESHOLD:
        risk_tier = "high"
    elif prob >= RISK_TIER_MEDIUM_THRESHOLD:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    return PredictionResponse(
        probability=prob,
        label_high_precision=int(prob >= thr_hp),
        label_max_f1=int(prob >= thr_f1),
        label_cost_sensitive=int(prob >= thr_cost),
        risk_tier=risk_tier,
        threshold_high_precision=thr_hp,
        threshold_max_f1=thr_f1,
        threshold_cost_sensitive=thr_cost,
        cost_threshold_source=threshold_sources["cost_sensitive"],
        cost_threshold_fallback_used=bool(cost_fallback),
        alerts=alerts,
    )


def mount_gradio(app_: FastAPI) -> FastAPI:
    ui = build_ui()
    return gr.mount_gradio_app(app_, ui, path="/ui", css=BACKGROUND_CSS)


app = mount_gradio(app)
