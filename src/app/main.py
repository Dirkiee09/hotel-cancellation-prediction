"""FastAPI app entry point."""

from __future__ import annotations

import logging

import gradio as gr
from fastapi import FastAPI, HTTPException, status

from src.app.schemas import BookingRequest, ModelInfoResponse, PredictionResponse
from src.config import MODEL_SELECTION_POLICY, RISK_TIER_HIGH_THRESHOLD, RISK_TIER_MEDIUM_THRESHOLD
from src.serving.inference import explain_prediction, get_cached_artifacts, predict_proba
from src.utils.thresholds import resolve_thresholds

from .ui import BACKGROUND_CSS, build_ui

logger = logging.getLogger(__name__)

app = FastAPI(title="Hotel Booking Cancellation")


def get_artifacts():
    """Delegate to the shared singleton in inference.py."""
    return get_cached_artifacts()


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


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    try:
        artifacts = get_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    thresholds, threshold_sources, _, alerts = resolve_thresholds(artifacts.thresholds or {})
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

    try:
        probs, feature_df = predict_proba(payload.model_dump(exclude={"arrival_date"}), artifacts)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed due to invalid input: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model inference unavailable: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error. Check server logs for details.",
        ) from exc
    prob = float(probs[0])
    thresholds, threshold_sources, cost_fallback, alerts = resolve_thresholds(
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

    # Compute per-prediction feature explanations (non-blocking — empty list on failure)
    top_features = explain_prediction(feature_df, artifacts, top_n=5)

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
        top_features=top_features,
    )


def mount_gradio(app_: FastAPI) -> FastAPI:
    ui = build_ui()
    return gr.mount_gradio_app(app_, ui, path="/ui", css=BACKGROUND_CSS)


app = mount_gradio(app)
