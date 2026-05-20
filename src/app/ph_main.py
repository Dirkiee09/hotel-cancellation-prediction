"""FastAPI server for the PH (Philippine dataset) sub-study.

Runs independently of the Portugal server (src/app/main.py) on a different
port. Loads its own artifact bundle from ``artifacts/ph/`` via
``src/serving/inference_ph.py``. Every /predict call is appended to
``data/predictions/ph_predictions.sqlite`` and auto-exported to
``ph_predictions_live.csv`` for Power BI.

**Launch**::

    uvicorn src.app.ph_main:app --port 8001

Mounted Gradio UI at ``/ui`` (port 8001 also serves the form).
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr
from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from src.app.ph_schemas import (
    PHBookingRequest,
    PHModelInfoResponse,
    PHPredictionResponse,
)
from src.config import (
    PH_PREDICTION_LOG_DB,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
)
from src.serving.inference_ph import (
    PHModelArtifacts,
    explain_ph_prediction,
    get_cached_ph_artifacts,
    predict_ph,
)
from src.serving.prediction_log_ph import export_ph_to_csv, log_ph_prediction

logger = logging.getLogger(__name__)

DATASET_CAVEAT = (
    "This server runs the PH sub-study model trained on the real Punta Villa "
    "Resort PMS export (193 booking records, 2022-2025). The training set is "
    "small (n_test ≈ 20 rows) so bootstrap 95% CIs on PR-AUC span roughly "
    "±15 percentage points — treat displayed metrics as directional rather "
    "than as production-grade headlines."
)

app = FastAPI(
    title="Hotel Cancellation — Philippine Resort Sub-Study",
    description=(
        "Demonstration server for the Philippine resort sub-study.\n\n"
        f"⚠️ **{DATASET_CAVEAT}**\n\n"
        "See `CLAUDE.md` § *PH Sub-Study* and "
        "`notebooks/ph/11_transferability.ipynb` for the full context."
    ),
    version="0.2.0",
)


def _get_ph_artifacts() -> PHModelArtifacts:
    try:
        return get_cached_ph_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.get("/")
def root() -> dict[str, Any]:
    """Health probe + dataset caveat for any client opening the root."""
    return {
        "status": "ok",
        "server": "ph-philippine-substudy",
        "caveat": DATASET_CAVEAT,
        "endpoints": ["/predict", "/model-info", "/healthz", "/ui (Gradio)"],
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Readiness probe: confirms the PH artifacts can be loaded."""
    try:
        artifacts = _get_ph_artifacts()
    except HTTPException as exc:
        return {"ready": False, "reason": exc.detail}
    return {
        "ready": True,
        "model_family": artifacts.metadata.get("model_family", "unknown"),
        "feature_count": len(artifacts.feature_columns),
        "has_calibrator": artifacts.calibrator is not None,
        "caveat": DATASET_CAVEAT,
    }


@app.get("/model-info", response_model=PHModelInfoResponse)
def model_info() -> PHModelInfoResponse:
    artifacts = _get_ph_artifacts()
    thresholds = artifacts.thresholds or {}
    report = artifacts.transferability_report or {}
    return PHModelInfoResponse(
        model_family=artifacts.metadata.get("model_family", "unknown"),
        feature_count=len(artifacts.feature_columns),
        has_calibrator=artifacts.calibrator is not None,
        n_train=int(report.get("n_train", 0)),
        n_test=int(report.get("n_test", 0)),
        test_roc_auc=report.get("roc_auc_test"),
        test_pr_auc=report.get("pr_auc_test"),
        thresholds={
            "max_f1": float(thresholds.get("max_f1", {}).get("threshold", 0.5)),
            "high_precision": float(thresholds.get("high_precision", {}).get("threshold", 0.9)),
        },
        risk_tier_thresholds={
            "medium": float(RISK_TIER_MEDIUM_THRESHOLD),
            "high": float(RISK_TIER_HIGH_THRESHOLD),
        },
        dataset_caveat=DATASET_CAVEAT,
        dataset_diagnostics=report.get("dataset_diagnostics", {}),
    )


@app.post("/predict", response_model=PHPredictionResponse)
def predict(payload: PHBookingRequest, background_tasks: BackgroundTasks) -> PHPredictionResponse:
    artifacts = _get_ph_artifacts()

    try:
        prob, feature_df = predict_ph(payload.to_inference_dict(), artifacts)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PH prediction failed due to invalid input: {exc}",
        ) from exc
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("Unexpected PH prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal PH prediction error. Check server logs.",
        ) from exc

    thresholds = artifacts.thresholds or {}
    thr_f1 = float(thresholds.get("max_f1", {}).get("threshold", 0.5))
    thr_hp = float(thresholds.get("high_precision", {}).get("threshold", 0.9))

    if prob >= RISK_TIER_HIGH_THRESHOLD:
        risk_tier = "high"
    elif prob >= RISK_TIER_MEDIUM_THRESHOLD:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    top_features = explain_ph_prediction(feature_df, artifacts, top_n=5)

    response = PHPredictionResponse(
        probability=prob,
        label_max_f1=int(prob >= thr_f1),
        label_high_precision=int(prob >= thr_hp),
        risk_tier=risk_tier,
        threshold_max_f1=thr_f1,
        threshold_high_precision=thr_hp,
        alerts=[DATASET_CAVEAT],
        top_features=top_features,
    )

    # Persist the (request, response) pair to SQLite asynchronously so the
    # response is never delayed by disk I/O. log_ph_prediction is non-raising;
    # if the DB is unavailable the API keeps serving but a WARNING is logged.
    # Auto-refresh the CSV after each write so Power BI sees new predictions
    # on its next refresh without a manual export step.
    def _persist() -> None:
        log_ph_prediction(
            payload.model_dump(mode="json"),
            response.model_dump(),
            PH_PREDICTION_LOG_DB,
        )
        try:
            export_ph_to_csv()
        except Exception:  # pragma: no cover — non-fatal
            logger.exception("ph_predict_csv_export_failed (non-fatal)")

    background_tasks.add_task(_persist)

    return response


def _mount_gradio(app_: FastAPI) -> FastAPI:
    """Attach the PH Gradio UI at /ui (lazy import to avoid heavy startup cost)."""
    try:
        from src.app.ph_ui import PH_CSS, build_ph_ui
    except ImportError as exc:  # pragma: no cover — defensive
        logger.warning("PH Gradio UI not available: %s", exc)
        return app_
    ui = build_ph_ui()
    return gr.mount_gradio_app(app_, ui, path="/ui", css=PH_CSS)


app = _mount_gradio(app)
