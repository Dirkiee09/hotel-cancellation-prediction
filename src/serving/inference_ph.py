"""Inference utilities for the PH (Philippine dataset) sub-study server.

Parallel to src/serving/inference.py but loads from artifacts/ph/ and uses
the reduced PH feature schema (8 raw / 16 engineered features). The PH model
is a sklearn Pipeline (preprocessor + LightGBM); the calibrator is separate.

This module is deliberately isolated from the Portugal serving path:
src/app/main.py never imports from here, and src/app/ph_main.py never imports
from src/serving/inference.py. Each server caches its own artifact singleton.
"""

from __future__ import annotations

import json
import logging
import threading
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import PH_ARTIFACTS_DIR, PH_BOOKING_TIME_FEATURES, PH_TARGET_COL
from src.utils.validate_data import clean_raw_ph

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer",
    category=UserWarning,
)


class PHModelArtifacts:
    """Loaded PH model bundle. Simpler than Portugal ModelArtifacts (no ADR)."""

    def __init__(
        self,
        model_pipeline: Any,
        calibrator: Any,
        feature_columns: list[str],
        thresholds: dict[str, Any],
        metadata: dict[str, Any],
        transferability_report: dict[str, Any],
    ) -> None:
        self.model_pipeline = model_pipeline
        self.calibrator = calibrator
        self.feature_columns = feature_columns
        self.thresholds = thresholds
        self.metadata = metadata
        self.transferability_report = transferability_report


_CACHED_PH_ARTIFACTS: PHModelArtifacts | None = None
_CACHED_PH_LOCK = threading.Lock()


def get_cached_ph_artifacts() -> PHModelArtifacts:
    """Return the shared PH artifact singleton (thread-safe, lazy-loaded)."""
    global _CACHED_PH_ARTIFACTS
    if _CACHED_PH_ARTIFACTS is not None:
        return _CACHED_PH_ARTIFACTS
    with _CACHED_PH_LOCK:
        if _CACHED_PH_ARTIFACTS is None:
            _CACHED_PH_ARTIFACTS = load_ph_artifacts()
    return _CACHED_PH_ARTIFACTS


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ph_artifacts(artifacts_dir: Path = PH_ARTIFACTS_DIR) -> PHModelArtifacts:
    """Load PH model pipeline, calibrator, thresholds, and metadata."""
    artifacts_dir = artifacts_dir.resolve()
    logger.info("PH artifacts dir: %s", artifacts_dir)

    model_path = artifacts_dir / "ph_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"PH model artifact not found: {model_path}. "
            "Run `python scripts/train_ph.py` to generate the PH artifacts."
        )
    model_pipeline = joblib.load(model_path)

    calibrator_path = artifacts_dir / "ph_calibrator.pkl"
    calibrator: Any = None
    if calibrator_path.exists():
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception as exc:  # pragma: no cover - non-fatal
            logger.warning(
                "Failed to load PH calibrator %s: %s. Predictions will be uncalibrated.",
                calibrator_path,
                exc,
            )

    thresholds: dict[str, Any] = {}
    thresholds_path = artifacts_dir / "ph_thresholds.json"
    if thresholds_path.exists():
        thresholds = _load_json(thresholds_path)

    feature_columns: list[str] = list(PH_BOOKING_TIME_FEATURES)
    feature_cols_path = artifacts_dir / "ph_feature_columns.json"
    if feature_cols_path.exists():
        feature_columns = _load_json(feature_cols_path).get("features", feature_columns)

    metadata: dict[str, Any] = {}
    metadata_path = artifacts_dir / "ph_model_metadata.json"
    if metadata_path.exists():
        metadata = _load_json(metadata_path)

    transferability_report: dict[str, Any] = {}
    transfer_path = artifacts_dir.parent.parent / "reports" / "ph" / "ph_transferability.json"
    if transfer_path.exists():
        transferability_report = _load_json(transfer_path)

    return PHModelArtifacts(
        model_pipeline=model_pipeline,
        calibrator=calibrator,
        feature_columns=feature_columns,
        thresholds=thresholds,
        metadata=metadata,
        transferability_report=transferability_report,
    )


def _engineer_ph_features(raw: dict[str, Any]) -> pd.DataFrame:
    """Convert a raw PH request dict to the engineered 16-feature DataFrame.

    Wraps the dict with PH column names that ``clean_raw_ph`` recognises, then
    runs the same cleaner the training pipeline uses. The output has the
    engineered features (lead_time, total_stay, month_sin, etc.) that the
    saved pipeline expects.
    """
    # Build a 1-row DataFrame using PH source column names so clean_raw_ph
    # can rename + derive features identically to training time.
    row = {
        "Lead_Time_Days": raw.get("lead_time", 0),
        "Weekend_Nights": raw.get("weekend_nights", raw.get("stays_in_weekend_nights", 0)),
        "Week_Nights": raw.get("week_nights", raw.get("stays_in_week_nights", 0)),
        "Adults": raw.get("adults", 1),
        "Children": raw.get("children", 0),
        "Babies": raw.get("babies", 0),
        "ADR_Rate": raw.get("adr", 0.0),
        "Room_Type": raw.get("reserved_room_type", "Standard Room"),
        "Deposit_Type": raw.get("deposit_type", "No Deposit"),
        "Special_Requests": int(raw.get("total_of_special_requests", 0) or 0),
        "Booking_Status": "Checked-in",  # placeholder, cleaned out
        "Arrival_Date": raw.get("arrival_date"),
        # Constants the cleaner expects but drops anyway
        "Meals": "Breakfast (Complimentary)",
        "Guest_Type": "Walk-In",
        "Nights_Stayed": int(raw.get("weekend_nights", 0) or 0)
        + int(raw.get("week_nights", 0) or 0),
        "Booking_Date": raw.get("arrival_date"),
        "Booking_ID": "ph_inference",
    }
    df = pd.DataFrame([row])
    cleaned, _ = clean_raw_ph(df)
    return cleaned


def predict_ph(record: dict[str, Any], artifacts: PHModelArtifacts) -> tuple[float, pd.DataFrame]:
    """Predict P(cancel) for a single PH booking. Returns (calibrated_prob, feature_df)."""
    engineered = _engineer_ph_features(record)
    feature_cols = [c for c in artifacts.feature_columns if c in engineered.columns]
    if PH_TARGET_COL in engineered.columns:
        engineered = engineered.drop(columns=[PH_TARGET_COL])
    X = engineered[feature_cols]
    raw_prob = float(artifacts.model_pipeline.predict_proba(X)[:, 1][0])
    if artifacts.calibrator is not None:
        prob = float(np.clip(artifacts.calibrator.predict([raw_prob])[0], 0.0, 1.0))
    else:
        prob = raw_prob
    return prob, X


def explain_ph_prediction(
    feature_df: pd.DataFrame,
    artifacts: PHModelArtifacts,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-N TreeSHAP contributions for one PH prediction.

    Aggregates one-hot encoded SHAP contributions back to raw PH feature names.
    Falls back to an empty list if shap is missing or the model is unsupported.
    """
    try:
        import shap
    except ImportError:
        return []

    try:
        pipeline = artifacts.model_pipeline
        preprocessor = pipeline.named_steps.get("preprocessor") or pipeline.named_steps.get(
            "encode"
        )
        if preprocessor is None:
            # The PH pipeline wraps preprocessor under "preprocessor"; some
            # earlier versions used a single step. Try the first step.
            step_names = list(pipeline.named_steps.keys())
            preprocessor = pipeline.named_steps[step_names[0]]
            tree_model = pipeline.named_steps[step_names[-1]]
        else:
            tree_model = pipeline.named_steps["model"]

        X = preprocessor.transform(feature_df)
        explainer = shap.TreeExplainer(tree_model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        contributions = sv[0] if sv.ndim > 1 else sv

        encoded_names = list(
            preprocessor.named_steps["encode"].get_feature_names_out()
            if hasattr(preprocessor, "named_steps")
            else preprocessor.get_feature_names_out()
        )

        raw_features = list(feature_df.columns)
        agg: dict[str, float] = {f: 0.0 for f in raw_features}
        for enc_name, val in zip(encoded_names, contributions, strict=True):
            rest = enc_name.split("__", 1)[1] if "__" in enc_name else enc_name
            matched = next(
                (raw for raw in raw_features if rest == raw or rest.startswith(raw + "_")),
                None,
            )
            if matched:
                agg[matched] += float(val)

        ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        result: list[dict[str, Any]] = []
        for name, contrib in ranked:
            raw_val = feature_df[name].iloc[0] if name in feature_df.columns else None
            result.append(
                {
                    "feature": name,
                    "value": _safe_value(raw_val),
                    "contribution": round(float(contrib), 4),
                }
            )
        return result

    except Exception as exc:  # pragma: no cover — best-effort, non-fatal
        logger.debug("PH SHAP explanation failed (non-fatal): %s", exc)
        return []


def _safe_value(val: Any) -> Any:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return round(float(val), 4)
    if isinstance(val, np.bool_):
        return bool(val)
    return val
