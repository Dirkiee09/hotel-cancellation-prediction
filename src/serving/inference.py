"""Inference utilities for serving."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, BOOKING_TIME_FEATURES, LEAKAGE_COLS, MODEL_SELECTION_POLICY
from src.features.build import ensure_model_features

logger = logging.getLogger(__name__)
_LOGGED_ARTIFACT_DIRS: set[str] = set()


class ModelArtifacts:
    def __init__(
        self,
        model,
        preprocessor,
        calibrator,
        feature_columns,
        thresholds,
        metadata,
        is_pipeline: bool,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.calibrator = calibrator
        self.feature_columns = feature_columns
        self.thresholds = thresholds
        self.metadata = metadata
        self.is_pipeline = is_pipeline


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> ModelArtifacts:
    artifacts_dir = artifacts_dir.resolve()
    dir_key = str(artifacts_dir)
    if dir_key not in _LOGGED_ARTIFACT_DIRS:
        try:
            cwd = Path.cwd()
        except OSError:
            cwd = Path(".")
        logger.info("Inference cwd: %s", cwd)
        logger.info("Artifacts dir: %s", artifacts_dir)
        for name in ("best_model.pkl", "feature_columns.json", "thresholds.json"):
            path = artifacts_dir / name
            logger.info("Artifact exists %s: %s", path, path.exists())
        _LOGGED_ARTIFACT_DIRS.add(dir_key)

    thresholds: dict[str, Any] = {}
    thresholds_path = artifacts_dir / "thresholds.json"
    if thresholds_path.exists():
        thresholds = _load_json(thresholds_path)

    metadata: dict[str, Any] = {}
    metadata_json = artifacts_dir / "model_metadata.json"
    if metadata_json.exists():
        metadata = _load_json(metadata_json)
        policy = metadata.get("model_selection_policy")
        if policy and policy != MODEL_SELECTION_POLICY:
            logger.warning(
                "Artifact policy mismatch current=%s artifact=%s",
                MODEL_SELECTION_POLICY,
                policy,
            )

    feature_columns: list[str] = BOOKING_TIME_FEATURES
    feature_columns_path = artifacts_dir / "feature_columns.json"
    if feature_columns_path.exists():
        feature_columns = _load_json(feature_columns_path).get("features", feature_columns)

    leaking = sorted(set(feature_columns).intersection(LEAKAGE_COLS))
    if leaking:
        raise ValueError(f"Loaded feature columns contain leakage fields: {leaking}")

    pipeline_path = artifacts_dir / "best_model.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline artifact not found: {pipeline_path}. "
            "Run `python scripts/train.py` to generate best_model.pkl."
        )

    model = joblib.load(pipeline_path)

    preprocessor = None
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor") or model.named_steps.get("preprocess")
    calibrator = None
    calibrator_path = artifacts_dir / "probability_calibrator.pkl"
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
    return ModelArtifacts(
        model,
        preprocessor,
        calibrator,
        feature_columns,
        thresholds,
        metadata,
        is_pipeline=hasattr(model, "named_steps"),
    )


def _to_dataframe(records: Any) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, dict):
        return pd.DataFrame([records])
    if isinstance(records, list):
        return pd.DataFrame(records)
    raise TypeError(
        f"predict_proba expects dict, list[dict], or DataFrame, got {type(records).__name__}"
    )


def _prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    engineered = ensure_model_features(df)
    for col in feature_columns:
        if col not in engineered.columns:
            engineered[col] = df[col] if col in df.columns else None
    return engineered[feature_columns]


def predict_proba(records: Any, artifacts: ModelArtifacts) -> tuple[list[float], pd.DataFrame]:
    df_raw = _to_dataframe(records)
    df = _prepare_features(df_raw, artifacts.feature_columns)
    if artifacts.is_pipeline:
        probs = artifacts.model.predict_proba(df)[:, 1]
    else:
        X = artifacts.preprocessor.transform(df)
        probs = artifacts.model.predict_proba(X)[:, 1]
    if artifacts.calibrator is not None:
        probs = np.clip(artifacts.calibrator.predict(probs), 0.0, 1.0)
    return probs.tolist(), df
