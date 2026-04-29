"""Inference utilities for serving."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, BOOKING_TIME_FEATURES, LEAKAGE_COLS, MODEL_SELECTION_POLICY
from src.features.build import ensure_model_features

logger = logging.getLogger(__name__)

# Suppress harmless sklearn/LightGBM warnings that fire on every prediction.
# The ColumnTransformer outputs a numpy array (no feature names) which is correct
# behaviour — the warning is a false positive.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
_LOGGED_ARTIFACT_DIRS: set[str] = set()

# Shared singleton so FastAPI + Gradio share one copy in memory.
_CACHED_ARTIFACTS: ModelArtifacts | None = None
_CACHED_ARTIFACTS_LOCK = __import__("threading").Lock()


def get_cached_artifacts() -> ModelArtifacts:
    """Return the shared artifact singleton (thread-safe, lazy-loaded)."""
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is not None:
        return _CACHED_ARTIFACTS
    with _CACHED_ARTIFACTS_LOCK:
        if _CACHED_ARTIFACTS is None:
            _CACHED_ARTIFACTS = load_artifacts()
    return _CACHED_ARTIFACTS


class ModelArtifacts:
    """Container for loaded model artifacts, thresholds, and metadata."""

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
    """Load trained model pipeline, calibrator, and configuration from the artifacts directory."""
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

    try:
        model = joblib.load(pipeline_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model artifact {pipeline_path}: {exc}. "
            "The file may be corrupted. Re-run `python scripts/train.py`."
        ) from exc

    preprocessor = None
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor") or model.named_steps.get("preprocess")
    calibrator = None
    calibrator_path = artifacts_dir / "probability_calibrator.pkl"
    if calibrator_path.exists():
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception as exc:
            logger.warning(
                "Failed to load calibrator %s: %s. Predictions will not be calibrated.",
                calibrator_path,
                exc,
            )
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
    missing_cols = [
        col for col in feature_columns if col not in engineered.columns and col not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Cannot prepare features: columns {missing_cols} are missing after "
            f"feature engineering and not in raw input. This suggests a schema mismatch "
            f"between the trained model and the input data."
        )
    for col in feature_columns:
        if col not in engineered.columns:
            engineered[col] = df[col]
    return engineered[feature_columns]


def predict_proba(records: Any, artifacts: ModelArtifacts) -> tuple[list[float], pd.DataFrame]:
    df_raw = _to_dataframe(records)
    df = _prepare_features(df_raw, artifacts.feature_columns)
    if artifacts.is_pipeline:
        probs = artifacts.model.predict_proba(df)[:, 1]
    else:
        if artifacts.preprocessor is None:
            raise RuntimeError(
                "Model artifact is not a sklearn Pipeline but no separate preprocessor "
                "was found. Re-run `python scripts/train.py` to regenerate artifacts."
            )
        X = artifacts.preprocessor.transform(df)
        probs = artifacts.model.predict_proba(X)[:, 1]
    if artifacts.calibrator is not None:
        probs = np.clip(artifacts.calibrator.predict(probs), 0.0, 1.0)
    return probs.tolist(), df


def explain_prediction(
    feature_df: pd.DataFrame,
    artifacts: ModelArtifacts,
    top_n: int = 5,
) -> list[dict[str, object]]:
    """Return the top-N feature contributions for a single prediction using TreeSHAP.

    Falls back gracefully to an empty list if SHAP is unavailable or the model
    type is unsupported.  This keeps the /predict endpoint fast and never fails.
    """
    try:
        import shap  # noqa: F811
    except ImportError:
        logger.debug("shap not installed; skipping prediction explanation")
        return []

    try:
        # Extract the underlying tree model from the sklearn Pipeline
        model = artifacts.model
        if hasattr(model, "named_steps"):
            # Pipeline: preprocessor → model
            step_names = list(model.named_steps.keys())
            tree_model = model.named_steps[step_names[-1]]
            preprocessor = model.named_steps.get(step_names[0])
            if preprocessor is not None:
                X = preprocessor.transform(feature_df)
            else:
                X = feature_df.values
        else:
            tree_model = model
            if artifacts.preprocessor is not None:
                X = artifacts.preprocessor.transform(feature_df)
            else:
                X = feature_df.values

        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            sv = shap_values[1]  # contributions toward P(cancel)
        else:
            sv = shap_values

        # sv shape: (n_samples, n_features_encoded) — we want the first row
        contributions = sv[0] if sv.ndim > 1 else sv

        # Map encoded feature names back to raw feature names
        raw_features = list(feature_df.columns)
        if hasattr(X, "shape") and X.shape[1] != len(raw_features):
            # Encoded features (one-hot): aggregate contributions per raw feature
            encoded_names = _get_encoded_feature_names(artifacts)
            agg = _aggregate_encoded_contributions(contributions, encoded_names, raw_features)
        else:
            agg = dict(zip(raw_features, contributions))

        # Sort by absolute contribution and return top_n
        sorted_feats = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        result: list[dict[str, object]] = []
        for feat_name, contrib in sorted_feats:
            raw_val = feature_df[feat_name].iloc[0] if feat_name in feature_df.columns else None
            result.append(
                {
                    "feature": feat_name,
                    "value": _safe_value(raw_val),
                    "contribution": round(float(contrib), 4),
                }
            )
        return result

    except Exception as exc:
        logger.debug("SHAP explanation failed (non-fatal): %s", exc)
        return []


def _get_encoded_feature_names(artifacts: ModelArtifacts) -> list[str]:
    """Extract encoded feature names from the preprocessor's OneHotEncoder."""
    model = artifacts.model
    preprocessor = None
    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())
        preprocessor = model.named_steps.get(step_names[0])
    elif artifacts.preprocessor is not None:
        preprocessor = artifacts.preprocessor

    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except (AttributeError, ValueError) as exc:
            logger.debug("encoded_feature_names_unavailable error=%s", exc)
    return []


def _aggregate_encoded_contributions(
    contributions: Any,
    encoded_names: list[str],
    raw_features: list[str],
) -> dict[str, float]:
    """Sum SHAP contributions of one-hot encoded columns back to raw feature level."""
    agg: dict[str, float] = {f: 0.0 for f in raw_features}

    if len(encoded_names) != len(contributions):
        # Fallback: can't map — return empty
        return agg

    for enc_name, val in zip(encoded_names, contributions):
        # Encoded names look like "cat__feature_value" or "num__feature"
        matched = False
        for raw in raw_features:
            if raw in enc_name:
                agg[raw] += float(val)
                matched = True
                break
        if not matched:
            # Put unmapped contributions under a generic key
            agg.setdefault("_other", 0.0)
            agg["_other"] += float(val)

    # Remove the _other bucket if it exists but is negligible
    if "_other" in agg and abs(agg["_other"]) < 1e-6:
        del agg["_other"]

    return agg


def _safe_value(val: Any) -> object:
    """Convert a feature value to a JSON-safe type."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return round(float(val), 4)
    if isinstance(val, np.bool_):
        return bool(val)
    return val
