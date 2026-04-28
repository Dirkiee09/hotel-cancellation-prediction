from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import MODEL_SELECTION_POLICY, TARGET_COL
from src.models.train import is_lightgbm_available
from src.pipelines import run_training_pipeline
from src.pipelines.train import FALLBACK_MODEL_FAMILY, _select_model_family
from src.serving.inference import load_artifacts


def test_training_pipeline_produces_fitted_artifacts(tmp_path) -> None:
    outputs = run_training_pipeline(
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        max_rows=5000,
    )
    assert outputs.model_path.exists()
    assert "high_precision" in outputs.metrics
    assert "max_f1" in outputs.thresholds
    assert (outputs.artifacts_dir / "probability_calibrator.pkl").exists()
    assert (outputs.artifacts_dir / "hashes.json").exists()

    artifacts = load_artifacts(outputs.artifacts_dir)
    assert hasattr(artifacts.model, "predict_proba")
    assert artifacts.calibrator is not None

    metadata_path = outputs.artifacts_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_selection_policy"] == MODEL_SELECTION_POLICY
    allowed = {"gradient_boosting", "xgboost"}
    if is_lightgbm_available():
        allowed.add("lightgbm")
    assert metadata["model_type"] in allowed
    assert metadata["model_selection"]["winner"] in allowed
    assert "lineage" in metadata
    assert "calibration" in metadata

    selection_path = outputs.reports_dir / "model_selection_summary.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["winner"] in allowed
    assert "candidates" in selection

    segment_path = outputs.reports_dir / "segment_metrics.json"
    segment = json.loads(segment_path.read_text(encoding="utf-8"))
    assert "rows" in segment
    assert isinstance(segment["rows"], list)


def test_rolling_selection_fallback_for_tiny_dataset() -> None:
    """When data is too small for rolling windows, fallback model is returned."""
    rng = np.random.default_rng(42)
    n = 50  # Well below ROLLING_SELECTION_MIN_TRAIN_ROWS (1500)
    tiny_df = pd.DataFrame(
        {
            TARGET_COL: rng.integers(0, 2, size=n),
            "lead_time": rng.integers(0, 365, size=n),
        }
    )
    result = _select_model_family(tiny_df, feature_cols=["lead_time"])
    assert result["winner"] == FALLBACK_MODEL_FAMILY
    assert "fallback_reason" in result
    assert result["candidates"] == []
    assert result["folds"] == []
