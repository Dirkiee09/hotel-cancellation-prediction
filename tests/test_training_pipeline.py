from __future__ import annotations

import json

from src.config import MODEL_SELECTION_POLICY
from src.models.train import is_lightgbm_available
from src.pipelines import run_training_pipeline
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
