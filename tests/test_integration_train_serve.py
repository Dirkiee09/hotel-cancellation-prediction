from __future__ import annotations

from src.pipelines import run_training_pipeline
from src.serving.inference import load_artifacts, predict_proba


def test_integration_train_save_load_predict(tmp_path, sample_record) -> None:
    outputs = run_training_pipeline(
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        max_rows=4000,
    )
    artifacts = load_artifacts(outputs.artifacts_dir)
    probs, aligned = predict_proba([sample_record], artifacts)

    assert len(probs) == 1
    assert aligned.shape[0] == 1
    assert 0.0 <= float(probs[0]) <= 1.0
