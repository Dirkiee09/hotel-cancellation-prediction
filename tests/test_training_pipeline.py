from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_SELECTION_POLICY, TARGET_COL
from src.models.train import is_lightgbm_available
from src.pipelines import run_training_pipeline
from src.pipelines.train import (
    FALLBACK_MODEL_FAMILY,
    _h2_baseline_comparison,
    _runtime_environment,
    _select_model_family,
)
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
    environment = metadata["lineage"]["environment"]
    assert environment["python"]
    assert "scikit-learn" in environment["packages"]

    selection_path = outputs.reports_dir / "model_selection_summary.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["winner"] in allowed
    assert "candidates" in selection

    segment_path = outputs.reports_dir / "segment_metrics.json"
    segment = json.loads(segment_path.read_text(encoding="utf-8"))
    assert "rows" in segment
    assert isinstance(segment["rows"], list)


def test_thresholds_have_sane_ordering_after_train(tmp_path) -> None:
    """high_precision threshold must be >= max_f1, and all three must lie in [0, 1].

    A precision-tuned policy must use a stricter cutoff than F1-balanced — if it doesn't,
    something has gone wrong with the threshold sweep or calibration. Cost-sensitive is
    free to be more aggressive (lower) than max_f1 because it weights FN cost.
    """
    outputs = run_training_pipeline(
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        max_rows=5000,
    )
    thr_hp = float(outputs.thresholds["high_precision"]["threshold"])
    thr_f1 = float(outputs.thresholds["max_f1"]["threshold"])
    thr_cost = float(outputs.thresholds["cost_sensitive"]["threshold"])
    for name, value in [
        ("high_precision", thr_hp),
        ("max_f1", thr_f1),
        ("cost_sensitive", thr_cost),
    ]:
        assert 0.0 <= value <= 1.0, f"{name} threshold {value} out of [0, 1]"
    assert thr_hp >= thr_f1, (
        f"high_precision threshold {thr_hp} should be >= max_f1 {thr_f1}; "
        "calibration or threshold sweep may be inverted"
    )


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


def test_gbt_challengers_have_matched_capacity() -> None:
    """XGBoost and LightGBM must compete under identical capacity budgets.

    Regression guard: the rolling-origin champion selection previously compared
    LightGBM at 300 trees / depth 7 against XGBoost at 100 trees / depth 5,
    which confounded the selection result with hyperparameter allocation.
    """
    from src.models.train import get_default_lgbm_params, get_default_xgb_params

    xgb_params = get_default_xgb_params()
    lgbm_params = get_default_lgbm_params()
    for key in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        assert xgb_params[key] == lgbm_params[key], f"capacity mismatch on {key}"


def _imbalanced_toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    n = 400
    y = (rng.rand(n) < 0.15).astype(int)
    x = y * 1.5 + rng.randn(n)
    return x.reshape(-1, 1), y


def test_train_lgbm_applies_scale_pos_weight() -> None:
    """train_lgbm must forward scale_pos_weight so class weighting is symmetric."""
    if not is_lightgbm_available():
        return
    from src.models.train import train_lgbm

    X, y = _imbalanced_toy_data()
    model = train_lgbm(X, y, scale_pos_weight=3.5, params={"n_estimators": 10, "max_depth": 2})
    assert model.get_params()["scale_pos_weight"] == 3.5


def test_train_gb_applies_scale_pos_weight_via_sample_weight() -> None:
    """Weighting positives must raise GB's average predicted probability."""
    from src.models.train import train_gb

    X, y = _imbalanced_toy_data()
    small = {"n_estimators": 20, "max_depth": 2}
    unweighted = train_gb(X, y, params=small)
    weighted = train_gb(X, y, scale_pos_weight=8.0, params=small)
    assert weighted.predict_proba(X)[:, 1].mean() > unweighted.predict_proba(X)[:, 1].mean()


def test_hypothesis_summary_is_not_circular(tmp_path) -> None:
    """H4 must be judged on the test set; H2 against a real baseline model.

    Regression guard: H4 was previously declared "supported" using validation-set
    savings at a threshold optimised on that same validation set, and H2 by
    checking which GBT family won a GBT-only comparison.
    """
    outputs = run_training_pipeline(
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        max_rows=3500,
    )
    hyp = json.loads((outputs.reports_dir / "hypothesis_summary.json").read_text(encoding="utf-8"))

    h4 = hyp["H4"]
    assert h4["evaluation_dataset"] == "test"
    assert "savings_vs_threshold_050" in h4
    assert "savings_vs_intervene_all" in h4

    h2 = hyp["H2"]
    assert h2["evaluation_dataset"] == "test"
    assert h2["baseline_model"] == "logistic_regression"
    assert "both models" in h2["calibration"]
    assert "champion_pr_auc" in h2
    assert "baseline_pr_auc" in h2
    assert "p_value" in h2

    # The test-set cost report must also be persisted in metrics.json
    metrics = json.loads((outputs.reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cost_thresholding"]["dataset"] == "validation"
    assert metrics["cost_thresholding_test"]["dataset"] == "test"
    assert "p_value" in metrics["h2_baseline_comparison"]["uncalibrated"]


def test_h2_baseline_is_calibrated_like_the_champion() -> None:
    """The H2 baseline must get the champion's isotonic step before the paired test.

    Regression guard: the baseline was compared uncalibrated against the calibrated
    champion. Calibration lowers PR-AUC for any model, so that pairing understated
    the gap (per-record stream: +0.005, p = 0.177 instead of +0.019, p < 0.001).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    rng = np.random.RandomState(3)
    X = rng.randn(900, 5)
    y = (X[:, 0] + 0.8 * rng.randn(900) > 0.3).astype(int)
    X_tr, X_va, X_te = X[:500], X[500:700], X[700:]
    y_tr, y_va, y_te = pd.Series(y[:500]), pd.Series(y[500:700]), y[700:]
    champion_raw = 1.0 / (1.0 + np.exp(-X_te[:, 0]))
    champion_cal = np.clip(
        IsotonicRegression(out_of_bounds="clip")
        .fit(1.0 / (1.0 + np.exp(-X_va[:, 0])), y_va)
        .predict(champion_raw),
        0.0,
        1.0,
    )

    result = _h2_baseline_comparison(X_tr, y_tr, X_va, y_va, X_te, y_te, champion_raw, champion_cal)

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42).fit(X_tr, y_tr)
    lr_cal = IsotonicRegression(out_of_bounds="clip").fit(lr.predict_proba(X_va)[:, 1], y_va)
    expected = average_precision_score(
        y_te, np.clip(lr_cal.predict(lr.predict_proba(X_te)[:, 1]), 0.0, 1.0)
    )
    assert result["baseline_pr_auc"] == pytest.approx(expected)
    assert result["uncalibrated"]["baseline_pr_auc"] == pytest.approx(
        average_precision_score(y_te, lr.predict_proba(X_te)[:, 1])
    )
    assert result["champion_pr_auc"] == pytest.approx(average_precision_score(y_te, champion_cal))
    for block in (result, result["uncalibrated"]):
        assert {"observed_delta", "p_value", "significant_at_05"} <= set(block)


def test_runtime_environment_names_interpreter_and_libraries() -> None:
    env = _runtime_environment()
    assert env["python"].count(".") == 2
    assert env["platform"]
    assert env["packages"]["numpy"]


def _noisy_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(7)
    X = rng.randn(600, 4)
    y = (rng.rand(600) < 0.4).astype(int)  # pure noise: val metric stalls fast
    return X[:400], y[:400], X[400:], y[400:]


def test_train_xgb_early_stopping_actually_stops() -> None:
    """Passing an eval_set must engage early stopping, not silently fit all trees.

    Regression guard: eval_set was previously forwarded without
    early_stopping_rounds, so the "early stopping" path was a no-op.
    """
    from src.models.train import train_xgb

    X_tr, y_tr, X_val, y_val = _noisy_split()
    model = train_xgb(X_tr, y_tr, X_val=X_val, y_val=y_val, params={"n_estimators": 500})
    assert model.best_iteration < 499


def test_train_lgbm_early_stopping_actually_stops() -> None:
    """LightGBM eval_set path must register an early-stopping callback."""
    if not is_lightgbm_available():
        return
    from src.models.train import train_lgbm

    X_tr, y_tr, X_val, y_val = _noisy_split()
    model = train_lgbm(X_tr, y_tr, X_val=X_val, y_val=y_val, params={"n_estimators": 500})
    assert model.best_iteration_ is not None
    assert model.best_iteration_ < 500
