from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from src.eval.thesis import _run_hypothesis_mapping

_CHECK_PATH = Path(__file__).resolve().parents[1] / "scripts" / "utils" / "check.py"


def _load_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location("quality_check", _CHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(roc: float = 0.86, family: str = "lightgbm") -> dict:
    scores = {"roc_auc": roc, "pr_auc": 0.76, "f1": 0.74, "precision": 0.62, "recall": 0.89}
    return {
        "selected_model_family": family,
        "max_f1": scores,
        "cost_sensitive": {**scores, "f1": 0.68, "precision": 0.52, "recall": 0.99},
        "calibration": {"test": {"ece_calibrated": 0.031}},
        "cost_thresholding_test": {"intervene_all_cost": 111240.0, "total_cost": 71135.5},
    }


def _thresholds(max_f1: float = 0.41) -> dict:
    return {
        "max_f1": {"threshold": max_f1},
        "high_precision": {"threshold": 0.98},
        "cost_sensitive": {"threshold": 0.06},
    }


def _write_run(root: Path, metrics: dict, thresholds: dict) -> tuple[Path, Path]:
    artifacts, reports = root / "artifacts", root / "reports"
    artifacts.mkdir(parents=True)
    reports.mkdir(parents=True)
    (reports / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (artifacts / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    return artifacts, reports


def _baseline(root: Path) -> Path:
    baseline = root / "committed"
    baseline.mkdir()
    (baseline / "metrics.json").write_text(json.dumps(_metrics()), encoding="utf-8")
    (baseline / "thresholds.json").write_text(json.dumps(_thresholds()), encoding="utf-8")
    return baseline


def test_reproduce_exact_passes_only_on_identical_results(tmp_path) -> None:
    check = _load_check()
    baseline = _baseline(tmp_path)
    artifacts, reports = _write_run(tmp_path / "same", _metrics(), _thresholds())
    assert check.run_reproduce(baseline, artifacts, reports, mode="exact") == []

    artifacts, reports = _write_run(tmp_path / "tiny", _metrics(roc=0.8601), _thresholds())
    failures = check.run_reproduce(baseline, artifacts, reports, mode="exact")
    assert any("max_f1.roc_auc" in f for f in failures)


def test_reproduce_drift_reports_a_family_flip_without_failing(tmp_path) -> None:
    """Cross-platform drift can tip LightGBM vs XGBoost; that is reported, not failed.

    This is what CI saw on Linux / Python 3.11: XGBoost selected, ROC-AUC +0.001,
    PR-AUC +0.003, recall at the max-F1 threshold -0.035.
    """
    check = _load_check()
    artifacts, reports = _write_run(
        tmp_path / "linux", _metrics(roc=0.8610, family="xgboost"), _thresholds(0.38)
    )
    baseline = _baseline(tmp_path)
    assert check.run_reproduce(baseline, artifacts, reports, mode="drift") == []
    failures = check.run_reproduce(baseline, artifacts, reports, mode="exact")
    joined = " ".join(failures)
    assert "selected_model_family" in joined
    assert "threshold.max_f1" in joined


def test_reproduce_drift_fails_when_scores_break(tmp_path) -> None:
    check = _load_check()
    artifacts, reports = _write_run(tmp_path / "broken", _metrics(roc=0.84), _thresholds())
    failures = check.run_reproduce(_baseline(tmp_path), artifacts, reports, mode="drift")
    assert any("max_f1.roc_auc" in f for f in failures)


def _mapping(delta: float, significant: bool, family: str = "lightgbm") -> dict:
    return _run_hypothesis_mapping(
        champion_family=family,
        family_scores={},
        cost_summary={"savings_vs_050": 1.0},
        shap_summary=None,
        baseline_comparison={
            "champion_vs_lr_pr_auc": {
                "observed_delta": delta,
                "p_value": 0.0 if significant else 0.4,
                "significant_at_05": significant,
            }
        },
    )


def test_thesis_h2_follows_the_baseline_test_not_the_winning_family() -> None:
    """A boosted champion alone must not mark H2 supported.

    Regression guard: H2 was "supported" whenever XGBoost or LightGBM won a
    selection among boosted models only, which says nothing about baselines.
    """
    assert _mapping(0.02, significant=True)["H2"]["status"] == "supported"
    assert _mapping(0.004, significant=False)["H2"]["status"] == "not_supported"
    assert _mapping(-0.01, significant=True)["H2"]["status"] == "not_supported"
