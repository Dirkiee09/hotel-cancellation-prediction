"""Tests for deterministic multi-model benchmark tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.benchmark import run_model_benchmark
from src.models.train import is_lightgbm_available


def test_run_model_benchmark_writes_expected_tables(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    outputs = run_model_benchmark(
        reports_dir=reports_dir,
        max_rows=1800,
        n_bootstraps=20,
        temporal_buckets=4,
    )

    assert outputs.reports_dir == reports_dir / "benchmarks"
    assert len(outputs.table_paths) == 16

    expected_tables = {
        "01_dataset_split_summary",
        "02_model_specs",
        "03_holdout_probability_metrics",
        "04_holdout_threshold_metrics_050",
        "05_holdout_threshold_metrics_max_f1",
        "06_holdout_threshold_metrics_high_precision",
        "07_thresholds_per_model",
        "08_confusion_matrix_counts_per_model",
        "09_confusion_matrix_rates_per_model",
        "10_rolling_origin_fold_metrics",
        "11_rolling_origin_summary",
        "12_temporal_stability_by_bucket",
        "13_bootstrap_confidence_intervals",
        "14_paired_significance_vs_champion",
        "15_training_inference_cost",
        "16_rankings",
    }
    assert set(outputs.table_paths.keys()) == expected_tables

    for table_name, csv_path in outputs.table_paths.items():
        assert csv_path.exists(), f"Missing CSV table {table_name}"
        md_path = csv_path.with_suffix(".md")
        assert md_path.exists(), f"Missing Markdown table {table_name}"

    prob_df = pd.read_csv(outputs.table_paths["03_holdout_probability_metrics"])
    expected_models = {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    }
    if is_lightgbm_available():
        expected_models.add("lightgbm")
    assert len(prob_df) == len(expected_models)
    assert set(prob_df["model"]) == expected_models

    rank_df = pd.read_csv(outputs.table_paths["16_rankings"])
    assert rank_df["rank"].tolist() == list(range(1, len(expected_models) + 1))
    assert outputs.champion_model in set(rank_df["model"])
