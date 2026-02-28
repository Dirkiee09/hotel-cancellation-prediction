"""Shared helpers for thesis notebooks.

This module keeps notebook code short and reproducible by loading persisted
artifacts/reports and centralizing plotting behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    BOOKING_TIME_FEATURES,
    LEAKAGE_COLS,
    MIN_POSITIVE_RATE,
    MIN_RECALL_FOR_HIGH_PRECISION,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
    TARGET_COL,
    TEMPORAL_STABILITY_BUCKETS,
)
from src.data.load import load_raw_data
from src.features.build import add_arrival_date, build_preprocessor, split_time_aware
from src.utils.validate_data import clean_raw, validate_raw

PALETTE = {
    "champion": "#1f77b4",
    "gradient_boosting": "#1f77b4",
    "xgboost": "#ff7f0e",
    "logistic_regression": "#2ca02c",
    "dummy": "#7f7f7f",
    "roc": "#1f77b4",
    "pr": "#ff7f0e",
    "f1": "#2ca02c",
    "precision": "#9467bd",
    "recall": "#d62728",
    "positive_rate": "#8c564b",
}


BENCHMARK_TABLES = {
    "03_holdout_probability_metrics": "03_holdout_probability_metrics.csv",
    "05_holdout_threshold_metrics_max_f1": "05_holdout_threshold_metrics_max_f1.csv",
    "07_thresholds_per_model": "07_thresholds_per_model.csv",
    "14_paired_significance_vs_champion": "14_paired_significance_vs_champion.csv",
    "15_training_inference_cost": "15_training_inference_cost.csv",
    "16_rankings": "16_rankings.csv",
}


def project_root() -> Path:
    """Return the repository root regardless of notebook working directory."""
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_plotting() -> dict[str, Any]:
    """Apply shared plotting style and return plotting config."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        }
    )
    fig_dir = project_root() / "reports" / "figures" / "thesis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"palette": PALETTE, "fig_dir": fig_dir}


def save_thesis_figure(fig: plt.Figure, fig_no: int, stem: str, fig_dir: Path) -> None:
    """Save figure as PNG and PDF with thesis numbering."""
    base = f"fig_{fig_no:02d}_{stem}"
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{base}.{ext}", bbox_inches="tight")


def _prepare_df() -> pd.DataFrame:
    df = load_raw_data()
    df, _ = clean_raw(df)
    validation = validate_raw(df)
    if not validation.passed:
        raise ValueError(f"Data validation failed: {validation.messages}")
    df["arrival_date"] = add_arrival_date(df)
    df = df.sort_values("arrival_date").reset_index(drop=True)
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    keep_cols = BOOKING_TIME_FEATURES + [TARGET_COL, "arrival_date"]
    if "company" in df.columns and "company" not in keep_cols:
        keep_cols.append("company")
    df = df[keep_cols].copy()
    return df


def load_main_context() -> dict[str, Any]:
    """Load report/artifact-backed context for main thesis notebook."""
    root = project_root()
    required = [
        root / "artifacts" / "best_model.pkl",
        root / "artifacts" / "probability_calibrator.pkl",
        root / "artifacts" / "thresholds.json",
        root / "artifacts" / "threshold_sweep.csv",
        root / "reports" / "metrics.json",
        root / "reports" / "model_selection_summary.json",
        root / "reports" / "thesis" / "confidence_intervals.json",
        root / "reports" / "thesis" / "temporal_stability.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    metrics = load_json(root / "reports" / "metrics.json")
    model_selection = load_json(root / "reports" / "model_selection_summary.json")
    thresholds = load_json(root / "artifacts" / "thresholds.json")
    threshold_sweep = pd.read_csv(root / "artifacts" / "threshold_sweep.csv")
    ci_report = load_json(root / "reports" / "thesis" / "confidence_intervals.json")
    stability_report = load_json(root / "reports" / "thesis" / "temporal_stability.json")
    benchmark_tables = _load_benchmark_tables(root)

    model_pipeline = joblib.load(root / "artifacts" / "best_model.pkl")
    calibrator = joblib.load(root / "artifacts" / "probability_calibrator.pkl")

    df = _prepare_df()
    train_df, val_df, test_df = split_time_aware(df)
    feature_cols = BOOKING_TIME_FEATURES.copy()
    feature_columns_path = root / "artifacts" / "feature_columns.json"
    if feature_columns_path.exists():
        feature_cols = load_json(feature_columns_path).get("features", feature_cols)
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = None
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)
    y_test_np = y_test.to_numpy()

    y_prob_raw = model_pipeline.predict_proba(X_test)[:, 1]
    y_prob = np.clip(calibrator.predict(y_prob_raw), 0.0, 1.0)

    threshold_max_f1 = float(thresholds["max_f1"]["threshold"])
    threshold_high_precision = float(thresholds["high_precision"]["threshold"])
    y_pred_max_f1 = (y_prob >= threshold_max_f1).astype(int)

    return {
        "root": root,
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_np": y_test_np,
        "y_prob": y_prob,
        "metrics": metrics,
        "model_selection": model_selection,
        "thresholds": thresholds,
        "threshold_sweep": threshold_sweep,
        "ci_report": ci_report,
        "stability_report": stability_report,
        "model_pipeline": model_pipeline,
        "calibrator": calibrator,
        "threshold_max_f1": threshold_max_f1,
        "threshold_high_precision": threshold_high_precision,
        "y_pred_max_f1": y_pred_max_f1,
        "benchmark_tables": benchmark_tables,
        "benchmark_available": benchmark_tables is not None,
    }


def main_summary_table(ctx: dict[str, Any]) -> pd.DataFrame:
    metrics = ctx["metrics"]
    return pd.DataFrame(
        {
            "selected_model_family": [metrics.get("selected_model_family")],
            "policy": [metrics.get("model_selection", {}).get("policy")],
            "test_roc_auc": [metrics.get("max_f1", {}).get("roc_auc")],
            "test_pr_auc": [metrics.get("max_f1", {}).get("pr_auc")],
            "max_f1_threshold": [ctx["threshold_max_f1"]],
            "high_precision_threshold": [ctx["threshold_high_precision"]],
        }
    )


def _load_benchmark_tables(root: Path) -> dict[str, pd.DataFrame] | None:
    benchmark_dir = root / "reports" / "benchmarks"
    if not benchmark_dir.exists():
        return None
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in BENCHMARK_TABLES.items():
        path = benchmark_dir / filename
        if not path.exists():
            return None
        tables[name] = pd.read_csv(path)
    return tables


def benchmark_rankings_table(ctx: dict[str, Any]) -> pd.DataFrame:
    tables = ctx.get("benchmark_tables")
    if not tables:
        raise FileNotFoundError("Benchmark tables not found. Run: python scripts/benchmark.py")
    return tables["16_rankings"].copy()


def benchmark_significance_table(ctx: dict[str, Any]) -> pd.DataFrame:
    tables = ctx.get("benchmark_tables")
    if not tables:
        raise FileNotFoundError("Benchmark tables not found. Run: python scripts/benchmark.py")
    df = tables["14_paired_significance_vs_champion"].copy()
    return df.sort_values(["metric", "challenger_model"]).reset_index(drop=True)


def plot_benchmark_model_comparison(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 6) -> None:
    tables = ctx.get("benchmark_tables")
    if not tables:
        raise FileNotFoundError("Benchmark tables not found. Run: python scripts/benchmark.py")
    df = tables["03_holdout_probability_metrics"].copy()
    df = df.sort_values("pr_auc", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6), sharey=True)
    ax = axes[0]
    bars = ax.barh(df["model"], df["pr_auc"], color=PALETTE["pr"], alpha=0.85)
    ax.set_title("Figure 6A. Benchmark PR-AUC by Model")
    ax.set_xlabel("PR-AUC")
    for bar, val in zip(bars, df["pr_auc"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    ax = axes[1]
    bars = ax.barh(df["model"], df["roc_auc"], color=PALETTE["roc"], alpha=0.85)
    ax.set_title("Figure 6B. Benchmark ROC-AUC by Model")
    ax.set_xlabel("ROC-AUC")
    for bar, val in zip(bars, df["roc_auc"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "benchmark_model_probability_comparison", fig_dir)
    plt.show()


def plot_benchmark_threshold_heatmap(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 9) -> None:
    tables = ctx.get("benchmark_tables")
    if not tables:
        raise FileNotFoundError("Benchmark tables not found. Run: python scripts/benchmark.py")
    df = tables["05_holdout_threshold_metrics_max_f1"].copy()
    metric_cols = ["precision", "recall", "f1", "balanced_accuracy", "pr_auc", "roc_auc"]
    heat = df.set_index("model")[metric_cols]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_title("Figure 9. Max-F1 Policy Metrics by Model (Benchmark Heatmap)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "benchmark_threshold_heatmap", fig_dir)
    plt.show()


def plot_benchmark_cost_vs_performance(
    ctx: dict[str, Any], fig_dir: Path, fig_no: int = 10
) -> pd.DataFrame:
    tables = ctx.get("benchmark_tables")
    if not tables:
        raise FileNotFoundError("Benchmark tables not found. Run: python scripts/benchmark.py")
    perf = tables["03_holdout_probability_metrics"].copy()
    cost = tables["15_training_inference_cost"].copy()
    merged = perf.merge(cost, on="model", how="inner")

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sizes = np.clip(merged["bundle_size_mb"] * 120, 80, 1200)
    ax.scatter(
        merged["fit_seconds"],
        merged["pr_auc"],
        s=sizes,
        alpha=0.75,
        color=PALETTE["champion"],
        edgecolor="black",
    )
    for _, row in merged.iterrows():
        ax.text(row["fit_seconds"] + 0.01, row["pr_auc"] + 0.001, row["model"], fontsize=9)
    ax.set_title("Figure 10. Cost vs Performance (Bubble Size = Model Bundle MB)")
    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("PR-AUC")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "benchmark_cost_vs_performance", fig_dir)
    plt.show()
    return merged


def main_ci_table(ctx: dict[str, Any]) -> pd.DataFrame:
    return (
        pd.DataFrame(ctx["ci_report"])
        .T[["point_estimate", "ci_lower", "ci_upper", "n_bootstraps"]]
        .rename_axis("metric")
        .reset_index()
    )


def plot_main_roc_pr(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 1) -> None:
    y_true = ctx["y_test_np"]
    y_prob = ctx["y_prob"]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    prevalence = float(np.mean(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(fpr, tpr, color=PALETTE["roc"], linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_title("Figure 1A. ROC Curve (Champion Model)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.text(
        0.62,
        0.08,
        f"AUC = {roc_auc:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax = axes[1]
    ax.plot(recall_curve, precision_curve, color=PALETTE["pr"], linewidth=2.5)
    ax.hlines(prevalence, 0, 1, color="gray", linestyle="--", linewidth=1.5)
    ax.set_title("Figure 1B. Precision-Recall Curve (Champion Model)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.text(
        0.62,
        0.08,
        f"PR-AUC = {pr_auc:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "roc_pr_curves", fig_dir)
    plt.show()


def plot_main_calibration_hist(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 2) -> None:
    y_true = ctx["y_test_np"]
    y_prob = ctx["y_prob"]
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(mean_pred, frac_pos, marker="o", color=PALETTE["champion"], linewidth=2.2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_title("Figure 2A. Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")

    ax = axes[1]
    sns.histplot(
        y_prob[y_true == 0],
        bins=30,
        stat="density",
        color="#1f77b4",
        alpha=0.45,
        label="Actual 0",
        ax=ax,
    )
    sns.histplot(
        y_prob[y_true == 1],
        bins=30,
        stat="density",
        color="#d62728",
        alpha=0.45,
        label="Actual 1",
        ax=ax,
    )
    ax.set_title("Figure 2B. Predicted Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "calibration_with_probability_histogram", fig_dir)
    plt.show()


def plot_main_confusion(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 3) -> None:
    cm_norm = confusion_matrix(ctx["y_test_np"], ctx["y_pred_max_f1"], normalize="true")
    cm_df = pd.DataFrame(
        cm_norm,
        index=["Actual: Not Canceled", "Actual: Canceled"],
        columns=["Pred: Not Canceled", "Pred: Canceled"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Figure 3. Normalized Confusion Matrix (Max-F1 Policy)")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "normalized_confusion_matrix_max_f1", fig_dir)
    plt.show()


def plot_main_threshold_tradeoff(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 4) -> None:
    sweep = ctx["threshold_sweep"]
    fig, ax1 = plt.subplots(figsize=(11.5, 6))
    ax1.plot(
        sweep["threshold"],
        sweep["precision"],
        label="Precision",
        color=PALETTE["precision"],
        linewidth=2.2,
    )
    ax1.plot(
        sweep["threshold"], sweep["recall"], label="Recall", color=PALETTE["recall"], linewidth=2.2
    )
    ax1.plot(sweep["threshold"], sweep["f1"], label="F1", color=PALETTE["f1"], linewidth=2.2)

    ax2 = ax1.twinx()
    ax2.plot(
        sweep["threshold"],
        sweep["positive_rate"],
        label="Predicted Positive Rate",
        color=PALETTE["positive_rate"],
        linewidth=2.0,
        linestyle="--",
    )
    ax1.axhline(
        MIN_RECALL_FOR_HIGH_PRECISION, color=PALETTE["recall"], linestyle=":", linewidth=1.7
    )
    ax2.axhline(MIN_POSITIVE_RATE, color=PALETTE["positive_rate"], linestyle=":", linewidth=1.7)
    ax1.axvline(ctx["threshold_high_precision"], color="black", linestyle="--", linewidth=1.6)
    ax1.axvline(ctx["threshold_max_f1"], color="black", linestyle="-.", linewidth=1.6)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Precision / Recall / F1")
    ax2.set_ylabel("Predicted Positive Rate")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax1.set_title("Figure 4. Threshold Tradeoffs and Policy Constraints")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "threshold_tradeoff_profile", fig_dir)
    plt.show()


def plot_main_temporal_stability(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 5) -> None:
    stability_df = pd.DataFrame(ctx["stability_report"]["buckets"]).copy()
    fig, ax1 = plt.subplots(figsize=(11.5, 6))
    ax1.plot(
        stability_df["bucket"],
        stability_df["roc_auc"],
        marker="o",
        linewidth=2.2,
        color=PALETTE["roc"],
        label="ROC-AUC",
    )
    ax1.plot(
        stability_df["bucket"],
        stability_df["pr_auc"],
        marker="o",
        linewidth=2.2,
        color=PALETTE["pr"],
        label="PR-AUC",
    )
    ax1.set_xlabel("Chronological Bucket (Test Split)")
    ax1.set_ylabel("Metric Value")
    ax1.set_ylim(0.45, 1.0)
    ax2 = ax1.twinx()
    ax2.bar(
        stability_df["bucket"],
        stability_df["cancel_rate"],
        alpha=0.18,
        color=PALETTE["positive_rate"],
        label="Cancel Rate",
    )
    ax2.set_ylabel("Cancel Rate")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax1.set_title("Figure 5. Temporal Stability on the Test Timeline")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower left")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "temporal_stability", fig_dir)
    plt.show()


def load_analysis_context() -> dict[str, Any]:
    """Load context for appendix notebook."""
    ctx = load_main_context()
    root = ctx["root"]
    ctx["selection_summary"] = load_json(root / "reports" / "model_selection_summary.json")
    ctx["rolling_cv"] = pd.read_csv(root / "reports" / "model_selection_rolling.csv")
    ctx["selection_df"] = pd.concat([ctx["train_df"], ctx["val_df"]], ignore_index=True)
    alerts = list(ctx.get("alerts", []))
    thesis_dir = root / "reports" / "thesis"
    if (thesis_dir / "cost_sensitive_threshold.json").exists():
        ctx["cost_sensitive_threshold"] = load_json(thesis_dir / "cost_sensitive_threshold.json")
    elif (root / "reports" / "cost_threshold_summary.json").exists():
        ctx["cost_sensitive_threshold"] = load_json(
            root / "reports" / "cost_threshold_summary.json"
        )
    else:
        alerts.append(
            "cost-sensitive threshold artifact missing; API falls back to max_f1 threshold."
        )
    if (thesis_dir / "late_window_analysis.json").exists():
        ctx["late_window_analysis"] = load_json(thesis_dir / "late_window_analysis.json")
    elif (root / "reports" / "late_window_metrics.json").exists():
        ctx["late_window_analysis"] = load_json(root / "reports" / "late_window_metrics.json")
    if (thesis_dir / "hypothesis_mapping.json").exists():
        ctx["hypothesis_mapping"] = load_json(thesis_dir / "hypothesis_mapping.json")
    elif (root / "reports" / "hypothesis_summary.json").exists():
        ctx["hypothesis_mapping"] = load_json(root / "reports" / "hypothesis_summary.json")
    if (thesis_dir / "cost_threshold_sweep.csv").exists():
        ctx["cost_threshold_sweep"] = pd.read_csv(thesis_dir / "cost_threshold_sweep.csv")
    elif (root / "reports" / "cost_threshold_sweep.csv").exists():
        ctx["cost_threshold_sweep"] = pd.read_csv(root / "reports" / "cost_threshold_sweep.csv")
    else:
        alerts.append("cost-threshold sweep CSV not found; cost curve plot will be unavailable.")
    thresholds = ctx.get("thresholds", {})
    if not isinstance(thresholds.get("cost_sensitive"), dict):
        alerts.append("thresholds.json has no cost_sensitive policy; max_f1 fallback applies.")
    ctx["alerts"] = alerts
    return ctx


def context_alerts_table(ctx: dict[str, Any]) -> pd.DataFrame:
    alerts = list(ctx.get("alerts", []))
    if not alerts:
        return pd.DataFrame([{"severity": "info", "message": "No context alerts."}])
    return pd.DataFrame([{"severity": "warning", "message": message} for message in alerts])


def roadmap_status_table(ctx: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cost = ctx.get("cost_sensitive_threshold", {})
    late = ctx.get("late_window_analysis", {})
    rows.append(
        {
            "component": "cost_sensitive_threshold",
            "status": bool(cost),
            "value": float(cost.get("threshold", np.nan)) if cost else np.nan,
            "details": "decision threshold",
        }
    )
    rows.append(
        {
            "component": "savings_vs_050",
            "status": bool(cost),
            "value": float(cost.get("savings_vs_050", np.nan)) if cost else np.nan,
            "details": "expected savings vs default 0.50",
        }
    )
    rows.append(
        {
            "component": "late_window_cancel_rate",
            "status": bool(late),
            "value": float(late.get("cancel_rate_late_window", np.nan)) if late else np.nan,
            "details": "lead_time <= 3 days",
        }
    )
    rows.append(
        {
            "component": "hypothesis_mapping",
            "status": bool(ctx.get("hypothesis_mapping")),
            "value": np.nan,
            "details": "H1-H4 summary present",
        }
    )
    return pd.DataFrame(rows)


def hypothesis_mapping_table(ctx: dict[str, Any]) -> pd.DataFrame:
    payload = ctx.get("hypothesis_mapping", {})
    if not payload:
        return pd.DataFrame(columns=["hypothesis", "status", "statement"])
    rows = []
    for key, value in payload.items():
        rows.append(
            {
                "hypothesis": key,
                "status": value.get("status"),
                "statement": value.get("statement"),
            }
        )
    return pd.DataFrame(rows)


def late_window_metrics_table(ctx: dict[str, Any]) -> pd.DataFrame:
    late = ctx.get("late_window_analysis", {})
    if not late:
        return pd.DataFrame(columns=["metric", "value"])
    rows = [
        {"metric": "n_rows_late_window", "value": late.get("n_rows_late_window")},
        {"metric": "late_window_share", "value": late.get("late_window_share")},
        {"metric": "cancel_rate_overall_test", "value": late.get("cancel_rate_overall_test")},
        {"metric": "cancel_rate_late_window", "value": late.get("cancel_rate_late_window")},
    ]
    return pd.DataFrame(rows)


def _risk_tier_series(probs: np.ndarray) -> pd.Series:
    tiers = np.where(
        probs >= float(RISK_TIER_HIGH_THRESHOLD),
        "high",
        np.where(probs >= float(RISK_TIER_MEDIUM_THRESHOLD), "medium", "low"),
    )
    return pd.Series(tiers, dtype="object")


def risk_tier_calibration_tables(
    ctx: dict[str, Any],
    n_buckets: int = TEMPORAL_STABILITY_BUCKETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_df = ctx["test_df"].copy().reset_index(drop=True)
    probs = np.asarray(ctx["y_prob"], dtype=float)
    y_true = np.asarray(ctx["y_test_np"], dtype=int)
    tiers = _risk_tier_series(probs)

    base = pd.DataFrame(
        {
            "arrival_date": pd.to_datetime(test_df["arrival_date"]),
            "prob": probs,
            "y": y_true,
            "risk_tier": tiers,
        }
    ).sort_values("arrival_date", kind="mergesort")

    overall = (
        base.groupby("risk_tier", as_index=False)
        .agg(
            n_rows=("y", "size"),
            predicted_mean=("prob", "mean"),
            observed_rate=("y", "mean"),
        )
        .assign(calibration_gap=lambda d: d["observed_rate"] - d["predicted_mean"])
    )
    tier_order = {"low": 0, "medium": 1, "high": 2}
    overall = overall.sort_values("risk_tier", key=lambda s: s.map(tier_order)).reset_index(
        drop=True
    )

    bucket_indices = np.array_split(np.arange(len(base)), int(n_buckets))
    by_time_rows: list[dict[str, Any]] = []
    for bucket_id, idx in enumerate(bucket_indices, start=1):
        if len(idx) == 0:
            continue
        sub = base.iloc[idx]
        for tier in ("low", "medium", "high"):
            seg = sub[sub["risk_tier"] == tier]
            if seg.empty:
                continue
            by_time_rows.append(
                {
                    "bucket": bucket_id,
                    "tier": tier,
                    "n_rows": int(len(seg)),
                    "predicted_mean": float(seg["prob"].mean()),
                    "observed_rate": float(seg["y"].mean()),
                    "calibration_gap": float(seg["y"].mean() - seg["prob"].mean()),
                    "date_min": str(seg["arrival_date"].min().date()),
                    "date_max": str(seg["arrival_date"].max().date()),
                }
            )
    by_time = pd.DataFrame(by_time_rows)
    return overall, by_time


def plot_cost_threshold_curve(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 11) -> None:
    sweep = ctx.get("cost_threshold_sweep")
    if sweep is None or sweep.empty:
        raise FileNotFoundError("cost_threshold_sweep data not found in context.")
    cost_summary = ctx.get("cost_sensitive_threshold", {})
    threshold = float(cost_summary.get("threshold", np.nan))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(sweep["threshold"], sweep["total_cost"], linewidth=2.5, color=PALETTE["champion"])
    if not np.isnan(threshold):
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.6)
        ax.text(
            threshold + 0.01,
            float(np.nanmin(sweep["total_cost"])) * 1.01,
            f"opt={threshold:.2f}",
            fontsize=10,
        )
    ax.set_title("Figure 11. Cost-Sensitive Threshold Sweep")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Expected Total Cost")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "cost_sensitive_threshold_sweep", fig_dir)
    plt.show()


def plot_risk_tier_calibration_over_time(
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 12,
    n_buckets: int = TEMPORAL_STABILITY_BUCKETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, by_time = risk_tier_calibration_tables(ctx, n_buckets=n_buckets)
    if by_time.empty:
        raise ValueError("No per-tier rows available for risk-tier calibration plot.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=False)
    tier_palette = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}

    ax = axes[0]
    for tier in ("low", "medium", "high"):
        sub = by_time[by_time["tier"] == tier]
        if sub.empty:
            continue
        ax.plot(
            sub["bucket"],
            sub["calibration_gap"],
            marker="o",
            linewidth=2.0,
            label=tier,
            color=tier_palette[tier],
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Figure 12A. Risk-Tier Calibration Gap Over Time")
    ax.set_xlabel("Chronological Bucket")
    ax.set_ylabel("Observed - Predicted")
    ax.legend(frameon=False, title="Tier")

    ax = axes[1]
    bars = ax.bar(
        overall["risk_tier"],
        overall["n_rows"],
        color=[tier_palette.get(t, "#7f7f7f") for t in overall["risk_tier"]],
        alpha=0.75,
    )
    ax.set_title("Figure 12B. Risk-Tier Volume on Test Set")
    ax.set_xlabel("Risk Tier")
    ax.set_ylabel("Rows")
    for bar, (_, row) in zip(bars, overall.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"gap={row['calibration_gap']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "risk_tier_calibration_over_time", fig_dir)
    plt.show()
    return overall, by_time


def plot_monthly_trend(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 1) -> None:
    df = ctx["df"]
    monthly = (
        df.set_index("arrival_date")
        .resample("MS")
        .agg(cancel_rate=(TARGET_COL, "mean"), bookings=(TARGET_COL, "size"))
        .reset_index()
    )
    monthly["cancel_rate_roll3"] = monthly["cancel_rate"].rolling(3, min_periods=1).mean()
    train_end = ctx["train_df"]["arrival_date"].max()
    val_end = ctx["val_df"]["arrival_date"].max()

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()
    ax2.bar(
        monthly["arrival_date"],
        monthly["bookings"],
        width=20,
        alpha=0.16,
        color="#7f7f7f",
        label="Bookings",
    )
    ax1.plot(
        monthly["arrival_date"],
        monthly["cancel_rate"],
        linewidth=1.8,
        color=PALETTE["pr"],
        alpha=0.45,
        label="Monthly cancel rate",
    )
    ax1.plot(
        monthly["arrival_date"],
        monthly["cancel_rate_roll3"],
        linewidth=2.6,
        color=PALETTE["roc"],
        label="3-month rolling mean",
    )
    ax1.axvspan(monthly["arrival_date"].min(), train_end, color="#d9edf7", alpha=0.25)
    ax1.axvspan(train_end, val_end, color="#fcf8e3", alpha=0.30)
    ax1.axvspan(val_end, monthly["arrival_date"].max(), color="#f2dede", alpha=0.25)
    ax1.text(monthly["arrival_date"].min(), 0.96 * ax1.get_ylim()[1], "Train", fontsize=11)
    ax1.text(train_end, 0.96 * ax1.get_ylim()[1], "Val", fontsize=11)
    ax1.text(val_end, 0.96 * ax1.get_ylim()[1], "Test", fontsize=11)
    ax1.set_title("Figure 1. Monthly Cancellation Rate, Rolling Mean, and Booking Volume")
    ax1.set_xlabel("Arrival Month")
    ax1.set_ylabel("Cancellation Rate")
    ax2.set_ylabel("Booking Volume")
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "monthly_trend_with_volume_and_splits", fig_dir)
    plt.show()


def plot_model_dumbbell(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 2) -> None:
    candidates = pd.DataFrame(ctx["selection_summary"]["candidates"]).copy()
    candidates = candidates.sort_values("rolling_roc_auc_mean", ascending=True).reset_index(
        drop=True
    )
    candidates["label"] = candidates["model_family"].str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    y = np.arange(len(candidates))
    for i, row in candidates.iterrows():
        ax.plot(
            [row["rolling_pr_auc_mean"], row["rolling_roc_auc_mean"]],
            [i, i],
            color="#9e9e9e",
            linewidth=2.2,
            alpha=0.9,
        )
        ax.scatter(row["rolling_pr_auc_mean"], i, color=PALETTE["pr"], s=90, zorder=3)
        ax.scatter(row["rolling_roc_auc_mean"], i, color=PALETTE["roc"], s=90, zorder=3)
        ax.text(
            row["rolling_pr_auc_mean"] - 0.0015,
            i + 0.12,
            f"PR {row['rolling_pr_auc_mean']:.3f}",
            ha="right",
            fontsize=10,
        )
        ax.text(
            row["rolling_roc_auc_mean"] + 0.0015,
            i + 0.12,
            f"ROC {row['rolling_roc_auc_mean']:.3f}",
            ha="left",
            fontsize=10,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(candidates["label"])
    ax.set_xlabel("Rolling Mean Score")
    ax.set_title("Figure 2. Rolling-Origin Ranking: PR-AUC vs ROC-AUC")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="PR-AUC",
            markerfacecolor=PALETTE["pr"],
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="ROC-AUC",
            markerfacecolor=PALETTE["roc"],
            markersize=9,
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "ranked_dumbbell_model_selection", fig_dir)
    plt.show()


def plot_pr_isof1(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 3) -> None:
    y_true = ctx["y_test_np"]
    y_prob = ctx["y_prob"]
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob))
    prevalence = float(np.mean(y_true))

    def pr_point_at_threshold(threshold: float) -> tuple[float, float]:
        pred = (y_prob >= threshold).astype(int)
        return float(recall_score(y_true, pred, zero_division=0)), float(
            precision_score(y_true, pred, zero_division=0)
        )

    recall_f1, precision_f1 = pr_point_at_threshold(ctx["threshold_max_f1"])
    recall_hp, precision_hp = pr_point_at_threshold(ctx["threshold_high_precision"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall_curve, precision_curve, color=PALETTE["pr"], linewidth=2.6)
    ax.hlines(prevalence, 0, 1, color="gray", linestyle="--", linewidth=1.5)
    for f1_level in [0.3, 0.5, 0.7, 0.85]:
        r = np.linspace(0.01, 1.0, 300)
        p = (f1_level * r) / (2 * r - f1_level)
        p[(2 * r - f1_level) <= 0] = np.nan
        p[(p < 0) | (p > 1)] = np.nan
        ax.plot(r, p, color="#bdbdbd", linestyle=":", linewidth=1.2)
    ax.scatter(recall_f1, precision_f1, color=PALETTE["f1"], s=85, zorder=4)
    ax.scatter(recall_hp, precision_hp, color=PALETTE["precision"], s=85, zorder=4)
    ax.text(
        recall_f1,
        precision_f1 + 0.03,
        f"max_f1 @ {ctx['threshold_max_f1']:.2f}",
        color=PALETTE["f1"],
        fontsize=10,
    )
    ax.text(
        recall_hp,
        precision_hp - 0.05,
        f"high_precision @ {ctx['threshold_high_precision']:.2f}",
        color=PALETTE["precision"],
        fontsize=10,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Figure 3. Precision-Recall with Iso-F1 Guides (PR-AUC={pr_auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "pr_curve_with_iso_f1", fig_dir)
    plt.show()


def plot_threshold_diagnostics(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 4) -> None:
    sweep = ctx["threshold_sweep"]
    fig, ax1 = plt.subplots(figsize=(12, 6.2))
    ax1.plot(
        sweep["threshold"],
        sweep["precision"],
        color=PALETTE["precision"],
        linewidth=2.1,
        label="Precision",
    )
    ax1.plot(
        sweep["threshold"], sweep["recall"], color=PALETTE["recall"], linewidth=2.1, label="Recall"
    )
    ax1.plot(sweep["threshold"], sweep["f1"], color=PALETTE["f1"], linewidth=2.1, label="F1")
    ax2 = ax1.twinx()
    ax2.plot(
        sweep["threshold"],
        sweep["positive_rate"],
        color=PALETTE["positive_rate"],
        linewidth=2.0,
        linestyle="--",
        label="Predicted Positive Rate",
    )
    ax1.axhline(
        MIN_RECALL_FOR_HIGH_PRECISION, color=PALETTE["recall"], linestyle=":", linewidth=1.7
    )
    ax2.axhline(MIN_POSITIVE_RATE, color=PALETTE["positive_rate"], linestyle=":", linewidth=1.7)
    ax1.axvline(ctx["threshold_high_precision"], color="black", linestyle="--", linewidth=1.4)
    ax1.axvline(ctx["threshold_max_f1"], color="black", linestyle="-.", linewidth=1.4)
    ax1.text(
        ctx["threshold_high_precision"] + 0.01,
        0.18,
        f"high_precision={ctx['threshold_high_precision']:.2f}",
        fontsize=10,
    )
    ax1.text(
        ctx["threshold_max_f1"] + 0.01, 0.11, f"max_f1={ctx['threshold_max_f1']:.2f}", fontsize=10
    )
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Precision / Recall / F1")
    ax2.set_ylabel("Predicted Positive Rate")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax1.set_title("Figure 4. Threshold Policy Profile with Hard Constraints")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "threshold_policy_profile", fig_dir)
    plt.show()


def plot_calibration_deep(ctx: dict[str, Any], fig_dir: Path, fig_no: int = 5) -> None:
    y_true = ctx["y_test_np"]
    y_prob = ctx["y_prob"]
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=12, strategy="quantile")
    ece_test = float(
        ctx["metrics"].get("calibration", {}).get("test", {}).get("ece_calibrated", np.nan)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2.2, color=PALETTE["champion"])
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_title("Figure 5A. Reliability Diagram")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.text(
        0.05,
        0.90,
        f"ECE={ece_test:.3f}",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax = axes[1]
    sns.histplot(
        y_prob[y_true == 0],
        bins=35,
        stat="density",
        color="#1f77b4",
        alpha=0.45,
        label="Actual 0",
        ax=ax,
    )
    sns.histplot(
        y_prob[y_true == 1],
        bins=35,
        stat="density",
        color="#d62728",
        alpha=0.45,
        label="Actual 1",
        ax=ax,
    )
    ax.set_title("Figure 5B. Probability Histogram by Class")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "calibration_reliability_and_histogram", fig_dir)
    plt.show()


def grouped_permutation_stats(
    ctx: dict[str, Any], n_repeats: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return grouped permutation stats and repeat-level grouped data."""
    preprocessor = ctx["model_pipeline"].named_steps["preprocessor"]
    model = ctx["model_pipeline"].named_steps.get("model")
    if model is None:
        model = ctx["model_pipeline"].named_steps.get("classifier")
    if model is None:
        raise KeyError("Expected model step named 'model' or 'classifier' in pipeline.")

    try:
        X_test_t = preprocessor.transform(ctx["X_test"])
        feature_names = preprocessor.get_feature_names_out()
        perm = permutation_importance(
            model,
            X_test_t,
            ctx["y_test_np"],
            n_repeats=n_repeats,
            random_state=42,
            scoring="roc_auc",
            n_jobs=1,
        )
        repeat_df = pd.DataFrame(perm.importances.T, columns=feature_names)

        def feature_group(name: str) -> str:
            if name.startswith("categorical__"):
                raw = name.split("categorical__", 1)[1]
                return raw.split("_", 1)[0]
            if name.startswith("numeric__"):
                return name.split("numeric__", 1)[1]
            return name

        group_map = {c: feature_group(c) for c in repeat_df.columns}
        grouped_repeat = repeat_df.T.groupby(group_map).sum().T
    except Exception:
        # Fallback for preprocessors that do not expose transformed feature names.
        # In this mode, importance is already at original feature granularity.
        perm = permutation_importance(
            ctx["model_pipeline"],
            ctx["X_test"],
            ctx["y_test_np"],
            n_repeats=n_repeats,
            random_state=42,
            scoring="roc_auc",
            n_jobs=1,
        )
        repeat_df = pd.DataFrame(perm.importances.T, columns=list(ctx["X_test"].columns))
        grouped_repeat = repeat_df.copy()
    group_stats = pd.DataFrame(
        {
            "mean_importance": grouped_repeat.mean(axis=0),
            "std_importance": grouped_repeat.std(axis=0),
            "ci_low": grouped_repeat.quantile(0.025, axis=0),
            "ci_high": grouped_repeat.quantile(0.975, axis=0),
        }
    ).sort_values("mean_importance", ascending=False)
    return group_stats, grouped_repeat


def plot_grouped_permutation(
    ctx: dict[str, Any], fig_dir: Path, fig_no_start: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot grouped permutation mean+CI and stability distribution."""
    group_stats, grouped_repeat = grouped_permutation_stats(ctx, n_repeats=20)
    top_k = 15
    top_stats = group_stats.head(top_k).sort_values("mean_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(
        top_stats.index,
        top_stats["mean_importance"],
        xerr=[
            top_stats["mean_importance"] - top_stats["ci_low"],
            top_stats["ci_high"] - top_stats["mean_importance"],
        ],
        color="#4e79a7",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("Figure 6. Top Feature Groups by Permutation Importance (95% CI)")
    ax.set_xlabel("ROC-AUC Decrease")
    ax.set_ylabel("Original Feature Group")
    for idx, value in enumerate(top_stats["mean_importance"]):
        ax.text(value + 0.0008, idx, f"{value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no_start, "grouped_permutation_importance_ci", fig_dir)
    plt.show()

    top_groups = group_stats.head(10).index.tolist()
    stability_long = grouped_repeat[top_groups].melt(
        var_name="feature_group", value_name="importance"
    )
    fig, ax = plt.subplots(figsize=(11.5, 7))
    sns.violinplot(
        data=stability_long,
        y="feature_group",
        x="importance",
        inner=None,
        color="#a0cbe8",
        linewidth=0,
        ax=ax,
    )
    sns.stripplot(
        data=stability_long,
        y="feature_group",
        x="importance",
        color="#1f4e79",
        alpha=0.45,
        size=3.8,
        ax=ax,
    )
    ax.set_title("Figure 7. Importance Stability Across Permutation Repeats")
    ax.set_xlabel("ROC-AUC Decrease")
    ax.set_ylabel("Original Feature Group")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no_start + 1, "grouped_permutation_importance_stability", fig_dir)
    plt.show()
    return group_stats, grouped_repeat


def plot_cv_violin_strip(
    ctx: dict[str, Any], fig_dir: Path, fig_no: int = 8, sample_cap: int = 25000
) -> pd.DataFrame:
    """Plot rolling-origin vs stratified-kfold metric distributions."""
    rolling_long = ctx["rolling_cv"][["model_family", "roc_auc", "pr_auc"]].copy()
    rolling_long["method"] = "Rolling-Origin"

    selection_sample = ctx["selection_df"].tail(min(sample_cap, len(ctx["selection_df"]))).copy()
    X_sel = selection_sample[BOOKING_TIME_FEATURES].reset_index(drop=True)
    y_sel = selection_sample[TARGET_COL].astype(int).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    skf_rows: list[dict[str, Any]] = []
    model_specs = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        ),
    }
    for model_name, estimator in model_specs.items():
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y_sel), start=1):
            preprocessor = build_preprocessor()
            X_tr = preprocessor.fit_transform(X_sel.iloc[tr_idx])
            X_val = preprocessor.transform(X_sel.iloc[val_idx])
            y_tr = y_sel.iloc[tr_idx].to_numpy()
            y_val = y_sel.iloc[val_idx].to_numpy()
            model = clone(estimator)
            model.fit(X_tr, y_tr)
            p_val = model.predict_proba(X_val)[:, 1]
            skf_rows.append(
                {
                    "model_family": model_name,
                    "fold": fold,
                    "roc_auc": float(roc_auc_score(y_val, p_val)),
                    "pr_auc": float(average_precision_score(y_val, p_val)),
                    "method": "Stratified K-Fold",
                }
            )
    skf_long = pd.DataFrame(skf_rows)
    cv_compare = pd.concat([rolling_long, skf_long], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6), sharey=False)
    for ax, metric, title in [
        (axes[0], "roc_auc", "Figure 8A. ROC-AUC Distribution by CV Method"),
        (axes[1], "pr_auc", "Figure 8B. PR-AUC Distribution by CV Method"),
    ]:
        sns.violinplot(
            data=cv_compare,
            x="model_family",
            y=metric,
            hue="method",
            dodge=True,
            inner=None,
            cut=0,
            ax=ax,
        )
        sns.stripplot(
            data=cv_compare,
            x="model_family",
            y=metric,
            hue="method",
            dodge=True,
            palette={"Rolling-Origin": "#1f1f1f", "Stratified K-Fold": "#1f1f1f"},
            alpha=0.55,
            size=4,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Model Family")
        ax.set_ylabel(metric.upper())
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, loc="lower right")

    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "cv_violin_strip_comparison", fig_dir)
    plt.show()
    return cv_compare
