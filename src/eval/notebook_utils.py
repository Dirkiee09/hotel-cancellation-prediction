"""Shared helpers for thesis notebooks.

This module keeps notebook code short and reproducible by loading persisted
artifacts/reports and centralizing plotting behavior.
"""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

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


def save_thesis_figure(fig: plt.Figure, fig_no: int | str, stem: str, fig_dir: Path) -> None:
    """Save figure as PNG and PDF with thesis numbering."""
    fig_tag = f"{fig_no:02d}" if isinstance(fig_no, int) else str(fig_no)
    base = f"fig_{fig_tag}_{stem}"
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


def benchmark_rankings_table(ctx: dict[str, Any], top_n: int | None = None) -> pd.DataFrame:
    tables = ctx.get("benchmark_tables")
    if not tables:
        raise FileNotFoundError("Benchmark tables not found. Run: python scripts/benchmark.py")
    df = tables["16_rankings"].copy()
    if top_n is not None:
        df = df.head(top_n)
    return df


def benchmark_significance_table(ctx: dict[str, Any], top_n: int | None = None) -> pd.DataFrame:
    tables = ctx.get("benchmark_tables")
    if not tables:
        raise FileNotFoundError("Benchmark tables not found. Run: python scripts/benchmark.py")
    df = tables["14_paired_significance_vs_champion"].copy()
    if top_n is not None:
        rankings = tables.get("16_rankings")
        if rankings is not None:
            top_models = rankings.head(top_n + 1)["model"].tolist()
            df = df[df["challenger_model"].isin(top_models)]
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


def plot_benchmark_threshold_heatmap(
    ctx: dict[str, Any], fig_dir: Path, fig_no: int = 9, top_n: int | None = None
) -> None:
    tables = ctx.get("benchmark_tables")
    if not tables:
        raise FileNotFoundError("Benchmark tables not found. Run: python scripts/benchmark.py")
    df = tables["05_holdout_threshold_metrics_max_f1"].copy()
    if top_n is not None:
        rankings = tables.get("16_rankings")
        if rankings is not None:
            top_models = rankings.head(top_n)["model"].tolist()
            df = df[df["model"].isin(top_models)]
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
    # support both load_main_context() (key="model_selection") and load_analysis_context() (key="selection_summary")
    _sel = ctx.get("selection_summary") or ctx.get("model_selection", {})
    candidates = pd.DataFrame(_sel.get("candidates", [])).copy()
    if candidates.empty or "rolling_roc_auc_mean" not in candidates.columns:
        print("plot_model_dumbbell: no candidate data available — skipping figure.")
        return
    candidates = candidates.sort_values("rolling_roc_auc_mean", ascending=True).reset_index(
        drop=True
    )
    candidates["label"] = candidates["model_family"].str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    y = np.arange(len(candidates))
    for pos, (_, row) in enumerate(candidates.iterrows()):
        ax.plot(
            [row["rolling_pr_auc_mean"], row["rolling_roc_auc_mean"]],
            [pos, pos],
            color="#9e9e9e",
            linewidth=2.2,
            alpha=0.9,
        )
        ax.scatter(row["rolling_pr_auc_mean"], pos, color=PALETTE["pr"], s=90, zorder=3)
        ax.scatter(row["rolling_roc_auc_mean"], pos, color=PALETTE["roc"], s=90, zorder=3)
        ax.text(
            row["rolling_pr_auc_mean"] - 0.0015,
            pos + 0.12,
            f"PR {row['rolling_pr_auc_mean']:.3f}",
            ha="right",
            fontsize=10,
        )
        ax.text(
            row["rolling_roc_auc_mean"] + 0.0015,
            pos + 0.12,
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
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            ct = preprocessor.named_steps["encode"]
            _names: list[str] = []
            for _tname, _trans, _ in ct.transformers_:
                try:
                    _out = list(_trans.get_feature_names_out())
                except AttributeError:
                    _out = list(_trans.named_steps["onehot"].get_feature_names_out())
                _names.extend(f"{_tname}__{n}" for n in _out)
            feature_names = _names
        X_test_t_named = pd.DataFrame(X_test_t, columns=feature_names)
        perm = permutation_importance(
            model,
            X_test_t_named,
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
        grouped_repeat = repeat_df.T.groupby(group_map).sum().T  # type: ignore[arg-type, unused-ignore]
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
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no_start: int = 6,
    style: str = "box",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot grouped permutation mean+CI and stability distribution.

    Parameters
    ----------
    style : str
        ``"box"`` (default) uses box plots for the stability panel.
        ``"violin"`` uses the legacy violin+strip overlay.
    """
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
    if style == "violin":
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
    else:
        sns.boxplot(
            data=stability_long,
            y="feature_group",
            x="importance",
            color="#a0cbe8",
            fliersize=3,
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
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 8,
    sample_cap: int = 25000,
    style: str = "box",
) -> pd.DataFrame:
    """Plot rolling-origin vs stratified-kfold metric distributions.

    Parameters
    ----------
    style : str
        ``"box"`` (default) uses box plots. ``"violin"`` uses violin+strip.
    """
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
        if style == "violin":
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
        else:
            sns.boxplot(
                data=cv_compare,
                x="model_family",
                y=metric,
                hue="method",
                dodge=True,
                fliersize=3,
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


# ---------------------------------------------------------------------------
# SHAP & Segment utilities (used by Notebook 05)
# ---------------------------------------------------------------------------


def _feature_group_name(name: str) -> str:
    """Map a transformed feature name back to its original feature group."""
    if name.startswith("categorical__"):
        raw = name.split("categorical__", 1)[1]
        return raw.split("_", 1)[0]
    if name.startswith("numeric__"):
        return name.split("numeric__", 1)[1]
    return name


def load_shap_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Compute SHAP values for the champion model on the test set.

    Returns a dict with keys:
      - shap_values      : np.ndarray shape (n_test, n_features)
      - X_test_t_named   : pd.DataFrame with transformed feature names
      - feature_names    : list[str] of 94 transformed feature names
      - feature_importance_df : pd.DataFrame with mean_abs_shap + group
      - expected_value   : float baseline (log-odds of mean prediction)
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("pip install shap to use load_shap_context()") from exc

    preprocessor = ctx["model_pipeline"].named_steps["preprocessor"]
    model = ctx["model_pipeline"].named_steps.get("model") or ctx["model_pipeline"].named_steps.get(
        "classifier"
    )
    if model is None:
        raise KeyError("Pipeline must have a step named 'model' or 'classifier'.")

    X_test_t = preprocessor.transform(ctx["X_test"])
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for artifacts built before FunctionTransformer had feature_names_out="one-to-one".
        # Navigate past the blocking to_str step by reading OHE and imputer outputs directly.
        ct = preprocessor.named_steps["encode"]  # ColumnTransformer
        _names: list[str] = []
        for _tname, _trans, _ in ct.transformers_:
            try:
                _out = list(_trans.get_feature_names_out())
            except AttributeError:
                # cat_pipeline blocked by FunctionTransformer — bypass via OHE step
                _out = list(_trans.named_steps["onehot"].get_feature_names_out())
            _names.extend(f"{_tname}__{n}" for n in _out)
        feature_names = _names
    X_test_t_named = pd.DataFrame(X_test_t, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_test_t_named)
    # For binary classifiers some backends return list[pos_class, neg_class]
    shap_values = raw[1] if isinstance(raw, list) and len(raw) == 2 else raw

    mean_abs = np.abs(shap_values).mean(axis=0)
    groups = [_feature_group_name(f) for f in feature_names]
    importance_df = (
        pd.DataFrame({"feature_name": feature_names, "group": groups, "mean_abs_shap": mean_abs})
        .groupby("group", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)  # type: ignore[call-overload, unused-ignore]
        .reset_index(drop=True)
    )

    expected_value = float(
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    return {
        "shap_values": shap_values,
        "X_test_t_named": X_test_t_named,
        "feature_names": feature_names,
        "feature_importance_df": importance_df,
        "expected_value": expected_value,
        "explainer": explainer,
    }


def plot_shap_bar(
    shap_ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 13,
    top_k: int = 20,
) -> None:
    """Horizontal bar chart: top-k feature groups by mean |SHAP value|."""
    importance_df = (
        shap_ctx["feature_importance_df"].head(top_k).sort_values("mean_abs_shap", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, max(5, top_k * 0.42)))
    bars = ax.barh(
        importance_df["group"],
        importance_df["mean_abs_shap"],
        color="#4e79a7",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    for bar, val in zip(bars, importance_df["mean_abs_shap"]):
        ax.text(
            bar.get_width() + importance_df["mean_abs_shap"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax.set_title(f"Figure {fig_no}. SHAP Feature Importance — Top {top_k} Groups")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "shap_feature_importance_bar", fig_dir)
    plt.show()


def plot_shap_beeswarm(
    shap_ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 14,
    top_k: int = 15,
    sample_cap: int = 2000,
) -> None:
    """Beeswarm-style SHAP summary: direction + magnitude per feature group."""
    shap_values = shap_ctx["shap_values"]
    feature_names = shap_ctx["feature_names"]
    X_test_t = shap_ctx["X_test_t_named"].to_numpy()

    importance_df = shap_ctx["feature_importance_df"]
    top_groups = importance_df.head(top_k)["group"].tolist()

    # Aggregate SHAP values to original feature groups and track if categorical
    group_shap: dict[str, np.ndarray] = {}
    group_feat: dict[str, np.ndarray] = {}
    group_is_categorical: dict[str, bool] = {}
    for i, fname in enumerate(feature_names):
        g = _feature_group_name(fname)
        if g not in top_groups:
            continue
        is_cat = fname.startswith("categorical__")
        if g not in group_shap:
            group_shap[g] = shap_values[:, i].copy()
            group_feat[g] = X_test_t[:, i].copy()
            group_is_categorical[g] = is_cat
        else:
            group_shap[g] += shap_values[:, i]
            # A group is categorical if any constituent feature is categorical
            group_is_categorical[g] = group_is_categorical[g] or is_cat

    # Subsample for plotting performance
    n = min(sample_cap, len(shap_values))
    idx = np.random.default_rng(42).choice(len(shap_values), size=n, replace=False)

    fig, ax = plt.subplots(figsize=(11, max(6, top_k * 0.52)))
    y_positions = {g: i for i, g in enumerate(reversed(top_groups))}

    for group in top_groups:
        if group not in group_shap:
            continue
        sv = group_shap[group][idx]
        y = y_positions[group]
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(sv))
        if group_is_categorical.get(group, False):
            # Categorical features: color by SHAP magnitude (direction already on x-axis)
            ax.scatter(
                sv,
                np.full_like(sv, y) + jitter,
                c=np.abs(sv),
                cmap="Purples",
                alpha=0.35,
                s=8,
                linewidths=0,
            )
        else:
            fv = group_feat[group][idx]
            # Normalize feature values to [0, 1] for coloring
            fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
            ax.scatter(
                sv,
                np.full_like(sv, y) + jitter,
                c=fv_norm,
                cmap="coolwarm",
                alpha=0.35,
                s=8,
                linewidths=0,
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(f"Figure {fig_no}. SHAP Beeswarm — Direction & Magnitude")
    # Colorbar legend for numeric features
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Numeric feature value (low → high)\nCategorical: shaded by |SHAP|", fontsize=9)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "shap_beeswarm", fig_dir)
    plt.show()


def plot_segment_heatmap(
    fig_dir: Path,
    fig_no: int = 15,
    policy: str = "max_f1",
) -> pd.DataFrame:
    """Load segment_metrics.csv and render a metric heatmap per dimension."""
    seg_path = project_root() / "reports" / "segment_metrics.csv"
    if not seg_path.exists():
        raise FileNotFoundError("segment_metrics.csv not found. Run `make train` first.")
    seg = pd.read_csv(seg_path)
    seg = seg[seg["policy"] == policy].copy()

    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    dimensions = seg["dimension"].unique().tolist()

    n_dims = len(dimensions)
    fig, axes = plt.subplots(1, n_dims, figsize=(5.5 * n_dims, 7), squeeze=False)

    for ax, dim in zip(axes[0], dimensions):
        sub = seg[seg["dimension"] == dim].copy()
        sub = sub.sort_values("roc_auc", ascending=False).reset_index(drop=True)
        labels = sub.apply(
            lambda r: f"{r['segment']}\n(n={r['n_rows']:,})" + (" ⚠" if not r["gated"] else ""),
            axis=1,
        )
        heat = sub[metrics].copy()
        heat.index = labels
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0.4,
            vmax=1.0,
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"{dim.replace('_', ' ').title()}", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(
        f"Figure {fig_no}. Segment Performance Heatmap ({policy} policy)\n"
        "⚠ = segment did not meet quality gates (treat metrics with caution)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "segment_performance_heatmap", fig_dir)
    plt.show()
    return seg


# ---------------------------------------------------------------------------
# Thesis baseline comparison helpers
# ---------------------------------------------------------------------------

_BASELINE_DISPLAY_NAMES: dict[str, str] = {
    "dummy_most_frequent": "Dummy (Most Frequent)",
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "decision_tree_depth5": "Decision Tree (depth≤5)",
    "champion": "LightGBM Champion",
}


def plot_baseline_comparison(
    fig_dir: Path,
    fig_no: str | int,
    thesis_dir: Path | None = None,
) -> pd.DataFrame:
    """Bar chart comparing ROC-AUC and PR-AUC across all baselines vs champion.

    Loads ``reports/thesis/baseline_comparison.json`` (written by thesis.py).
    Returns the comparison DataFrame for display.
    """
    from src.config import REPORTS_DIR

    if thesis_dir is None:
        thesis_dir = REPORTS_DIR / "thesis"

    baseline_path = thesis_dir / "baseline_comparison.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"{baseline_path} not found — run `make thesis` first.")

    raw: dict[str, Any] = json.loads(baseline_path.read_text(encoding="utf-8"))

    # Extract only the per-model metric dicts (skip paired-bootstrap sub-keys)
    metric_keys = [k for k in raw if not k.startswith("champion_vs_")]
    rows = []
    for key in metric_keys:
        val = raw[key]
        if isinstance(val, dict) and "roc_auc" in val:
            rows.append(
                {
                    "model": _BASELINE_DISPLAY_NAMES.get(key, key),
                    "ROC-AUC": float(val["roc_auc"]),
                    "PR-AUC": float(val["pr_auc"]),
                }
            )

    df = pd.DataFrame(rows)
    # Sort by PR-AUC descending so champion is on top
    df = df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["ROC-AUC", "PR-AUC"]):
        colors = ["#e15759" if m == "LightGBM Champion" else "#76b7b2" for m in df["model"]]
        bars = ax.barh(df["model"], df[metric], color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.5)")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Figure {fig_no}. Baseline Comparison — Test-Set Performance",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "baseline_comparison", fig_dir)
    plt.show()
    return df


def plot_thesis_dt(
    fig_dir: Path,
    fig_no: str | int,
    artifacts_dir: Path | None = None,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    max_display_depth: int = 4,
) -> Any:
    """Visualise the pruned Decision Tree baseline saved by thesis analysis.

    Loads ``artifacts/thesis_baseline_dt.pkl`` and renders the tree using
    sklearn's ``plot_tree``.  The tree is deliberately shallow (max_depth=5)
    so every decision path is readable.

    Parameters
    ----------
    fig_dir:
        Directory to save the figure.
    fig_no:
        Figure number prefix for the filename.
    artifacts_dir:
        Override the default ``ARTIFACTS_DIR`` location.
    feature_names:
        Post-encoding feature names for the x-axis labels.  If None, loads
        them from the preprocessor stored inside ``artifacts/best_model.pkl``.
    class_names:
        Class labels, default ``["Not Cancelled", "Cancelled"]``.
    max_display_depth:
        Depth limit for the plot (independent of the model's max_depth).
    """
    from sklearn.tree import plot_tree

    from src.config import ARTIFACTS_DIR as _DEFAULT_ARTIFACTS

    if artifacts_dir is None:
        artifacts_dir = _DEFAULT_ARTIFACTS

    dt_path = artifacts_dir / "thesis_baseline_dt.pkl"
    if not dt_path.exists():
        raise FileNotFoundError(f"{dt_path} not found — run `make thesis` first.")
    dt = joblib.load(dt_path)

    # Resolve feature names from the champion pipeline's preprocessor
    if feature_names is None:
        pipeline_path = artifacts_dir / "best_model.pkl"
        if pipeline_path.exists():
            pipeline = joblib.load(pipeline_path)
            prep = pipeline.named_steps.get("preprocessor")
            if prep is not None:
                try:
                    feature_names = list(prep.get_feature_names_out())
                except Exception:
                    feature_names = None

    if class_names is None:
        class_names = ["Not Cancelled", "Cancelled"]

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_display_depth,
        fontsize=7,
        ax=ax,
        impurity=False,
        precision=3,
    )
    ax.set_title(
        f"Figure {fig_no}. Decision Tree Baseline (max_depth=5, min_samples_leaf=50)\n"
        "Pruned for interpretability — each leaf shows predicted class and sample count.",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "decision_tree_baseline", fig_dir)
    plt.show()
    return dt


# ---------------------------------------------------------------------------
# Chi-Squared Independence Test
# ---------------------------------------------------------------------------


def chi_squared_independence(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target: str = TARGET_COL,
) -> pd.DataFrame:
    """Run chi-squared independence tests between categorical features and the target.

    Returns a DataFrame with columns: Feature, Chi2 Statistic, p-value, Degrees of Freedom,
    Cramér's V, and a Significant? flag (at alpha = 0.05).
    """
    from scipy.stats import chi2_contingency

    rows: list[dict[str, object]] = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[col], df[target])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        n = ct.values.sum()
        k = min(ct.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0
        rows.append(
            {
                "Feature": col,
                "Chi2 Statistic": round(float(chi2), 2),
                "p-value": float(p),
                "Degrees of Freedom": int(dof),
                "Cramér's V": round(cramers_v, 4),
                "Significant?": "Yes" if p < 0.05 else "No",
            }
        )
    return pd.DataFrame(rows).sort_values("Chi2 Statistic", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gradient-boosted tree deep-dive helpers (Notebook 02 — Section 2.10+)
# ---------------------------------------------------------------------------


def _get_model_and_preprocessor(
    ctx: dict[str, Any],
) -> tuple[Any, Any, list[str]]:
    """Extract the inner model, preprocessor, and feature names from context."""
    preprocessor = ctx["model_pipeline"].named_steps["preprocessor"]
    model = ctx["model_pipeline"].named_steps.get("model") or ctx["model_pipeline"].named_steps.get(
        "classifier"
    )
    if model is None:
        raise KeyError("Pipeline must have a step named 'model' or 'classifier'.")
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        ct = preprocessor.named_steps["encode"]
        _names: list[str] = []
        for _tname, _trans, _ in ct.transformers_:
            try:
                _out = list(_trans.get_feature_names_out())
            except AttributeError:
                _out = list(_trans.named_steps["onehot"].get_feature_names_out())
            _names.extend(f"{_tname}__{n}" for n in _out)
        feature_names = _names
    return model, preprocessor, feature_names


def champion_hyperparameters_table(ctx: dict[str, Any]) -> pd.DataFrame:
    """Return a styled table of the champion model's key hyperparameters."""
    model, _, _ = _get_model_and_preprocessor(ctx)
    model_type = type(model).__name__
    params = model.get_params()

    key_params = [
        ("Model Class", model_type),
        ("n_estimators", params.get("n_estimators")),
        ("max_depth", params.get("max_depth")),
        ("learning_rate", params.get("learning_rate")),
        ("subsample", params.get("subsample", params.get("colsample_bytree", "N/A"))),
        (
            "min_child_weight",
            params.get("min_child_weight", params.get("min_child_samples", "N/A")),
        ),
        ("reg_alpha (L1)", params.get("reg_alpha", "N/A")),
        ("reg_lambda (L2)", params.get("reg_lambda", "N/A")),
        ("num_leaves", params.get("num_leaves", "N/A")),
        ("random_state", params.get("random_state")),
    ]
    rows = [{"Parameter": k, "Value": v} for k, v in key_params if v is not None]
    return pd.DataFrame(rows)


def plot_split_feature_importance(
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 16,
    top_k: int = 20,
) -> pd.DataFrame:
    """Bar chart of built-in split-based feature importance (gain/weight)."""
    model, _, feature_names = _get_model_and_preprocessor(ctx)
    importances = model.feature_importances_

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    # Map to original feature groups
    fi_df["group"] = fi_df["feature"].apply(_feature_group_name)
    grouped = (
        fi_df.groupby("group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)  # type: ignore[call-overload, unused-ignore]
        .reset_index(drop=True)
    )

    top = grouped.head(top_k).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5, top_k * 0.38)))
    bars = ax.barh(
        top["group"],
        top["importance"],
        color="#4e79a7",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    for bar, val in zip(bars, top["importance"]):
        ax.text(
            val + grouped["importance"].max() * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )
    model_name = type(model).__name__
    ax.set_xlabel("Feature Importance (split gain)")
    ax.set_title(f"Figure {fig_no}. {model_name} Built-in Feature Importance — Top {top_k}")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "split_feature_importance", fig_dir)
    plt.show()
    return grouped


def plot_learning_dynamics(
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 17,
    n_checkpoints: int = 10,
) -> pd.DataFrame:
    """Train incremental models to show how performance improves with more trees."""
    from sklearn.metrics import log_loss

    model, preprocessor, _ = _get_model_and_preprocessor(ctx)
    model_type = type(model).__name__
    params = model.get_params()

    max_trees = params.get("n_estimators", 100)
    checkpoints = sorted(
        set([1, 5, 10] + list(np.linspace(10, max_trees, n_checkpoints, dtype=int)))
    )

    X_val = ctx["val_df"][BOOKING_TIME_FEATURES]
    y_val = ctx["val_df"][TARGET_COL].astype(int).to_numpy()
    X_val_t = preprocessor.transform(X_val)

    X_train = ctx["train_df"][BOOKING_TIME_FEATURES]
    y_train = ctx["train_df"][TARGET_COL].astype(int).to_numpy()
    X_train_t = preprocessor.transform(X_train)

    rows: list[dict[str, Any]] = []
    for n_est in checkpoints:
        clone_params = {k: v for k, v in params.items() if k != "n_estimators"}
        clone_params["n_estimators"] = int(n_est)
        # Suppress verbose output
        clone_params["verbose"] = 0
        if "verbosity" in clone_params:
            clone_params["verbosity"] = 0
        try:
            m = type(model)(**clone_params)
            m.fit(X_train_t, y_train)
            p_val = m.predict_proba(X_val_t)[:, 1]
            p_train = m.predict_proba(X_train_t)[:, 1]
            rows.append(
                {
                    "n_estimators": int(n_est),
                    "val_roc_auc": float(roc_auc_score(y_val, p_val)),
                    "val_pr_auc": float(average_precision_score(y_val, p_val)),
                    "val_log_loss": float(log_loss(y_val, p_val)),
                    "train_roc_auc": float(roc_auc_score(y_train, p_train)),
                    "train_log_loss": float(log_loss(y_train, p_train)),
                }
            )
        except (ValueError, RuntimeError, MemoryError) as exc:
            logger.debug("learning_curve_skip n_estimators=%d error=%s", int(n_est), exc)
            continue

    curve_df = pd.DataFrame(rows)
    if curve_df.empty:
        return curve_df

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.plot(
        curve_df["n_estimators"],
        curve_df["val_roc_auc"],
        "o-",
        color=PALETTE["roc"],
        linewidth=2.2,
        label="Validation",
    )
    ax.plot(
        curve_df["n_estimators"],
        curve_df["train_roc_auc"],
        "s--",
        color=PALETTE["roc"],
        alpha=0.5,
        linewidth=1.5,
        label="Train",
    )
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Figure {fig_no}A. ROC-AUC vs Trees")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(
        curve_df["n_estimators"],
        curve_df["val_pr_auc"],
        "o-",
        color=PALETTE["pr"],
        linewidth=2.2,
        label="Validation",
    )
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"Figure {fig_no}B. PR-AUC vs Trees")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(
        curve_df["n_estimators"],
        curve_df["val_log_loss"],
        "o-",
        color=PALETTE["f1"],
        linewidth=2.2,
        label="Validation",
    )
    ax.plot(
        curve_df["n_estimators"],
        curve_df["train_log_loss"],
        "s--",
        color=PALETTE["f1"],
        alpha=0.5,
        linewidth=1.5,
        label="Train",
    )
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Log Loss")
    ax.set_title(f"Figure {fig_no}C. Log Loss vs Trees")
    ax.legend(frameon=False)

    fig.suptitle(
        f"{model_type} Learning Dynamics — Performance vs Ensemble Size",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "learning_dynamics", fig_dir)
    plt.show()
    return curve_df


def plot_prediction_distribution_by_class(
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 18,
) -> None:
    """KDE + rug plot showing predicted probability distribution per true class."""
    y_prob = ctx["y_prob"]
    y_true = ctx["y_test_np"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: overlapping KDE
    ax = axes[0]
    for label, color, name in [(0, "#1f77b4", "Kept Bookings"), (1, "#d62728", "Cancelled")]:
        mask = y_true == label
        sns.kdeplot(
            y_prob[mask], ax=ax, color=color, fill=True, alpha=0.3, linewidth=2.2, label=name
        )
    ax.axvline(
        ctx["threshold_max_f1"],
        color="black",
        linestyle="-.",
        linewidth=1.5,
        label=f"Max-F1 @ {ctx['threshold_max_f1']:.2f}",
    )
    ax.axvline(
        ctx["threshold_high_precision"],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"High-Prec @ {ctx['threshold_high_precision']:.2f}",
    )
    ax.set_xlabel("Predicted Probability of Cancellation")
    ax.set_ylabel("Density")
    ax.set_title(f"Figure {fig_no}A. Prediction Distribution by True Class")
    ax.legend(frameon=False, fontsize=9)

    # Panel B: cumulative distribution
    ax = axes[1]
    for label, color, name in [(0, "#1f77b4", "Kept Bookings"), (1, "#d62728", "Cancelled")]:
        mask = y_true == label
        sorted_probs = np.sort(y_prob[mask])
        cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        ax.plot(sorted_probs, cdf, color=color, linewidth=2.2, label=name)
    ax.axvline(ctx["threshold_max_f1"], color="black", linestyle="-.", linewidth=1.5)
    ax.set_xlabel("Predicted Probability of Cancellation")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title(f"Figure {fig_no}B. Cumulative Distribution (Separation Quality)")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "prediction_distribution_by_class", fig_dir)
    plt.show()


def plot_tree_depth_analysis(
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 19,
) -> pd.DataFrame | None:
    """Visualise tree structure statistics from the gradient-boosted ensemble."""
    model, _, _ = _get_model_and_preprocessor(ctx)
    model_type = type(model).__name__

    # XGBoost: extract from booster dump
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        dump = booster.get_dump(with_stats=True)
        n_trees = len(dump)
        depths: list[int] = []
        leaves_per_tree: list[int] = []
        for tree_str in dump:
            tree_depths = [line.count("\t") for line in tree_str.strip().split("\n")]
            depths.append(max(tree_depths) if tree_depths else 0)
            leaves_per_tree.append(
                sum(1 for line in tree_str.strip().split("\n") if "leaf" in line)
            )
        summary = {
            "n_trees": n_trees,
            "mean_depth": float(np.mean(depths)),
            "max_depth": int(np.max(depths)),
            "mean_leaves": float(np.mean(leaves_per_tree)),
            "total_leaves": int(np.sum(leaves_per_tree)),
        }
    # LightGBM: extract from booster
    elif hasattr(model, "booster_"):
        booster = model.booster_
        model_dump = booster.dump_model()
        tree_info = model_dump.get("tree_info", [])
        n_trees = len(tree_info)
        depths = [t.get("max_depth", 0) for t in tree_info]
        leaves_per_tree = [t.get("num_leaves", 0) for t in tree_info]
        summary = {
            "n_trees": n_trees,
            "mean_depth": float(np.mean(depths)) if depths else 0,
            "max_depth": int(np.max(depths)) if depths else 0,
            "mean_leaves": float(np.mean(leaves_per_tree)) if leaves_per_tree else 0,
            "total_leaves": int(np.sum(leaves_per_tree)) if leaves_per_tree else 0,
        }
    else:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.hist(
        depths,
        bins=range(0, max(depths) + 2),
        color=PALETTE["champion"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(
        np.mean(depths),
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {np.mean(depths):.1f}",
    )
    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Number of Trees")
    ax.set_title(f"Figure {fig_no}A. Distribution of Tree Depths")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(
        leaves_per_tree, bins=20, color=PALETTE["pr"], alpha=0.8, edgecolor="black", linewidth=0.5
    )
    ax.axvline(
        np.mean(leaves_per_tree),
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {np.mean(leaves_per_tree):.1f}",
    )
    ax.set_xlabel("Leaves per Tree")
    ax.set_ylabel("Number of Trees")
    ax.set_title(f"Figure {fig_no}B. Distribution of Leaves per Tree")
    ax.legend(frameon=False)

    fig.suptitle(
        f"{model_type}: {n_trees} Trees, Mean Depth {np.mean(depths):.1f}, "
        f"Total Leaves {np.sum(leaves_per_tree):,}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "tree_depth_analysis", fig_dir)
    plt.show()
    return pd.DataFrame([summary])


def plot_gain_vs_cover(
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int = 20,
    top_k: int = 15,
) -> pd.DataFrame | None:
    """Compare feature importance by gain vs cover (XGBoost) or split vs gain (LightGBM)."""
    model, _, feature_names = _get_model_and_preprocessor(ctx)

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        cover_scores = booster.get_score(importance_type="cover")
        weight_scores = booster.get_score(importance_type="weight")

        # Map booster feature names (f0, f1, ...) to real names
        fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
        rows = []
        for fkey in gain_scores:
            real_name = fmap.get(fkey, fkey)
            group = _feature_group_name(real_name)
            rows.append(
                {
                    "feature": real_name,
                    "group": group,
                    "gain": gain_scores.get(fkey, 0),
                    "cover": cover_scores.get(fkey, 0),
                    "weight": weight_scores.get(fkey, 0),
                }
            )
        df = pd.DataFrame(rows)
        # Aggregate by group
        grouped = df.groupby("group", as_index=False)[["gain", "cover", "weight"]].sum()
        grouped = grouped.sort_values("gain", ascending=False).reset_index(drop=True)
        metric_a, metric_b = "gain", "cover"
        label_a, label_b = "Total Gain", "Total Cover"
    elif hasattr(model, "booster_"):
        booster = model.booster_
        fi = booster.feature_importance(importance_type="gain")
        fi_split = booster.feature_importance(importance_type="split")
        rows = []
        for i, name in enumerate(feature_names):
            group = _feature_group_name(name)
            rows.append({"feature": name, "group": group, "gain": fi[i], "split": fi_split[i]})
        df = pd.DataFrame(rows)
        grouped = df.groupby("group", as_index=False)[["gain", "split"]].sum()
        grouped = grouped.sort_values("gain", ascending=False).reset_index(drop=True)
        metric_a, metric_b = "gain", "split"
        label_a, label_b = "Total Gain", "Split Count"
    else:
        return None

    top = grouped.head(top_k)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, top_k * 0.38)), sharey=True)

    top_sorted = top.sort_values(metric_a, ascending=True)
    ax = axes[0]
    ax.barh(top_sorted["group"], top_sorted[metric_a], color=PALETTE["champion"], alpha=0.85)
    ax.set_xlabel(label_a)
    ax.set_title(f"Figure {fig_no}A. Feature Importance by {label_a}")

    top_sorted_b = top.sort_values(metric_b, ascending=True)
    ax = axes[1]
    ax.barh(top_sorted_b["group"], top_sorted_b[metric_b], color=PALETTE["pr"], alpha=0.85)
    ax.set_xlabel(label_b)
    ax.set_title(f"Figure {fig_no}B. Feature Importance by {label_b}")

    model_name = type(model).__name__
    fig.suptitle(
        f"{model_name}: {label_a} vs {label_b} — Top {top_k} Feature Groups",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "gain_vs_cover", fig_dir)
    plt.show()
    return grouped


# ---------------------------------------------------------------------------
# Simplified visualisation helpers (added for thesis-panel readability)
# ---------------------------------------------------------------------------


def plot_model_grouped_bar(ctx: dict[str, Any], fig_dir: Path, fig_no: int | str = 2) -> None:
    """Grouped horizontal bar chart — PR-AUC and ROC-AUC per model family.

    Replaces the dumbbell chart with a universally understood chart type.
    """
    _sel = ctx.get("selection_summary") or ctx.get("model_selection", {})
    candidates = pd.DataFrame(_sel.get("candidates", [])).copy()
    if candidates.empty or "rolling_roc_auc_mean" not in candidates.columns:
        print("plot_model_grouped_bar: no candidate data available — skipping figure.")
        return
    candidates = candidates.sort_values("rolling_pr_auc_mean", ascending=True).reset_index(
        drop=True
    )
    candidates["label"] = candidates["model_family"].str.replace("_", " ").str.title()

    y = np.arange(len(candidates))
    bar_h = 0.35
    fig, ax = plt.subplots(figsize=(11, max(4, len(candidates) * 1.5)))
    bars_pr = ax.barh(
        y - bar_h / 2,
        candidates["rolling_pr_auc_mean"],
        height=bar_h,
        color=PALETTE["pr"],
        label="PR-AUC",
        alpha=0.85,
    )
    bars_roc = ax.barh(
        y + bar_h / 2,
        candidates["rolling_roc_auc_mean"],
        height=bar_h,
        color=PALETTE["roc"],
        label="ROC-AUC",
        alpha=0.85,
    )
    ax.bar_label(bars_pr, fmt="%.3f", padding=4, fontsize=9)
    ax.bar_label(bars_roc, fmt="%.3f", padding=4, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(candidates["label"])
    ax.set_xlabel("Rolling Mean Score")
    ax.set_title(
        f"Figure {fig_no}. Rolling-Origin Model Comparison: PR-AUC vs ROC-AUC",
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "grouped_bar_model_selection", fig_dir)
    plt.show()


def plot_pr_curve_plain(ctx: dict[str, Any], fig_dir: Path, fig_no: int | str = 3) -> None:
    """Plain precision-recall curve without iso-F1 contour lines.

    Keeps the two operating-point markers (max_f1 and high_precision) but removes
    the grey iso-F1 contour overlay that confuses non-ML audiences.
    """
    y_true = ctx["y_test_np"]
    y_prob = ctx["y_prob"]
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob))
    prevalence = float(np.mean(y_true))

    def _pr_at(threshold: float) -> tuple[float, float]:
        pred = (y_prob >= threshold).astype(int)
        return float(recall_score(y_true, pred, zero_division=0)), float(
            precision_score(y_true, pred, zero_division=0)
        )

    recall_f1, precision_f1 = _pr_at(ctx["threshold_max_f1"])
    recall_hp, precision_hp = _pr_at(ctx["threshold_high_precision"])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(rec_curve, prec_curve, color=PALETTE["pr"], linewidth=2.6)
    ax.hlines(prevalence, 0, 1, color="gray", linestyle="--", linewidth=1.5, label="Baseline")
    ax.scatter(recall_f1, precision_f1, color=PALETTE["f1"], s=100, zorder=4, edgecolor="black")
    ax.scatter(
        recall_hp, precision_hp, color=PALETTE["precision"], s=100, zorder=4, edgecolor="black"
    )
    ax.annotate(
        f"Max-F1 @ {ctx['threshold_max_f1']:.2f}",
        xy=(recall_f1, precision_f1),
        xytext=(recall_f1 + 0.05, precision_f1 + 0.05),
        fontsize=10,
        color=PALETTE["f1"],
        arrowprops={"arrowstyle": "->", "color": PALETTE["f1"]},
    )
    ax.annotate(
        f"High-Precision @ {ctx['threshold_high_precision']:.2f}",
        xy=(recall_hp, precision_hp),
        xytext=(recall_hp + 0.05, precision_hp - 0.08),
        fontsize=10,
        color=PALETTE["precision"],
        arrowprops={"arrowstyle": "->", "color": PALETTE["precision"]},
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Figure {fig_no}. Precision-Recall Curve (PR-AUC = {pr_auc:.3f})", fontweight="bold"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "pr_curve_plain", fig_dir)
    plt.show()


def plot_top_correlations_bar(
    df: pd.DataFrame,
    target: str,
    fig_dir: Path,
    fig_no: int | str = 5,
    top_k: int = 10,
) -> pd.DataFrame:
    """Horizontal bar chart of features most correlated with the target.

    Replaces the dense 16x16 annotated heatmap with a clear ranked bar chart.
    Returns the correlation series.
    """
    numeric_df = df.select_dtypes(include="number")
    if target not in numeric_df.columns:
        print(f"plot_top_correlations_bar: target '{target}' not in numeric columns — skipping.")
        return pd.DataFrame()
    corr = numeric_df.corr()[target].drop(target, errors="ignore").dropna()
    top = corr.abs().nlargest(top_k)
    top_corr = corr[top.index].sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e15759" if v > 0 else "#4e79a7" for v in top_corr.values]
    bars = ax.barh(top_corr.index, top_corr.values.tolist(), color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="{:+.3f}", padding=4, fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Correlation with Cancellation")
    ax.set_title(
        f"Figure {fig_no}. Top {top_k} Features Correlated with Cancellation",
        fontweight="bold",
    )
    legend_handles = [
        Line2D([0], [0], color="#e15759", lw=8, label="Positive (more → more cancellations)"),
        Line2D([0], [0], color="#4e79a7", lw=8, label="Negative (more → fewer cancellations)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "top_correlations_bar", fig_dir)
    plt.show()
    return top_corr.to_frame("correlation")


def plot_metric_forest(
    ci_df: pd.DataFrame,
    fig_dir: Path,
    fig_no: int | str = 6,
    *,
    point_col: str = "point_estimate",
    lower_col: str = "ci_lower",
    upper_col: str = "ci_upper",
    metric_col: str = "metric",
    title: str | None = None,
    stem: str = "bootstrap_ci_forest",
    x_label: str = "Metric value (95% bootstrap CI)",
    reference_line: float | None = None,
    sig_col: str | None = None,
) -> None:
    """Forest plot of point estimates with 95% CIs — one row per metric.

    Handles both absolute metrics (Round 1 use: bootstrap CI table at 0-1
    scale) and signed deltas (Round 2 use: paired challenger-vs-champion
    deltas which can be negative).  Auto-detects x-limits from the data
    so it does not clip negative values.

    Optional features:
        reference_line: draw a vertical line at this x (e.g. 0 for "no
            difference" when plotting deltas).
        sig_col: name of a boolean column.  Rows where this is True are
            drawn in red (significant); other rows stay blue.
    """
    df = ci_df.copy().reset_index(drop=True)
    if df.empty:
        print("plot_metric_forest: empty input — skipping.")
        return
    n = len(df)
    fig, ax = plt.subplots(figsize=(9, max(3.5, n * 0.55 + 1.2)))
    y_pos = np.arange(n)

    # Determine x-limits with padding so annotations don't get clipped
    all_lo = float(df[lower_col].astype(float).min())
    all_hi = float(df[upper_col].astype(float).max())
    span = max(all_hi - all_lo, 1e-9)
    pad_left = max(span * 0.10, 0.005)
    pad_right = max(span * 0.30, 0.05)  # extra right padding for annotations

    cap = 0.18
    for i, (_, row) in enumerate(df.iterrows()):
        lo, hi, pt = float(row[lower_col]), float(row[upper_col]), float(row[point_col])
        is_sig = bool(row[sig_col]) if sig_col and sig_col in df.columns else False
        ci_color = "#e15759" if is_sig else "#4e79a7"
        # CI bar + end caps
        ax.plot([lo, hi], [i, i], color=ci_color, linewidth=2.2, alpha=0.75)
        ax.plot([lo, lo], [i - cap, i + cap], color=ci_color, linewidth=2.2)
        ax.plot([hi, hi], [i - cap, i + cap], color=ci_color, linewidth=2.2)
        # Annotation to the right of the upper end
        sig_marker = "  *" if is_sig else ""
        ax.text(
            hi + pad_right * 0.08,
            i,
            f"{pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig_marker}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    # Point estimates as filled dots on top
    dot_colors = [
        "#9B1C1F"
        if (sig_col and sig_col in df.columns and bool(df[sig_col].iloc[i]))
        else "#1b3a5c"
        for i in range(n)
    ]
    ax.scatter(
        df[point_col].astype(float),
        y_pos,
        c=dot_colors,
        s=90,
        zorder=5,
        edgecolor="white",
        linewidth=1.3,
        label="Point estimate",
    )

    if reference_line is not None:
        ax.axvline(reference_line, color="#444444", linewidth=1.0, linestyle="--", alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[metric_col])
    ax.set_xlabel(x_label)
    ax.set_title(
        title or f"Figure {fig_no}. Bootstrap 95% Confidence Intervals",
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(all_lo - pad_left, all_hi + pad_right)
    ax.invert_yaxis()  # first row at the top
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, stem, fig_dir)
    plt.show()


def plot_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    fig_dir: Path,
    fig_no: int | str,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    lower_x_better: bool = True,
    higher_y_better: bool = True,
    title: str | None = None,
    stem: str = "pareto_frontier",
    highlight: str | None = None,
    annotate: bool = True,
) -> pd.DataFrame:
    """Scatter with Pareto-optimal points highlighted and connected by a line.

    Points are split into two layers:
      * Pareto-optimal (no other point dominates them on both axes) → dark dots
        connected with a dashed frontier line.
      * Dominated (some other point is better on both axes) → grey dots.

    If `highlight` matches a row's label_col value, that row gets a red star on
    top — useful for marking the champion.  Returns the frontier subset.
    """
    data = df.copy().reset_index(drop=True)
    if len(data) < 2:
        print(f"plot_pareto_frontier: need >=2 points, got {len(data)}; skipping.")
        return data

    x = data[x_col].astype(float).to_numpy()
    y = data[y_col].astype(float).to_numpy()

    # Compute Pareto frontier (O(n^2) but n is always small here)
    n = len(data)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = (x[j] <= x[i]) if lower_x_better else (x[j] >= x[i])
            y_better = (y[j] >= y[i]) if higher_y_better else (y[j] <= y[i])
            strict = (x[j] != x[i]) or (y[j] != y[i])
            x_strict = (x[j] < x[i]) if lower_x_better else (x[j] > x[i])
            y_strict = (y[j] > y[i]) if higher_y_better else (y[j] < y[i])
            if x_better and y_better and strict and (x_strict or y_strict):
                is_optimal[i] = False
                break

    frontier = data[is_optimal].sort_values(x_col, ascending=lower_x_better).reset_index(drop=True)
    dominated = data[~is_optimal]

    fig, ax = plt.subplots(figsize=(10, 6))

    if not dominated.empty:
        ax.scatter(
            dominated[x_col],
            dominated[y_col],
            s=85,
            color="#cccccc",
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
            label="Dominated",
        )
        if annotate:
            for _, row in dominated.iterrows():
                ax.annotate(
                    str(row[label_col]).replace("_", " ").title(),
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(8, 4),
                    fontsize=9,
                    color="#888888",
                )

    if not frontier.empty:
        ax.scatter(
            frontier[x_col],
            frontier[y_col],
            s=130,
            color="#1b3a5c",
            edgecolor="white",
            linewidth=1.8,
            zorder=5,
            label="Pareto-optimal",
        )
        if annotate:
            for _, row in frontier.iterrows():
                ax.annotate(
                    str(row[label_col]).replace("_", " ").title(),
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(8, 4),
                    fontsize=10,
                    fontweight="bold",
                )
        # Frontier line connecting the optimal points
        if len(frontier) >= 2:
            ax.plot(
                frontier[x_col],
                frontier[y_col],
                color="#4e79a7",
                linewidth=1.8,
                linestyle="--",
                alpha=0.6,
                zorder=4,
            )

    # Champion star
    if highlight is not None:
        hl = data[data[label_col].astype(str) == str(highlight)]
        if not hl.empty:
            ax.scatter(
                hl[x_col],
                hl[y_col],
                s=260,
                color="#e15759",
                marker="*",
                edgecolor="white",
                linewidth=1.8,
                zorder=6,
                label=f"Champion ({highlight})",
            )

    ax.set_xlabel(x_label or x_col, fontsize=11)
    ax.set_ylabel(y_label or y_col, fontsize=11)
    ax.set_title(
        title or f"Figure {fig_no}. Pareto Frontier: {x_col} vs {y_col}",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.92, fontsize=9)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, stem, fig_dir)
    plt.show()
    return frontier


def plot_parallel_coordinates(
    df: pd.DataFrame,
    value_cols: list[str],
    label_col: str,
    fig_dir: Path,
    fig_no: int | str,
    *,
    higher_better: list[bool] | None = None,
    highlight: str | None = None,
    title: str | None = None,
    stem: str = "parallel_coordinates",
    column_labels: list[str] | None = None,
) -> None:
    """Parallel-coordinates plot — one line per row across multiple normalised axes.

    Each axis is rescaled to [0, 1] where 1 is always "best" (flipping when the
    metric is one where lower is better, e.g. training time, business cost).
    Useful for visualising multi-criteria trade-offs across N models when no
    single composite score captures the picture.

    If `highlight` matches a row, that line is drawn in red on top of greyed
    others — directly answers "where does the champion win and lose?"
    """
    data = df.copy().reset_index(drop=True)
    n_cols = len(value_cols)
    if n_cols < 2:
        print("plot_parallel_coordinates: need >=2 value columns; skipping.")
        return
    if higher_better is None:
        higher_better = [True] * n_cols

    # Normalize each column to [0, 1] in the "better" direction
    normed = pd.DataFrame(index=data.index)
    for col, hib in zip(value_cols, higher_better):
        vals = data[col].astype(float)
        lo, hi = float(vals.min()), float(vals.max())
        if hi == lo:
            normed[col] = 0.5
        elif hib:
            normed[col] = (vals - lo) / (hi - lo)
        else:
            normed[col] = (hi - vals) / (hi - lo)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.6), 6))
    x_pos = np.arange(n_cols)

    # Distinct fallback colours for the no-highlight case so each model line is
    # visually separable in the legend. Used positionally; PALETTE is keyed by
    # metric/family name and can't be indexed by integer position.
    _LINE_CYCLE = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9da7",
    ]

    # Two-pass draw: greyed lines first, highlighted last
    for i, (_, row) in enumerate(data.iterrows()):
        label = str(row[label_col])
        is_hl = highlight is not None and label == str(highlight)
        if is_hl:
            continue  # draw last
        color = _LINE_CYCLE[i % len(_LINE_CYCLE)] if highlight is None else "#cccccc"
        ax.plot(
            x_pos,
            normed.iloc[i].to_numpy(),
            color=color,
            linewidth=1.4,
            alpha=0.75,
            zorder=2,
            marker="o",
            markersize=6,
            label=label.replace("_", " ").title() if highlight is None else None,
        )

    if highlight is not None:
        # Iterate the full data once and draw highlight matches on top. Using
        # enumerate gives a positional `i` for `normed.iloc[i]` rather than the
        # original index — important because `hl_rows` filtering would keep the
        # parent index, breaking the positional lookup if any caller passes a
        # non-default-indexed DataFrame.
        for i, (_, row) in enumerate(data.iterrows()):
            if str(row[label_col]) != str(highlight):
                continue
            ax.plot(
                x_pos,
                normed.iloc[i].to_numpy(),
                color="#e15759",
                linewidth=2.8,
                alpha=0.95,
                zorder=5,
                marker="o",
                markersize=9,
                label=f"{str(row[label_col]).replace('_', ' ').title()} (champion)",
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(column_labels or value_cols, rotation=20, ha="right")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Worst", "", "Mid", "", "Best"])
    ax.set_ylabel("Normalised rank per metric (1.0 = best across models)")
    ax.set_ylim(-0.05, 1.08)

    # Light vertical lines at each axis
    for xp in x_pos:
        ax.axvline(float(xp), color="#888888", linewidth=0.6, alpha=0.4)

    ax.set_title(
        title or f"Figure {fig_no}. Multi-Criteria Model Comparison",
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.92)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, stem, fig_dir)
    plt.show()


def plot_significance_bar(
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int | str = "9B",
    metric: str = "pr_auc",
) -> None:
    """Simple bar chart showing champion advantage over each challenger.

    Replaces the 3-panel forest plot with a single, easy-to-read bar chart.
    Bars are colored by significance. Error bars show 95% CI.
    """
    tables = ctx.get("benchmark_tables")
    if not tables:
        print("plot_significance_bar: benchmark tables not found — skipping.")
        return
    sig = tables["14_paired_significance_vs_champion"].copy()
    subset = sig[sig["metric"] == metric].sort_values("observed_delta", ascending=True)
    if subset.empty:
        print(f"plot_significance_bar: no data for metric '{metric}' — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(subset) * 1.2)))
    y = np.arange(len(subset))
    colors = ["#e15759" if s else "#bdbdbd" for s in subset["significant_at_05"]]
    xerr_low = subset["observed_delta"] - subset["delta_ci_lower"]
    xerr_high = subset["delta_ci_upper"] - subset["observed_delta"]
    ax.barh(
        y,
        subset["observed_delta"],
        xerr=[xerr_low.values, xerr_high.values],
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(subset["challenger_model"].str.replace("_", " ").str.title())
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    nice = metric.replace("_", "-").upper()
    ax.set_xlabel(f"Champion Advantage ({nice})")
    ax.set_title(
        f"Figure {fig_no}. Champion vs Challengers — {nice} (Paired Bootstrap, n=2000)",
        fontweight="bold",
    )
    for i, (_, row) in enumerate(subset.iterrows()):
        label = "sig." if row["significant_at_05"] else f"p={row['p_value_two_sided']:.2f}"
        ax.text(
            row["observed_delta"] + xerr_high.iloc[i] + 0.001,
            i,
            label,
            va="center",
            fontsize=9,
            fontstyle="italic",
        )
    legend_handles = [
        Line2D([0], [0], color="#e15759", lw=8, label="Significant (p < 0.05)"),
        Line2D([0], [0], color="#bdbdbd", lw=8, label="Not significant"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, f"significance_bar_{metric}", fig_dir)
    plt.show()


def plot_rolling_stability_top3(
    ctx: dict[str, Any],
    fig_dir: Path,
    fig_no: int | str = "9G",
    top_n: int = 3,
) -> pd.DataFrame:
    """Rolling-origin stability plot showing only the top N models.

    Replaces the cluttered 6-line version with a cleaner view focused on
    the most competitive models.
    """
    tables = ctx.get("benchmark_tables")
    if not tables:
        print("plot_rolling_stability_top3: benchmark tables not found — skipping.")
        return pd.DataFrame()

    rankings = tables.get("16_rankings")
    if rankings is not None:
        top_models = rankings.head(top_n)["model"].tolist()
    else:
        top_models = ["lightgbm", "gradient_boosting", "xgboost"][:top_n]

    rolling_key = "10_rolling_origin_fold_metrics"
    if rolling_key in tables:
        rolling = tables[rolling_key].copy()
    else:
        from src.eval.notebook_utils import project_root

        rolling = pd.read_csv(
            project_root() / "reports" / "benchmarks" / "10_rolling_origin_fold_metrics.csv"
        )

    rolling = rolling[rolling["model"].isin(top_models)]
    model_colors = {
        "lightgbm": "#e15759",
        "gradient_boosting": "#4e79a7",
        "xgboost": "#f28e2b",
        "random_forest": "#59a14f",
        "logistic_regression": "#b07aa1",
        "decision_tree": "#9c755f",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for metric, ax in zip(["roc_auc", "pr_auc"], axes):
        for model_name in rolling["model"].unique():
            subset = rolling[rolling["model"] == model_name]
            color = model_colors.get(model_name, "gray")
            ax.plot(
                subset["cutoff_frac"],
                subset[metric],
                marker="o",
                linewidth=2.5,
                markersize=9,
                label=model_name.replace("_", " ").title(),
                color=color,
            )
        nice = metric.replace("_", "-").upper()
        ax.set_title(f"{nice} Across Rolling-Origin Folds", fontweight="bold")
        ax.set_xlabel("Training Cutoff Fraction")
        ax.set_ylabel(nice)
        ax.legend(fontsize=10, loc="lower left")
        ax.set_xticks([0.6, 0.7, 0.8])

    fig.suptitle(
        f"Figure {fig_no}. Temporal Stability — Top {top_n} Models",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "rolling_origin_stability_top", fig_dir)
    plt.show()
    return rolling


# ---------------------------------------------------------------------------
# Sensitivity analysis helpers (Notebook 10)
# ---------------------------------------------------------------------------


def load_sensitivity_context() -> dict[str, Any]:
    """Load context for sensitivity analysis — extends main context with val predictions."""
    ctx = load_main_context()
    root = ctx["root"]
    feature_cols = BOOKING_TIME_FEATURES.copy()
    feature_columns_path = root / "artifacts" / "feature_columns.json"
    if feature_columns_path.exists():
        feature_cols = load_json(feature_columns_path).get("features", feature_cols)

    val_df = ctx["val_df"]
    train_df = ctx["train_df"]
    for col in feature_cols:
        if col not in val_df.columns:
            val_df[col] = None
        if col not in train_df.columns:
            train_df[col] = None

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL].astype(int)
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)

    model = ctx["model_pipeline"]
    calibrator = ctx["calibrator"]
    val_probs_raw = model.predict_proba(X_val)[:, 1]
    val_probs = np.clip(calibrator.predict(val_probs_raw), 0.0, 1.0)

    ctx.update(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "val_probs": val_probs,
            "val_probs_calibrated": val_probs,
            "val_labels": y_val.to_numpy(),
            "feature_cols": feature_cols,
        }
    )
    return ctx


def cost_sensitivity_sweep(
    ctx: dict[str, Any],
    fp_costs: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep FP intervention costs and return optimal cost-sensitive thresholds.

    For each FP cost, re-runs the cost threshold sweep on the validation set
    and records the optimal threshold and total cost.
    """
    from src.utils.thresholds import cost_threshold_sweep, select_min_cost_threshold

    if fp_costs is None:
        fp_costs = [1.0, 5.0, 10.0, 15.0, 25.0, 50.0, 75.0, 100.0]

    _vp = ctx.get("val_probs_calibrated")
    if _vp is None:
        _vp = ctx.get("val_probs", [])
    val_probs = np.array(_vp)
    val_labels = np.array(ctx.get("val_labels", []))

    # Compute per-sample FN cost (revenue at risk)
    val_df = ctx.get("val_df")
    if val_df is not None and "revenue_at_risk" in val_df.columns:
        fn_cost = val_df["revenue_at_risk"].values
    else:
        fn_cost = np.full(len(val_labels), 100.0)

    rows = []
    for fp_cost in fp_costs:
        sweep_df = cost_threshold_sweep(val_labels, val_probs, fn_cost, fp_cost=fp_cost)
        best = select_min_cost_threshold(sweep_df)
        rows.append(
            {
                "FP Cost": fp_cost,
                "Optimal Threshold": best["threshold"],
                "Total Cost": best["total_cost"],
                "FP Count": best["fp_count"],
                "FN Count": best["fn_count"],
            }
        )
    return pd.DataFrame(rows)


def plot_cost_sensitivity(
    sweep_df: pd.DataFrame,
    fig_dir: Path,
    fig_no: int | str = 101,
) -> None:
    """Plot how the optimal threshold shifts as FP intervention cost changes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sweep_df["FP Cost"], sweep_df["Optimal Threshold"], "o-", color="#4e79a7", lw=2)
    ax1.set_xlabel("FP Intervention Cost (EUR)")
    ax1.set_ylabel("Optimal Threshold")
    ax1.set_title("(a) How FP Cost Shifts the Optimal Threshold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(sweep_df["FP Cost"], sweep_df["Total Cost"], "s-", color="#e15759", lw=2)
    ax2.set_xlabel("FP Intervention Cost (EUR)")
    ax2.set_ylabel("Total Misclassification Cost (EUR)")
    ax2.set_title("(b) Total Cost at Optimal Threshold")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle(
        f"Figure {fig_no}. Cost Sensitivity Analysis",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "cost_sensitivity", fig_dir)
    plt.show()


def dataset_size_sensitivity(
    ctx: dict[str, Any],
    fractions: list[float] | None = None,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Train on progressively smaller subsets and measure holdout performance.

    Uses the pipeline's preprocessor to transform raw features into numeric
    arrays before fitting standalone classifiers on each subset.

    Returns a DataFrame with columns: Fraction, Rows, ROC-AUC (mean/std),
    PR-AUC (mean/std), F1 (mean/std).
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    X_train = ctx.get("X_train")
    y_train = ctx.get("y_train")
    X_val = ctx.get("X_val")
    y_val = ctx.get("y_val")

    if X_train is None or y_train is None or X_val is None or y_val is None:
        return pd.DataFrame()

    # Extract the preprocessor from the loaded pipeline to transform raw features
    pipeline = ctx.get("model_pipeline")
    preprocessor = None
    if pipeline is not None and hasattr(pipeline, "named_steps"):
        step_names = list(pipeline.named_steps.keys())
        preprocessor = pipeline.named_steps.get(step_names[0])

    # Pre-transform the full training and validation sets once
    if preprocessor is not None:
        X_train_enc = preprocessor.transform(X_train)
        X_val_enc = preprocessor.transform(X_val)
    else:
        X_train_enc = X_train.values if hasattr(X_train, "values") else X_train
        X_val_enc = X_val.values if hasattr(X_val, "values") else X_val

    rows = []
    for frac in fractions:
        roc_aucs, pr_aucs, f1s = [], [], []
        n_rows = int(len(X_train_enc) * frac)
        for seed in range(n_repeats):
            rng = np.random.RandomState(42 + seed)
            idx = rng.choice(len(X_train_enc), size=n_rows, replace=False)
            X_sub = X_train_enc[idx]
            y_sub = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]

            try:
                from src.models.train import is_lightgbm_available

                if is_lightgbm_available():
                    import lightgbm as lgb

                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42 + seed,
                        verbosity=-1,
                    )
                else:
                    from sklearn.ensemble import GradientBoostingClassifier

                    model = GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42 + seed,
                    )
                model.fit(X_sub, y_sub)
                probs = model.predict_proba(X_val_enc)[:, 1]
                preds = (probs >= 0.5).astype(int)
                roc_aucs.append(roc_auc_score(y_val, probs))
                pr_aucs.append(average_precision_score(y_val, probs))
                f1s.append(f1_score(y_val, preds))
            except (ValueError, RuntimeError, MemoryError) as exc:
                logger.debug("data_hunger_skip frac=%s seed=%d error=%s", frac, seed, exc)
                continue

        if roc_aucs:
            rows.append(
                {
                    "Fraction": frac,
                    "Rows": n_rows,
                    "ROC-AUC Mean": np.mean(roc_aucs),
                    "ROC-AUC Std": np.std(roc_aucs),
                    "PR-AUC Mean": np.mean(pr_aucs),
                    "PR-AUC Std": np.std(pr_aucs),
                    "F1 Mean": np.mean(f1s),
                    "F1 Std": np.std(f1s),
                }
            )
    return pd.DataFrame(rows)


def plot_dataset_size_sensitivity(
    size_df: pd.DataFrame,
    fig_dir: Path,
    fig_no: int | str = 102,
) -> None:
    """Plot how model performance degrades with less training data."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("ROC-AUC", "#4e79a7"), ("PR-AUC", "#e15759"), ("F1", "#59a14f")]

    for ax, (metric, color) in zip(axes, metrics):
        mean_col = f"{metric} Mean"
        std_col = f"{metric} Std"
        ax.errorbar(
            size_df["Rows"],
            size_df[mean_col],
            yerr=size_df[std_col],
            fmt="o-",
            color=color,
            capsize=4,
            lw=2,
        )
        ax.set_xlabel("Training Rows")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    fig.suptitle(
        f"Figure {fig_no}. Dataset Size Sensitivity",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "dataset_size_sensitivity", fig_dir)
    plt.show()


def threshold_policy_comparison(
    ctx: dict[str, Any],
) -> pd.DataFrame:
    """Compare the three threshold policies side by side on test set."""
    test_metrics = {}
    for policy in ["max_f1", "high_precision", "cost_sensitive"]:
        m = ctx.get("metrics", {}).get(policy, {})
        if m:
            test_metrics[policy] = {
                "Threshold": ctx.get("thresholds", {}).get(policy, {}).get("threshold", "N/A"),
                "Precision": m.get("precision", 0),
                "Recall": m.get("recall", 0),
                "F1": m.get("f1", 0),
                "Balanced Accuracy": m.get("balanced_accuracy", 0),
            }
    if not test_metrics:
        return pd.DataFrame()
    df = pd.DataFrame(test_metrics).T
    df.index.name = "Policy"
    return df.reset_index()


def plot_threshold_policy_comparison(
    policy_df: pd.DataFrame,
    fig_dir: Path,
    fig_no: int | str = 103,
) -> None:
    """Grouped bar chart comparing metrics across threshold policies."""
    if policy_df.empty:
        return

    metrics = ["Precision", "Recall", "F1", "Balanced Accuracy"]
    x = np.arange(len(policy_df))
    width = 0.18
    colors = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric in policy_df.columns:
            ax.bar(x + i * width, policy_df[metric], width, label=metric, color=color)

    ax.set_xlabel("Threshold Policy")
    ax.set_ylabel("Score")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(policy_df["Policy"])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Figure {fig_no}. Threshold Policy Trade-offs",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_thesis_figure(fig, fig_no, "threshold_policy_comparison", fig_dir)
    plt.show()
