"""Generate the Model Deployment Framework diagram for Chapter IV.

Produces a publication-grade architecture diagram showing the full live-
serving pipeline: how a single booking enters at the front desk, runs
through the model, gets logged to SQLite, materialises as a CSV, and
reaches the Power BI dashboard. Includes the side-channel drift loop
that triggers retraining.

Output: ``reports/figures/thesis/fig_deployment_framework.png`` (+ PDF).

Run once after any architectural change. The diagram intentionally uses
only matplotlib primitives (no graphviz dependency) so the figure
regenerates on any laptop with the project's existing requirements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures" / "thesis"

# Project palette (matches the Power BI dashboard guide colors).
PRIMARY = "#1F4E79"
DANGER = "#A6192E"
SAFE = "#107C41"
ACCENT = "#F5A623"
NEUTRAL = "#3B3B3B"
SURFACE = "#F4F4F4"


def _box(ax, xy, w, h, text, *, fc=SURFACE, ec=PRIMARY, fontsize=9.5, fontweight="normal"):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (xy[0], xy[1]),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=NEUTRAL,
        zorder=3,
        wrap=True,
    )


def _arrow(ax, xy_from, xy_to, *, color=NEUTRAL, label="", style="->", linewidth=1.4):
    """Draw a labeled arrow between two anchor points."""
    arr = FancyArrowPatch(
        xy_from,
        xy_to,
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        mx = (xy_from[0] + xy_to[0]) / 2
        my = (xy_from[1] + xy_to[1]) / 2
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9),
            zorder=4,
        )


def build_figure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        }
    )
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── Top row: user touch points ─────────────────────────────────
    _box(
        ax, (0.4, 8.4), 2.4, 1.0, "Front-desk staff\n(via PMS UI)", fc="#E6EEF7", fontweight="bold"
    )
    _box(
        ax,
        (3.4, 8.4),
        2.4,
        1.0,
        "External integrator\n(POST /predict)",
        fc="#E6EEF7",
        fontweight="bold",
    )

    # ── FastAPI box (large container) ─────────────────────────────
    _box(ax, (0.4, 5.2), 5.4, 2.7, "", fc="#FFFFFF", ec=PRIMARY)
    ax.text(
        0.6, 7.6, "FastAPI Server (localhost:8000)", fontsize=11, fontweight="bold", color=PRIMARY
    )
    # Endpoints sub-row
    _box(ax, (0.6, 6.8), 1.5, 0.55, "/predict", fc="#DFEAF5", fontsize=9)
    _box(ax, (2.25, 6.8), 1.5, 0.55, "/model-info", fc="#DFEAF5", fontsize=9)
    _box(ax, (3.9, 6.8), 1.5, 0.55, "/healthz", fc="#DFEAF5", fontsize=9)
    # Gradio UI mounted inside FastAPI
    _box(
        ax,
        (0.6, 6.1),
        5.0,
        0.55,
        "Gradio UI (mounted at /ui) — same model, web form",
        fc="#FFF4DE",
        fontsize=9,
    )
    # Inference pipeline (nested)
    _box(
        ax,
        (0.6, 5.32),
        5.0,
        0.65,
        "Inference: Pydantic validate → feature engineer → LightGBM →\n"
        "isotonic calibrate → threshold resolve → risk tier → SHAP top-5 → ADR",
        fc=SURFACE,
        fontsize=8.2,
    )

    # ── Artifact layer (left side) ────────────────────────────────
    _box(ax, (6.4, 6.8), 3.0, 0.65, "artifacts/best_model.pkl", fc="#EDF7EE", ec=SAFE, fontsize=9)
    _box(
        ax,
        (6.4, 6.05),
        3.0,
        0.65,
        "artifacts/probability_calibrator.pkl",
        fc="#EDF7EE",
        ec=SAFE,
        fontsize=9,
    )
    _box(ax, (6.4, 5.3), 3.0, 0.65, "artifacts/thresholds.json", fc="#EDF7EE", ec=SAFE, fontsize=9)

    # ── BackgroundTask + persistence row ─────────────────────────
    _box(
        ax,
        (0.4, 3.4),
        2.6,
        1.0,
        "BackgroundTask\n(non-blocking)",
        fc="#FDECEA",
        ec=DANGER,
        fontsize=9,
    )
    _box(
        ax,
        (3.4, 3.4),
        3.2,
        1.0,
        "SQLite audit log\ndata/predictions/predictions.sqlite\n(43-col schema)",
        fc="#FDECEA",
        ec=DANGER,
        fontsize=9,
    )
    _box(
        ax,
        (7.0, 3.4),
        3.0,
        1.0,
        "predictions_live.csv\n(auto-refreshed)",
        fc="#FDECEA",
        ec=DANGER,
        fontsize=9,
        fontweight="bold",
    )

    # ── Power BI box (right side) ────────────────────────────────
    _box(
        ax,
        (10.6, 3.4),
        3.0,
        4.0,
        "Power BI Desktop\n\n8-page decision-support dashboard\n(refresh on demand)",
        fc="#E6EEF7",
        ec=PRIMARY,
        fontsize=10,
        fontweight="bold",
    )

    # ── Bottom row: monitoring + retrain loop ────────────────────
    _box(
        ax,
        (0.4, 1.0),
        3.0,
        1.4,
        "compute_live_drift.py\n(scheduled — e.g., weekly)\nPSI per feature vs baseline",
        fc="#FFF4DE",
        ec=ACCENT,
        fontsize=9,
    )
    _box(
        ax,
        (3.8, 1.0),
        3.0,
        1.4,
        "drift_metrics.csv\n(zones: safe / watch / retrain)",
        fc="#FFF4DE",
        ec=ACCENT,
        fontsize=9,
    )
    _box(
        ax,
        (7.2, 1.0),
        3.0,
        1.4,
        "Retraining trigger\n(if ≥ 2 features cross PSI 0.25 →\n scripts/train.py)",
        fc="#FFF4DE",
        ec=ACCENT,
        fontsize=9,
        fontweight="bold",
    )

    # ── Arrows ────────────────────────────────────────────────────
    # Users → FastAPI endpoints
    _arrow(ax, (1.6, 8.4), (1.35, 7.4), color=PRIMARY, label="HTTP")
    _arrow(ax, (4.6, 8.4), (3.0, 7.4), color=PRIMARY, label="HTTP/JSON")

    # FastAPI inference → loads artifacts (dotted-ish but solid)
    _arrow(ax, (5.8, 6.35), (6.4, 6.35), color=SAFE, label="load")

    # FastAPI inference → response back to caller
    ax.annotate(
        "JSON response\n(≤ 500 ms)",
        xy=(2.5, 4.5),
        xytext=(2.5, 5.1),
        ha="center",
        fontsize=8,
        color=PRIMARY,
        arrowprops=dict(arrowstyle="<-", color=PRIMARY, linewidth=1.4),
    )

    # FastAPI → BackgroundTask (after response)
    _arrow(ax, (3.1, 5.2), (1.7, 4.4), color=DANGER, label="add_task")

    # BackgroundTask → SQLite
    _arrow(ax, (3.0, 3.9), (3.4, 3.9), color=DANGER, label="INSERT")

    # SQLite → CSV
    _arrow(ax, (6.6, 3.9), (7.0, 3.9), color=DANGER, label="export_to_csv()")

    # CSV → Power BI (refresh)
    _arrow(ax, (10.0, 3.9), (10.6, 4.5), color=PRIMARY, label="Power BI\nRefresh")

    # SQLite → baseline reference + live CSV → drift compute
    _arrow(ax, (8.5, 3.4), (1.9, 2.4), color=ACCENT, label="weekly read")

    # Drift compute → drift CSV
    _arrow(ax, (3.4, 1.7), (3.8, 1.7), color=ACCENT)

    # Drift CSV → Power BI Page 8 (monitoring page)
    _arrow(ax, (6.8, 1.7), (10.6, 5.0), color=ACCENT, label="Page 8")

    # Drift CSV → Retrain trigger
    _arrow(ax, (6.8, 1.7), (7.2, 1.7), color=ACCENT)

    # Retrain trigger → artifacts (full cycle)
    _arrow(ax, (10.2, 2.2), (7.9, 6.6), color=SAFE, label="regenerates\nartifacts/*", linewidth=1.6)

    # ── Title + caption ──────────────────────────────────────────
    ax.text(
        7.0,
        9.7,
        "Figure 4.8 — Model Deployment Framework",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=PRIMARY,
    )
    ax.text(
        7.0,
        9.35,
        "Live serving pipeline: from a single booking entry to the "
        "Power BI dashboard and back via drift-triggered retraining",
        ha="center",
        fontsize=10,
        color=NEUTRAL,
        style="italic",
    )

    # Legend
    legend_y = 0.3
    ax.text(0.4, legend_y, "Flow legend:", fontsize=9, fontweight="bold")
    ax.plot([1.6, 2.0], [legend_y + 0.05, legend_y + 0.05], color=PRIMARY, linewidth=1.6)
    ax.text(2.1, legend_y, "request / response", fontsize=8, va="center")
    ax.plot([4.0, 4.4], [legend_y + 0.05, legend_y + 0.05], color=DANGER, linewidth=1.6)
    ax.text(4.5, legend_y, "async persistence", fontsize=8, va="center")
    ax.plot([6.5, 6.9], [legend_y + 0.05, legend_y + 0.05], color=ACCENT, linewidth=1.6)
    ax.text(7.0, legend_y, "drift / retrain loop", fontsize=8, va="center")
    ax.plot([9.0, 9.4], [legend_y + 0.05, legend_y + 0.05], color=SAFE, linewidth=1.6)
    ax.text(9.5, legend_y, "artifact load / regenerate", fontsize=8, va="center")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_deployment_framework.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig_deployment_framework.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_deployment_framework.png'}")
    print(f"wrote {OUT_DIR / 'fig_deployment_framework.pdf'}")


if __name__ == "__main__":
    build_figure()
