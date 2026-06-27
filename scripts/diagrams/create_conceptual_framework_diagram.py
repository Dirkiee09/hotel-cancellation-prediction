"""Generate the Developed Conceptual Framework diagram (Figure 1.2).

Renders the study's Dynamic Capabilities framework as a Sense -> Seize ->
Transform cycle, operationalised for hotel booking-cancellation
prediction. This supersedes the earlier hand-drawn box diagram, whose
SEIZE stage listed "Decision Tree, Random Forest, XGBoost"; the model
roster, calibration, threshold policies, and deployment stack here match
the current methodology (LightGBM champion).

Output: ``reports/figures/thesis/fig_conceptual_framework_dct.png`` (+ PDF).

Matplotlib-only — no graphviz dependency. Style mirrors
``create_conceptual_systems_diagram.py`` (serif, project palette).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures" / "thesis"

# Project palette (shared with the technical + systems-positioning views).
PRIMARY = "#1F4E79"
DANGER = "#A6192E"
SAFE = "#107C41"
ACCENT = "#B5740A"
NEUTRAL = "#3B3B3B"
MUTED = "#7A7A7A"


def _stage(ax, xy, w, h, *, title, subtitle, bullets, fc, ec):
    """A titled stage box with a subtitle and left-aligned bullet lines."""
    box = FancyBboxPatch(
        (xy[0], xy[1]),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    pad = 0.28
    ax.text(
        xy[0] + pad,
        xy[1] + h - 0.42,
        title,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=ec,
        zorder=3,
    )
    ax.text(
        xy[0] + pad,
        xy[1] + h - 0.86,
        subtitle,
        ha="left",
        va="center",
        fontsize=9.5,
        style="italic",
        color=MUTED,
        zorder=3,
    )
    ax.text(
        xy[0] + pad,
        xy[1] + h - 1.24,
        "\n".join(bullets),
        ha="left",
        va="top",
        fontsize=9.7,
        color=NEUTRAL,
        linespacing=1.5,
        zorder=3,
    )


def _arrow(ax, xy_from, xy_to, *, color, linestyle="-", rad=0.0, lw=1.8):
    arr = FancyArrowPatch(
        xy_from,
        xy_to,
        arrowstyle="-|>",
        mutation_scale=18,
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arr)


def build_figure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        }
    )
    fig, ax = plt.subplots(figsize=(13.0, 8.6), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(
        6.5,
        8.85,
        "Figure 1.2 — Developed Conceptual Framework",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=PRIMARY,
    )
    ax.text(
        6.5,
        8.42,
        "Dynamic Capabilities Theory (Teece): Sense → Seize → Transform, "
        "operationalised for cancellation prediction.",
        ha="center",
        fontsize=10,
        style="italic",
        color=NEUTRAL,
    )

    _stage(
        ax,
        (0.5, 4.55),
        5.7,
        3.5,
        title="SENSE",
        subtitle="Sensing — spot data & revenue opportunities",
        bullets=[
            "•  EDA, data cleaning, leakage removal",
            "•  Booking-time signals: lead time, ADR,",
            "    deposit type, market segment, requests",
            "•  119,390 Portugal bookings +",
            "    193-booking Punta Villa (PH) pilot",
            "•  Identify cancellation drivers and",
            "    sources of revenue instability",
        ],
        fc="#E6EEF7",
        ec=PRIMARY,
    )

    _stage(
        ax,
        (6.8, 4.55),
        5.7,
        3.5,
        title="SEIZE",
        subtitle="Seizing — build & calibrate the model",
        bullets=[
            "•  Chronological 80 / 10 / 10 split",
            "•  Ladder: Dummy → NB → LR → DT → RF → GB → XGB",
            "•  Champion: LightGBM (rolling-origin PR-AUC)",
            "•  Isotonic probability calibration",
            "•  Cost-sensitive threshold policies",
            "    (max_f1 / high_precision / cost_sensitive)",
            "•  Interpret: SHAP, ROC-AUC, PR-AUC",
        ],
        fc="#FFF4DE",
        ec=ACCENT,
    )

    _stage(
        ax,
        (3.15, 0.55),
        6.7,
        3.0,
        title="TRANSFORM",
        subtitle="Transforming — embed insights into hotel strategy",
        bullets=[
            "•  Deploy: FastAPI + Gradio + Power BI decision support",
            "•  Risk tiers → cost-aware interventions " "(deposit tiers, overbooking buffers)",
            "•  Monitoring (PSI drift) → retraining trigger",
            "•  Outcome: fewer cancellations, protected revenue",
            "    (36% cost saving versus intervene-on-everyone)",
        ],
        fc="#EDF7EE",
        ec=SAFE,
    )

    # Forward flow: Sense -> Seize (top), Seize -> Transform (right).
    _arrow(ax, (6.2, 6.3), (6.8, 6.3), color=NEUTRAL)
    _arrow(ax, (9.0, 4.55), (7.6, 3.55), color=NEUTRAL)

    # Feedback loop: Transform -> Sense (left, dashed) — the retraining cycle.
    # Starts on the Transform box's left edge and curves up to the Sense box's
    # lower-left so the loop visibly closes the cycle.
    _arrow(ax, (3.15, 2.05), (1.95, 4.55), color=ACCENT, linestyle="--", rad=0.34, lw=1.6)
    ax.text(
        0.95,
        3.25,
        "monitor drift →\nretrain (feedback)",
        ha="center",
        va="center",
        fontsize=9,
        color=ACCENT,
        style="italic",
    )

    # Legend.
    ly = 0.18
    ax.plot([4.7, 5.15], [ly + 0.04, ly + 0.04], color=NEUTRAL, linewidth=1.8)
    ax.text(5.25, ly, "forward flow", fontsize=8.5, va="center")
    ax.plot([7.0, 7.45], [ly + 0.04, ly + 0.04], color=ACCENT, linewidth=1.8, linestyle="--")
    ax.text(7.55, ly, "retraining feedback loop", fontsize=8.5, va="center")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_conceptual_framework_dct.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig_conceptual_framework_dct.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_conceptual_framework_dct.png'}")
    print(f"wrote {OUT_DIR / 'fig_conceptual_framework_dct.pdf'}")


if __name__ == "__main__":
    build_figure()
