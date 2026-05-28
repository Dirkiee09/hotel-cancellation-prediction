"""Generate the Conceptual Systems Positioning diagram for Chapter IV.

Renders Figure 4.9 — the conceptual *operational positioning* view of
the cancellation model, adapted from Antonio, Almeida, & Nunes (2017,
Figure 6, "Model deployment framework", *Tourism & Management Studies*
13(2):25-39). Pair this with `create_deployment_diagram.py` (Figure
4.10), which renders the technical *implementation* view.

What this figure shows: where the LightGBM cancellation model sits in
the hotel's existing IT ecosystem and which message types it exchanges
with the PMS, the channel manager, the OTAs/GDSs/website, the other
distribution channels, and (in dashed feedback lines) the channel
manager again with revised inventory/price signals.

This level of abstraction answers the question a revenue manager or
distribution executive will ask: "where in our stack does this
prediction model plug in?" The companion Figure 4.10 answers the
follow-on question a serving engineer will ask: "how does a single
prediction call flow through the deployment?"

Output: ``reports/figures/thesis/fig_conceptual_systems_positioning.png``
(+ PDF).

Matplotlib-only — no graphviz dependency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures" / "thesis"

# Project palette (matches the technical view + Power BI dashboard).
PRIMARY = "#1F4E79"
DANGER = "#A6192E"
SAFE = "#107C41"
ACCENT = "#F5A623"
NEUTRAL = "#3B3B3B"
SURFACE = "#F4F4F4"


def _box(ax, xy, w, h, text, *, fc=SURFACE, ec=PRIMARY, fontsize=10, fontweight="normal"):
    """Rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (xy[0], xy[1]),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4,
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


def _ellipse(ax, xy_center, w, h, text, *, fc=SURFACE, ec=PRIMARY, fontsize=10):
    """Ellipse with centered text — Antonio et al. use ellipses for the
    external distribution endpoints (OTAs, channel ecosystems).
    """
    e = Ellipse(
        xy_center,
        w,
        h,
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(e)
    ax.text(
        xy_center[0],
        xy_center[1],
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NEUTRAL,
        zorder=3,
        wrap=True,
    )


def _arrow(
    ax,
    xy_from,
    xy_to,
    *,
    color=NEUTRAL,
    label="",
    label_pos=0.5,
    label_offset=(0.0, 0.25),
    linewidth=1.4,
    linestyle="-",
):
    """Labelled arrow. `label_offset` shifts the text away from the
    midpoint so it doesn't sit on top of the line.
    """
    arr = FancyArrowPatch(
        xy_from,
        xy_to,
        arrowstyle="->",
        mutation_scale=14,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        mx = xy_from[0] + (xy_to[0] - xy_from[0]) * label_pos
        my = xy_from[1] + (xy_to[1] - xy_from[1]) * label_pos
        ax.text(
            mx + label_offset[0],
            my + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.95),
            zorder=4,
        )


def build_figure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        }
    )
    fig, ax = plt.subplots(figsize=(14.5, 9.0), dpi=150)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # ─────────────────── Title + Antonio attribution ───────────────────
    ax.text(
        7.25,
        9.0,
        "Figure 4.9 — Conceptual Systems Positioning of the Cancellation Model",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=PRIMARY,
    )
    ax.text(
        7.25,
        8.55,
        "Where the LightGBM model and its Power BI decision layer sit inside the hotel's existing distribution ecosystem.",
        ha="center",
        fontsize=10,
        style="italic",
        color=NEUTRAL,
    )
    ax.text(
        7.25,
        8.20,
        "Adapted from António, Almeida, & Nunes (2017, Figure 6). The CRS-side box has been expanded to show the full BI stack.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#7A7A7A",
    )

    # ──────────────────────── Top-row entities ─────────────────────────
    # Left distribution endpoint (ellipse, blue tint).
    _ellipse(
        ax,
        (1.6, 6.4),
        2.4,
        1.5,
        "OTAs / GDSs /\nhotel website",
        fc="#E6EEF7",
        ec=PRIMARY,
        fontsize=10,
    )
    # Channel Manager (middle-left rectangle).
    _box(
        ax,
        (4.3, 5.85),
        2.0,
        1.1,
        "Channel\nManager",
        fc=SURFACE,
        ec=PRIMARY,
        fontsize=11,
        fontweight="bold",
    )
    # Hotel PMS (middle rectangle — the hub).
    _box(
        ax,
        (7.5, 5.85),
        2.4,
        1.1,
        "Hotel PMS",
        fc=SURFACE,
        ec=PRIMARY,
        fontsize=12,
        fontweight="bold",
    )
    # Right distribution endpoint (ellipse).
    _ellipse(
        ax,
        (12.8, 6.4),
        2.6,
        1.5,
        "Other\ndistribution channels",
        fc="#E6EEF7",
        ec=PRIMARY,
        fontsize=10,
    )

    # ─────────────────── Bottom: CRS + BI stack box ────────────────────
    # Outer CRS container (large, navy bordered).
    _box(
        ax,
        (4.5, 1.0),
        8.5,
        3.4,
        "",
        fc="#F8FBFE",
        ec=PRIMARY,
        fontsize=11,
    )
    ax.text(
        8.75,
        4.0,
        "Central Reservation System (CRS)",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY,
    )
    ax.text(
        8.75,
        3.65,
        "— this thesis's contribution sits inside this layer —",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#7A7A7A",
    )

    # Inner stack inside the CRS — three rows, BI-style:
    # Row A (model layer): LightGBM classifier + ADR regressor.
    _box(
        ax,
        (4.85, 2.65),
        3.7,
        0.75,
        "LightGBM cancellation classifier\n+ isotonic calibration",
        fc="#EDF7EE",
        ec=SAFE,
        fontsize=9,
    )
    _box(
        ax,
        (8.95, 2.65),
        3.7,
        0.75,
        "Gradient-Boosted ADR regressor\n(parallel pricing forecast)",
        fc="#EDF7EE",
        ec=SAFE,
        fontsize=9,
    )

    # Row B (decision layer): threshold policies + SHAP.
    _box(
        ax,
        (4.85, 1.80),
        3.7,
        0.75,
        "Threshold policies\n(max_f1 / high_precision / cost_sensitive)",
        fc="#FFF4DE",
        ec=ACCENT,
        fontsize=8.5,
    )
    _box(
        ax,
        (8.95, 1.80),
        3.7,
        0.75,
        "TreeSHAP per-prediction\nexplanation (top-5 features)",
        fc="#FFF4DE",
        ec=ACCENT,
        fontsize=8.5,
    )

    # Row C (presentation layer): SQLite audit log + Power BI dashboard.
    _box(
        ax,
        (4.85, 1.10),
        3.7,
        0.62,
        "SQLite audit log\n(every prediction is recorded)",
        fc="#FDECEA",
        ec=DANGER,
        fontsize=8.5,
    )
    _box(
        ax,
        (8.95, 1.10),
        3.7,
        0.62,
        "Power BI 8-page\ndecision-support dashboard",
        fc="#E6EEF7",
        ec=PRIMARY,
        fontsize=8.5,
        fontweight="bold",
    )

    # ───────────────────────── Top-row arrows ──────────────────────────
    # Forward (inventory + prices) right-to-left across the top:
    # PMS → Channel Manager → OTAs/GDSs/website.
    _arrow(
        ax,
        (7.5, 6.65),
        (6.3, 6.65),
        color=PRIMARY,
        label="inventory and prices",
        label_pos=0.5,
        label_offset=(0.0, 0.30),
    )
    _arrow(
        ax,
        (4.3, 6.65),
        (2.8, 6.65),
        color=PRIMARY,
        label="inventory and prices",
        label_pos=0.5,
        label_offset=(0.0, 0.30),
    )
    # Backward (new bookings + cancellations) left-to-right under the
    # forward arrows.
    _arrow(
        ax,
        (2.8, 6.15),
        (4.3, 6.15),
        color=DANGER,
        label="new bookings,\nchanges, cancellations",
        label_pos=0.5,
        label_offset=(0.0, -0.55),
    )
    _arrow(
        ax,
        (6.3, 6.15),
        (7.5, 6.15),
        color=DANGER,
        label="new bookings,\nchanges, cancellations",
        label_pos=0.5,
        label_offset=(0.0, -0.55),
    )

    # PMS ↔ Other distribution channels (right side, paired arrows).
    _arrow(
        ax,
        (9.9, 6.65),
        (11.5, 6.65),
        color=PRIMARY,
        label="inventory and prices",
        label_pos=0.5,
        label_offset=(0.0, 0.30),
    )
    _arrow(
        ax,
        (11.5, 6.15),
        (9.9, 6.15),
        color=DANGER,
        label="new bookings,\nchanges, cancellations",
        label_pos=0.5,
        label_offset=(0.0, -0.55),
    )

    # ───────────────── PMS ↔ CRS (the model boundary) ──────────────────
    # PMS → CRS: "all bookings" (every reservation enters the model).
    _arrow(
        ax,
        (8.1, 5.85),
        (8.1, 4.10),
        color=SAFE,
        label="all bookings",
        label_pos=0.5,
        label_offset=(-0.85, 0.0),
        linewidth=1.6,
    )
    # CRS → PMS: model output (cancellation classification + forecast).
    _arrow(
        ax,
        (9.4, 4.10),
        (9.4, 5.85),
        color=SAFE,
        label="cancellation classification\n+ forecasted room nights",
        label_pos=0.5,
        label_offset=(1.45, 0.0),
        linewidth=1.6,
    )

    # ─────────────── Dashed feedback: Channel Manager ↔ CRS ────────────
    # Channel Manager → CRS: inventory + prices (dashed, weaker signal).
    _arrow(
        ax,
        (5.3, 5.85),
        (5.3, 4.40),
        color=ACCENT,
        label="inventory\nand prices",
        label_pos=0.5,
        label_offset=(-0.65, 0.0),
        linestyle="--",
        linewidth=1.4,
    )
    # CRS → Channel Manager: revised booking signals (dashed).
    _arrow(
        ax,
        (6.0, 4.40),
        (6.0, 5.85),
        color=ACCENT,
        label="new bookings,\nchanges, cancellations",
        label_pos=0.5,
        label_offset=(1.05, 0.0),
        linestyle="--",
        linewidth=1.4,
    )

    # ──────────────────────────── Legend ───────────────────────────────
    legend_y = 0.30
    ax.text(0.4, legend_y, "Flow legend:", fontsize=9, fontweight="bold")
    ax.plot([1.6, 2.05], [legend_y + 0.05, legend_y + 0.05], color=PRIMARY, linewidth=1.6)
    ax.text(2.15, legend_y, "inventory / prices", fontsize=8, va="center")
    ax.plot([4.2, 4.65], [legend_y + 0.05, legend_y + 0.05], color=DANGER, linewidth=1.6)
    ax.text(4.75, legend_y, "new bookings / cancellations", fontsize=8, va="center")
    ax.plot([7.7, 8.15], [legend_y + 0.05, legend_y + 0.05], color=SAFE, linewidth=1.6)
    ax.text(8.25, legend_y, "model input / output", fontsize=8, va="center")
    ax.plot(
        [10.6, 11.05],
        [legend_y + 0.05, legend_y + 0.05],
        color=ACCENT,
        linewidth=1.6,
        linestyle="--",
    )
    ax.text(11.15, legend_y, "dashed = revised feedback", fontsize=8, va="center")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_conceptual_systems_positioning.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig_conceptual_systems_positioning.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_conceptual_systems_positioning.png'}")
    print(f"wrote {OUT_DIR / 'fig_conceptual_systems_positioning.pdf'}")


if __name__ == "__main__":
    build_figure()
