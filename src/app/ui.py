"""Simplified Gradio UI for hotel booking cancellation prediction.

Public exports kept stable for src/app/main.py:
    - build_ui() -> gr.Blocks
    - BACKGROUND_CSS: str

Design goals:
    - Plain Gradio components, minimal CSS — easy to scan in under two minutes.
    - Three tabs: Predict | Try Examples | Help & Troubleshooting.
    - Built-in B1 fix: when the isotonic calibrator floors a low raw probability
      to exactly 0.0, the UI shows ``<0.01%`` and a "Very Low" band instead of
      a misleading "0.00%".
    - One-click example buttons populate every field for high/medium/low risk
      so users immediately see how inputs change the prediction.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.app.schemas import BookingRequest
from src.config import (
    ADR_MAX_VALID,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
)
from src.serving.inference import (
    explain_prediction,
    get_cached_artifacts,
    predict_proba,
)
from src.utils.thresholds import resolve_thresholds

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
TEST_PREDS_PATH = PROJECT_ROOT / "reports" / "test_predictions_for_powerbi.csv"


@lru_cache(maxsize=1)
def _load_cohort_probs() -> np.ndarray | None:
    """Load and sort the held-out test-set cancel probabilities once.

    Used for cohort-percentile lookup: at inference time we can answer
    "this booking is riskier than X% of test bookings" with an O(log N)
    np.searchsorted call.  The artifact is the standard
    test_predictions_for_powerbi.csv (~12k rows), produced by
    scripts/train.py — no new artifact required.

    Returns None if the file is missing or malformed; callers must guard.
    """
    if not TEST_PREDS_PATH.exists():
        return None
    try:
        df = pd.read_csv(TEST_PREDS_PATH, usecols=["cancel_probability"])
        probs = df["cancel_probability"].dropna().to_numpy(dtype=float)
        if probs.size == 0:
            return None
        return np.sort(probs)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        logger.warning("cohort_probs_load_failed error=%s", exc)
        return None


def _cohort_percentile(prob: float) -> int | None:
    """Percentile rank (0-100) of `prob` in the test-set distribution."""
    sorted_probs = _load_cohort_probs()
    if sorted_probs is None or sorted_probs.size == 0:
        return None
    idx = int(np.searchsorted(sorted_probs, prob, side="right"))
    return int(round(idx / sorted_probs.size * 100))


def _cohort_chip(prob: float) -> str:
    """Render the cohort percentile as an inline chip.  Empty if unavailable."""
    pct = _cohort_percentile(prob)
    if pct is None:
        return ""
    # Phrase it relative to the population — easier mental anchor than the
    # raw probability for non-technical readers.
    if pct >= 50:
        phrase = f"Riskier than ~{pct}% of bookings"
    else:
        phrase = f"Safer than ~{100 - pct}% of bookings"
    return (
        "<div class='cohort-chip'>"
        f"<span class='cohort-label'>Cohort percentile</span>"
        f"<span class='cohort-value'>{pct}th</span>"
        f"<span class='cohort-phrase'>{phrase}</span>"
        "</div>"
    )


BACKGROUND_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* =========================================================================
   Design system — modern analytics SaaS (Stripe/Linear/Vercel-adjacent)
   Single font, single accent, light surfaces, 8px spacing scale.
   ========================================================================= */
:root {
    --bg:           #F8F7F4;
    --surface:      #FFFFFF;
    --surface-2:    #FCFBF9;
    --border:       #E8E6E1;
    --border-soft:  #F1EFEA;
    --text-1:       #0F172A;
    --text-2:       #475569;
    --text-3:       #94A3B8;
    --accent:       #0F766E;
    --accent-dim:   #0D5752;
    --accent-soft:  #CCFBF1;
    --good:         #059669;
    --warn:         #D97706;
    --bad:          #DC2626;
    --radius-sm:    6px;
    --radius:       8px;
    --radius-lg:    12px;
    --shadow-1:     0 1px 2px rgba(15, 23, 42, 0.04);
    --shadow-2:     0 4px 14px rgba(15, 23, 42, 0.06);
}

/* =========================================================================
   Override Gradio's theme tokens so every component picks up our palette
   regardless of system dark-mode detection.  These vars are referenced by
   Gradio's internal Svelte-hashed classes, so editing them is the only
   reliable way to repaint everything (forms, blocks, inputs, tabs, buttons).
   We apply at .gradio-container scope so they win over Gradio's :root vars.
   ========================================================================= */
.gradio-container,
.dark .gradio-container,
.gradio-container.dark {
    color-scheme: light !important;

    /* --- Body & surface fills --- */
    --body-background-fill: #F8F7F4 !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: #FCFBF9 !important;
    --block-background-fill: #FFFFFF !important;
    --block-secondary-background-fill: #FCFBF9 !important;
    --block-border-color: #E8E6E1 !important;
    --block-border-width: 1px !important;
    --block-radius: 12px !important;
    --block-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;

    /* --- Labels & text --- */
    --block-label-background-fill: transparent !important;
    --block-label-text-color: #475569 !important;
    --block-label-text-weight: 500 !important;
    --block-title-text-color: #0F172A !important;
    --block-title-text-weight: 600 !important;
    --body-text-color: #0F172A !important;
    --body-text-color-subdued: #94A3B8 !important;
    --body-text-weight: 400 !important;

    /* --- Inputs --- */
    --input-background-fill: #FFFFFF !important;
    --input-background-fill-focus: #FFFFFF !important;
    --input-border-color: #E8E6E1 !important;
    --input-border-color-focus: #0F766E !important;
    --input-border-width: 1px !important;
    --input-shadow: none !important;
    --input-shadow-focus: 0 0 0 3px rgba(15, 118, 110, 0.12) !important;
    --input-radius: 8px !important;
    --input-padding: 8px 12px !important;
    --input-text-size: 14px !important;

    /* --- Accent (teal) replaces Gradio's default orange --- */
    --color-accent: #0F766E !important;
    --color-accent-soft: #CCFBF1 !important;
    --primary-50:  #F0FDFA !important;
    --primary-100: #CCFBF1 !important;
    --primary-200: #99F6E4 !important;
    --primary-300: #5EEAD4 !important;
    --primary-400: #2DD4BF !important;
    --primary-500: #14B8A6 !important;
    --primary-600: #0D9488 !important;
    --primary-700: #0F766E !important;
    --primary-800: #115E59 !important;
    --primary-900: #134E4A !important;
    --primary-950: #042F2E !important;

    /* --- Neutral scale (warm light) replaces Gradio's defaults --- */
    --neutral-50:  #FAFAF9 !important;
    --neutral-100: #F5F5F4 !important;
    --neutral-200: #E8E6E1 !important;
    --neutral-300: #D6D3D1 !important;
    --neutral-400: #A8A29E !important;
    --neutral-500: #78716C !important;
    --neutral-600: #57534E !important;
    --neutral-700: #44403C !important;
    --neutral-800: #292524 !important;
    --neutral-900: #1C1917 !important;
    --neutral-950: #0C0A09 !important;

    /* --- Buttons --- */
    --button-primary-background-fill: #0F766E !important;
    --button-primary-background-fill-hover: #0D5752 !important;
    --button-primary-text-color: #FFFFFF !important;
    --button-primary-text-color-hover: #FFFFFF !important;
    --button-primary-border-color: #0F766E !important;
    --button-primary-border-color-hover: #0D5752 !important;
    --button-secondary-background-fill: #FFFFFF !important;
    --button-secondary-background-fill-hover: #FCFBF9 !important;
    --button-secondary-text-color: #0F172A !important;
    --button-secondary-text-color-hover: #0F172A !important;
    --button-secondary-border-color: #E8E6E1 !important;
    --button-secondary-border-color-hover: #94A3B8 !important;
    --button-shadow: var(--shadow-1) !important;
    --button-shadow-hover: var(--shadow-2) !important;
    --button-large-radius: 8px !important;
    --button-small-radius: 8px !important;

    /* --- Borders --- */
    --border-color-primary: #E8E6E1 !important;
    --border-color-accent: #0F766E !important;
}

/* Force light color scheme for Gradio components even if the user's system
   is in dark mode — Gradio's auto-detect adds a .dark class to body. */
body.dark, html.dark { color-scheme: light !important; }
body.dark .gradio-container, html.dark .gradio-container {
    --body-background-fill: #F8F7F4 !important;
    --block-background-fill: #FFFFFF !important;
}

/* ---------- Page ---------------------------------------------------------- */
.gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    color: var(--text-1) !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 32px 40px 64px !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

/* ---------- Typography hierarchy ----------------------------------------- */
.gradio-container h1 {
    font-size: 30px !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-1) !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.2 !important;
}
.gradio-container h2 {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: var(--text-1) !important;
    margin: 24px 0 12px 0 !important;
}
.gradio-container h3 {
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: var(--text-1) !important;
    margin: 0 0 12px 0 !important;
}

/* Hero subtitle (right under H1) */
.hero-subtitle {
    color: var(--text-2);
    font-size: 15px;
    margin: 0 0 20px 0;
    font-weight: 400;
}

/* ---------- KPI chip strip ----------------------------------------------- */
.kpi-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 0 0 32px 0;
}
.kpi-chip {
    display: inline-flex;
    align-items: baseline;
    gap: 10px;
    background: var(--surface);
    padding: 10px 14px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-1);
}
.kpi-label {
    color: var(--text-3);
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 10px;
}
.kpi-value {
    color: var(--text-1);
    font-weight: 600;
    font-size: 15px;
    font-variant-numeric: tabular-nums;
}
.kpi-muted {
    color: var(--text-3);
    font-size: 13px;
    margin: 0 0 24px 0;
}

/* ---------- Cards / form surfaces ---------------------------------------- */
.gradio-container .gr-form,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .form {
    background: var(--surface) !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-1) !important;
}

/* ---------- Buttons (scoped tightly so Reset stays ghosted) -------------- */
.gradio-container button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    border-radius: var(--radius) !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    transition: all 150ms ease !important;
    cursor: pointer !important;
}
/* Primary CTA — the strongest element on the page */
.gradio-container button.primary,
.gradio-container .gr-button-primary {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent) !important;
    padding: 10px 18px !important;
    min-height: 40px !important;
    box-shadow: var(--shadow-1) !important;
}
.gradio-container button.primary:hover,
.gradio-container .gr-button-primary:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent-dim) !important;
    box-shadow: var(--shadow-2) !important;
    transform: translateY(-1px);
}
/* Secondary / ghost — Reset, tab buttons, etc. */
.gradio-container button.secondary,
.gradio-container .gr-button-secondary {
    background: var(--surface) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border) !important;
    padding: 10px 16px !important;
    min-height: 40px !important;
    box-shadow: var(--shadow-1) !important;
}
.gradio-container button.secondary:hover,
.gradio-container .gr-button-secondary:hover {
    background: var(--surface-2) !important;
    border-color: var(--text-3) !important;
}

/* ---------- Inputs ------------------------------------------------------- */
/* Scope to text-like inputs only — :not() exclusions prevent the rule from
   stomping native checkbox / radio / range / file styling (which previously
   made the "Repeated guest" checkbox render as a blank 36px square and
   "disappear" when toggled).  */
.gradio-container input:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="file"]):not([type="color"]),
.gradio-container textarea,
.gradio-container select {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    min-height: 36px !important;
    box-sizing: border-box !important;
    transition: border-color 150ms ease, box-shadow 150ms ease !important;
}
.gradio-container input:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="file"]):not([type="color"]):focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12) !important;
    outline: none !important;
}

/* Native checkbox / radio: keep browser-default rendering, just retint the
   check/dot to our accent color so they stay visible against the cream bg.  */
.gradio-container input[type="checkbox"],
.gradio-container input[type="radio"] {
    accent-color: var(--accent) !important;
    width: 16px !important;
    height: 16px !important;
    min-height: 16px !important;
    margin: 0 6px 0 0 !important;
    padding: 0 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    background: initial !important;
    border: initial !important;
    border-radius: initial !important;
}
.gradio-container label > span,
.gradio-container .label-wrap > span,
.gradio-container span[data-testid="block-label"] {
    color: var(--text-2) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
/* Inline info-text under inputs ("info=" prop) */
.gradio-container .info {
    color: var(--text-3) !important;
    font-size: 12px !important;
    margin-top: 4px !important;
    line-height: 1.4 !important;
}

/* ---------- Tabs --------------------------------------------------------- */
.gradio-container .tab-nav,
.gradio-container .tab-buttons,
.gradio-container [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 24px !important;
    gap: 4px !important;
    background: transparent !important;
}
/* Reset Gradio's tab button styling — they share .button-like classes. */
.gradio-container .tab-nav button,
.gradio-container .tab-buttons button,
.gradio-container [role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    color: var(--text-2) !important;
    padding: 10px 14px !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    min-height: 0 !important;
    box-shadow: none !important;
    transition: color 150ms ease, border-color 150ms ease !important;
}
.gradio-container .tab-nav button:hover,
.gradio-container .tab-buttons button:hover,
.gradio-container [role="tab"]:hover {
    color: var(--text-1) !important;
    background: transparent !important;
}
.gradio-container .tab-nav button.selected,
.gradio-container .tab-buttons button.selected,
.gradio-container [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ---------- Accordions --------------------------------------------------- */
.gradio-container .accordion {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    box-shadow: none !important;
    margin: 8px 0 !important;
}
.gradio-container .accordion .label-wrap {
    background: transparent !important;
    border: none !important;
    padding: 10px 14px !important;
    font-weight: 500 !important;
    color: var(--text-1) !important;
}

/* ---------- Risk assessment card --------------------------------------- */
.result-card {
    background: var(--surface);
    padding: 24px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-1);
}

/* Big probability number */
.prob-block {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.prob-label {
    color: var(--text-3);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.prob-number {
    font-size: 44px;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
    color: var(--text-1);
}
.prob-number.result-good { color: var(--good); }
.prob-number.result-warn { color: var(--warn); }
.prob-number.result-bad  { color: var(--bad); }

/* Probability bar visual */
.prob-bar {
    background: var(--border-soft);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    margin: 12px 0 14px 0;
}
.prob-bar-fill {
    height: 100%;
    transition: width 250ms ease;
    border-radius: 999px;
}
.prob-bar-fill.result-good { background: var(--good); }
.prob-bar-fill.result-warn { background: var(--warn); }
.prob-bar-fill.result-bad  { background: var(--bad); }

/* Risk badge pill */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.risk-badge.result-good { background: #ECFDF5; color: var(--good); border: 1px solid #A7F3D0; }
.risk-badge.result-warn { background: #FFFBEB; color: var(--warn); border: 1px solid #FDE68A; }
.risk-badge.result-bad  { background: #FEF2F2; color: var(--bad);  border: 1px solid #FECACA; }

/* Recommended action callout */
.action-block {
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin: 18px 0;
}
.action-label {
    color: var(--text-3);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 4px 0;
}
.action-text {
    color: var(--text-1);
    font-size: 14px;
    font-weight: 500;
    margin: 0;
}

/* Top factors — horizontal mini-bars */
.factors-block { margin: 18px 0 8px 0; }
.factors-title {
    color: var(--text-3);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 10px 0;
}
.factor-row {
    display: grid;
    grid-template-columns: 1fr 90px 56px;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid var(--border-soft);
}
.factor-row:last-child { border-bottom: none; }
.factor-name {
    color: var(--text-1);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.factor-name code {
    background: var(--surface-2);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
    color: var(--text-2);
}
.factor-bar {
    background: var(--border-soft);
    border-radius: 999px;
    height: 5px;
    overflow: hidden;
    position: relative;
}
.factor-bar-fill {
    position: absolute;
    top: 0;
    bottom: 0;
    height: 5px;
    border-radius: 999px;
}
.factor-bar-fill.cancel { background: var(--bad); right: 50%; }
.factor-bar-fill.stay   { background: var(--good); left: 50%; }
.factor-bar-mid {
    position: absolute;
    left: 50%;
    top: -1px;
    bottom: -1px;
    width: 1px;
    background: var(--border);
}
.factor-contrib {
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--text-2);
    font-size: 12px;
}

/* Policy decisions row — compact grid */
.policy-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin: 18px 0 0 0;
}
.policy-cell {
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 10px 12px;
}
.policy-name {
    color: var(--text-3);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 4px 0;
}
.policy-decision {
    color: var(--text-1);
    font-size: 13px;
    font-weight: 500;
    margin: 0;
}
.policy-thr {
    color: var(--text-3);
    font-size: 11px;
    margin: 4px 0 0 0;
    font-variant-numeric: tabular-nums;
}

/* Risk-band color classes kept (referenced from _risk_band() in Python). */
.result-good { color: var(--good); font-weight: 600; }
.result-warn { color: var(--warn); font-weight: 600; }
.result-bad  { color: var(--bad);  font-weight: 600; }

/* ---------- Decision axis: the chart that justifies every policy call ---- */
.decision-axis-block {
    margin: 22px 0 6px 0;
}
.axis-block-label {
    color: var(--text-3);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 22px 0;
}
.axis-strip {
    position: relative;
    height: 14px;
    background: var(--border-soft);
    border-radius: 999px;
    margin: 0 14px;
}
/* Risk-band tinted zones (Low / Medium / High) */
.band-zone {
    position: absolute;
    top: 0;
    bottom: 0;
    height: 100%;
}
.band-zone.low  { background: rgba(5, 150, 105, 0.18); border-radius: 999px 0 0 999px; }
.band-zone.med  { background: rgba(217, 119, 6, 0.18); }
.band-zone.high { background: rgba(220, 38, 38, 0.18); border-radius: 0 999px 999px 0; }

/* Vertical tick marks for the 3 policy thresholds */
.thr-tick {
    position: absolute;
    top: -3px;
    bottom: -3px;
    width: 2px;
    background: var(--text-2);
    border-radius: 1px;
    transform: translateX(-1px);
}
.thr-tick.cost { background: #64748B; }
.thr-tick.f1   { background: #475569; }
.thr-tick.hp   { background: #1E293B; }

/* The prominent prediction marker (circle on the axis + label balloon above) */
.axis-marker {
    position: absolute;
    top: -5px;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--accent);
    border: 3px solid #FFFFFF;
    box-shadow: 0 2px 6px rgba(15, 118, 110, 0.35);
    transform: translateX(-50%);
    z-index: 3;
}
.marker-label {
    position: absolute;
    bottom: 34px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--accent);
    color: #FFFFFF;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
    box-shadow: 0 2px 4px rgba(15, 118, 110, 0.25);
}
.marker-label::after {
    /* downward-pointing arrow under the label */
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 4px solid transparent;
    border-top-color: var(--accent);
}

/* Tick labels (0% / 40% / 70% / 100%) under the strip */
.axis-ticks-row {
    position: relative;
    height: 16px;
    margin: 8px 14px 0;
    font-variant-numeric: tabular-nums;
}
.axis-ticks-row span {
    position: absolute;
    transform: translateX(-50%);
    font-size: 10px;
    color: var(--text-3);
    white-space: nowrap;
}

/* Legend showing what each tick means */
.thr-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 14px 14px 0;
    padding: 10px 12px;
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    font-size: 11px;
    color: var(--text-2);
}
.thr-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.thr-legend-swatch {
    display: inline-block;
    width: 2px;
    height: 10px;
    border-radius: 1px;
}
.thr-legend-swatch.cost { background: #64748B; }
.thr-legend-swatch.f1   { background: #475569; }
.thr-legend-swatch.hp   { background: #1E293B; }
.thr-legend-swatch.you  { background: var(--accent); border-radius: 50%; width: 8px; height: 8px; }

/* Borderline indicator: surfaced when prob is within 5pp of a band threshold */
.borderline-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 12px;
    padding: 6px 10px;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    color: #92400E;
    border-radius: var(--radius);
    font-size: 12px;
    font-weight: 500;
}
.borderline-chip::before {
    content: '⚠';
    font-size: 13px;
}

/* ---------- Cohort percentile chip (population-context trust signal) ----- */
.cohort-chip {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin: 14px 0 0 0;
    padding: 10px 14px;
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    font-size: 12px;
}
.cohort-label {
    color: var(--text-3);
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 10px;
}
.cohort-value {
    color: var(--accent);
    font-weight: 700;
    font-size: 15px;
    font-variant-numeric: tabular-nums;
}
.cohort-phrase {
    color: var(--text-2);
    flex: 1;
}

/* ---------- Waterfall summary header (above the factor bars) ------------- */
.waterfall-summary {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 14px 0;
    padding: 10px 14px;
    background: var(--surface-2);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
}
.waterfall-segment {
    color: var(--text-2);
}
.waterfall-segment .num {
    color: var(--text-1);
    font-weight: 600;
}
.waterfall-arrow {
    color: var(--text-3);
}
.waterfall-delta {
    margin-left: auto;
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 11px;
}
.waterfall-delta.up    { background: #FEF2F2; color: var(--bad); }
.waterfall-delta.down  { background: #ECFDF5; color: var(--good); }
.waterfall-delta.flat  { background: var(--surface-2); color: var(--text-3); }

/* ---------- Loading skeleton (shimmer during scoring) -------------------- */
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
}
.skeleton {
    background: linear-gradient(
        90deg,
        var(--border-soft) 0%,
        #FFFFFF 50%,
        var(--border-soft) 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.6s linear infinite;
    border-radius: var(--radius);
    color: transparent !important;
    border: 1px solid var(--border-soft) !important;
}
.skeleton-line {
    height: 14px;
    margin: 8px 0;
}
.skeleton-line.lg   { height: 44px; width: 60%; }
.skeleton-line.sm   { height: 10px; width: 40%; }
.skeleton-line.full { width: 100%; }
.skeleton-label {
    color: var(--text-3);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 12px 0;
}

/* ---------- Help-note callout (kept for B1 zero-probability path) -------- */
.help-note {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: var(--radius);
    padding: 12px 14px;
    margin: 12px 0;
    font-size: 13px;
    color: #92400E;
}

/* ---------- Help-tab Markdown tables ------------------------------------- */
.gradio-container table {
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 13px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}
.gradio-container th {
    background: var(--surface-2) !important;
    color: var(--text-2) !important;
    font-weight: 500 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    padding: 8px 12px !important;
    text-align: left !important;
    border-bottom: 1px solid var(--border) !important;
}
.gradio-container td {
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--border-soft) !important;
    color: var(--text-1) !important;
}
.gradio-container tr:last-child td { border-bottom: none !important; }

/* ---------- Code blocks: subtle, not in-your-face ------------------------ */
.gradio-container code {
    background: var(--surface-2);
    color: var(--text-1);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12.5px;
    font-family: 'JetBrains Mono', ui-monospace, 'SFMono-Regular', Consolas, monospace;
}
"""


# ---------------------------------------------------------------------------
# Dropdown data — auto-detected from the training CSV (plug-and-play)
# ---------------------------------------------------------------------------

_FALLBACK_CHOICES: dict[str, list[str]] = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "market_segment": [
        "Online TA",
        "Offline TA/TO",
        "Direct",
        "Groups",
        "Corporate",
        "Complementary",
        "Aviation",
        "Undefined",
    ],
    "distribution_channel": ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
    "customer_type": ["Transient", "Transient-Party", "Contract", "Group"],
    "meal": ["BB", "HB", "FB", "SC", "Undefined"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "reserved_room_type": ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
    "country": ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "USA", "BRA", "IRL"],
}


def _load_categorical_choices() -> dict[str, list[str]]:
    cols = list(_FALLBACK_CHOICES.keys())
    result: dict[str, list[str]] = {}
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH, usecols=cols)
            for col in cols:
                vals = sorted({str(v).strip() for v in df[col].dropna() if str(v).strip()})
                if vals:
                    result[col] = vals
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            logger.warning("categorical_choices_load_failed error=%s; using fallbacks", exc)
    for col in cols:
        if col not in result or not result[col]:
            result[col] = _FALLBACK_CHOICES[col]
    return result


_CAT = _load_categorical_choices()


def _hero_metrics_line() -> str:
    """Render the headline metrics as compact KPI chips.

    Four chips when calibration is available: ROC-AUC, PR-AUC, F1, ECE.
    ECE (expected calibration error) is the small one panelists care about
    most for a calibrated classifier — the average gap between predicted
    and observed cancellation frequency, in probability units.  Surfacing
    it next to the discriminative metrics signals "the % we show below
    is trustworthy, not just a ranking."
    """
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            mf = data.get("max_f1", {})
            roc, pr, f1 = mf.get("roc_auc"), mf.get("pr_auc"), mf.get("f1")
            cal_test = data.get("calibration", {}).get("test", {})
            ece = cal_test.get("ece_calibrated")
            if roc is not None and pr is not None and f1 is not None:
                chips = [
                    ("ROC-AUC", f"{float(roc):.3f}"),
                    ("PR-AUC", f"{float(pr):.3f}"),
                    ("F1", f"{float(f1):.3f}"),
                ]
                if ece is not None:
                    chips.append(("Calibration (ECE)", f"{float(ece):.3f}"))
                return (
                    "<div class='kpi-row'>"
                    + "".join(
                        f"<span class='kpi-chip'><span class='kpi-label'>{label}</span>"
                        f"<span class='kpi-value'>{value}</span></span>"
                        for label, value in chips
                    )
                    + "</div>"
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("hero_metrics_load_failed error=%s", exc)
    return (
        "<p class='kpi-muted'>Model metrics not available — "
        "run <code>python scripts/train.py</code>.</p>"
    )


# ---------------------------------------------------------------------------
# Example scenarios for the "Try Examples" tab
# ---------------------------------------------------------------------------

# NOTE: arrival_date values MUST be ISO strings ("YYYY-MM-DD"), not date objects.
# gr.DateTime in Gradio 6.x calls int()/arithmetic on its value during postprocess
# and raises "datetime.date object cannot be interpreted as an integer" if passed
# a raw date.  _form_defaults() already uses .isoformat() — keep these consistent.
EXAMPLES: dict[str, dict[str, Any]] = {
    "high_risk": {
        "label": "🔴 High cancellation risk",
        "hint": "Long lead time, group booking, prior cancellation, no deposit.",
        "values": {
            "hotel": "City Hotel",
            "arrival_date": date(2025, 10, 15).isoformat(),
            "lead_time": 200,
            "weekend_nights": 0,
            "week_nights": 3,
            "adults": 1,
            "children": 0,
            "babies": 0,
            "country": "PRT",
            "meal": "BB",
            "market_segment": "Groups",
            "distribution_channel": "TA/TO",
            "customer_type": "Transient",
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "previous_cancellations": 1,
            "previous_bookings_not_canceled": 0,
            "adr": 80.0,
            "parking": 0,
            "special_requests": 0,
        },
    },
    "medium_risk": {
        "label": "🟠 Borderline cancellation risk",
        "hint": (
            "Long lead time, online TA, single guest, no special requests "
            "— around ~40% probability, near the decision boundary."
        ),
        "values": {
            "hotel": "City Hotel",
            "arrival_date": date(2025, 9, 15).isoformat(),
            "lead_time": 180,
            "weekend_nights": 2,
            "week_nights": 3,
            "adults": 1,
            "children": 0,
            "babies": 0,
            "country": "GBR",
            "meal": "BB",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "customer_type": "Transient",
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "adr": 100.0,
            "parking": 0,
            "special_requests": 0,
        },
    },
    "low_risk": {
        "label": "🟢 Low cancellation risk",
        "hint": "Repeated guest, short lead time, parking + special requests.",
        "values": {
            "hotel": "Resort Hotel",
            "arrival_date": date(2025, 8, 5).isoformat(),
            "lead_time": 5,
            "weekend_nights": 1,
            "week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "country": "PRT",
            "meal": "BB",
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "customer_type": "Transient",
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 10,
            "adr": 90.0,
            "parking": 1,
            "special_requests": 2,
        },
    },
}

INPUT_KEYS: tuple[str, ...] = (
    "hotel",
    "arrival_date",
    "lead_time",
    "weekend_nights",
    "week_nights",
    "adults",
    "children",
    "babies",
    "country",
    "meal",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "reserved_room_type",
    "deposit_type",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "parking",
    "special_requests",
)


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------


def _coerce_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).date()
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip()).date()
        except ValueError as exc:
            raise ValueError(f"Unparseable arrival_date: {value!r}") from exc
    raise ValueError("Arrival date is required.")


def _to_int(value: Any, default: int = 0) -> int:
    """Coerce a Gradio Number value to int, defaulting if the user cleared it.

    gr.Number passes ``None`` when the field is empty; passing that straight to
    ``int()`` raises ``TypeError: int() argument must be a real number...``.
    """
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce a Gradio Number value to float with a safe default on empty/None."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_booking(values: dict[str, Any]) -> BookingRequest:
    payload: dict[str, Any] = {
        "hotel": values["hotel"],
        "lead_time": _to_int(values["lead_time"]),
        "arrival_date": _coerce_date(values["arrival_date"]),
        "stays_in_weekend_nights": _to_int(values["weekend_nights"]),
        "stays_in_week_nights": _to_int(values["week_nights"]),
        "adults": max(1, _to_int(values["adults"], default=1)),
        "children": _to_int(values["children"]),
        "babies": _to_int(values["babies"]),
        "meal": values["meal"] or None,
        "country": (str(values["country"]).strip().upper() or None) if values["country"] else None,
        "market_segment": values["market_segment"] or None,
        "distribution_channel": values["distribution_channel"] or None,
        # is_repeated_guest is derived, not user-supplied: any prior history
        # (a previous cancellation OR a previous successful stay) marks the
        # booker as a returning guest.  Eliminates the redundant checkbox and
        # prevents the user from entering inconsistent values.
        "previous_cancellations": _to_int(values["previous_cancellations"]),
        "previous_bookings_not_canceled": _to_int(values["previous_bookings_not_canceled"]),
        "is_repeated_guest": int(
            _to_int(values["previous_cancellations"]) > 0
            or _to_int(values["previous_bookings_not_canceled"]) > 0
        ),
        "reserved_room_type": values["reserved_room_type"] or None,
        "deposit_type": values["deposit_type"] or None,
        "agent": "Direct",
        "customer_type": values["customer_type"] or None,
        "adr": _to_float(values["adr"]),
        "required_car_parking_spaces": _to_int(values["parking"]),
        "total_of_special_requests": _to_int(values["special_requests"]),
    }
    return BookingRequest.model_validate(payload)


def _format_pct(prob: float) -> str:
    """B1 fix: display ``<0.01%`` when the calibrator floors a raw prob to zero."""
    if prob <= 0.0:
        return "&lt;0.01%"
    return f"{prob * 100:.2f}%"


def _risk_band(prob: float) -> tuple[str, str]:
    if prob <= 0.0:
        return "Very Low", "result-good"
    if prob >= RISK_TIER_HIGH_THRESHOLD:
        return "High", "result-bad"
    if prob >= RISK_TIER_MEDIUM_THRESHOLD:
        return "Medium", "result-warn"
    return "Low", "result-good"


def _why_zero_note() -> str:
    return (
        "<div class='help-note'>"
        "The model is highly confident this booking will <b>not</b> cancel "
        "(raw probability below the calibrator's resolution floor). "
        "Try the <b>High risk</b> example in the next tab to see a non-zero result."
        "</div>"
    )


@lru_cache(maxsize=1)
def _population_base_rate() -> float | None:
    """Mean cancel probability across the held-out test set (empirical base rate).

    Used as the "starting point" reference in the waterfall summary header.
    None if cohort artifact is unavailable; callers must guard.
    """
    sorted_probs = _load_cohort_probs()
    if sorted_probs is None or sorted_probs.size == 0:
        return None
    return float(sorted_probs.mean())


def _waterfall_summary(prob: float) -> str:
    """Render the 'population baseline → this booking' summary line.

    Shows base rate vs predicted probability with the signed delta.  The
    factor bars below explain *which* features drove that delta.  Not a
    strict SHAP waterfall in probability space (SHAP additivity holds in
    log-odds, not probabilities) — instead a population-relative anchor
    so non-technical readers can frame the prediction in context.
    """
    base = _population_base_rate()
    if base is None:
        return ""
    base_pct = base * 100.0
    pred_pct = prob * 100.0
    delta_pp = pred_pct - base_pct
    if delta_pp > 0.5:
        delta_cls, delta_label = "up", f"↑ +{delta_pp:.1f}pp"
    elif delta_pp < -0.5:
        delta_cls, delta_label = "down", f"↓ {delta_pp:.1f}pp"
    else:
        delta_cls, delta_label = "flat", f"≈ {delta_pp:+.1f}pp"
    return (
        "<div class='waterfall-summary'>"
        "<span class='waterfall-segment'>"
        "Base rate "
        f"<span class='num'>{base_pct:.1f}%</span></span>"
        "<span class='waterfall-arrow'>→</span>"
        "<span class='waterfall-segment'>"
        "This booking "
        f"<span class='num'>{pred_pct:.2f}%</span></span>"
        f"<span class='waterfall-delta {delta_cls}'>{delta_label}</span>"
        "</div>"
    )


def _format_top_features(features: list[dict[str, object]]) -> str:
    """Render top contributing features as horizontal mini-bars."""
    if not features:
        return (
            "<div class='factors-block'>"
            "<p class='factors-title'>Top contributing factors</p>"
            "<p style='color:var(--text-3);font-size:13px;margin:0;'>"
            "Feature attributions not available for this prediction.</p>"
            "</div>"
        )

    # Normalize all contributions against the largest absolute value for
    # bar widths — gives a relative sense of which factor dominated.
    contribs: list[float] = []
    for item in features:
        try:
            contribs.append(float(item.get("contribution", 0.0)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            contribs.append(0.0)
    peak = max((abs(c) for c in contribs), default=1.0) or 1.0

    rows: list[str] = []
    for item, contrib in zip(features, contribs):
        feat = str(item.get("feature", "?"))
        val = item.get("value")
        width_pct = min(48.0, abs(contrib) / peak * 48.0)
        side_cls = "cancel" if contrib > 0 else "stay"
        if side_cls == "cancel":
            fill_style = f"width:{width_pct:.1f}%; right:50%;"
        else:
            fill_style = f"width:{width_pct:.1f}%; left:50%;"
        rows.append(
            "<div class='factor-row'>"
            f"<div class='factor-name'>{feat} <code>{val}</code></div>"
            "<div class='factor-bar'>"
            "<div class='factor-bar-mid'></div>"
            f"<div class='factor-bar-fill {side_cls}' style='{fill_style}'></div>"
            "</div>"
            f"<div class='factor-contrib'>{contrib:+.3f}</div>"
            "</div>"
        )
    return (
        "<div class='factors-block'>"
        "<p class='factors-title'>Top contributing factors</p>" + "".join(rows) + "</div>"
    )


def _recommended_action(prob: float, band: str) -> str:
    """Business-meaningful next step for the front desk, keyed off risk band."""
    if prob <= 0.0:
        action = "No action needed — model is highly confident this guest will arrive."
    elif band == "High":
        action = (
            "Request a deposit or reconfirm by phone before arrival. "
            "Hold back-up inventory for over-sell protection."
        )
    elif band == "Medium":
        action = (
            "Schedule a courtesy reminder closer to arrival. " "Monitor for further risk signals."
        )
    else:  # Low
        action = "No outreach needed at this time."
    return (
        "<div class='action-block'>"
        "<p class='action-label'>Recommended action</p>"
        f"<p class='action-text'>{action}</p>"
        "</div>"
    )


def _decision_axis_html(prob: float, thr_f1: float, thr_hp: float, thr_cost: float) -> str:
    """Render the horizontal 0-100% axis showing every decision boundary.

    Justification chart: places the prediction on the same scale as the three
    UI risk bands (Low/Medium/High) AND the three policy thresholds (Cost /
    Balanced / High-precision).  A single glance proves every policy decision
    below follows from the math — left of a tick = "stay", right = "cancel".
    """
    prob_pct = max(0.0, min(100.0, prob * 100.0))
    med_pct = RISK_TIER_MEDIUM_THRESHOLD * 100.0
    high_pct = RISK_TIER_HIGH_THRESHOLD * 100.0
    cost_pct = max(0.0, min(100.0, thr_cost * 100.0))
    f1_pct = max(0.0, min(100.0, thr_f1 * 100.0))
    hp_pct = max(0.0, min(100.0, thr_hp * 100.0))
    return (
        "<div class='decision-axis-block'>"
        "<p class='axis-block-label'>Where this prediction falls</p>"
        "<div class='axis-strip'>"
        # risk-band tinted zones
        f"<div class='band-zone low'  style='left:0%; width:{med_pct:.1f}%'></div>"
        f"<div class='band-zone med'  style='left:{med_pct:.1f}%; "
        f"width:{high_pct - med_pct:.1f}%'></div>"
        f"<div class='band-zone high' style='left:{high_pct:.1f}%; "
        f"width:{100 - high_pct:.1f}%'></div>"
        # policy threshold tick marks
        f"<div class='thr-tick cost' style='left:{cost_pct:.1f}%' "
        f"title='Cost-optimal threshold: {thr_cost:.2f}'></div>"
        f"<div class='thr-tick f1'   style='left:{f1_pct:.1f}%' "
        f"title='Balanced (max-F1) threshold: {thr_f1:.2f}'></div>"
        f"<div class='thr-tick hp'   style='left:{hp_pct:.1f}%' "
        f"title='High-precision threshold: {thr_hp:.2f}'></div>"
        # prediction marker (with label balloon above)
        f"<div class='axis-marker' style='left:{prob_pct:.1f}%'>"
        f"<span class='marker-label'>{prob_pct:.2f}%</span>"
        "</div>"
        "</div>"
        # axis tick labels below
        "<div class='axis-ticks-row'>"
        "<span style='left:0%'>0%</span>"
        f"<span style='left:{med_pct:.1f}%'>{int(med_pct)}%</span>"
        f"<span style='left:{high_pct:.1f}%'>{int(high_pct)}%</span>"
        "<span style='left:100%'>100%</span>"
        "</div>"
        # legend for the four marker types
        "<div class='thr-legend'>"
        "<span class='thr-legend-item'>"
        "<span class='thr-legend-swatch you'></span>This booking"
        "</span>"
        "<span class='thr-legend-item'>"
        f"<span class='thr-legend-swatch cost'></span>Cost-opt ({thr_cost:.2f})"
        "</span>"
        "<span class='thr-legend-item'>"
        f"<span class='thr-legend-swatch f1'></span>Balanced ({thr_f1:.2f})"
        "</span>"
        "<span class='thr-legend-item'>"
        f"<span class='thr-legend-swatch hp'></span>High-prec ({thr_hp:.2f})"
        "</span>"
        "</div>"
        "</div>"
    )


def _borderline_chip(prob: float, margin_pp: float = 5.0) -> str:
    """Show a small warning when prob is within `margin_pp` of a risk-band boundary.

    Tells the operator the call could flip with small input changes — useful
    framing for cases like prob=39.2% sitting just below Medium=40%.
    """
    pct = prob * 100.0
    if pct <= 0.0:
        return ""
    boundaries = [
        ("Medium", RISK_TIER_MEDIUM_THRESHOLD * 100.0),
        ("High", RISK_TIER_HIGH_THRESHOLD * 100.0),
    ]
    closest_label, closest_diff = "", 9999.0
    for label, b in boundaries:
        diff = abs(pct - b)
        if diff < closest_diff:
            closest_label, closest_diff = label, diff
    if closest_diff > margin_pp:
        return ""
    direction = (
        "below"
        if pct
        < (
            RISK_TIER_MEDIUM_THRESHOLD * 100.0
            if closest_label == "Medium"
            else RISK_TIER_HIGH_THRESHOLD * 100.0
        )
        else "above"
    )
    return (
        f"<div><span class='borderline-chip'>"
        f"Borderline · {closest_diff:.1f}pp {direction} the {closest_label}-risk threshold"
        f"</span></div>"
    )


def predict_one(*args: Any) -> tuple[str, str, str]:
    """Run a single prediction from the UI inputs.

    Returns (headline_markdown, details_markdown, raw_json_str).
    """
    values = dict(zip(INPUT_KEYS, args))

    try:
        booking = _build_booking(values)
    except ValidationError as exc:
        msg = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}"
            for err in exc.errors()
        )
        return f"### ❌ Invalid input\n{msg}", "", ""
    except (ValueError, TypeError, KeyError) as exc:
        return f"### ❌ Could not score booking\n{exc}", "", ""

    try:
        artifacts = get_cached_artifacts()
        record = booking.model_dump(exclude={"arrival_date"})
        probs, feature_df = predict_proba(record, artifacts)
        prob = float(probs[0])
    except FileNotFoundError as exc:
        return (
            (
                "### ❌ Model artifacts not loaded\n"
                f"`{exc}`\n\n"
                "**Fix:** run `python scripts/train.py` from the project root."
            ),
            "",
            "",
        )
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        logger.exception("Prediction failed")
        return f"### ❌ Prediction failed\n`{exc}`", "", ""

    resolved, _, _, _ = resolve_thresholds(artifacts.thresholds or {})
    thr_f1 = resolved["max_f1"]
    thr_hp = resolved["high_precision"]
    thr_cost = resolved["cost_sensitive"]
    pct_str = _format_pct(prob)
    band, css = _risk_band(prob)

    verdict_f1 = "Likely to cancel" if prob >= thr_f1 else "Likely to stay"
    verdict_hp = "Cancel (high-confidence)" if prob >= thr_hp else "Stay (high-confidence)"
    verdict_cost = "Flag for outreach" if prob >= thr_cost else "No outreach needed"

    # Headline = big probability number + horizontal bar + risk badge pill
    # + decision-axis viz + (conditional) borderline indicator.  The axis is
    # the central "justify-the-results" chart: it visually positions the
    # prediction against every UI band AND every policy threshold, so the
    # three decisions in the policy grid below need no further explanation.
    bar_pct = max(0.0, min(100.0, prob * 100.0))
    headline = (
        "<div class='prob-block'>"
        "<p class='prob-label'>Cancellation probability</p>"
        f"<div class='prob-number {css}'>{pct_str}</div>"
        "<div class='prob-bar'>"
        f"<div class='prob-bar-fill {css}' style='width:{bar_pct:.1f}%'></div>"
        "</div>"
        f"<div><span class='risk-badge {css}'>● {band} risk</span></div>"
        + _borderline_chip(prob)
        + _cohort_chip(prob)
        + _decision_axis_html(prob, thr_f1, thr_hp, thr_cost)
        + "</div>"
    )

    # Details = recommended action + top-factor bars + zero-note (if applicable)
    # + compact 3-up policy grid.
    parts: list[str] = [_recommended_action(prob, band)]
    if prob <= 0.0:
        parts.append(_why_zero_note())
    top = explain_prediction(feature_df, artifacts, top_n=5)
    parts.append(_waterfall_summary(prob))
    parts.append(_format_top_features(top))
    parts.append(
        "<div class='policy-grid'>"
        "<div class='policy-cell'>"
        "<p class='policy-name'>Balanced</p>"
        f"<p class='policy-decision'>{verdict_f1}</p>"
        f"<p class='policy-thr'>threshold {thr_f1:.2f}</p>"
        "</div>"
        "<div class='policy-cell'>"
        "<p class='policy-name'>High-precision</p>"
        f"<p class='policy-decision'>{verdict_hp}</p>"
        f"<p class='policy-thr'>threshold {thr_hp:.2f}</p>"
        "</div>"
        "<div class='policy-cell'>"
        "<p class='policy-name'>Cost-optimal</p>"
        f"<p class='policy-decision'>{verdict_cost}</p>"
        f"<p class='policy-thr'>threshold {thr_cost:.2f}</p>"
        "</div>"
        "</div>"
    )

    raw = {
        "probability": prob,
        "probability_display": pct_str.replace("&lt;", "<"),
        "risk_band": band,
        "thresholds": {
            "max_f1": thr_f1,
            "high_precision": thr_hp,
            "cost_sensitive": thr_cost,
        },
        "decisions": {
            "balanced": verdict_f1,
            "high_precision": verdict_hp,
            "cost_optimal": verdict_cost,
        },
        "top_features": top,
        "scored_utc": datetime.now(timezone.utc).isoformat(),
    }
    return headline, "\n".join(parts), json.dumps(raw, indent=2, default=str)


def _populate_from_example(key: str) -> tuple[Any, ...]:
    v = EXAMPLES[key]["values"]
    return tuple(v[k] for k in INPUT_KEYS)


_SKELETON_HEADLINE = (
    "<div class='prob-block'>"
    "<p class='skeleton-label'>Scoring booking</p>"
    "<div class='skeleton skeleton-line lg'>&nbsp;</div>"
    "<div class='skeleton skeleton-line full'>&nbsp;</div>"
    "<div class='skeleton skeleton-line sm'>&nbsp;</div>"
    "</div>"
)
_SKELETON_DETAILS = (
    "<div class='action-block'>"
    "<p class='skeleton-label'>Computing explanation</p>"
    "<div class='skeleton skeleton-line full'>&nbsp;</div>"
    "<div class='skeleton skeleton-line full'>&nbsp;</div>"
    "<div class='skeleton skeleton-line' style='width:75%'>&nbsp;</div>"
    "</div>"
)


def _set_predicting_state() -> tuple[Any, str, str, str]:
    """First step of the prediction click chain — disable the button and
    swap the headline / details panels for animated shimmer skeletons so
    the user has visible feedback that scoring is in progress."""
    return (
        gr.update(value="Scoring…", interactive=False),
        _SKELETON_HEADLINE,
        _SKELETON_DETAILS,
        "",
    )


def _set_ready_state() -> Any:
    """Final step of the prediction click chain — re-enable the button."""
    return gr.update(value="Predict cancellation risk", interactive=True)


# ---------------------------------------------------------------------------
# R5 — Form defaults + Reset
# R7 — Demo mode (?demo=1 auto-loads the high-risk scenario)
# ---------------------------------------------------------------------------


# Default values used at first page load AND when the Reset button is pressed.
# arrival_date is computed dynamically so it does not go stale if the server
# stays up for days.
def _form_defaults() -> dict[str, Any]:
    return {
        "hotel": _CAT["hotel"][0],
        "arrival_date": date.today().isoformat(),
        "lead_time": 30,
        "weekend_nights": 1,
        "week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "country": "PRT"
        if "PRT" in _CAT["country"]
        else (_CAT["country"][0] if _CAT["country"] else ""),
        "meal": "BB" if "BB" in _CAT["meal"] else (_CAT["meal"][0] if _CAT["meal"] else ""),
        "market_segment": "Online TA"
        if "Online TA" in _CAT["market_segment"]
        else _CAT["market_segment"][0],
        "distribution_channel": "TA/TO"
        if "TA/TO" in _CAT["distribution_channel"]
        else _CAT["distribution_channel"][0],
        "customer_type": "Transient"
        if "Transient" in _CAT["customer_type"]
        else _CAT["customer_type"][0],
        "reserved_room_type": "A"
        if "A" in _CAT["reserved_room_type"]
        else _CAT["reserved_room_type"][0],
        "deposit_type": "No Deposit"
        if "No Deposit" in _CAT["deposit_type"]
        else _CAT["deposit_type"][0],
        "is_repeated_guest": False,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "adr": 100.0,
        "parking": 0,
        "special_requests": 0,
    }


def _defaults_tuple() -> tuple[Any, ...]:
    d = _form_defaults()
    return tuple(d[k] for k in INPUT_KEYS)


def _reset_to_defaults() -> tuple[Any, ...]:
    """R5 — clear the form to defaults and reset the result panel."""
    return (
        *_defaults_tuple(),
        "_Run a prediction to see results._",
        "",
        "",
    )


def _on_page_load(request: gr.Request) -> tuple[Any, ...]:
    """R7 — auto-populate the high-risk scenario when ?demo=1 is in the URL."""
    try:
        qs = dict(getattr(request, "query_params", {}) or {})
    except (AttributeError, TypeError):
        qs = {}
    demo_flag = str(qs.get("demo", "")).lower()
    if demo_flag in {"1", "true", "yes", "high"}:
        return _populate_from_example("high_risk")
    return _defaults_tuple()


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

HELP_MARKDOWN = """
## How to read the result

The model returns the **probability** that the booking will be cancelled.
That probability is mapped to a risk band and three policy decisions.

| Risk band     | Probability    | Suggested action                          |
|---------------|----------------|-------------------------------------------|
| **Very Low**  | exactly 0%     | No action — model is highly confident     |
| **Low**       | up to 40%      | Standard flow — monitor                   |
| **Medium**    | 40% – 70%      | Manual review, send reminder              |
| **High**      | 70% – 100%     | Contact guest now, hold a backup room     |

---

## ❓ Why do I sometimes see "<0.01%" (Very Low)?

The isotonic calibrator was fit on real validation data. Below a raw model probability
of about **1.25%** the calibrator returns *exactly* zero — meaning the model is
*confident* the booking will not cancel. That is the expected behaviour, not a bug.

If you'd like to see a non-zero result, open the **Try Examples** tab and click
**High risk**, or change one of the cancel signals listed below.

The model is also strongly **bimodal** — most predictions fall under 10% or above 80%.
Getting a result in the Medium band (40%–70%) is genuinely rare with this dataset.
That is a property of the underlying booking history, not a UI bug.

---

## 🎯 What inputs drive the prediction?

The model learned these patterns from 119 000 historical bookings.

**↑ Strong cancel signals**
- Long lead time (`lead_time > 100` days)
- `deposit_type = "Non Refund"` — counterintuitive but consistent in this dataset
- `market_segment = "Groups"`
- `previous_cancellations > 0`
- Single-adult booking with zero special requests

**↓ Strong stay signals**
- Repeated guest with prior successful stays
- `total_of_special_requests > 0`
- `required_car_parking_spaces > 0`
- Short lead time (booked within a week of arrival)
- `market_segment = "Direct"`

---

## 🛠 Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `0.00%` for many bookings | Calibrator zero-plateau on confident "stay" bookings | Expected — see above |
| `Model artifacts not loaded` | Missing `artifacts/best_model.pkl` | Run `python scripts/train.py` |
| Server returns "Connection refused" | Stale uvicorn or port 8000 in use | `Stop-Process -Name python -Force` then `python demo/start_server.py` |
| `Validation error` on Predict | Required field missing / out of range | Check inputs and try again |

The fastest way to confirm the server is healthy is to open `http://localhost:8000/healthz`
in a browser — a green JSON `{"status":"ok"}` means the model is loaded and ready.

---

## ⚡ Demo shortcuts

- **`?demo=1`** in the URL auto-populates the high-risk scenario on page load.
  Open `http://localhost:8000/ui?demo=1` and the form starts pre-filled — handy for
  walking into a presentation with a non-zero result already on screen.
- **Reset** button next to Predict clears the form back to defaults and the result
  panel — useful between demo scenarios.
- **Try Examples** tab fills the form with high/borderline/low scenarios in one click.
"""


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Hotel Booking Risk") as ui:
        gr.Markdown("# Hotel Booking Risk")
        gr.Markdown(
            "<p class='hero-subtitle'>Calibrated cancellation predictions at the moment "
            "of reservation, with explainable contributing factors.</p>"
        )
        gr.Markdown(_hero_metrics_line())

        d = _form_defaults()

        with gr.Tab("Predict"):
            with gr.Row():
                # ---------- Input column ----------
                with gr.Column(scale=3):
                    gr.Markdown("### Reservation details")

                    hotel = gr.Dropdown(_CAT["hotel"], label="Hotel", value=d["hotel"])
                    arrival_date = gr.DateTime(
                        label="Arrival date",
                        include_time=False,
                        value=d["arrival_date"],
                    )
                    lead_time = gr.Number(
                        label="Lead time (days)",
                        value=d["lead_time"],
                        precision=0,
                        minimum=0,
                        info=(
                            "Days between booking and arrival. "
                            "Long lead times (>100) strongly predict cancellation."
                        ),
                    )

                    with gr.Row():
                        weekend_nights = gr.Number(
                            label="Weekend nights",
                            value=d["weekend_nights"],
                            precision=0,
                            minimum=0,
                        )
                        week_nights = gr.Number(
                            label="Week nights",
                            value=d["week_nights"],
                            precision=0,
                            minimum=0,
                        )

                    with gr.Row():
                        adults = gr.Number(
                            label="Adults",
                            value=d["adults"],
                            precision=0,
                            minimum=1,
                            maximum=20,
                        )
                        children = gr.Number(
                            label="Children",
                            value=d["children"],
                            precision=0,
                            minimum=0,
                            maximum=20,
                        )
                        babies = gr.Number(
                            label="Babies",
                            value=d["babies"],
                            precision=0,
                            minimum=0,
                            maximum=20,
                        )

                    country = gr.Dropdown(
                        _CAT["country"],
                        label="Country (ISO-3)",
                        value=d["country"],
                        allow_custom_value=True,
                    )

                    with gr.Accordion("Booking source", open=False):
                        market_segment = gr.Dropdown(
                            _CAT["market_segment"],
                            label="Market segment",
                            value=d["market_segment"],
                            info=(
                                "Booking channel. 'Groups' is the strongest cancel signal; "
                                "'Direct' is the strongest stay signal."
                            ),
                        )
                        distribution_channel = gr.Dropdown(
                            _CAT["distribution_channel"],
                            label="Distribution channel",
                            value=d["distribution_channel"],
                        )
                        customer_type = gr.Dropdown(
                            _CAT["customer_type"],
                            label="Customer type",
                            value=d["customer_type"],
                        )

                    with gr.Accordion("Room and rate", open=False):
                        reserved_room_type = gr.Dropdown(
                            _CAT["reserved_room_type"],
                            label="Reserved room type",
                            value=d["reserved_room_type"],
                        )
                        meal = gr.Dropdown(_CAT["meal"], label="Meal plan", value=d["meal"])
                        adr = gr.Number(
                            label="Average daily rate (€)",
                            value=d["adr"],
                            minimum=0.0,
                            maximum=float(ADR_MAX_VALID),
                        )
                        special_requests = gr.Number(
                            label="Special requests",
                            value=d["special_requests"],
                            precision=0,
                            minimum=0,
                            info=(
                                "Each special request signals investment in the booking. "
                                ">0 is a strong stay signal."
                            ),
                        )

                    with gr.Accordion("Guest history & deposit", open=False):
                        deposit_type = gr.Dropdown(
                            _CAT["deposit_type"],
                            label="Deposit type",
                            value=d["deposit_type"],
                            info=(
                                "Counterintuitive in this dataset: 'Non Refund' often "
                                "signals HIGH cancel rate, not low."
                            ),
                        )
                        previous_cancellations = gr.Number(
                            label="Previous cancellations",
                            value=d["previous_cancellations"],
                            precision=0,
                            minimum=0,
                            info=(
                                "Prior cancellations by this guest. Even 1 dramatically "
                                "increases risk. Leaving this and 'Previous bookings' both "
                                "at 0 marks the booker as a new guest."
                            ),
                        )
                        previous_bookings_not_canceled = gr.Number(
                            label="Previous bookings (not cancelled)",
                            value=d["previous_bookings_not_canceled"],
                            precision=0,
                            minimum=0,
                            info=(
                                "Prior successful stays. Any value > 0 here OR in "
                                "'Previous cancellations' flags the booker as a "
                                "returning guest in the model input."
                            ),
                        )
                        parking = gr.Number(
                            label="Parking spaces requested",
                            value=d["parking"],
                            precision=0,
                            minimum=0,
                        )

                    with gr.Row():
                        predict_btn = gr.Button(
                            "Predict cancellation risk",
                            variant="primary",
                            size="lg",
                            scale=4,
                        )
                        reset_btn = gr.Button("Reset", variant="secondary", size="sm", scale=1)

                # ---------- Output column ----------
                with gr.Column(scale=2):
                    gr.Markdown("### Risk assessment")
                    headline_out = gr.Markdown(
                        # Skeleton placeholder — same structure the prediction
                        # will fill in, so the panel doesn't feel empty.
                        "<div class='prob-block'>"
                        "<p class='prob-label'>Cancellation probability</p>"
                        "<div class='prob-number' style='color:var(--text-3);'>—%</div>"
                        "<div class='prob-bar'>"
                        "<div class='prob-bar-fill' "
                        "style='width:0%;background:var(--border);'></div>"
                        "</div>"
                        "<div><span class='risk-badge' style='background:var(--surface-2);"
                        "color:var(--text-3);border:1px solid var(--border);'>"
                        "● Awaiting input</span></div>"
                        "</div>",
                        elem_classes=["result-card"],
                    )
                    details_out = gr.Markdown(
                        "<div class='action-block'>"
                        "<p class='action-label'>Recommended action</p>"
                        "<p class='action-text' style='color:var(--text-3);'>"
                        "Fill in the reservation details on the left, then press "
                        "<b>Predict</b> to receive a calibrated risk assessment, "
                        "the top contributing factors, and the recommended next step."
                        "</p></div>"
                    )
                    with gr.Accordion("Technical details (JSON)", open=False):
                        raw_out = gr.Code(label="response", language="json")

            input_components = [
                hotel,
                arrival_date,
                lead_time,
                weekend_nights,
                week_nights,
                adults,
                children,
                babies,
                country,
                meal,
                market_segment,
                distribution_channel,
                customer_type,
                reserved_room_type,
                deposit_type,
                previous_cancellations,
                previous_bookings_not_canceled,
                adr,
                parking,
                special_requests,
            ]
            predict_btn.click(
                fn=_set_predicting_state,
                outputs=[predict_btn, headline_out, details_out, raw_out],
            ).then(
                fn=predict_one,
                inputs=input_components,
                outputs=[headline_out, details_out, raw_out],
            ).then(
                fn=_set_ready_state,
                outputs=predict_btn,
            )

            reset_btn.click(
                fn=_reset_to_defaults,
                outputs=[*input_components, headline_out, details_out, raw_out],
            )

        with gr.Tab("Examples"):
            gr.Markdown("### Pre-loaded scenarios")
            gr.Markdown(
                "Pick a sample guest profile to fill the form on the **Predict** tab, "
                "then switch over and press **Predict**."
            )
            for key, ex in EXAMPLES.items():
                with gr.Row():
                    btn = gr.Button(ex["label"], size="lg")
                    gr.Markdown(ex["hint"])
                    btn.click(
                        fn=lambda k=key: _populate_from_example(k),
                        outputs=input_components,
                    )

        with gr.Tab("Help"):
            gr.Markdown(HELP_MARKDOWN)

        # R7 — on page load, honour the ?demo=1 query param.
        ui.load(fn=_on_page_load, outputs=input_components)

    return ui
