"""Gradio UI for the PH (Philippine resort) sub-study.

Redesigned to mirror the Portugal UI (src/app/ui.py) so the two servers
present a consistent visual identity for a side-by-side defense demo.
The PH UI is intentionally narrower in scope:

  * 10 input fields (vs Portugal's 21) — no country / market_segment /
    customer_type / agent / previous cancellations / parking / meal.
  * 2 threshold policies — max_f1 and high_precision (no cost_sensitive,
    because n_val ≈ 19 is too thin to fit a reliable cost curve).
  * Small-sample caveat strip immediately below the hero metrics — the
    most important UX signal panelists need.
  * Help tab cross-references the 11-notebook PH suite, so a curious
    reader can follow up on any number they see in the UI.

The Gradio submit handler appends every prediction to
``data/predictions/ph_predictions.sqlite`` (and refreshes the CSV) so
Power BI sees activity regardless of whether the call came from HTTP or
from this UI button — same contract as the Portugal UI.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from pydantic import ValidationError

from src.app.ph_schemas import PHBookingRequest
from src.app.ui import BACKGROUND_CSS  # reuse Portugal's design system
from src.config import (
    PH_DATA_PATH,
    PH_PREDICTION_LOG_DB,
    PH_REPORTS_DIR,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
)
from src.serving.inference_ph import (
    explain_ph_prediction,
    get_cached_ph_artifacts,
    predict_ph,
)
from src.serving.prediction_log_ph import export_ph_to_csv, log_ph_prediction

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PH_TRANSFERABILITY_PATH = PH_REPORTS_DIR / "ph_transferability.json"
PH_SHAP_IMPORTANCE_PATH = PH_REPORTS_DIR / "shap_feature_importance.csv"


# ---------------------------------------------------------------------------
# Categorical-choice loading (auto-detected from CSV, with fallbacks)
# ---------------------------------------------------------------------------


def _load_categorical_choices(*column_aliases: str, fallback: list[str]) -> list[str]:
    """Distinct values for a column, scanning test predictions then raw CSV."""
    candidates: list[Path] = [
        PH_REPORTS_DIR / "ph_test_predictions.csv",
        PH_DATA_PATH,
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:  # nosec B112  # pragma: no cover — defensive: skip malformed CSV and try the next alias path
            continue
        for col in column_aliases:
            if col in df.columns:
                values = sorted({str(v).strip() for v in df[col].dropna() if str(v).strip()})
                if values:
                    return values
    return fallback


_ROOM_TYPES = _load_categorical_choices(
    "reserved_room_type",
    "Room_Type",
    fallback=["Standard Room", "De Luxe Room", "Group Room", "Presidential Room"],
)
_DEPOSIT_TYPES = _load_categorical_choices(
    "deposit_type",
    "Deposit_Type",
    fallback=["No Deposit", "Partial", "Non-Refundable"],
)


# ---------------------------------------------------------------------------
# Hero metrics + caveat strip
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_ph_metrics() -> dict[str, Any] | None:
    """Read ph_transferability.json once; return None if missing."""
    if not PH_TRANSFERABILITY_PATH.exists():
        return None
    try:
        return json.loads(PH_TRANSFERABILITY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("ph_metrics_load_failed error=%s", exc)
        return None


def _hero_metrics_line() -> str:
    """Render PH headline metrics as compact KPI chips.

    Mirrors src/app/ui.py::_hero_metrics_line but reads the PH artefact.
    Four chips: ROC-AUC, PR-AUC, F1@max_f1, ECE — all from
    reports/ph/ph_transferability.json.
    """
    data = _load_ph_metrics()
    if data is None:
        return (
            "<p class='kpi-muted'>PH model metrics not available — "
            "run <code>python scripts/train_ph.py</code>.</p>"
        )
    try:
        roc = data["roc_auc_test"]
        pr = data["pr_auc_test"]
        f1 = data["max_f1"]["f1"]
        ece = data.get("ece_test")
        chips: list[tuple[str, str]] = [
            ("ROC-AUC", f"{float(roc):.3f}"),
            ("PR-AUC", f"{float(pr):.3f}"),
            ("F1 @ max_f1", f"{float(f1):.3f}"),
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
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("ph_hero_metrics_render_failed error=%s", exc)
        return "<p class='kpi-muted'>PH metrics shape unexpected — re-run training.</p>"


def _model_lineage_chip() -> str:
    """One-line lineage strip — family, n_train/n_val/n_test, trained at."""
    data = _load_ph_metrics()
    if data is None:
        return ""
    parts = []
    family = (data.get("selected_model_family") or "lightgbm").upper()
    n_train = data.get("n_train", "?")
    n_val = data.get("n_val", "?")
    n_test = data.get("n_test", "?")
    parts.append(f"Champion: <strong>{family}</strong>")
    parts.append(f"Train / Val / Test: {n_train} / {n_val} / {n_test}")
    return "<p class='hero-meta'>" + " &nbsp;·&nbsp; ".join(parts) + "</p>"


CAVEAT_STRIP_HTML = """
<div class='caveat-strip'>
    <strong>Small-sample sub-study —</strong>
    n_test ≈ 20 rows. Bootstrap 95 % CIs on PR-AUC span roughly ±15 percentage
    points. Treat displayed metrics as <em>directional</em>, not as
    production-grade headlines. The Portugal main study at
    <code>http://localhost:8000/ui</code> (119k bookings) is the headline result.
</div>
"""


# Additional PH-only CSS layered on top of the imported Portugal design system.
# Exposed as PH_CSS (BACKGROUND_CSS + PH_EXTRA_CSS) so src/app/ph_main.py can
# pass it to gr.mount_gradio_app() — Gradio 6 prefers css there over the
# deprecated gr.Blocks(css=...) constructor argument.
PH_EXTRA_CSS = """
.hero-meta {
    color: var(--text-2);
    font-size: 13px;
    margin: -16px 0 24px 0;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}
.caveat-strip {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    color: #78350F;
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 13px;
    line-height: 1.5;
    margin: 0 0 28px 0;
}
.caveat-strip strong { color: #92400E; }
.caveat-strip code {
    background: #FEF3C7;
    color: #78350F;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
}
.kpi-muted {
    color: var(--text-2);
    font-size: 13px;
    margin: 0 0 24px 0;
}
"""

PH_CSS = BACKGROUND_CSS + PH_EXTRA_CSS


# ---------------------------------------------------------------------------
# Example scenarios for the "Examples" tab (PH-specific defaults)
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, dict[str, Any]] = {
    "high_risk": {
        "label": "🔴 High cancellation risk",
        "hint": "No Deposit + long lead + zero special requests — the model's worst-case profile.",
        "values": {
            "lead_time": 120,
            "arrival_date": date(2025, 12, 15).isoformat(),
            "weekend_nights": 2,
            "week_nights": 3,
            "adults": 1,
            "children": 0,
            "babies": 0,
            "adr": 4500.0,
            "reserved_room_type": "De Luxe Room",
            "deposit_type": "No Deposit",
            "total_of_special_requests": 0,
        },
    },
    "medium_risk": {
        "label": "🟠 Borderline cancellation risk",
        "hint": "Partial deposit + moderate lead + one special request — near the decision boundary.",
        "values": {
            "lead_time": 45,
            "arrival_date": date(2025, 11, 1).isoformat(),
            "weekend_nights": 1,
            "week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "adr": 3200.0,
            "reserved_room_type": "Standard Room",
            "deposit_type": "Partial",
            "total_of_special_requests": 1,
        },
    },
    "low_risk": {
        "label": "🟢 Low cancellation risk",
        "hint": "Non-Refundable + short lead + multiple special requests — strongly committed booking.",
        "values": {
            "lead_time": 7,
            "arrival_date": date(2025, 10, 10).isoformat(),
            "weekend_nights": 1,
            "week_nights": 1,
            "adults": 2,
            "children": 1,
            "babies": 0,
            "adr": 5000.0,
            "reserved_room_type": "Presidential Room",
            "deposit_type": "Non-Refundable",
            "total_of_special_requests": 3,
        },
    },
}

INPUT_KEYS = (
    "lead_time",
    "arrival_date",
    "weekend_nights",
    "week_nights",
    "adults",
    "children",
    "babies",
    "adr",
    "reserved_room_type",
    "deposit_type",
    "total_of_special_requests",
)


def _form_defaults() -> dict[str, Any]:
    """Sensible defaults populated on first load."""
    return EXAMPLES["medium_risk"]["values"]


# ---------------------------------------------------------------------------
# Result rendering helpers
# ---------------------------------------------------------------------------


def _band_classname(prob: float) -> tuple[str, str]:
    """Return (CSS-class, human-readable band) for a probability."""
    if prob >= RISK_TIER_HIGH_THRESHOLD:
        return "result-bad", "High"
    if prob >= RISK_TIER_MEDIUM_THRESHOLD:
        return "result-warn", "Medium"
    return "result-good", "Low"


def _format_probability(prob: float) -> str:
    """B1 fix from Portugal: surface very-low probs as '<0.01%' not '0.00%'."""
    if prob <= 0.0:
        return "&lt;0.01%"
    return f"{prob * 100:.2f}%"


def _recommended_action(prob: float, band: str) -> str:
    """The action block shown below the big probability."""
    if band == "High":
        text = (
            "Contact the guest within 24 hours to confirm intent. "
            "Consider an overbooking buffer for the arrival date."
        )
    elif band == "Medium":
        text = (
            "Send a gentle reminder one week before arrival. "
            "Monitor cancellation rate in this cohort during the next quarter."
        )
    else:
        text = "Standard handling. No proactive outreach required."
    return (
        "<div class='action-block'>"
        "<p class='action-label'>Recommended action</p>"
        f"<p class='action-text'>{text}</p>"
        "</div>"
    )


def _format_top_features(top: list[dict[str, Any]]) -> str:
    """Mini-bar visualisation of the top SHAP contributors.

    Mirrors src/app/ui.py's factor-row layout but with PH-friendly captions.
    """
    if not top:
        return (
            "<div class='factors-block'>"
            "<p class='factors-title'>Top contributing features</p>"
            "<p style='color:var(--text-2);font-size:13px;margin:0;'>"
            "No feature contributions returned (SHAP unavailable).</p></div>"
        )
    # Normalise bar widths against the largest absolute contribution.
    max_abs = max(abs(float(t.get("contribution", 0.0) or 0.0)) for t in top) or 1.0
    rows: list[str] = []
    for entry in top:
        feature = str(entry.get("feature", "?"))
        value = str(entry.get("value", "?"))
        contrib = float(entry.get("contribution", 0.0) or 0.0)
        width_pct = min(50.0, abs(contrib) / max_abs * 50.0)
        direction = "cancel" if contrib > 0 else "stay"
        bar_html = (
            f"<div class='factor-bar-fill {direction}' "
            f"style='width:{width_pct:.1f}%'></div>"
            "<div class='factor-bar-mid'></div>"
        )
        rows.append(
            "<div class='factor-row'>"
            f"<div class='factor-name'><code>{feature}</code> "
            f"<span style='color:var(--text-2);'>= {value}</span></div>"
            f"<div class='factor-bar'>{bar_html}</div>"
            f"<div class='factor-contrib'>{contrib:+.3f}</div>"
            "</div>"
        )
    return (
        "<div class='factors-block'>"
        "<p class='factors-title'>Top contributing features "
        "(red = pushes toward cancel · green = pushes toward stay)</p>" + "".join(rows) + "</div>"
    )


def _format_policy_grid(prob: float, thr_f1: float, thr_hp: float) -> str:
    """2-cell policy grid (PH has no cost_sensitive policy)."""
    verdict_f1 = "Flag as cancel" if prob >= thr_f1 else "Treat as stay"
    verdict_hp = "Flag (high-confidence)" if prob >= thr_hp else "Treat as stay"
    return (
        "<div class='policy-grid' style='grid-template-columns:1fr 1fr;'>"
        "<div class='policy-cell'>"
        "<p class='policy-name'>Balanced (max F1)</p>"
        f"<p class='policy-decision'>{verdict_f1}</p>"
        f"<p class='policy-thr'>threshold {thr_f1:.2f}</p>"
        "</div>"
        "<div class='policy-cell'>"
        "<p class='policy-name'>High-precision</p>"
        f"<p class='policy-decision'>{verdict_hp}</p>"
        f"<p class='policy-thr'>threshold {thr_hp:.2f}</p>"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Prediction handler
# ---------------------------------------------------------------------------


def predict_one(
    lead_time: float,
    arrival_date_val: Any,
    weekend_nights: float,
    week_nights: float,
    adults: float,
    children: float,
    babies: float,
    adr: float,
    reserved_room_type: str,
    deposit_type: str,
    total_of_special_requests: float,
) -> tuple[str, str, str]:
    """Score one booking and render (headline, details, raw_json) HTML/JSON.

    Same return contract as src/app/ui.py::predict_one so the UI layout
    mirrors Portugal's exactly.
    """
    try:
        artifacts = get_cached_ph_artifacts()
    except FileNotFoundError as exc:
        msg = (
            f"<div class='prob-block'><p class='prob-label'>Error</p>"
            f"<p style='color:var(--bad)'>PH artifacts unavailable: {exc}<br>"
            "Run <code>python scripts/train_ph.py</code> first.</p></div>"
        )
        return msg, "", ""

    # Parse arrival_date — gr.DateTime returns either a datetime, a date, or
    # a string depending on the Gradio version. Handle each.
    arrival: date
    try:
        if isinstance(arrival_date_val, (datetime,)):
            arrival = arrival_date_val.date()
        elif isinstance(arrival_date_val, date):
            arrival = arrival_date_val
        elif isinstance(arrival_date_val, str):
            arrival = date.fromisoformat(arrival_date_val[:10])
        elif arrival_date_val is None:
            arrival = date.today()
        else:
            # gr.DateTime in Gradio 6.x can yield a float (epoch seconds).
            arrival = datetime.fromtimestamp(float(arrival_date_val)).date()
    except (TypeError, ValueError) as exc:
        return (
            f"<div class='prob-block'><p class='prob-label'>Invalid input</p>"
            f"<p style='color:var(--bad)'>Could not parse arrival date: {exc}</p></div>",
            "",
            "",
        )

    try:
        booking = PHBookingRequest(
            lead_time=int(lead_time),
            arrival_date=arrival,
            weekend_nights=int(weekend_nights),
            week_nights=int(week_nights),
            adults=int(adults),
            children=int(children),
            babies=int(babies),
            adr=float(adr),
            reserved_room_type=str(reserved_room_type),
            deposit_type=str(deposit_type),
            total_of_special_requests=int(total_of_special_requests),
        )
    except (ValueError, TypeError, ValidationError) as exc:
        return (
            f"<div class='prob-block'><p class='prob-label'>Invalid input</p>"
            f"<p style='color:var(--bad)'>{exc}</p></div>",
            "",
            "",
        )

    prob, feature_df = predict_ph(booking.to_inference_dict(), artifacts)
    thresholds = artifacts.thresholds or {}
    thr_f1 = float(thresholds.get("max_f1", {}).get("threshold", 0.5))
    thr_hp = float(thresholds.get("high_precision", {}).get("threshold", 0.9))
    band_css, band = _band_classname(prob)
    pct_str = _format_probability(prob)

    bar_pct = max(0.0, min(100.0, prob * 100.0))
    headline = (
        "<div class='prob-block'>"
        "<p class='prob-label'>Cancellation Risk</p>"
        f"<div class='prob-number {band_css}'>{pct_str}</div>"
        "<div class='prob-bar'>"
        f"<div class='prob-bar-fill {band_css}' style='width:{bar_pct:.1f}%'></div>"
        "</div>"
        f"<div><span class='risk-badge {band_css}'>● {band} risk</span></div>"
        "</div>"
    )

    top = explain_ph_prediction(feature_df, artifacts, top_n=5)
    parts = [
        _recommended_action(prob, band),
        _format_top_features(top),
        _format_policy_grid(prob, thr_f1, thr_hp),
    ]

    raw: dict[str, Any] = {
        "probability": prob,
        "probability_display": pct_str.replace("&lt;", "<"),
        "risk_band": band,
        "thresholds": {"max_f1": thr_f1, "high_precision": thr_hp},
        "decisions": {
            "balanced": "Flag as cancel" if prob >= thr_f1 else "Treat as stay",
            "high_precision": "Flag (high-confidence)" if prob >= thr_hp else "Treat as stay",
        },
        "top_features": top,
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "lightgbm",
        "dataset": "Punta Villa Resort PH 2022-2025 (193 rows)",
    }

    # Log to SQLite + refresh CSV (mirrors Portugal pattern). The Gradio
    # path doesn't go through /predict, so we have to log explicitly.
    try:
        if prob >= RISK_TIER_HIGH_THRESHOLD:
            risk_tier = "high"
        elif prob >= RISK_TIER_MEDIUM_THRESHOLD:
            risk_tier = "medium"
        else:
            risk_tier = "low"
        response_for_log = {
            "probability": prob,
            "label_max_f1": int(prob >= thr_f1),
            "label_high_precision": int(prob >= thr_hp),
            "risk_tier": risk_tier,
            "threshold_max_f1": thr_f1,
            "threshold_high_precision": thr_hp,
            "alerts": [],
            "top_features": top,
        }
        log_ph_prediction(booking.model_dump(mode="json"), response_for_log, PH_PREDICTION_LOG_DB)
        export_ph_to_csv()
    except Exception:
        logger.exception("ph_ui_prediction_log_failed (non-fatal)")

    return headline, "\n".join(parts), json.dumps(raw, indent=2, default=str)


def _populate_from_example(key: str) -> tuple[Any, ...]:
    v = EXAMPLES[key]["values"]
    return tuple(v[k] for k in INPUT_KEYS)


def _reset_to_defaults() -> tuple[Any, ...]:
    """Reset all fields + result panels to the medium-risk default scenario."""
    d = _form_defaults()
    return (
        *(d[k] for k in INPUT_KEYS),
        _SKELETON_HEADLINE,
        _SKELETON_DETAILS,
        "",
    )


_SKELETON_HEADLINE = (
    "<div class='prob-block'>"
    "<p class='prob-label'>Cancellation Risk</p>"
    "<div class='prob-number' style='color:var(--text-3);'>—%</div>"
    "<div class='prob-bar'>"
    "<div class='prob-bar-fill' style='width:0%;background:var(--border);'></div>"
    "</div>"
    "<div><span class='risk-badge' style='background:var(--surface-2);"
    "color:var(--text-3);border:1px solid var(--border);'>"
    "● Awaiting input</span></div>"
    "</div>"
)
_SKELETON_DETAILS = (
    "<div class='action-block'>"
    "<p class='action-label'>Recommended action</p>"
    "<p class='action-text' style='color:var(--text-3);'>"
    "Fill in the booking details on the left, then press "
    "<b>Predict</b> to receive a calibrated risk assessment, "
    "the top contributing features, and the policy decisions."
    "</p></div>"
)


# ---------------------------------------------------------------------------
# Help-tab markdown
# ---------------------------------------------------------------------------


def _top_shap_features_markdown(n: int = 8) -> str:
    """Render the top-N SHAP features as a markdown bullet list."""
    if not PH_SHAP_IMPORTANCE_PATH.exists():
        return "*(SHAP importance not available — retrain via `scripts/train_ph.py`.)*"
    try:
        df = pd.read_csv(PH_SHAP_IMPORTANCE_PATH).head(n)
        return "\n".join(
            f"{pos + 1}. **`{row['feature']}`** — mean(|SHAP|) = {row['mean_abs_shap']:.3f}"
            for pos, (_, row) in enumerate(df.reset_index(drop=True).iterrows())
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("shap_importance_load_failed error=%s", exc)
        return "*(SHAP importance unreadable.)*"


def _help_markdown() -> str:
    """Compose the Help tab markdown at UI build time."""
    data = _load_ph_metrics() or {}
    family = (data.get("selected_model_family") or "lightgbm").upper()
    n_train = data.get("n_train", "?")
    n_val = data.get("n_val", "?")
    n_test = data.get("n_test", "?")

    def _fmt(value: Any, spec: str = ".3f") -> str:
        if value is None:
            return "—"
        try:
            return format(float(value), spec)
        except (TypeError, ValueError):
            return "—"

    roc_str = _fmt(data.get("roc_auc_test"))
    pr_str = _fmt(data.get("pr_auc_test"))
    ece_str = _fmt(data.get("ece_test"))
    return f"""\
## 📊 What this model is

Calibrated cancellation classifier trained on the **real Punta Villa Resort PMS
export** (193 bookings, 2022-2025). The model is the same family as Portugal's
champion (LightGBM) so SHAP rankings and calibration are directly comparable
across studies.

| | |
|---|---|
| Champion family | **{family}** |
| Train / Val / Test rows | {n_train} / {n_val} / {n_test} |
| Test ROC-AUC | {roc_str} |
| Test PR-AUC | {pr_str} |
| Test ECE (calibration gap) | {ece_str} |
| Threshold policies | `max_f1`, `high_precision` (no `cost_sensitive` — n_val too small) |

## 📐 How to read the result

- **Cancellation Risk**: the percentage the model gives you is an
  honest estimate of how likely this booking is to be cancelled. We
  apply a calibration step during training so that, for example, a
  "30 %" result means about 30 out of 100 similar bookings really do
  cancel — the percentage matches reality.
- **Risk tier**: Low (< {RISK_TIER_MEDIUM_THRESHOLD:.0%}), Medium
  ({RISK_TIER_MEDIUM_THRESHOLD:.0%}-{RISK_TIER_HIGH_THRESHOLD:.0%}),
  High (≥ {RISK_TIER_HIGH_THRESHOLD:.0%}). The tier is the
  recommended action level for the front-desk team.
- **Top contributing factors**: each red bar shows a booking detail
  that pushed this prediction toward "cancel"; each green bar shows a
  detail that pushed it toward "will arrive". The longer the bar, the
  bigger the influence on this particular booking.
- **Threshold policies**: each policy is a different decision rule the
  hotel can adopt. *Balanced (max F1)* flags bookings that are
  more-likely-than-not to cancel; *High-precision* only flags
  bookings where the model is very confident, so almost every flag is
  a real cancellation but many true cancellations slip through.

## ⚠ Small-sample caveats (read before quoting any number)

The Philippine sub-study trains on **154 bookings** and tests on
**20 bookings**. At this sample size:

- The numbers in the hero chips at the top are best treated as
  **directional indicators** rather than headline figures. If we
  re-ran the study on a different random selection of 20 test
  bookings, the displayed PR-AUC could shift by roughly ±15-30
  percentage points just from the small sample.
- A single mis-classified test booking moves the recall by about 33
  percentage points, so per-cell performance numbers should always
  be reported with their uncertainty range.
- The "balanced" decision cut-off (currently 0.190) was learned on
  only 19 validation bookings, so the *exact* cut-off is statistically
  noisy. The **risk tiers** (Low / Medium / High) are more stable
  because they use the calibrated probability directly rather than a
  single cut-off, and are the recommended operational tool for the
  Philippine deployment until more bookings are collected.
- The Portugal main study (119,210 bookings, port 8000) is the
  headline result of the thesis. The Philippine study is a
  **transferability probe** — it shows the same methodology produces
  a working model on a real Philippine resort dataset with only a
  fraction of the data.

## 🎯 Top booking details the model relies on

(Global ranking of the booking details the model treats as most
influential, across the test set. Top 8 shown.)

{_top_shap_features_markdown(8)}

**Deposit policy is the #1 driver on the Philippine model** — exactly
the same pattern observed in Portugal, where deposit policy was also a
top driver. This is one of the strongest cross-dataset findings in
the study: the same single booking detail dominates cancellation
prediction across two different countries.

## 🔗 Cross-reference to the PH notebook suite

The 11-notebook PH suite under `notebooks/ph/` documents the full
methodology. Notable cross-references:

- **`notebooks/ph/02_modeling.ipynb`** — ROC/PR curves, calibration,
  threshold sweep, confusion matrix
- **`notebooks/ph/05_explainability.ipynb`** — SHAP feature importance,
  beeswarm, per-row explanations, Portugal vs PH SHAP comparison
- **`notebooks/ph/07_model_selection.ipynb`** — 3-way model comparison
  (LightGBM/XGBoost/GradientBoosting) with bootstrap CIs
- **`notebooks/ph/08_model_monitoring.ipynb`** — runnable monitoring
  template; reads the live SQLite log every prediction made through
  this UI feeds
- **`notebooks/ph/11_transferability.ipynb`** — pre-flight diagnostic
  outcome + real-data findings + defense framing

## 🛠 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Hero chips say "metrics not available" | Missing `reports/ph/ph_transferability.json` | Run `python scripts/train_ph.py` |
| `PH artifacts unavailable` error in result panel | Missing `artifacts/ph/ph_model.pkl` | Run `python scripts/train_ph.py` |
| Many predictions return ≈ 0 % | Calibrator zero-plateau on confident "stay" bookings | Expected — UI shows `<0.01 %` for these |
| Server returns "Connection refused" | Port 8001 in use or uvicorn not started | `python demo/start_server_ph.py` |
| `Validation error` on Predict | Required field missing / out of range | Check inputs (adults ≥ 1, ADR < 100 000) |

A green JSON at `http://localhost:8001/healthz` confirms the server
is healthy.
"""


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


def build_ph_ui() -> gr.Blocks:
    """Construct the PH Gradio Blocks UI (mirrors Portugal's structure).

    CSS is NOT attached here — Gradio 6 expects CSS at mount time
    (``gr.mount_gradio_app(app, ui, css=PH_CSS)``) or at launch time
    (``ui.launch(css=PH_CSS)``). The PH_CSS module-level constant
    contains BACKGROUND_CSS + PH_EXTRA_CSS for both call sites.
    """
    with gr.Blocks(
        title="PH Cancellation Risk — Real Punta Villa Resort Sub-Study",
    ) as ui:
        gr.Markdown("# PH Cancellation Risk")
        gr.Markdown(
            "<p class='hero-subtitle'>Calibrated cancellation predictions on the "
            "real Punta Villa Resort PMS export (193 bookings, 2022-2025), with "
            "explainable contributing features.</p>"
        )
        gr.Markdown(_hero_metrics_line())
        lineage = _model_lineage_chip()
        if lineage:
            gr.Markdown(lineage)
        gr.HTML(CAVEAT_STRIP_HTML)

        d = _form_defaults()

        with gr.Tab("Predict"):
            with gr.Row():
                # ---------- Input column ----------
                with gr.Column(scale=3):
                    gr.Markdown("### Booking details")

                    arrival_date_in = gr.DateTime(
                        label="Arrival date",
                        include_time=False,
                        value=d["arrival_date"],
                    )
                    lead_time_in = gr.Number(
                        label="Lead time (days)",
                        value=d["lead_time"],
                        precision=0,
                        minimum=0,
                        info=(
                            "Days between booking and arrival. Long lead times "
                            "increase cancellation risk in this dataset."
                        ),
                    )

                    with gr.Row():
                        weekend_nights_in = gr.Number(
                            label="Weekend nights",
                            value=d["weekend_nights"],
                            precision=0,
                            minimum=0,
                        )
                        week_nights_in = gr.Number(
                            label="Week nights",
                            value=d["week_nights"],
                            precision=0,
                            minimum=0,
                        )

                    with gr.Row():
                        adults_in = gr.Number(
                            label="Adults",
                            value=d["adults"],
                            precision=0,
                            minimum=1,
                            maximum=10,
                        )
                        children_in = gr.Number(
                            label="Children",
                            value=d["children"],
                            precision=0,
                            minimum=0,
                            maximum=10,
                        )
                        babies_in = gr.Number(
                            label="Babies",
                            value=d["babies"],
                            precision=0,
                            minimum=0,
                            maximum=10,
                        )

                    with gr.Accordion("Room and rate", open=True):
                        reserved_room_type_in = gr.Dropdown(
                            _ROOM_TYPES,
                            label="Reserved room type",
                            value=d["reserved_room_type"],
                        )
                        adr_in = gr.Number(
                            label="Average daily rate (PHP)",
                            value=d["adr"],
                            minimum=0.0,
                            maximum=100_000.0,
                        )

                    with gr.Accordion("Deposit & special requests", open=True):
                        deposit_type_in = gr.Dropdown(
                            _DEPOSIT_TYPES,
                            label="Deposit policy",
                            value=d["deposit_type"],
                            info=(
                                "The #1 SHAP feature on this PH model. "
                                "'No Deposit' strongly predicts cancellation; "
                                "'Non-Refundable' / 'Partial' predict stay."
                            ),
                        )
                        total_of_special_requests_in = gr.Number(
                            label="Special requests",
                            value=d["total_of_special_requests"],
                            precision=0,
                            minimum=0,
                            maximum=10,
                            info=(
                                "Each special request signals investment in the "
                                "booking. > 0 is a stay signal."
                            ),
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
                        _SKELETON_HEADLINE,
                        elem_classes=["result-card"],
                    )
                    details_out = gr.Markdown(_SKELETON_DETAILS)
                    with gr.Accordion("Technical details (JSON)", open=False):
                        raw_out = gr.Code(label="response", language="json")

            input_components = [
                lead_time_in,
                arrival_date_in,
                weekend_nights_in,
                week_nights_in,
                adults_in,
                children_in,
                babies_in,
                adr_in,
                reserved_room_type_in,
                deposit_type_in,
                total_of_special_requests_in,
            ]
            predict_btn.click(
                fn=predict_one,
                inputs=input_components,
                outputs=[headline_out, details_out, raw_out],
            )
            reset_btn.click(
                fn=_reset_to_defaults,
                outputs=[*input_components, headline_out, details_out, raw_out],
            )

        with gr.Tab("Examples"):
            gr.Markdown("### Pre-loaded scenarios")
            gr.Markdown(
                "Pick a booking profile to fill the form on the **Predict** tab, "
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
            gr.Markdown(_help_markdown())

    return ui


# Keep the legacy export name for any external caller that imports it.
build_ui = build_ph_ui


def main() -> None:  # pragma: no cover — manual launch only
    """Launch the PH UI standalone on port 7861."""
    logging.basicConfig(level=logging.INFO)
    demo = build_ph_ui()
    demo.launch(
        server_name="0.0.0.0",  # nosec B104 — intentional: demo server must be reachable from host network for defense panel walk-through
        server_port=7861,
        share=False,
        css=PH_CSS,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
