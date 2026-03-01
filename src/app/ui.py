"""Gradio UI for hotel booking cancellation prediction."""

from __future__ import annotations

import csv
import html
import json
import logging
import os
import re
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import gradio as gr
import pandas as pd
from pydantic import ValidationError

from src.app.schemas import BookingRequest
from src.config import (
    ARTIFACTS_DIR,
    BOOKING_TIME_FEATURES,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
)
from src.serving.inference import ModelArtifacts, load_artifacts, predict_proba
from src.utils.thresholds import resolve_thresholds

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGED_PATH = PROJECT_ROOT / ".gradio" / "flagged" / "predictions.csv"
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"

_ARTIFACTS: ModelArtifacts | None = None
_ARTIFACTS_LOCK = threading.Lock()
_GLOBAL_DRIVER_LINES: list[str] | None = None
_GLOBAL_DRIVER_MODEL_ID: str | None = None
_GLOBAL_DRIVER_LOCK = threading.Lock()


def _get_artifacts() -> ModelArtifacts:
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS
    with _ARTIFACTS_LOCK:
        if _ARTIFACTS is None:
            _ARTIFACTS = load_artifacts()
    return _ARTIFACTS


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

RISK_BANDS = [
    (RISK_TIER_MEDIUM_THRESHOLD, "Low"),
    (RISK_TIER_HIGH_THRESHOLD, "Medium"),
    (1.01, "High"),
]
RISK_TONES = {
    "Low": "safe",
    "Medium": "watch",
    "High": "danger",
    "Unavailable": "neutral",
}


def _load_country_choices() -> list[str]:
    if DATA_PATH.exists():
        try:
            series = pd.read_csv(DATA_PATH, usecols=["country"])["country"]
            countries = sorted({str(c).strip() for c in series.dropna() if str(c).strip()})
        except Exception:
            countries = []
    else:
        countries = []

    if "UNKNOWN" not in countries:
        countries.insert(0, "UNKNOWN")
    return countries


COUNTRY_CHOICES = _load_country_choices()
COUNTRY_OPTIONAL_CHOICES = [""] + COUNTRY_CHOICES

REQUIRED_FIELD_ORDER = [
    "hotel",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "lead_time",
    "arrival_date",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adr",
    "deposit_type",
]
REQUIRED_FIELD_LABELS = {
    "hotel": "Hotel",
    "market_segment": "Market segment",
    "distribution_channel": "Distribution channel",
    "customer_type": "Customer type",
    "lead_time": "Lead time",
    "arrival_date": "Arrival date",
    "stays_in_weekend_nights": "Weekend nights",
    "stays_in_week_nights": "Week nights",
    "adr": "ADR",
    "deposit_type": "Deposit type",
}


def _default_arrival_date() -> datetime:
    base = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    return base + timedelta(days=30)


def _risk_bucket(prob: float) -> str:
    for cutoff, label in RISK_BANDS:
        if prob < cutoff:
            return label
    return "High"


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _derive_metrics(values: Dict[str, Any]) -> tuple[float, int, float]:
    weekend = _to_float(values.get("stays_in_weekend_nights")) or 0.0
    week = _to_float(values.get("stays_in_week_nights")) or 0.0
    adults = _to_int(values.get("adults"), default=0)
    children = _to_int(values.get("children"), default=0)
    babies = _to_int(values.get("babies"), default=0)
    adr = _to_float(values.get("adr")) or 0.0

    total_nights = max(0.0, weekend + week)
    party_size = max(0, adults + children + babies)
    adr_per_person = float(adr / party_size) if party_size > 0 else 0.0
    return total_nights, party_size, adr_per_person


def _validate_required(values: Dict[str, Any]) -> tuple[int, int, list[str], list[str]]:
    missing: list[str] = []
    errors: list[str] = []
    completed = 0
    total = len(REQUIRED_FIELD_ORDER)

    for field in REQUIRED_FIELD_ORDER:
        raw = values.get(field)
        if field in {"lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adr"}:
            num = _to_float(raw)
            if num is None:
                missing.append(REQUIRED_FIELD_LABELS[field])
                continue
            if field == "adr" and num <= 0:
                errors.append("ADR must be greater than 0.")
                continue
            if field != "adr" and num < 0:
                errors.append(f"{REQUIRED_FIELD_LABELS[field]} must be 0 or higher.")
                continue
            completed += 1
            continue

        if field == "arrival_date":
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                missing.append(REQUIRED_FIELD_LABELS[field])
                continue
            try:
                if isinstance(raw, datetime):
                    arr_date = raw.date()
                elif hasattr(raw, "date"):
                    arr_date = raw
                else:
                    arr_date = datetime.fromisoformat(str(raw)).date()
                today = datetime.now(timezone.utc).date()
                if arr_date < today:
                    errors.append("Arrival date must be today or in the future.")
                    continue
            except (TypeError, ValueError):
                errors.append("Invalid arrival date format.")
                continue
            completed += 1
            continue

        if raw is None or (isinstance(raw, str) and not raw.strip()):
            missing.append(REQUIRED_FIELD_LABELS[field])
            continue
        completed += 1

    total_nights, party_size, _ = _derive_metrics(values)
    if total_nights < 0:
        errors.append("Total nights must be 0 or higher.")
    if party_size < 1:
        errors.append("Party size must be at least 1 (adults + children + babies).")

    adults_val = _to_int(values.get("adults"), default=0)
    if adults_val < 1:
        errors.append("Number of adults must be at least 1.")

    return completed, total, missing, errors


def _required_status_markdown(
    completed: int,
    total: int,
    missing: list[str],
    errors: list[str],
) -> str:
    if not missing and not errors:
        return f"### Required Fields Complete: {completed}/{total}\nAll required fields are valid."

    lines = [f"### Required Fields Complete: {completed}/{total}"]
    if missing:
        lines.append("Missing fields:")
        lines.extend([f"- {item}" for item in missing])
    if errors:
        lines.append("Validation issues:")
        lines.extend([f"- {item}" for item in errors])
    return "\n".join(lines)


def _ready_summary(missing: list[str], errors: list[str]) -> str:
    if not missing and not errors:
        return (
            "### Ready to score\n"
            "All required fields are valid. Click **Predict** for a decision-ready output."
        )

    lines = [
        "### Ready to score",
        "Complete required fields to enable prediction.",
    ]
    if missing:
        lines.append("Missing checklist:")
        lines.extend([f"- [ ] {item}" for item in missing])
    if errors:
        lines.append("Validation checklist:")
        lines.extend([f"- [ ] {item}" for item in errors])
    return "\n".join(lines)


def _risk_drivers(record: Dict[str, Any], prob: float) -> list[str]:
    drivers: list[str] = []
    lead_time = _to_float(record.get("lead_time")) or 0.0
    deposit_type = str(record.get("deposit_type") or "").strip().lower()
    repeated = bool(record.get("is_repeated_guest", 0))
    specials = _to_int(record.get("total_of_special_requests"), default=0)
    parking = _to_int(record.get("required_car_parking_spaces"), default=0)
    market_segment = str(record.get("market_segment") or "").strip()
    prev_cxl = _to_int(record.get("previous_cancellations"), default=0)

    if deposit_type == "non refund":
        drivers.append("Non-Refund deposit — in booking data this deposit type correlates with higher dropout rates.")
    if lead_time >= 120:
        drivers.append(f"Booked {int(lead_time)} days in advance — very long lead times increase dropout risk.")
    if not repeated:
        drivers.append("First-time guest — no prior booking history to indicate loyalty.")
    if specials == 0:
        drivers.append("No special requests made — lower engagement signal.")
    if parking == 0:
        drivers.append("No parking needed — slightly lower commitment signal.")
    if market_segment in {"Groups", "Online TA"}:
        drivers.append(f"{market_segment} bookings tend to have higher cancellation rates in our data.")
    if prev_cxl > 0:
        drivers.append(f"The guest has {prev_cxl} previous cancellation(s) on record.")

    if not drivers:
        drivers.append("No strong warning signal was triggered for this booking.")
    return drivers[:5]


def _artifact_model_id(artifacts: ModelArtifacts) -> str:
    metadata = artifacts.metadata or {}
    lineage_sha = (
        metadata.get("lineage", {})
        .get("artifacts", {})
        .get("bundle_sha256")
    )
    if isinstance(lineage_sha, str) and lineage_sha:
        return lineage_sha
    model_type = metadata.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type
    return type(artifacts.model).__name__


def _format_model_feature_name(name: str) -> str:
    if name.startswith("categorical__"):
        raw = name.split("categorical__", 1)[1]
        head, sep, tail = raw.partition("_")
        if sep and head in BOOKING_TIME_FEATURES:
            return f"`{head}` = `{tail}`"
        return f"`{raw}` (categorical)"
    if name.startswith("numeric__"):
        raw = name.split("numeric__", 1)[1]
        return f"`{raw}`"
    return f"`{name}`"


def _load_global_model_drivers(artifacts: ModelArtifacts, top_k: int = 5) -> list[str]:
    """Load overall top drivers from thesis SHAP artifact and map them to feature names."""
    global _GLOBAL_DRIVER_LINES, _GLOBAL_DRIVER_MODEL_ID
    model_id = _artifact_model_id(artifacts)
    with _GLOBAL_DRIVER_LOCK:
        if _GLOBAL_DRIVER_LINES is not None and _GLOBAL_DRIVER_MODEL_ID == model_id:
            return list(_GLOBAL_DRIVER_LINES)

        shap_path = PROJECT_ROOT / "reports" / "thesis" / "shap_feature_importance.csv"
        if not shap_path.exists():
            _GLOBAL_DRIVER_LINES = []
            _GLOBAL_DRIVER_MODEL_ID = model_id
            return []

        lines: list[str] = []
        try:
            shap_df = pd.read_csv(shap_path)
            if shap_df.empty or "feature_index" not in shap_df.columns:
                _GLOBAL_DRIVER_LINES = []
                _GLOBAL_DRIVER_MODEL_ID = model_id
                return []

            top = shap_df.head(max(1, int(top_k))).copy()
            feature_names: list[str] = []
            preprocessor = None
            if artifacts.is_pipeline and hasattr(artifacts.model, "named_steps"):
                preprocessor = artifacts.model.named_steps.get("preprocessor") or artifacts.model.named_steps.get(
                    "preprocess"
                )
            elif artifacts.preprocessor is not None:
                preprocessor = artifacts.preprocessor

            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                feature_names = list(preprocessor.get_feature_names_out())

            for rank, row in enumerate(top.itertuples(index=False), start=1):
                idx = int(getattr(row, "feature_index"))
                importance = float(getattr(row, "mean_abs_shap", 0.0))
                if feature_names and 0 <= idx < len(feature_names):
                    label = _format_model_feature_name(feature_names[idx])
                else:
                    label = f"`feature_{idx}`"
                lines.append(f"{rank}. {label} (mean |SHAP| `{importance:.3f}`)")
        except Exception:
            logger.exception("Failed to load global SHAP drivers from %s", shap_path)
            lines = []

        _GLOBAL_DRIVER_LINES = lines
        _GLOBAL_DRIVER_MODEL_ID = model_id
        return list(lines)


def _intervention_suggestions(
    record: Dict[str, Any],
    risk_label: str,
    prob: float,
    thr_f1: float,
    thr_hp: float,
) -> list[str]:
    """Generate concrete, booking-level action suggestions for operations."""
    lead_time = _to_float(record.get("lead_time")) or 0.0
    deposit_type = str(record.get("deposit_type") or "").strip().lower()
    repeated = bool(record.get("is_repeated_guest", 0))
    specials = _to_int(record.get("total_of_special_requests"), default=0)
    prev_cxl = _to_int(record.get("previous_cancellations"), default=0)

    suggestions: list[str] = []
    if risk_label == "High":
        suggestions.append("Call or message the guest now to confirm they are still coming.")
        suggestions.append("Hold a small room buffer for this arrival date in case this booking falls through.")
    elif risk_label == "Medium":
        suggestions.append("Send automated reminders at 72 h and 24 h before arrival.")
        suggestions.append("Check room availability — review this booking manually if inventory is tight.")
    else:
        suggestions.append("No action needed — continue with the standard booking flow.")

    if lead_time >= 120:
        suggestions.append("Set up a reconfirmation schedule: contact the guest at 90, 30 and 7 days before arrival.")
    if deposit_type in {"no deposit", "refundable"} and prob >= thr_f1:
        suggestions.append("Request a deposit or card pre-authorisation to strengthen the guest's commitment.")
    if prev_cxl > 0:
        suggestions.append("Call or email this guest personally — they have cancelled a booking before.")
    if specials == 0:
        suggestions.append("Ask the guest if they have any special requests — this increases booking engagement.")
    if not repeated and prob >= thr_f1:
        suggestions.append("Consider a small loyalty incentive (room upgrade or F&B voucher) to build commitment.")
    if prob >= thr_hp:
        suggestions.append("Flag this booking for daily monitoring in your reservations system.")

    seen: set[str] = set()
    deduped = [s for s in suggestions if not (s in seen or seen.add(s))]
    return deduped[:6]


def _risk_meter_html(prob: float | None, *, label: str, note: str) -> str:
    tone = RISK_TONES.get(label, "neutral")
    medium_pct = max(0.0, min(100.0, RISK_TIER_MEDIUM_THRESHOLD * 100.0))
    high_pct = max(0.0, min(100.0, RISK_TIER_HIGH_THRESHOLD * 100.0))
    if prob is None:
        return f"""
<div class="risk-card risk-{tone} risk-idle">
  <div class="risk-topline">
    <span class="risk-pill">{label}</span>
    <span class="risk-percent risk-percent-idle">&#x2014;</span>
  </div>
  <div class="risk-track with-markers">
    <span class="risk-marker medium" style="left:{medium_pct:.1f}%"></span>
    <span class="risk-marker high" style="left:{high_pct:.1f}%"></span>
  </div>
  <div class="risk-threshold-labels">
    <span>Medium @{RISK_TIER_MEDIUM_THRESHOLD:.2f}</span>
    <span>High @{RISK_TIER_HIGH_THRESHOLD:.2f}</span>
  </div>
  <p class="risk-note">{note}</p>
</div>
"""
    pct = max(0.0, min(100.0, prob * 100.0))
    return f"""
<div class="risk-card risk-{tone}">
  <div class="risk-topline">
    <span class="risk-pill">{label}</span>
    <span class="risk-percent">{pct:.1f}%</span>
  </div>
  <div class="risk-track with-markers">
    <span class="risk-marker medium" style="left:{medium_pct:.1f}%"></span>
    <span class="risk-marker high" style="left:{high_pct:.1f}%"></span>
    <div class="risk-fill" style="width:{pct:.1f}%"></div>
    <span class="risk-dot" style="left:{pct:.1f}%"></span>
  </div>
  <div class="risk-threshold-labels">
    <span>Medium @{RISK_TIER_MEDIUM_THRESHOLD:.2f}</span>
    <span>High @{RISK_TIER_HIGH_THRESHOLD:.2f}</span>
  </div>
  <p class="risk-note">{note}</p>
</div>
"""


def _idle_summary() -> str:
    return _ready_summary([], [])


def _idle_decision_notes() -> str:
    priority_label, priority_tone = _priority_action_for_risk("Unavailable")
    return _decision_explanation_card(
        risk_label="Unavailable",
        priority_label=priority_label,
        priority_tone=priority_tone,
        priority_instruction=_ops_instruction_for_risk("Unavailable"),
        headline="How to read this result",
        subheadline="After prediction, this card will explain the risk in plain language.",
        why_items=[
            "You will see why this booking looks risky.",
            "The explanation combines model-learned patterns and booking details.",
        ],
        action_items=[
            "You will get clear next steps for your team.",
            "Use the JSON section only if you need technical details.",
        ],
        model_items=[
            "Decision thresholds are chosen from validation data.",
            "Suggestions are decision support, not hard business rules.",
        ],
        policy_alignment=None,
    )


def _loading_summary() -> str:
    return "### Running prediction\nCalculating calibrated probability and decision thresholds..."


def _loading_decision_notes() -> str:
    priority_label, priority_tone = _priority_action_for_risk("Unavailable")
    return _decision_explanation_card(
        risk_label="Unavailable",
        priority_label=priority_label,
        priority_tone=priority_tone,
        priority_instruction=_ops_instruction_for_risk("Unavailable"),
        headline="Preparing explanation",
        subheadline="Calculating policy decisions and recommended actions...",
        why_items=["Scoring this booking now."],
        action_items=["Suggestions will appear after prediction completes."],
        model_items=["Global model drivers are loading."],
        policy_alignment=None,
    )


def _idle_risk_card() -> str:
    return _risk_meter_html(
        None,
        label="Unavailable",
        note="Prediction risk meter will appear here.",
    )


def _html_bullets(items: list[str], *, fallback: str) -> str:
    if not items:
        return f"<li>{html.escape(fallback)}</li>"
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def _top_n_dedup(items: list[str], n: int = 3) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= n:
            break
    return out


def _plain_global_driver(line: str) -> str:
    cleaned = re.sub(r"^\s*\d+\.\s*", "", line).strip()
    cleaned = re.sub(r"\s*\(mean \|SHAP\|\s*`?[0-9.]+`?\)\s*$", "", cleaned).strip()
    cleaned = cleaned.replace("`", "")
    if cleaned.startswith("feature_"):
        return f"Model signal: {cleaned.replace('_', ' ')}"
    if "=" in cleaned:
        return f"Model often uses: {cleaned}"
    return f"Model often uses: {cleaned}"


def _decision_explanation_card(
    *,
    risk_label: str,
    priority_label: str,
    priority_tone: str,
    priority_instruction: str,
    headline: str,
    subheadline: str,
    why_items: list[str],
    action_items: list[str],
    model_items: list[str],
    policy_alignment: str | None,
    borderline: bool = False,
) -> str:
    tone = RISK_TONES.get(risk_label, "neutral")
    chips: list[str] = []
    if policy_alignment == "Agree":
        chips.append('<span class="explain-chip">Both assessments agree</span>')
    elif policy_alignment == "Disagree":
        chips.append(
            '<span class="explain-chip">Assessments differ — strict filter did not trigger</span>'
        )
    if borderline:
        chips.append('<span class="explain-chip">&#9888; Near decision boundary</span>')
    chips_html = "".join(chips) if chips else '<span class="explain-chip">Awaiting score</span>'

    return f"""
<div class="explain-card explain-{tone}">
  <div class="explain-priority-row">
    <span class="explain-priority-title">Recommended action</span>
    <div class="explain-priority-actions">
      <span class="explain-priority-pill explain-priority-{html.escape(priority_tone)}">{html.escape(priority_label)}</span>
      <button
        class="explain-copy-btn"
        type="button"
        data-copy-text="{html.escape(priority_instruction, quote=True)}"
        onclick="(function(btn) {{ var toast = btn.nextElementSibling; var showToast = function(msg) {{ if (!toast) return; toast.textContent = msg; toast.classList.add('show'); if (toast._hideTimer) {{ clearTimeout(toast._hideTimer); }} toast._hideTimer = setTimeout(function() {{ toast.classList.remove('show'); }}, 1400); }}; if (navigator.clipboard && navigator.clipboard.writeText) {{ navigator.clipboard.writeText(btn.dataset.copyText).then(function() {{ showToast('Copied'); }}, function() {{ showToast('Copy failed'); }}); }} else {{ showToast('Clipboard blocked'); }} }})(this);"
      >Copy instruction</button>
      <span class="explain-copy-toast" role="status" aria-live="polite"></span>
    </div>
  </div>
  <div class="explain-head">
    <div>
      <div class="explain-title">{html.escape(headline)}</div>
      <div class="explain-subtitle">{html.escape(subheadline)}</div>
    </div>
    <span class="explain-badge">{html.escape(risk_label)} Risk</span>
  </div>
  <div class="explain-chips">{chips_html}</div>
  <div class="explain-divider"></div>
  <div class="explain-grid">
    <section class="explain-section">
      <h4 class="explain-h explain-h-why"><span class="explain-ico">?</span>Why it was flagged</h4>
      <ul>{_html_bullets(why_items, fallback="No strong warning signals were detected.")}</ul>
    </section>
    <section class="explain-section">
      <h4 class="explain-h explain-h-action"><span class="explain-ico">!</span>Recommended actions</h4>
      <ul>{_html_bullets(action_items, fallback="Continue with normal booking flow.")}</ul>
    </section>
    <section class="explain-section">
      <h4 class="explain-h explain-h-model"><span class="explain-ico">i</span>What the model looks for</h4>
      <ul>{_html_bullets(model_items, fallback="Model context is not available.")}</ul>
    </section>
  </div>
</div>
"""


def _priority_action_for_risk(risk_label: str) -> tuple[str, str]:
    if risk_label == "High":
        return "Act Now", "danger"
    if risk_label == "Medium":
        return "Review Soon", "watch"
    if risk_label == "Low":
        return "Monitor", "safe"
    return "Waiting", "neutral"


def _ops_instruction_for_risk(risk_label: str) -> str:
    if risk_label == "High":
        return "Act Now: Contact guest now and keep a small room buffer."
    if risk_label == "Medium":
        return "Review Soon: Send 72h and 24h reminders; review manually if inventory is tight."
    if risk_label == "Low":
        return "Monitor: Keep normal flow and watch for booking changes."
    return "Waiting: Complete required fields and run Predict."


def _verdict_badge(label: str) -> str:
    css_class = "verdict-cancel" if label == "Likely to cancel" else "verdict-safe"
    return f'<span class="verdict-badge {css_class}">{html.escape(label)}</span>'


def _format_prediction_output(
    prob: float,
    thr_f1: float,
    thr_hp: float,
    timestamp_utc: str,
    model_utc: str | None,
    record: Dict[str, Any],
    artifacts: ModelArtifacts,
) -> tuple[str, str, str, str]:
    risk_label = _risk_bucket(prob)
    pct = prob * 100.0
    label_f1 = "Likely to cancel" if prob >= thr_f1 else "Low concern"
    label_hp = "Likely to cancel" if prob >= thr_hp else "Low concern"
    borderline = abs(prob - thr_f1) <= 0.05

    # Revenue at risk from record values
    _adr = float(record.get("adr") or 0)
    _nights = max(
        0.0,
        float(record.get("stays_in_weekend_nights") or 0)
        + float(record.get("stays_in_week_nights") or 0),
    )
    rev_at_risk = _adr * _nights
    rev_line = f"- Revenue at risk: **€{rev_at_risk:,.0f}**\n" if rev_at_risk > 0 else ""

    # Timestamp: drop seconds, keep "UTC" suffix
    scored_ts = timestamp_utc[:16] if len(timestamp_utc) >= 16 else timestamp_utc

    borderline_line = (
        '\n<div class="borderline-banner">'
        "&#9888; <strong>Borderline</strong> \u2014 This booking is near the decision boundary. "
        "Small changes could flip the verdict."
        "</div>\n"
        if borderline
        else ""
    )
    summary = (
        "### Prediction result\n"
        f"**Cancellation risk: {pct:.1f}% — {risk_label} risk**\n\n"
        f"- Standard verdict: {_verdict_badge(label_f1)}\n"
        f"- High-confidence verdict: {_verdict_badge(label_hp)}\n"
        f"{rev_line}"
        f"- Scored: {scored_ts} UTC"
        f"{borderline_line}"
    )
    meter_note = f"Standard: {label_f1} · High-confidence: {label_hp}"
    if borderline:
        meter_note += " · ⚠ Near boundary"
    risk_html = _risk_meter_html(prob, label=risk_label, note=meter_note)

    if risk_label == "High":
        action = "Trigger retention playbook now and prioritise outreach."
    elif risk_label == "Medium":
        action = "Queue for manual review and monitor booking changes."
    else:
        action = "No intervention needed — continue standard guest journey."

    policy_alignment = "Agree" if label_f1 == label_hp else "Disagree"
    global_drivers = _load_global_model_drivers(artifacts, top_k=5)
    booking_drivers = _risk_drivers(record, prob)
    suggestions = _intervention_suggestions(record, risk_label, prob, thr_f1, thr_hp)
    plain_model_items = [_plain_global_driver(item) for item in global_drivers]
    why_items = _top_n_dedup(booking_drivers, n=3)
    action_items = _top_n_dedup([action] + suggestions, n=3)
    model_items = _top_n_dedup(
        plain_model_items + ["Decision rules are calibrated on historical booking data."],
        n=3,
    )
    priority_label, priority_tone = _priority_action_for_risk(risk_label)
    global_driver_lines = (
        "\n".join([f"- {item}" for item in global_drivers])
        if global_drivers
        else "- SHAP summary artifact not available."
    )
    booking_driver_lines = "\n".join([f"- {item}" for item in booking_drivers])
    suggestion_lines = "\n".join([f"- {item}" for item in suggestions])
    export_hint = (
        f"`risk_band={risk_label}`, `probability={pct:.1f}%`, "
        f"`decision_standard={label_f1}`, `decision_high_confidence={label_hp}`"
    )
    decision_notes = _decision_explanation_card(
        risk_label=risk_label,
        priority_label=priority_label,
        priority_tone=priority_tone,
        priority_instruction=_ops_instruction_for_risk(risk_label),
        headline="Why this booking was flagged",
        subheadline="Key risk signals detected and what your team should do.",
        why_items=why_items,
        action_items=action_items,
        model_items=model_items,
        policy_alignment=policy_alignment,
        borderline=borderline,
    )

    details = {
        "timestamp_utc": timestamp_utc,
        "model_utc": model_utc,
        "probability": prob,
        "risk_percent": round(pct, 1),
        "risk_label": risk_label,
        "thresholds": {"max_f1": thr_f1, "high_precision": thr_hp},
        "decisions": {"max_f1": label_f1, "high_precision": label_hp},
        "policies_disagree": label_f1 != label_hp,
        "global_model_drivers": global_drivers,
        "booking_drivers": booking_drivers,
        "suggestions": suggestions,
        "global_driver_lines_markdown": global_driver_lines,
        "booking_driver_lines_markdown": booking_driver_lines,
        "suggestion_lines_markdown": suggestion_lines,
        "export_hint": export_hint,
        "input_record": record,
    }
    details_json = json.dumps(details, indent=2, sort_keys=True)
    return summary, details_json, risk_html, decision_notes


def _error_output(message: str, exc: Exception | None = None) -> tuple[str, str, str, str]:
    details = {"status": "error", "message": message}
    if exc is not None:
        details["exception"] = repr(exc)
        details["traceback"] = traceback.format_exc()
    summary = f"### Unable to score\n{message}"
    return (
        summary,
        json.dumps(details, indent=2, sort_keys=True),
        _idle_risk_card(),
        _idle_decision_notes(),
    )


def _format_validation_error(exc: ValidationError) -> str:
    messages: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "invalid value")
        messages.append(f"{loc}: {msg}" if loc else str(msg))
    return "; ".join(messages)


def _validated_record(values: Dict[str, Any]) -> Dict[str, Any]:
    payload = {field: values.get(field) for field in BookingRequest.model_fields}
    optional_text_fields = {
        "country",
        "meal",
        "reserved_room_type",
        "deposit_type",
        "agent",
        "company",
        "customer_type",
        "market_segment",
        "distribution_channel",
        "hotel",
    }
    for field in optional_text_fields:
        val = payload.get(field)
        if isinstance(val, str) and not val.strip():
            payload[field] = None

    if isinstance(payload.get("is_repeated_guest"), bool):
        payload["is_repeated_guest"] = int(payload["is_repeated_guest"])

    request = BookingRequest.model_validate(payload)
    return request.model_dump(exclude={"arrival_date"})


def _format_utc(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _model_timestamp_utc() -> str | None:
    try:
        model_path = ARTIFACTS_DIR / "best_model.pkl"
        if model_path.exists():
            return _format_utc(datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc))
    except OSError:
        return None
    return None


@contextmanager
def _exclusive_file_lock(handle):
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_log_row(record: Dict[str, Any], ordered_cols: list[str]) -> None:
    LOGGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {col: record.get(col, "") for col in ordered_cols}

    with LOGGED_PATH.open("a+", encoding="utf-8", newline="") as handle:
        with _exclusive_file_lock(handle):
            handle.seek(0, 2)
            write_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=ordered_cols, quoting=csv.QUOTE_ALL)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()


def _missing_fields_message(missing: list[str], errors: list[str]) -> str:
    items = [*missing, *errors]
    if not items:
        return ""
    return "Missing/invalid required fields: " + "; ".join(items)


def _form_feedback(
    values: Dict[str, Any],
) -> tuple[str, str, str, str, bool, str, float, int, float]:
    completed, total, missing, errors = _validate_required(values)
    total_nights, party_size, adr_per_person = _derive_metrics(values)
    ready = not missing and not errors

    required_status = _required_status_markdown(completed, total, missing, errors)
    summary = _ready_summary(missing, errors)
    decision_notes = _idle_decision_notes()
    risk_note = (
        "All required fields valid. Score to view risk and recommended action."
        if ready
        else "Complete required fields to unlock prediction."
    )
    risk_html = _risk_meter_html(None, label="Unavailable", note=risk_note)
    missing_message = _missing_fields_message(missing, errors)
    return (
        required_status,
        summary,
        decision_notes,
        risk_html,
        ready,
        missing_message,
        float(total_nights),
        int(party_size),
        float(adr_per_person),
    )


def _predict_output(values: Dict[str, Any]) -> tuple[str, str, str, str, Dict[str, Any] | None]:
    _, _, _, _, ready, missing_msg, _, _, _ = _form_feedback(values)
    if not ready:
        summary, details_json, risk_html, decision_notes = _error_output(
            missing_msg or "Input incomplete."
        )
        return summary, details_json, risk_html, decision_notes, None

    try:
        record = _validated_record(values)
    except ValidationError as exc:
        summary, details_json, risk_html, decision_notes = _error_output(
            "Input error: " + _format_validation_error(exc)
        )
        return summary, details_json, risk_html, decision_notes, None

    try:
        artifacts = _get_artifacts()
        prob = float(predict_proba(record, artifacts)[0][0])
        resolved, _, _, _ = resolve_thresholds(artifacts.thresholds or {})
        thr_f1 = resolved["max_f1"]
        thr_hp = resolved["high_precision"]
        timestamp_utc = _format_utc(datetime.now(timezone.utc))
        model_ts = _model_timestamp_utc() or "unknown"
        summary, details_json, risk_html, decision_notes = _format_prediction_output(
            prob, thr_f1, thr_hp, timestamp_utc, model_ts, record, artifacts
        )
        return summary, details_json, risk_html, decision_notes, record
    except Exception as exc:
        logger.exception("Prediction failed")
        summary, details_json, risk_html, decision_notes = _error_output(
            f"Prediction failed: {exc}", exc
        )
        return summary, details_json, risk_html, decision_notes, None


def _log_case(record: Dict[str, Any], label: str, flagged: bool = False) -> None:
    try:
        log_record = dict(record)
        try:
            year = int(log_record["arrival_date_year"])
            month = MONTHS.index(str(log_record["arrival_date_month"])) + 1
            day = int(log_record["arrival_date_day_of_month"])
            log_record["arrival_date"] = datetime(year, month, day).date().isoformat()
        except (KeyError, ValueError, TypeError):
            log_record["arrival_date"] = ""

        log_record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        log_record["prediction"] = label
        log_record["flagged"] = int(flagged)

        columns = ["timestamp_utc", "prediction", "flagged", "arrival_date"]
        columns.extend(BOOKING_TIME_FEATURES)
        seen: set[str] = set()
        ordered_cols = [c for c in columns if not (c in seen or seen.add(c))]
        _append_log_row(log_record, ordered_cols)
    except Exception:
        logger.exception("Failed to log prediction case to %s", LOGGED_PATH)


BACKGROUND_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-0: #0a1220;
  --bg-1: #101f35;
  --ink: #f4f7ff;
  --ink-muted: #c5d3ef;
  --surface: rgba(8, 16, 30, 0.78);
  --surface-soft: rgba(255, 255, 255, 0.06);
  --line: rgba(255, 255, 255, 0.16);
  --accent: #57d9a3;
  --accent-2: #3ca0ff;
  --warn: #ffb74d;
  --danger: #ff5f6d;
}

html, body { height: 100%; margin: 0; }

#app-bg {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(1000px 700px at 15% 10%, rgba(60, 160, 255, 0.22), transparent 65%),
    radial-gradient(900px 650px at 88% 78%, rgba(87, 217, 163, 0.20), transparent 64%),
    linear-gradient(150deg, var(--bg-0), var(--bg-1));
}

#app-noise {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0.15;
  background-image:
    linear-gradient(to right, rgba(255, 255, 255, 0.08) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255, 255, 255, 0.08) 1px, transparent 1px);
  background-size: 26px 26px;
}

.gradio-container {
  font-family: "Space Grotesk", "Segoe UI", sans-serif !important;
  background: transparent !important;
  position: relative;
  z-index: 1;
  min-height: 100vh;
  padding: 26px 16px 34px;
  color: var(--ink) !important;
}

.gradio-container > .wrap,
.gradio-container .main {
  max-width: 1220px;
  margin: 0 auto;
  background: var(--surface) !important;
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: 0 26px 60px rgba(0, 0, 0, 0.38);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 18px;
}

.hero-shell {
  padding: 8px 8px 16px;
}

.hero-title {
  margin: 0 0 6px;
  font-size: clamp(1.45rem, 2.4vw, 2rem);
  line-height: 1.12;
  letter-spacing: 0.2px;
}

.hero-subtitle {
  margin: 0 0 12px;
  color: var(--ink-muted);
  max-width: 760px;
}

.hero-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.hero-chip {
  font-size: 0.83rem;
  padding: 7px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: var(--ink);
}

.input-panel .block,
.result-panel .block,
.gradio-container .gr-box {
  border-radius: 16px !important;
  border: 1px solid var(--line) !important;
  background: var(--surface-soft) !important;
}

.layout-row {
  gap: 16px;
}

.result-panel .block {
  background: rgba(5, 17, 38, 0.84) !important;
  border-color: rgba(87, 217, 163, 0.45) !important;
  box-shadow: 0 16px 30px rgba(0, 0, 0, 0.28);
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label {
  color: var(--ink) !important;
}

.gradio-container label {
  font-weight: 600 !important;
  letter-spacing: 0.1px;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--ink) !important;
  background: rgba(0, 0, 0, 0.25) !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 12px !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--accent-2) !important;
  box-shadow: 0 0 0 3px rgba(60, 160, 255, 0.22) !important;
}

.gradio-container button.primary {
  background: linear-gradient(95deg, var(--accent), var(--accent-2)) !important;
  border: none !important;
  color: #031024 !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: 0.03em !important;
  border-radius: 12px !important;
  transition: filter 160ms ease, transform 120ms ease, box-shadow 160ms ease !important;
  box-shadow: 0 4px 14px rgba(87, 217, 163, 0.28) !important;
}
.gradio-container button.primary:hover:not(:disabled) {
  filter: brightness(1.10) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(87, 217, 163, 0.42) !important;
}
.gradio-container button.primary:active:not(:disabled) {
  transform: translateY(0px) !important;
  filter: brightness(0.96) !important;
  box-shadow: 0 2px 8px rgba(87, 217, 163, 0.18) !important;
}
.gradio-container button.primary:disabled {
  background: rgba(87, 217, 163, 0.20) !important;
  color: rgba(3, 16, 36, 0.50) !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
}

.gradio-container button:not(.primary) {
  border-color: var(--line) !important;
  color: var(--ink) !important;
}

/* Keep radio option labels readable in all states (fixes disappearing text on selection) */
.gradio-container [role="radiogroup"] button {
  background: rgba(0, 0, 0, 0.26) !important;
  border: 1px solid var(--line) !important;
  color: var(--ink) !important;
}

.gradio-container [role="radiogroup"] button *,
.gradio-container [role="radiogroup"] label {
  color: inherit !important;
}

.gradio-container [role="radiogroup"] button:hover {
  border-color: rgba(87, 217, 163, 0.55) !important;
}

.gradio-container [role="radiogroup"] button[aria-checked="true"],
.gradio-container [role="radiogroup"] button[aria-pressed="true"],
.gradio-container [role="radiogroup"] button.selected {
  background: linear-gradient(95deg, var(--accent), var(--accent-2)) !important;
  border-color: transparent !important;
  color: #031024 !important;
}

.gradio-container [role="radiogroup"] button[aria-checked="true"] *,
.gradio-container [role="radiogroup"] button[aria-pressed="true"] *,
.gradio-container [role="radiogroup"] button.selected * {
  color: #031024 !important;
}

/* Checkbox controls */
.gradio-container input[type="checkbox"] {
  accent-color: var(--accent) !important;
  width: 1.05rem;
  height: 1.05rem;
  cursor: pointer;
}

.gradio-container [role="checkbox"] {
  border: 1px solid var(--line) !important;
  background: rgba(0, 0, 0, 0.26) !important;
  color: var(--ink) !important;
  border-radius: 10px !important;
}

.gradio-container [role="checkbox"][aria-checked="true"] {
  border-color: transparent !important;
  background: linear-gradient(95deg, var(--accent), var(--accent-2)) !important;
  color: #031024 !important;
}

.gradio-container [role="checkbox"][aria-checked="true"] * {
  color: #031024 !important;
}

#result-summary {
  background: rgba(0, 0, 0, 0.22);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px 16px 14px;
}
#result-summary h3 {
  margin: 0 0 10px;
  font-size: clamp(1.1rem, 1.8vw, 1.3rem);
  font-weight: 800;
  letter-spacing: -0.2px;
  color: var(--ink) !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.10);
  padding-bottom: 8px;
}
#result-summary p,
#result-summary li { font-size: 0.92rem; line-height: 1.55; }

#required-status {
  margin-top: 8px;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
}

#decision-notes {
  margin-top: 10px;
  padding: 0;
}

.explain-card {
  margin-top: 10px;
  background: rgba(3, 14, 30, 0.86);
  border: 1px solid rgba(87, 217, 163, 0.35);
  border-radius: 14px;
  padding: 14px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.explain-priority-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.05);
}

.explain-priority-actions {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.explain-priority-title {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--ink-muted);
  font-weight: 700;
}

.explain-priority-pill {
  font-size: 0.82rem;
  font-weight: 800;
  padding: 5px 11px;
  border-radius: 999px;
  border: 1px solid var(--line);
}

.explain-priority-danger {
  color: #ffd9dd;
  border-color: rgba(255, 95, 109, 0.82);
  background: rgba(255, 95, 109, 0.18);
}

.explain-priority-watch {
  color: #ffe7bf;
  border-color: rgba(255, 183, 77, 0.82);
  background: rgba(255, 183, 77, 0.16);
}

.explain-priority-safe {
  color: #d5ffef;
  border-color: rgba(87, 217, 163, 0.82);
  background: rgba(87, 217, 163, 0.18);
}

.explain-priority-neutral {
  color: #dce8ff;
  border-color: rgba(167, 189, 227, 0.78);
  background: rgba(167, 189, 227, 0.16);
}

.explain-copy-btn {
  border: 1px solid var(--line);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  color: var(--ink);
  font-size: 0.72rem;
  font-weight: 600;
  padding: 4px 10px;
  cursor: pointer;
  transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
}

.explain-copy-btn:hover {
  border-color: rgba(143, 240, 200, 0.72);
  background: rgba(143, 240, 200, 0.12);
}

.explain-copy-btn:focus-visible {
  outline: 2px solid rgba(143, 240, 200, 0.6);
  outline-offset: 2px;
}

.explain-copy-btn:active {
  transform: translateY(1px);
}

.explain-copy-toast {
  min-width: 66px;
  font-size: 0.72rem;
  font-weight: 600;
  color: #b9f7df;
  opacity: 0;
  transform: translateY(2px);
  transition: opacity 140ms ease, transform 140ms ease;
  pointer-events: none;
}

.explain-copy-toast.show {
  opacity: 1;
  transform: translateY(0);
}

.explain-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.explain-title {
  font-size: 1.02rem;
  font-weight: 700;
  color: var(--ink);
}

.explain-subtitle {
  margin-top: 4px;
  font-size: 0.86rem;
  color: var(--ink-muted);
}

.explain-badge {
  white-space: nowrap;
  font-size: 0.78rem;
  font-weight: 700;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.08);
}

.explain-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.explain-chip {
  font-size: 0.76rem;
  padding: 4px 9px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.08);
  color: var(--ink);
}

.explain-divider {
  margin: 12px 0;
  height: 1px;
  background: linear-gradient(90deg, rgba(87, 217, 163, 0.05), rgba(87, 217, 163, 0.55), rgba(87, 217, 163, 0.05));
}

.explain-grid {
  display: grid;
  gap: 10px;
}

.explain-section {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  background: rgba(255, 255, 255, 0.04);
  border-left-width: 4px;
}

.explain-h {
  margin: 0 0 8px;
  font-size: 0.9rem;
  color: var(--ink);
  display: flex;
  align-items: center;
  gap: 8px;
}

.explain-ico {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.08);
  color: var(--ink);
}

.explain-h-why { color: #ffd78a; }
.explain-h-action { color: #8ff0c8; }
.explain-h-model { color: #9ec7ff; }

.explain-h-why .explain-ico {
  border-color: rgba(255, 215, 138, 0.8);
  background: rgba(255, 215, 138, 0.14);
}

.explain-h-action .explain-ico {
  border-color: rgba(143, 240, 200, 0.85);
  background: rgba(143, 240, 200, 0.14);
}

.explain-h-model .explain-ico {
  border-color: rgba(158, 199, 255, 0.85);
  background: rgba(158, 199, 255, 0.14);
}

.explain-section:nth-child(1) { border-left-color: rgba(255, 215, 138, 0.85); }
.explain-section:nth-child(2) { border-left-color: rgba(143, 240, 200, 0.9); }
.explain-section:nth-child(3) { border-left-color: rgba(158, 199, 255, 0.9); }

.explain-section:nth-child(1) li { color: #ffe9be; }
.explain-section:nth-child(2) li { color: #c7f8e7; }
.explain-section:nth-child(3) li { color: #d4e5ff; }

.explain-section li:last-child {
  margin-bottom: 0;
}

.explain-section ul {
  margin: 0;
  padding-left: 18px;
}

.explain-section li {
  margin: 0 0 6px;
  color: var(--ink-muted);
  line-height: 1.35;
}

.explain-safe .explain-badge { border-color: rgba(87, 217, 163, 0.65); }
.explain-watch .explain-badge { border-color: rgba(255, 183, 77, 0.72); }
.explain-danger .explain-badge { border-color: rgba(255, 95, 109, 0.75); }
.explain-neutral .explain-badge { border-color: rgba(167, 189, 227, 0.7); }

@media (min-width: 900px) {
  .explain-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

#result-details textarea {
  font-family: "IBM Plex Mono", Consolas, monospace !important;
  font-size: 0.82rem !important;
  line-height: 1.4 !important;
}

.risk-card {
  border-radius: 14px;
  border: 1px solid var(--line);
  padding: 12px;
  background: rgba(0, 0, 0, 0.26);
}

.risk-topline {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
}

.risk-pill {
  font-size: 0.8rem;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.risk-percent {
  font-size: clamp(2.4rem, 4vw, 3.2rem);
  font-weight: 800;
  letter-spacing: -0.5px;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.risk-safe  .risk-percent { color: #7fe2ba; }
.risk-watch .risk-percent { color: #ffca6b; }
.risk-danger .risk-percent { color: #ff7b7b; }

.risk-track {
  position: relative;
  margin-top: 10px;
  border-radius: 999px;
  height: 18px;
  background: rgba(255, 255, 255, 0.10);
  overflow: visible;
}

.risk-marker {
  position: absolute;
  top: -3px;
  width: 2px;
  height: 18px;
  background: rgba(255, 255, 255, 0.75);
  z-index: 3;
}

.risk-marker.high {
  background: rgba(255, 95, 109, 0.92);
}

.risk-threshold-labels {
  margin-top: 8px;
  display: flex;
  justify-content: space-between;
  color: var(--ink-muted);
  font-size: 0.76rem;
}

.risk-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 420ms ease;
}

.risk-note {
  margin: 10px 0 0;
  color: var(--ink-muted);
  font-size: 0.86rem;
  line-height: 1.38;
}

.risk-safe .risk-fill { background: linear-gradient(90deg, #56cc9d, #7fe2ba); }
.risk-watch .risk-fill { background: linear-gradient(90deg, #ffca6b, #ff9f3f); }
.risk-danger .risk-fill { background: linear-gradient(90deg, #ff7b7b, #ff4e6d); }
.risk-neutral .risk-fill { background: linear-gradient(90deg, #8ba7d8, #a7bde3); }

@media (min-width: 980px) {
  #result-col {
    position: sticky;
    top: 18px;
    align-self: flex-start;
    max-height: calc(100vh - 36px);
    overflow: auto;
  }
}

@media (max-width: 720px) {
  .gradio-container {
    padding: 12px 8px 20px;
  }
  .gradio-container > .wrap,
  .gradio-container .main {
    border-radius: 14px;
    padding: 10px;
  }
  .hero-title {
    font-size: 1.3rem;
  }
}

/* ── Risk meter: position indicator dot ─────────────────────────────── */
.risk-dot {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 2.5px solid rgba(255, 255, 255, 0.9);
  z-index: 5;
  transition: left 420ms ease;
  pointer-events: none;
}
.risk-safe  .risk-dot { background: #7fe2ba; box-shadow: 0 0 8px rgba(87, 217, 163, 0.6); }
.risk-watch .risk-dot { background: #ffca6b; box-shadow: 0 0 8px rgba(255, 183, 77, 0.6); }
.risk-danger .risk-dot { background: #ff7b7b; box-shadow: 0 0 8px rgba(255, 95, 109, 0.6); }
.risk-neutral .risk-dot { background: #8ba7d8; }

/* ── Risk meter: idle / awaiting state ──────────────────────────────── */
.risk-idle .risk-percent-idle {
  font-size: clamp(2.4rem, 4vw, 3.2rem);
  font-weight: 800;
  color: var(--ink-muted);
  opacity: 0.45;
}
.risk-idle .risk-track {
  overflow: hidden;
}
.risk-idle .risk-track::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: 999px;
  background: linear-gradient(
    90deg,
    transparent 20%,
    rgba(255, 255, 255, 0.07) 50%,
    transparent 80%
  );
  background-size: 200% 100%;
  animation: risk-shimmer 2.4s ease-in-out infinite;
}
@keyframes risk-shimmer {
  0%   { background-position: -100% 0; }
  100% { background-position: 220% 0; }
}

/* ── Borderline warning banner ─────────────────────────────────────── */
.borderline-banner {
  margin: 10px 0 0;
  padding: 9px 14px;
  border-radius: 10px;
  border: 1px solid rgba(255, 183, 77, 0.65);
  background: rgba(255, 183, 77, 0.12);
  color: #ffe7bf;
  font-size: 0.9rem;
  font-weight: 600;
  line-height: 1.45;
}

/* ── Verdict pill badges ─────────────────────────────────────────────── */
.verdict-badge {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 800;
  padding: 3px 10px;
  border-radius: 999px;
  vertical-align: middle;
  letter-spacing: 0.02em;
}
.verdict-cancel {
  background: rgba(255, 95, 109, 0.18);
  border: 1px solid rgba(255, 95, 109, 0.72);
  color: #ffd9dd;
}
.verdict-safe {
  background: rgba(87, 217, 163, 0.15);
  border: 1px solid rgba(87, 217, 163, 0.65);
  color: #d5ffef;
}

/* ── Explain card: border & divider colour per risk tier ─────────────── */
.explain-safe   { border-color: rgba(87,  217, 163, 0.45); }
.explain-watch  {
  border-color: rgba(255, 183, 77, 0.50);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 0 1px rgba(255,183,77,0.10);
}
.explain-danger {
  border-color: rgba(255, 95, 109, 0.55);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 0 1px rgba(255,95,109,0.12);
}
.explain-watch  .explain-divider {
  background: linear-gradient(
    90deg, rgba(255,183,77,0.05), rgba(255,183,77,0.55), rgba(255,183,77,0.05)
  );
}
.explain-danger .explain-divider {
  background: linear-gradient(
    90deg, rgba(255,95,109,0.05), rgba(255,95,109,0.55), rgba(255,95,109,0.05)
  );
}
"""


def build_ui() -> gr.Blocks:
    defaults: dict[str, Any] = {
        "hotel": "City Hotel",
        "lead_time": 30,
        "arrival_date": _default_arrival_date(),
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": False,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "deposit_type": "No Deposit",
        "agent": "UNKNOWN",
        "company": "UNKNOWN",
        "customer_type": "Transient",
        "adr": 100.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
    }
    (
        initial_required_status,
        initial_summary,
        initial_decision_notes,
        initial_risk_html,
        initial_ready,
        _,
        initial_total_nights,
        initial_party_size,
        initial_adr_per_person,
    ) = _form_feedback(defaults)

    with gr.Blocks(title="Hotel Booking Cancellation Prediction") as demo:
        gr.HTML('<div id="app-bg"></div><div id="app-noise"></div>')
        gr.HTML(
            f"""
<section class="hero-shell">
  <h1 class="hero-title">Hotel Booking Cancellation Predictor</h1>
  <p class="hero-subtitle">
    Enter booking details to see how likely this guest is to cancel.
    The model flags high-risk bookings so your team can act before arrival.
  </p>
  <div class="hero-chips">
    <span class="hero-chip">Uses booking details only — no post-check-in data needed</span>
    <span class="hero-chip">Medium risk: {int(RISK_TIER_MEDIUM_THRESHOLD * 100)}%+ chance of cancellation</span>
    <span class="hero-chip">High risk: {int(RISK_TIER_HIGH_THRESHOLD * 100)}%+ chance of cancellation</span>
  </div>
</section>
"""
        )
        with gr.Row(elem_classes=["layout-row"]):
            with gr.Column(scale=5, elem_classes=["input-panel"]):
                with gr.Accordion("1) Booking details (required)", open=True):
                    with gr.Row():
                        hotel = gr.Dropdown(
                            label="Hotel (required)",
                            choices=["City Hotel", "Resort Hotel"],
                            value=defaults["hotel"],
                            allow_custom_value=True,
                        )
                        customer_type = gr.Dropdown(
                            label="Guest type (required)",
                            choices=["Transient", "Transient-Party", "Contract", "Group"],
                            value=defaults["customer_type"],
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        market_segment = gr.Dropdown(
                            label="Market segment (required)",
                            info="How the booking was sourced — e.g., Online TA = online travel agency.",
                            choices=[
                                "Online TA",
                                "Offline TA/TO",
                                "Direct",
                                "Groups",
                                "Corporate",
                                "Complementary",
                                "Aviation",
                                "Undefined",
                            ],
                            value=defaults["market_segment"],
                            allow_custom_value=True,
                        )
                        distribution_channel = gr.Dropdown(
                            label="Booking platform (required)",
                            info="The platform or channel used to make this booking.",
                            choices=["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
                            value=defaults["distribution_channel"],
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        lead_time = gr.Number(
                            label="Days until arrival (required)",
                            value=defaults["lead_time"],
                            minimum=0,
                            maximum=5000,
                            step=1,
                            info="Number of days between today and the guest's arrival date.",
                        )
                        arrival_date = gr.DateTime(
                            label="Arrival date (required)",
                            value=defaults["arrival_date"],
                            include_time=False,
                            type="datetime",
                        )
                    with gr.Row():
                        stays_in_weekend_nights = gr.Number(
                            label="Weekend nights — Sat & Sun (required)",
                            value=defaults["stays_in_weekend_nights"],
                            minimum=0,
                            maximum=60,
                            step=1,
                        )
                        stays_in_week_nights = gr.Number(
                            label="Week nights — Mon to Fri (required)",
                            value=defaults["stays_in_week_nights"],
                            minimum=0,
                            maximum=120,
                            step=1,
                        )
                    with gr.Row():
                        adults = gr.Number(
                            label="Adults",
                            value=defaults["adults"],
                            minimum=1,
                            maximum=20,
                            step=1,
                        )
                        children = gr.Number(
                            label="Children",
                            value=defaults["children"],
                            minimum=0,
                            maximum=20,
                            step=1,
                        )
                        babies = gr.Number(
                            label="Babies",
                            value=defaults["babies"],
                            minimum=0,
                            maximum=20,
                            step=1,
                        )
                    with gr.Row():
                        adr = gr.Number(
                            label="Room rate — ADR (required)",
                            value=defaults["adr"],
                            minimum=0.01,
                            maximum=2000,
                            step=1,
                            info="Average Daily Rate in EUR. The nightly room price.",
                        )
                        deposit_type = gr.Dropdown(
                            label="Payment / deposit type (required)",
                            choices=["No Deposit", "Non Refund", "Refundable"],
                            value=defaults["deposit_type"],
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        total_nights_view = gr.Number(
                            label="Total nights (derived)",
                            value=initial_total_nights,
                            interactive=False,
                        )
                        party_size_view = gr.Number(
                            label="Party size (derived)",
                            value=initial_party_size,
                            interactive=False,
                        )
                        adr_per_person_view = gr.Number(
                            label="ADR per person (derived)",
                            value=round(initial_adr_per_person, 2),
                            interactive=False,
                        )

                with gr.Accordion("2) Guest preferences (optional — improves accuracy)", open=False):
                    with gr.Row():
                        country = gr.Dropdown(
                            label="Country (optional)",
                            choices=COUNTRY_OPTIONAL_CHOICES,
                            value=defaults["country"],
                            allow_custom_value=True,
                        )
                        meal = gr.Dropdown(
                            label="Meal plan (optional)",
                            choices=["BB", "HB", "FB", "SC", "Undefined"],
                            value=defaults["meal"],
                            allow_custom_value=True,
                            info="BB = Bed & Breakfast  ·  HB = Half Board  ·  FB = Full Board  ·  SC = Self-Catering",
                        )
                    with gr.Row():
                        reserved_room_type = gr.Dropdown(
                            label="Room category (optional)",
                            choices=["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
                            value=defaults["reserved_room_type"],
                            allow_custom_value=True,
                        )
                        total_of_special_requests = gr.Number(
                            label="No. of special requests (optional)",
                            value=defaults["total_of_special_requests"],
                            minimum=0,
                            maximum=10,
                            step=1,
                            info="Number of requests made at booking (e.g., cot, late check-in, floor preference). More = stronger engagement.",
                        )
                        required_car_parking_spaces = gr.Number(
                            label="Parking spaces needed (optional)",
                            value=defaults["required_car_parking_spaces"],
                            minimum=0,
                            maximum=10,
                            step=1,
                        )
                    with gr.Accordion("Advanced identifiers (optional)", open=False):
                        agent = gr.Textbox(label="Agent (optional)", value=defaults["agent"])
                        company = gr.Textbox(label="Company (optional)", value=defaults["company"])

                with gr.Accordion("3) Guest history (optional)", open=False):
                    with gr.Row():
                        is_repeated_guest = gr.Checkbox(
                            label="Returning guest",
                            value=defaults["is_repeated_guest"],
                            info="Tick if this guest has stayed with you before.",
                        )
                        previous_cancellations = gr.Number(
                            label="Previous cancellations",
                            value=defaults["previous_cancellations"],
                            minimum=0,
                            maximum=20,
                            step=1,
                        )
                        previous_bookings_not_canceled = gr.Number(
                            label="Past completed stays",
                            value=defaults["previous_bookings_not_canceled"],
                            minimum=0,
                            maximum=50,
                            step=1,
                        )

                required_status = gr.Markdown(
                    value=initial_required_status, elem_id="required-status"
                )

                with gr.Row():
                    predict_btn = gr.Button(
                        "Predict",
                        variant="primary",
                        interactive=bool(initial_ready),
                    )
                    flag_btn = gr.Button("Flag", interactive=bool(initial_ready))
                    reset_btn = gr.Button("Reset")

            with gr.Column(scale=6, elem_id="result-col", elem_classes=["result-panel"]):
                gr.Markdown("## Prediction result")
                gr.Markdown(
                    "Fill in the booking details and click **Predict** to get a risk score and recommended action."
                )
                risk_card = gr.HTML(value=initial_risk_html, elem_id="risk-card")
                result = gr.Markdown(value=initial_summary, elem_id="result-summary")
                decision_notes = gr.HTML(value=initial_decision_notes, elem_id="decision-notes")
                with gr.Accordion("Developer details (JSON)", open=False):
                    details = gr.Textbox(
                        label="Raw output",
                        value="",
                        interactive=False,
                        lines=9,
                        max_lines=14,
                        elem_id="result-details",
                        buttons=["copy"],
                    )

        inputs = {
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date": arrival_date,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type": reserved_room_type,
            "deposit_type": deposit_type,
            "agent": agent,
            "company": company,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests,
        }

        form_valid_state = gr.State(value=bool(initial_ready))
        last_prediction_state = gr.State(value=None)

        def _on_form_change(*vals):
            payload = dict(zip(inputs.keys(), vals))
            (
                req_status,
                summary,
                decision_md,
                risk_html,
                ready,
                _,
                total_nights,
                party_size,
                adr_pp,
            ) = _form_feedback(payload)
            return (
                req_status,
                summary,
                decision_md,
                risk_html,
                gr.update(interactive=bool(ready)),
                gr.update(interactive=bool(ready)),
                bool(ready),
                total_nights,
                party_size,
                round(adr_pp, 2),
                "",
                None,
            )

        def _predict(form_ready: bool, *vals):
            try:
                payload = dict(zip(inputs.keys(), vals))
                if not form_ready:
                    (
                        _,
                        summary,
                        decision_md,
                        risk_html,
                        _ready,
                        missing_msg,
                        _,
                        _,
                        _,
                    ) = _form_feedback(payload)
                    details_json = json.dumps(
                        {"status": "validation_error", "message": missing_msg},
                        indent=2,
                        sort_keys=True,
                    )
                    return summary, details_json, risk_html, decision_md, None
                summary, details_json, risk_html, decision_md, record = _predict_output(payload)
                if record is not None:
                    _log_case(record, summary, flagged=False)
                state_payload = {
                    "timestamp_utc": _format_utc(datetime.now(timezone.utc)),
                    "summary": summary,
                    "details": details_json,
                }
                return summary, details_json, risk_html, decision_md, state_payload
            except Exception as exc:
                logger.exception("Prediction handler failed")
                summary, details_json, risk_html, decision_md = _error_output(
                    f"Prediction failed: {exc}", exc
                )
                return summary, details_json, risk_html, decision_md, None

        def _flag(form_ready: bool, *vals):
            try:
                payload = dict(zip(inputs.keys(), vals))
                if not form_ready:
                    (
                        _,
                        summary,
                        decision_md,
                        risk_html,
                        _ready,
                        missing_msg,
                        _,
                        _,
                        _,
                    ) = _form_feedback(payload)
                    details_json = json.dumps(
                        {"status": "validation_error", "message": missing_msg},
                        indent=2,
                        sort_keys=True,
                    )
                    return summary, details_json, risk_html, decision_md, None
                summary, details_json, risk_html, decision_md, record = _predict_output(payload)
                if record is not None:
                    _log_case(record, summary, flagged=True)
                state_payload = {
                    "timestamp_utc": _format_utc(datetime.now(timezone.utc)),
                    "summary": summary,
                    "details": details_json,
                }
                return summary, details_json, risk_html, decision_md, state_payload
            except Exception as exc:
                logger.exception("Flag handler failed")
                summary, details_json, risk_html, decision_md = _error_output(
                    f"Prediction failed: {exc}", exc
                )
                return summary, details_json, risk_html, decision_md, None

        def _set_loading():
            return (
                _loading_summary(),
                "",
                _risk_meter_html(None, label="Unavailable", note="Scoring in progress..."),
                _loading_decision_notes(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        def _set_ready(is_ready: bool):
            return gr.update(interactive=bool(is_ready)), gr.update(interactive=bool(is_ready))

        reset_outputs = list(inputs.values()) + [
            required_status,
            result,
            decision_notes,
            risk_card,
            predict_btn,
            flag_btn,
            form_valid_state,
            total_nights_view,
            party_size_view,
            adr_per_person_view,
            details,
            last_prediction_state,
        ]

        def _reset():
            payload = dict(defaults)
            payload["arrival_date"] = _default_arrival_date()
            (
                req_status,
                summary,
                decision_md,
                risk_html,
                ready,
                _,
                total_nights,
                party_size,
                adr_pp,
            ) = _form_feedback(payload)
            vals = [payload.get(k) for k in inputs]
            vals.extend(
                [
                    req_status,
                    summary,
                    decision_md,
                    risk_html,
                    gr.update(interactive=bool(ready)),
                    gr.update(interactive=bool(ready)),
                    bool(ready),
                    total_nights,
                    party_size,
                    round(adr_pp, 2),
                    "",
                    None,
                ]
            )
            return vals

        validation_outputs = [
            required_status,
            result,
            decision_notes,
            risk_card,
            predict_btn,
            flag_btn,
            form_valid_state,
            total_nights_view,
            party_size_view,
            adr_per_person_view,
            details,
            last_prediction_state,
        ]
        for component in inputs.values():
            component.change(
                _on_form_change,
                inputs=list(inputs.values()),
                outputs=validation_outputs,
                queue=False,
            )

        predict_btn.click(
            _set_loading,
            outputs=[result, details, risk_card, decision_notes, predict_btn, flag_btn],
            queue=False,
        ).then(
            _predict,
            inputs=[form_valid_state, *list(inputs.values())],
            outputs=[result, details, risk_card, decision_notes, last_prediction_state],
            show_progress="full",
        ).then(
            _set_ready,
            inputs=[form_valid_state],
            outputs=[predict_btn, flag_btn],
        )

        flag_btn.click(
            _set_loading,
            outputs=[result, details, risk_card, decision_notes, predict_btn, flag_btn],
            queue=False,
        ).then(
            _flag,
            inputs=[form_valid_state, *list(inputs.values())],
            outputs=[result, details, risk_card, decision_notes, last_prediction_state],
            show_progress="full",
        ).then(
            _set_ready,
            inputs=[form_valid_state],
            outputs=[predict_btn, flag_btn],
        )
        reset_btn.click(
            _reset,
            outputs=reset_outputs,
            queue=False,
        )

    return demo
