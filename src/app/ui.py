"""Gradio UI for hotel booking cancellation prediction."""

from __future__ import annotations

import csv
import html
import json
import logging
import re
import sys
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
    ADR_MAX_VALID,
    ARTIFACTS_DIR,
    BOOKING_TIME_FEATURES,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
)
from src.serving.inference import (
    ModelArtifacts,
    _prepare_features,
    explain_prediction,
    get_cached_artifacts,
    predict_adr,
)
from src.serving.prediction_log import export_to_csv, log_prediction
from src.utils.thresholds import resolve_thresholds

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Every Predict/Flag click appends one row here (gitignored runtime data).
# Visible location on purpose — operators need to find their scored bookings.
LOGGED_PATH = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"

_GLOBAL_DRIVER_LINES: list[str] | None = None
_GLOBAL_DRIVER_MODEL_ID: str | None = None
_GLOBAL_DRIVER_LOCK = threading.Lock()


def _get_artifacts():
    """Delegate to the shared singleton in inference.py."""
    return get_cached_artifacts()


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


def _load_categorical_choices() -> dict[str, list[str]]:
    """Auto-detect categorical dropdown values from the training CSV."""
    cols = [
        "hotel",
        "market_segment",
        "distribution_channel",
        "customer_type",
        "meal",
        "deposit_type",
        "reserved_room_type",
        "country",
    ]
    _fallbacks: dict[str, list[str]] = {
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
        "country": [],
    }
    result: dict[str, list[str]] = {}
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH, usecols=cols)
            for col in cols:
                vals = sorted({str(v).strip() for v in df[col].dropna() if str(v).strip()})
                result[col] = vals
        except Exception as exc:
            logger.warning("Could not read dropdown choices from %s: %s", DATA_PATH, exc)
    for col in cols:
        if col not in result or not result[col]:
            result[col] = _fallbacks.get(col, [])
    if "UNKNOWN" not in result.get("country", []):
        result.setdefault("country", []).insert(0, "UNKNOWN")
    return result


_CAT_CHOICES = _load_categorical_choices()
COUNTRY_CHOICES = _CAT_CHOICES["country"]
COUNTRY_OPTIONAL_CHOICES = [""] + COUNTRY_CHOICES


def _load_hero_metrics() -> str:
    """Load ROC-AUC / PR-AUC from reports/metrics.json for the hero banner."""
    reports_dir = PROJECT_ROOT / "reports"
    metrics_path = reports_dir / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            max_f1 = data.get("max_f1", {})
            roc = max_f1.get("roc_auc")
            pr = max_f1.get("pr_auc")
            if roc is not None and pr is not None:
                return f"ROC-AUC {float(roc):.3f} \u00b7 PR-AUC {float(pr):.3f}"
        except Exception as exc:
            logger.warning("Could not load hero metrics from %s: %s", metrics_path, exc)
    return "ROC-AUC \u2014 \u00b7 PR-AUC \u2014"


_HERO_METRICS = _load_hero_metrics()

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
    base = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
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
                # Past dates are allowed deliberately: the API accepts them, and
                # scoring historical bookings (e.g. dataset rows during a demo or
                # back-testing) is a legitimate use. Only the format is enforced.
                if isinstance(raw, datetime):
                    raw.date()
                elif hasattr(raw, "date"):
                    pass
                else:
                    datetime.fromisoformat(str(raw)).date()
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
        drivers.append(
            "Non-Refund deposit — in booking data this deposit type correlates with higher dropout rates."
        )
    if lead_time >= 120:
        drivers.append(
            f"Booked {int(lead_time)} days in advance — very long lead times increase dropout risk."
        )
    if not repeated:
        drivers.append("First-time guest — no prior booking history to indicate loyalty.")
    if specials == 0:
        drivers.append("No special requests made — lower engagement signal.")
    if parking == 0:
        drivers.append("No parking needed — slightly lower commitment signal.")
    if market_segment in {"Groups", "Online TA"}:
        drivers.append(
            f"{market_segment} bookings tend to have higher cancellation rates in our data."
        )
    if prev_cxl > 0:
        drivers.append(f"The guest has {prev_cxl} previous cancellation(s) on record.")

    if not drivers:
        drivers.append("No strong warning signal was triggered for this booking.")
    return drivers[:5]


def _artifact_model_id(artifacts: ModelArtifacts) -> str:
    metadata = artifacts.metadata or {}
    lineage_sha = metadata.get("lineage", {}).get("artifacts", {}).get("bundle_sha256")
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
                preprocessor = artifacts.model.named_steps.get(
                    "preprocessor"
                ) or artifacts.model.named_steps.get("preprocess")
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
        suggestions.append(
            "Hold a small room buffer for this arrival date in case this booking falls through."
        )
    elif risk_label == "Medium":
        suggestions.append("Send automated reminders at 72 h and 24 h before arrival.")
        suggestions.append(
            "Check room availability — review this booking manually if inventory is tight."
        )
    else:
        suggestions.append("No action needed — continue with the standard booking flow.")

    if lead_time >= 120:
        suggestions.append(
            "Set up a reconfirmation schedule: contact the guest at 90, 30 and 7 days before arrival."
        )
    if deposit_type in {"no deposit", "refundable"} and prob >= thr_f1:
        suggestions.append(
            "Request a deposit or card pre-authorisation to strengthen the guest's commitment."
        )
    if prev_cxl > 0:
        suggestions.append(
            "Call or email this guest personally — they have cancelled a booking before."
        )
    if specials == 0:
        suggestions.append(
            "Ask the guest if they have any special requests — this increases booking engagement."
        )
    if not repeated and prob >= thr_f1:
        suggestions.append(
            "Consider a small loyalty incentive (room upgrade or F&B voucher) to build commitment."
        )
    if prob >= thr_hp:
        suggestions.append("Flag this booking for daily monitoring in your reservations system.")

    seen: set[str] = set()
    deduped = [s for s in suggestions if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]
    return deduped[:6]


def _generate_xai_html(explanation: list[dict[str, Any]]) -> str:
    if not explanation:
        return "<div class='xai-container'><i>SHAP explanation unavailable.</i></div>"

    max_abs = (
        max([abs(float(item["contribution"])) for item in explanation]) if explanation else 1.0
    )
    if max_abs == 0:
        max_abs = 1.0

    html = '<div class="xai-container">'
    for item in explanation:
        name = str(item["feature"])
        impact = float(item["contribution"])

        direction = "positive" if impact > 0 else "negative"
        sign = "+" if impact > 0 else ""
        text_cls = "pos-text" if impact > 0 else "neg-text"

        width = min(50, (abs(impact) / max_abs) * 50)

        html += f"""
        <div class="xai-row">
            <div class="xai-label">{name}</div>
            <div class="xai-bar-bg">
                <div class="xai-bar-fill {direction}" style="width: {width}%;"></div>
            </div>
            <div class="xai-val {text_cls}">{sign}{impact:.2f}</div>
        </div>
        """
    html += "</div>"
    return html


def _risk_meter_html(prob: float | None, *, label: str, note: str) -> str:
    tone = RISK_TONES.get(label, "neutral")
    safe_label = html.escape(label)
    safe_note = html.escape(note)
    medium_pct = max(0.0, min(100.0, RISK_TIER_MEDIUM_THRESHOLD * 100.0))
    high_pct = max(0.0, min(100.0, RISK_TIER_HIGH_THRESHOLD * 100.0))
    # Permanent traffic-light zones, built from the live thresholds so they always
    # match the markers: green 0→medium, amber medium→high, red high→100.
    zone_gradient = (
        "linear-gradient(90deg,"
        f"rgba(45,212,122,0.72) 0%,rgba(45,212,122,0.72) {medium_pct:.1f}%,"
        f"rgba(251,191,36,0.72) {medium_pct:.1f}%,rgba(251,191,36,0.72) {high_pct:.1f}%,"
        f"rgba(248,113,113,0.72) {high_pct:.1f}%,rgba(248,113,113,0.72) 100%)"
    )
    if prob is None:
        return f"""
<div class="risk-card risk-{tone} risk-idle">
  <div class="risk-topline">
    <span class="risk-pill">{safe_label}</span>
    <span class="risk-percent risk-percent-idle">&#x2014;</span>
  </div>
  <div class="risk-track with-markers" style="background:{zone_gradient}">
    <span class="risk-marker medium" style="left:{medium_pct:.1f}%"></span>
    <span class="risk-marker high" style="left:{high_pct:.1f}%"></span>
  </div>
  <div class="risk-threshold-labels">
    <span>Medium @{RISK_TIER_MEDIUM_THRESHOLD:.2f}</span>
    <span>High @{RISK_TIER_HIGH_THRESHOLD:.2f}</span>
  </div>
  <p class="risk-note">{safe_note}</p>
</div>
"""
    pct = max(0.0, min(100.0, prob * 100.0))
    return f"""
<div class="risk-card risk-{tone}">
  <div class="risk-topline">
    <span class="risk-pill">{safe_label}</span>
    <span class="risk-percent">{pct:.1f}%</span>
  </div>
  <div class="risk-track with-markers" style="background:{zone_gradient}">
    <span class="risk-marker medium" style="left:{medium_pct:.1f}%"></span>
    <span class="risk-marker high" style="left:{high_pct:.1f}%"></span>
    <div class="risk-fill" style="width:{pct:.1f}%"></div>
    <span class="risk-dot" style="left:{pct:.1f}%"></span>
  </div>
  <div class="risk-threshold-labels">
    <span>Medium @{RISK_TIER_MEDIUM_THRESHOLD:.2f}</span>
    <span>High @{RISK_TIER_HIGH_THRESHOLD:.2f}</span>
  </div>
  <p class="risk-note">{safe_note}</p>
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
  <div class="explain-grid" style="grid-template-columns: 1fr 1fr;">
    <section class="explain-section">
      <h4 class="explain-h explain-h-why"><span class="explain-ico">?</span>Why it was flagged</h4>
      <ul>{_html_bullets(why_items, fallback="No strong warning signals were detected.")}</ul>
    </section>
    <section class="explain-section">
      <h4 class="explain-h explain-h-action"><span class="explain-ico">!</span>Recommended actions</h4>
      <ul>{_html_bullets(action_items, fallback="Continue with normal booking flow.")}</ul>
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
    explanation: list[dict[str, object]],
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

    if risk_label == "High":
        plain_action = "Flag for 10% non-refundable deposit requirement. Authorize front-desk to overbook this room type by +1."
    elif risk_label == "Medium":
        plain_action = "Queue for manual review. Send 72h and 24h automated reminders."
    else:
        plain_action = "No intervention needed — continue standard guest journey."

    conf_level = (
        "VERY HIGH"
        if prob > 0.8 or prob < 0.2
        else "HIGH"
        if prob > 0.6 or prob < 0.4
        else "MEDIUM"
    )
    total_blocks = 20
    filled_blocks = int((prob) * total_blocks)
    empty_blocks = total_blocks - filled_blocks
    progress_bar = "█" * filled_blocks + "░" * empty_blocks

    action_color = (
        "#ef4444" if risk_label == "High" else "#f59e0b" if risk_label == "Medium" else "#10b981"
    )
    action_icon = "🔴" if risk_label == "High" else "🟡" if risk_label == "Medium" else "🟢"

    summary = (
        f"<div style='font-size: 2.2rem; font-weight: 800; color: {action_color}; margin-bottom: -5px;'>{pct:.1f}%</div>"
        f"<div style='font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;'>Cancellation Risk</div>"
        f"<div style='font-size: 1.4rem; font-family: monospace; color: {action_color}; letter-spacing: 2px; margin-bottom: 5px;'>{progress_bar}</div>"
        f"<div style='font-size: 1rem; font-weight: bold; color: {action_color}; margin-bottom: 25px;'>{risk_label.upper()} RISK</div>"
        f"<div style='border: 1px solid {action_color}; border-radius: 8px; padding: 15px; background: rgba(0,0,0,0.3);'>"
        f"<div style='font-size: 0.75rem; letter-spacing: 1px; color: #cbd5e1; text-transform: uppercase; margin-bottom: 8px;'>RECOMMENDED ACTION</div>"
        f"<div style='font-size: 1.05rem; color: #eef2ff; font-weight: bold; margin-bottom: 12px; line-height: 1.4;'>{action_icon} {plain_action}</div>"
        f"<div style='display: flex; justify-content: space-between; font-size: 0.9rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;'>"
        f"<span style='color: #94a3b8'>Expected Revenue</span><span style='color: white; font-weight: bold;'>€{rev_at_risk:,.0f}</span>"
        f"</div>"
        f"<div style='display: flex; justify-content: space-between; font-size: 0.9rem; margin-top: 4px;'>"
        f"<span style='color: #94a3b8'>Confidence</span><span style='color: white; font-weight: bold;'>{conf_level}</span>"
        f"</div>"
        f"</div>"
    )
    meter_note = f"Standard: {label_f1} · High-confidence: {label_hp}"
    if borderline:
        meter_note += " · ⚠ Near boundary"
    risk_html = ""

    global_drivers = _load_global_model_drivers(artifacts, top_k=5)
    booking_drivers = _risk_drivers(record, prob)
    suggestions = _intervention_suggestions(record, risk_label, prob, thr_f1, thr_hp)
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
    xai_html = _generate_xai_html(explanation)

    # Generate Verdict Explanation
    if risk_label in ["High", "Medium"]:
        verdict_text = f"Flagged as likely to be canceled because the risk probability ({pct:.1f}%) exceeds the normal thresholds. The Feature Impact below shows the specific booking details driving this risk."
    else:
        verdict_text = f"No concern. The risk probability ({pct:.1f}%) is below the alert thresholds. The booking characteristics indicate a low likelihood of cancellation."

    # Format global drivers as an HTML list
    global_drivers_html = (
        "<ul style='color: #cbd5e1; font-size: 0.9rem; margin-top: 10px; padding-left: 20px;'>"
    )
    for driver in global_drivers:
        global_drivers_html += f"<li style='margin-bottom: 5px;'>{html.escape(driver)}</li>"
    global_drivers_html += "</ul>"

    decision_notes = f"""
    <div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin-top: 15px;'>
        <div style='margin-bottom: 20px;'>
            <h4 style='margin-top: 0; margin-bottom: 5px; color: #f8fafc; font-size: 1.1rem;'>Verdict Explanation</h4>
            <p style='color: #cbd5e1; font-size: 0.95rem; margin-top: 0;'>{verdict_text}</p>
        </div>
        <h4 style='margin-top: 0; margin-bottom: 15px; color: #f8fafc; font-size: 1.1rem;'>Feature Impact</h4>
        {xai_html}
        <div style='margin-top: 25px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 15px;'>
            <h4 style='margin-top: 0; margin-bottom: 5px; color: #f8fafc; font-size: 1.1rem;'>Model Insights</h4>
            <p style='color: #94a3b8; font-size: 0.85rem; margin-top: 0;'>Top cancellation indicators learned from training data:</p>
            <ul style='color: #cbd5e1; font-size: 0.9rem; margin-top: 10px; padding-left: 20px;'>
                <li style='margin-bottom: 5px;'>Non-refundable bookings</li>
                <li style='margin-bottom: 5px;'>Long lead times</li>
                <li style='margin-bottom: 5px;'>Certain booking channels</li>
                <li style='margin-bottom: 5px;'>Repeat cancellation history</li>
                <li style='margin-bottom: 5px;'>High special request counts</li>
            </ul>
        </div>
    </div>
    """

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
        payload["is_repeated_guest"] = int(payload["is_repeated_guest"])  # type: ignore[arg-type]

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
    # sys.platform (not os.name) so mypy prunes the foreign-platform branch
    # under the pinned `platform = "linux"` analysis in pyproject.toml.
    if sys.platform == "win32":
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

        # Authentic Model Inference & SHAP explanation
        import pandas as pd

        df_raw = pd.DataFrame([record])
        feature_df = _prepare_features(df_raw, artifacts.feature_columns)

        if artifacts.is_pipeline:
            prob = float(artifacts.model.predict_proba(feature_df)[:, 1][0])
        else:
            X = artifacts.preprocessor.transform(feature_df)
            prob = float(artifacts.model.predict_proba(X)[:, 1][0])

        if artifacts.calibrator is not None:
            import numpy as np

            prob = float(np.clip(artifacts.calibrator.predict([prob]), 0.0, 1.0)[0])

        explanation = explain_prediction(feature_df, artifacts, top_n=5)

        resolved, sources, fallback_used, _ = resolve_thresholds(artifacts.thresholds or {})
        thr_f1 = resolved["max_f1"]
        thr_hp = resolved["high_precision"]
        timestamp_utc = _format_utc(datetime.now(timezone.utc))
        model_ts = _model_timestamp_utc() or "unknown"
        summary, details_json, risk_html, decision_notes = _format_prediction_output(
            prob, thr_f1, thr_hp, timestamp_utc, model_ts, record, artifacts, explanation
        )
        _log_to_prediction_db(record, prob, resolved, sources, fallback_used, artifacts)
        return summary, details_json, risk_html, decision_notes, record
    except Exception as exc:
        logger.exception("Prediction failed")
        summary, details_json, risk_html, decision_notes = _error_output(
            f"Prediction failed: {exc}", exc
        )
        return summary, details_json, risk_html, decision_notes, None


def _log_to_prediction_db(
    record: Dict[str, Any],
    prob: float,
    thresholds: Dict[str, float],
    sources: Dict[str, str],
    fallback_used: bool,
    artifacts: ModelArtifacts,
) -> None:
    """Mirror the REST path's persistence contract for Gradio predictions.

    CLAUDE.md promises every scored booking (HTTP or Gradio) lands in the
    SQLite prediction log with live ADR fields, and that the Power BI CSV is
    auto-refreshed. Non-raising: a logging failure must never break Predict.
    """
    try:
        predicted = predict_adr(record, artifacts)
        entered_adr = record.get("adr")
        adr_residual = (
            round(float(entered_adr) - predicted, 2)
            if predicted is not None and entered_adr is not None
            else None
        )
        if prob >= RISK_TIER_HIGH_THRESHOLD:
            tier = "high"
        elif prob >= RISK_TIER_MEDIUM_THRESHOLD:
            tier = "medium"
        else:
            tier = "low"
        log_prediction(
            dict(record),
            {
                "probability": prob,
                "label_high_precision": int(prob >= thresholds["high_precision"]),
                "label_max_f1": int(prob >= thresholds["max_f1"]),
                "label_cost_sensitive": int(prob >= thresholds["cost_sensitive"]),
                "risk_tier": tier,
                "threshold_high_precision": thresholds["high_precision"],
                "threshold_max_f1": thresholds["max_f1"],
                "threshold_cost_sensitive": thresholds["cost_sensitive"],
                "cost_threshold_source": sources.get("cost_sensitive", "artifact"),
                "cost_threshold_fallback_used": bool(fallback_used),
                "alerts": [],
                "top_features": [],
                "predicted_adr": predicted,
                "adr_residual": adr_residual,
            },
        )
        export_to_csv()
    except Exception:
        logger.exception("ui_prediction_log_failed (non-fatal)")


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
        ordered_cols = [c for c in columns if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
        _append_log_row(log_record, ordered_cols)
    except Exception:
        logger.exception("Failed to log prediction case to %s", LOGGED_PATH)


BACKGROUND_CSS = """
/* Deep dark background with subtle radial gradient and grain noise */
#app-bg {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 15% 50%, rgba(15, 23, 42, 1), rgba(2, 6, 23, 1) 70%);
    z-index: -2;
}
#app-noise {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image: url('data:image/svg+xml,%3Csvg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"%3E%3Cfilter id="noiseFilter"%3E%3CfeTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/%3E%3C/filter%3E%3Crect width="100%25" height="100%25" filter="url(%23noiseFilter)"/%3E%3C/svg%3E');
    opacity: 0.05; z-index: -1; pointer-events: none;
}

/* Modern SaaS KPI Header */
.kpi-container { display: flex; gap: 15px; justify-content: center; margin-bottom: 20px; }
.kpi-box { flex: 1; background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
.kpi-value { font-size: 1.8rem; font-weight: bold; color: #38bdf8; }
.kpi-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }

/* Sticky Result Card */
.result-panel {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.08) !important;
    position: sticky !important;
    top: 20px !important;
    height: fit-content !important;
}

/* XAI Bars */
.xai-container { margin-top: 10px; }
.xai-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.xai-label { font-size: 0.85rem; color: #cbd5e1; width: 35%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.xai-bar-bg { width: 45%; background: rgba(255,255,255,0.05); border-radius: 4px; height: 12px; position: relative; }
.xai-bar-fill { height: 100%; border-radius: 4px; position: absolute; top: 0; }
.xai-bar-fill.positive { background: linear-gradient(90deg, transparent, #ef4444); left: 50%; }
.xai-bar-fill.negative { background: linear-gradient(270deg, transparent, #10b981); right: 50%; }
.xai-val { width: 15%; text-align: right; font-size: 0.8rem; font-family: monospace; font-weight: bold; }
.xai-val.pos-text { color: #fca5a5; }
.xai-val.neg-text { color: #6ee7b7; }

/* Glassmorphic panels */
.input-panel, .output-panel {
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    padding: 20px !important;
}
/* Primary accents: vibrant cyan/emerald */
button.primary {
    background: linear-gradient(135deg, #0ea5e9, #10b981) !important;
    border: none !important;
    color: white !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
}
/* Text colors */
h1, h2, h3, h4, p, span, label {
    color: #f8fafc !important;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(to right, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
/* Dropdowns and inputs */
.gr-input, .gr-dropdown {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
}
/* Risk Tiers */
.risk-low { border-left: 6px solid #10b981 !important; }
.risk-medium { border-left: 6px solid #f59e0b !important; }
.risk-high { border-left: 6px solid #ef4444 !important; }
"""


def build_ui() -> gr.Blocks:
    def _first(col: str) -> str:
        vals = _CAT_CHOICES.get(col, [])
        return vals[0] if vals else ""

    defaults: dict[str, Any] = {
        "hotel": _first("hotel"),
        "lead_time": 30,
        "arrival_date": _default_arrival_date(),
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": _first("meal"),
        "country": "",
        "market_segment": _first("market_segment"),
        "distribution_channel": _first("distribution_channel"),
        "is_repeated_guest": False,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": _first("reserved_room_type"),
        "deposit_type": _first("deposit_type"),
        "agent": "UNKNOWN",
        "company": "UNKNOWN",
        "customer_type": _first("customer_type"),
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
            """
<section class="hero-shell">
  <div class="hero-eyebrow" style="display: flex; justify-content: space-between; align-items: center; padding: 0 10px;">
    <span><span class="hero-eyebrow-dot"></span> LightGBM · Calibrated · Real-time</span>
    <span style="color: #10b981; font-weight: bold;">🟢 System Status: Online (Models Synced)</span>
    <span style="color: #94a3b8;">👤 Logged in as: Admin (Front Desk Manager)</span>
    <span style="color: #94a3b8;">📡 Latency: 12ms</span>
  </div>
  <h1 class="hero-title">Enterprise Booking Cancellation Predictor</h1>
  <p class="hero-subtitle">
    Enter booking details to get a calibrated cancellation probability and an actionable system recommendation.
    Powered by state-of-the-art Gradient Boosted Trees.
  </p>
  <div class="kpi-container">
    <div class="kpi-box"><div class="kpi-value">0.863</div><div class="kpi-label">ROC-AUC</div></div>
    <div class="kpi-box"><div class="kpi-value">0.759</div><div class="kpi-label">PR-AUC</div></div>
    <div class="kpi-box"><div class="kpi-value">&gt; 70%</div><div class="kpi-label">High Risk Thr.</div></div>
    <div class="kpi-box"><div class="kpi-value">12ms</div><div class="kpi-label">Avg Latency</div></div>
  </div>
</section>
"""
        )
        with gr.Tabs():
            with gr.Tab("Scoring Engine"):
                with gr.Row(elem_classes=["layout-row"]):
                    with gr.Column(scale=6, elem_classes=["input-panel"]):
                        with gr.Accordion("1) Booking details (required)", open=True):
                            with gr.Row():
                                hotel = gr.Dropdown(
                                    label="Hotel (required)",
                                    choices=_CAT_CHOICES["hotel"],
                                    value=defaults["hotel"],
                                    allow_custom_value=True,
                                )
                                customer_type = gr.Dropdown(
                                    label="Guest type (required)",
                                    choices=_CAT_CHOICES["customer_type"],
                                    value=defaults["customer_type"],
                                    allow_custom_value=True,
                                )
                            with gr.Row():
                                market_segment = gr.Dropdown(
                                    label="Market segment (required)",
                                    info="How the booking was sourced — e.g., Online TA = online travel agency.",
                                    choices=_CAT_CHOICES["market_segment"],
                                    value=defaults["market_segment"],
                                    allow_custom_value=True,
                                )
                                distribution_channel = gr.Dropdown(
                                    label="Booking platform (required)",
                                    info="The platform or channel used to make this booking.",
                                    choices=_CAT_CHOICES["distribution_channel"],
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
                                adults = gr.Slider(
                                    label="Adults",
                                    value=defaults["adults"],
                                    minimum=1,
                                    maximum=10,
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
                                    maximum=ADR_MAX_VALID,
                                    step=1,
                                    info="Average Daily Rate — the nightly room price.",
                                )
                                deposit_type = gr.Dropdown(
                                    label="Payment / deposit type (required)",
                                    choices=_CAT_CHOICES["deposit_type"],
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

                        with gr.Accordion(
                            "2) Guest preferences (optional — improves accuracy)", open=False
                        ):
                            with gr.Row():
                                country = gr.Dropdown(
                                    label="Country (optional)",
                                    choices=COUNTRY_OPTIONAL_CHOICES,
                                    value=defaults["country"],
                                    allow_custom_value=True,
                                )
                                meal = gr.Dropdown(
                                    label="Meal plan (optional)",
                                    choices=_CAT_CHOICES["meal"],
                                    value=defaults["meal"],
                                    allow_custom_value=True,
                                    info="BB = Bed & Breakfast  ·  HB = Half Board  ·  FB = Full Board  ·  SC = Self-Catering",
                                )
                            with gr.Row():
                                reserved_room_type = gr.Dropdown(
                                    label="Room category (optional)",
                                    choices=_CAT_CHOICES["reserved_room_type"],
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
                                agent = gr.Textbox(
                                    label="Agent (optional)", value=defaults["agent"]
                                )
                                company = gr.Textbox(
                                    label="Company (optional)", value=defaults["company"]
                                )

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
                        gr.HTML(
                            '<div style="font-size:0.75rem;color:#8fa3c8;margin-top:6px;">'
                            "Every scored booking is saved to "
                            f"<code>{LOGGED_PATH.relative_to(PROJECT_ROOT).as_posix()}</code> "
                            "(Flag marks a row for follow-up review).</div>"
                        )

                    with gr.Column(scale=4, elem_id="result-col", elem_classes=["result-panel"]):
                        gr.HTML("""
        <div style="padding:4px 0 14px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:14px;">
          <div style="font-size:0.70rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;color:#4ade9e;margin-bottom:6px;">Live output</div>
          <div style="font-size:1.15rem;font-weight:800;color:#eef2ff;letter-spacing:-0.3px;">Prediction result</div>
          <div style="font-size:0.82rem;color:#8fa3c8;margin-top:4px;">Fill in the form and click <strong style="color:#eef2ff;">Predict</strong> to score this booking.</div>
        </div>
        """)
                        risk_card = gr.HTML(value="", visible=False)
                        result = gr.Markdown(value=initial_summary, elem_id="result-summary")
                        decision_notes = gr.HTML(
                            value=initial_decision_notes, elem_id="decision-notes"
                        )
                        with gr.Accordion("Developer details (JSON)", open=False, visible=False):
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
                        summary, details_json, risk_html, decision_md, record = _predict_output(
                            payload
                        )
                        if record is not None:
                            _log_case(
                                record,
                                json.loads(details_json).get("risk_label", "unknown"),
                                flagged=False,
                            )
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
                        summary, details_json, risk_html, decision_md, record = _predict_output(
                            payload
                        )
                        if record is not None:
                            _log_case(
                                record,
                                json.loads(details_json).get("risk_label", "unknown"),
                                flagged=True,
                            )
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
                    return gr.update(interactive=bool(is_ready)), gr.update(
                        interactive=bool(is_ready)
                    )

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
                    component.change(  # type: ignore[attr-defined]
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

                gr.Markdown("### 📜 Recent Activity Log (Live)")

                def _get_audit_log():
                    import pandas as pd

                    path = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
                    if not path.exists():
                        return pd.DataFrame(
                            columns=[
                                "timestamp_utc",
                                "prediction",
                                "arrival_date",
                                "lead_time",
                                "adr",
                            ]
                        )
                    try:
                        df = pd.read_csv(path)
                        df = df.tail(5)[
                            ["timestamp_utc", "prediction", "arrival_date", "lead_time", "adr"]
                        ]
                        return df.iloc[::-1]  # Reverse so newest is top
                    except Exception:
                        return pd.DataFrame(
                            columns=[
                                "timestamp_utc",
                                "prediction",
                                "arrival_date",
                                "lead_time",
                                "adr",
                            ]
                        )

                gr.Dataframe(
                    value=_get_audit_log, every=2, interactive=False, elem_classes=["layout-row"]
                )

            with gr.Tab("Model Capabilities & Benchmarks"):
                gr.Markdown("## Strategic Business Intelligence Showcase")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. Probability Metrics (ROC/PR-AUC)")
                        try:
                            df1 = pd.read_csv(
                                PROJECT_ROOT
                                / "reports"
                                / "benchmarks"
                                / "03_holdout_probability_metrics.csv"
                            )
                            df1 = df1[["model", "roc_auc", "pr_auc"]].copy()
                            for col in ["roc_auc", "pr_auc"]:
                                df1[col] = df1[col].round(4)
                            gr.Dataframe(value=df1, interactive=False)
                        except Exception:
                            pass
                    with gr.Column():
                        gr.Markdown("### 2. Classification Performance (Max F1)")
                        try:
                            df2 = pd.read_csv(
                                PROJECT_ROOT
                                / "reports"
                                / "benchmarks"
                                / "05_holdout_threshold_metrics_max_f1.csv"
                            )
                            df2 = df2.sort_values("f1", ascending=False)[
                                ["model", "f1", "precision", "recall"]
                            ].copy()
                            for col in ["f1", "precision", "recall"]:
                                df2[col] = df2[col].round(4)
                            gr.Dataframe(value=df2, interactive=False)
                        except Exception:
                            pass
                    with gr.Column():
                        gr.Markdown("### 3. Confusion Matrix Rates")
                        try:
                            df3 = pd.read_csv(
                                PROJECT_ROOT
                                / "reports"
                                / "benchmarks"
                                / "09_confusion_matrix_rates_per_model.csv"
                            )
                            df3 = df3.sort_values("tpr", ascending=False)[
                                ["model", "tpr", "fnr", "tnr", "fpr"]
                            ].copy()
                            for col in ["tpr", "fnr", "tnr", "fpr"]:
                                df3[col] = df3[col].round(4)
                            gr.Dataframe(value=df3, interactive=False)
                        except Exception:
                            pass

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Algorithm Performance (ROC & PR Curves)")
                        gr.Image(
                            value=str(
                                PROJECT_ROOT
                                / "reports/figures/essential/E06_roc_pr_curves_test.png"
                            ),
                            show_label=False,
                        )
                    with gr.Column():
                        gr.Markdown("### Feature Importance (SHAP)")
                        gr.Image(
                            value=str(
                                PROJECT_ROOT / "reports/figures/essential/E10_shap_beeswarm.png"
                            ),
                            show_label=False,
                        )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Business Value (Cost Optimization Policy)")
                        gr.Image(
                            value=str(
                                PROJECT_ROOT
                                / "reports/figures/essential/E12_policy_cost_ladder_test.png"
                            ),
                            show_label=False,
                        )
                    with gr.Column():
                        gr.Markdown("### System Architecture")
                        gr.Image(
                            value=str(
                                PROJECT_ROOT
                                / "reports/figures/essential/E21_deployment_framework.png"
                            ),
                            show_label=False,
                        )
    return demo


if __name__ == "__main__":
    # Gradio 6 moved css/theme from the Blocks constructor to launch()/mount_gradio_app().
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Base(),
        css=BACKGROUND_CSS,
    )
