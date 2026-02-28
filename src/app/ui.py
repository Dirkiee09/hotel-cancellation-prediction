"""Gradio UI for hotel booking cancellation prediction."""

from __future__ import annotations

import csv
import json
import logging
import os
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

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGED_PATH = PROJECT_ROOT / ".gradio" / "flagged" / "predictions.csv"
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"

_ARTIFACTS: ModelArtifacts | None = None
_ARTIFACTS_LOCK = threading.Lock()


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


def _default_arrival_date() -> datetime:
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return base + timedelta(days=30)


def _risk_bucket(prob: float) -> str:
    for cutoff, label in RISK_BANDS:
        if prob < cutoff:
            return label
    return "High"


def _risk_meter_html(prob: float | None, *, label: str, note: str) -> str:
    tone = RISK_TONES.get(label, "neutral")
    pct = max(0.0, min(100.0, (prob or 0.0) * 100.0))
    return f"""
<div class="risk-card risk-{tone}">
  <div class="risk-topline">
    <span class="risk-pill">{label}</span>
    <span class="risk-percent">{pct:.1f}%</span>
  </div>
  <div class="risk-track">
    <div class="risk-fill" style="width:{pct:.1f}%"></div>
  </div>
  <p class="risk-note">{note}</p>
</div>
"""


def _idle_summary() -> str:
    return (
        "### Ready to score\n"
        "Fill in booking-time features and run **Predict**.\n\n"
        "- `Max-F1` balances precision and recall.\n"
        "- `High-precision` is stricter and lowers false positives."
    )


def _idle_decision_notes() -> str:
    return (
        "### Decision guidance\n"
        "- Operational recommendation appears after scoring.\n"
        "- Policy alignment and threshold distance are shown for fast triage."
    )


def _loading_summary() -> str:
    return "### Running prediction\nCalculating calibrated probability and decision thresholds..."


def _loading_decision_notes() -> str:
    return "### Decision guidance\nPreparing policy comparison and recommended action..."


def _idle_risk_card() -> str:
    return _risk_meter_html(
        None,
        label="Unavailable",
        note="Prediction risk meter will appear here.",
    )


def _format_prediction_output(
    prob: float,
    thr_f1: float,
    thr_hp: float,
    timestamp_utc: str,
    model_utc: str | None,
    record: Dict[str, Any],
) -> tuple[str, str, str, str]:
    risk_label = _risk_bucket(prob)
    pct = prob * 100.0
    label_f1 = "Will Cancel" if prob >= thr_f1 else "Not Canceled"
    label_hp = "Will Cancel" if prob >= thr_hp else "Not Canceled"
    disagreement = (
        "\n- Policy note: decisions differ. `High-precision` is intentionally stricter."
        if label_f1 != label_hp
        else ""
    )
    summary = (
        "### Prediction summary\n"
        f"**Cancellation risk:** `{pct:.1f}%` ({risk_label})\n\n"
        f"- Max-F1 decision: **{label_f1}** at threshold `{thr_f1:.3f}`\n"
        f"- High-precision decision: **{label_hp}** at threshold `{thr_hp:.3f}`\n"
        f"- Scored at (UTC): `{timestamp_utc}`\n"
        f"- Model artifact timestamp (UTC): `{model_utc or 'unknown'}`"
        f"{disagreement}"
    )
    risk_html = _risk_meter_html(
        prob,
        label=risk_label,
        note=(f"Max-F1: {label_f1} @ {thr_f1:.3f} | " f"High-precision: {label_hp} @ {thr_hp:.3f}"),
    )
    if risk_label == "High":
        action = "Trigger retention playbook now and prioritize outreach."
    elif risk_label == "Medium":
        action = "Queue for manual review and monitor booking changes."
    else:
        action = "No intervention by default; continue standard journey."

    repeated_guest = "Yes" if bool(record.get("is_repeated_guest", 0)) else "No"
    deposit_type = str(record.get("deposit_type") or "unknown")
    lead_time = int(record.get("lead_time") or 0)
    prev_cxl = int(record.get("previous_cancellations") or 0)
    special_requests = int(record.get("total_of_special_requests") or 0)
    delta_f1 = prob - thr_f1
    delta_hp = prob - thr_hp
    policy_alignment = "Agree" if label_f1 == label_hp else "Disagree"
    decision_notes = (
        "### Decision details\n"
        f"**Recommended action:** {action}\n\n"
        f"- Policy alignment: **{policy_alignment}**\n"
        f"- Distance to Max-F1 threshold: `{delta_f1:+.3f}`\n"
        f"- Distance to High-precision threshold: `{delta_hp:+.3f}`\n"
        f"- Repeated guest: `{repeated_guest}`\n"
        f"- Deposit type: `{deposit_type}`\n"
        f"- Lead time: `{lead_time}` days\n"
        f"- Previous cancellations: `{prev_cxl}`\n"
        f"- Special requests: `{special_requests}`"
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
            writer = csv.DictWriter(handle, fieldnames=ordered_cols)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()


def _predict_output(values: Dict[str, Any]) -> tuple[str, str, str, str, Dict[str, Any] | None]:
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
        thresholds = artifacts.thresholds or {}
        thr_f1 = float(thresholds.get("max_f1", {}).get("threshold", 0.5))
        thr_hp = float(thresholds.get("high_precision", {}).get("threshold", 0.5))
        timestamp_utc = _format_utc(datetime.now(timezone.utc))
        model_ts = _model_timestamp_utc() or "unknown"
        summary, details_json, risk_html, decision_notes = _format_prediction_output(
            prob, thr_f1, thr_hp, timestamp_utc, model_ts, record
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
  padding: 12px;
}

#decision-notes {
  margin-top: 10px;
  background: rgba(0, 0, 0, 0.22);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px;
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
  font-size: 1.4rem;
  font-weight: 700;
}

.risk-track {
  margin-top: 10px;
  border-radius: 999px;
  height: 11px;
  background: rgba(255, 255, 255, 0.14);
  overflow: hidden;
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
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Hotel Booking Cancellation Prediction") as demo:
        gr.HTML('<div id="app-bg"></div><div id="app-noise"></div>')
        gr.HTML(
            f"""
<section class="hero-shell">
  <h1 class="hero-title">Hotel Booking Cancellation Predictor</h1>
  <p class="hero-subtitle">
    Decision-support form for booking-time cancellation risk. Use this before check-in planning or
    overbooking control.
  </p>
  <div class="hero-chips">
    <span class="hero-chip">Model input scope: booking-time only</span>
    <span class="hero-chip">Risk medium cutoff: {RISK_TIER_MEDIUM_THRESHOLD:.2f}</span>
    <span class="hero-chip">Risk high cutoff: {RISK_TIER_HIGH_THRESHOLD:.2f}</span>
  </div>
</section>
"""
        )
        with gr.Row():
            with gr.Column(scale=2, elem_classes=["input-panel"]):
                with gr.Group():
                    gr.Markdown("## Booking profile")
                    hotel = gr.Dropdown(
                        label="Hotel",
                        choices=["City Hotel", "Resort Hotel"],
                        value="City Hotel",
                        allow_custom_value=True,
                    )
                    with gr.Row():
                        market_segment = gr.Dropdown(
                            label="Market segment",
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
                            value="Online TA",
                            allow_custom_value=True,
                        )
                        distribution_channel = gr.Dropdown(
                            label="Distribution channel",
                            choices=["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
                            value="TA/TO",
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        customer_type = gr.Dropdown(
                            label="Customer type",
                            choices=["Transient", "Transient-Party", "Contract", "Group"],
                            value="Transient",
                            allow_custom_value=True,
                        )
                        country = gr.Dropdown(
                            label="Country",
                            choices=COUNTRY_CHOICES,
                            value="UNKNOWN",
                            allow_custom_value=True,
                        )

                with gr.Group():
                    gr.Markdown("## Timing and stay")
                    with gr.Row():
                        lead_time = gr.Number(
                            label="Lead time (days)",
                            value=30,
                            minimum=0,
                            info="Days between booking and arrival.",
                        )
                        arrival_date = gr.DateTime(
                            label="Arrival date",
                            value=_default_arrival_date(),
                            include_time=False,
                            type="datetime",
                            info="Week number is derived automatically.",
                        )
                    with gr.Row():
                        stays_in_weekend_nights = gr.Number(
                            label="Weekend nights", value=0, minimum=0
                        )
                        stays_in_week_nights = gr.Number(label="Week nights", value=2, minimum=0)
                    with gr.Row():
                        adults = gr.Number(label="Adults", value=2, minimum=1, maximum=20)
                        children = gr.Number(label="Children", value=0, minimum=0, maximum=20)
                        babies = gr.Number(label="Babies", value=0, minimum=0, maximum=20)
                    meal = gr.Dropdown(
                        label="Meal",
                        choices=["BB", "HB", "FB", "SC", "Undefined"],
                        value="BB",
                        allow_custom_value=True,
                    )

                with gr.Group():
                    gr.Markdown("## Price and policy")
                    with gr.Row():
                        reserved_room_type = gr.Dropdown(
                            label="Reserved room type",
                            choices=["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
                            value="A",
                            allow_custom_value=True,
                        )
                        deposit_type = gr.Dropdown(
                            label="Deposit type",
                            choices=["No Deposit", "Non Refund", "Refundable"],
                            value="No Deposit",
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        adr = gr.Number(label="ADR", value=100, minimum=0, maximum=2000)
                        required_car_parking_spaces = gr.Number(
                            label="Parking spaces", value=0, minimum=0, maximum=10
                        )
                        total_of_special_requests = gr.Number(
                            label="Special requests", value=0, minimum=0, maximum=10
                        )

                with gr.Group():
                    gr.Markdown("## Guest history")
                    with gr.Row():
                        is_repeated_guest = gr.Checkbox(
                            label="Repeated guest",
                            value=False,
                            info="Enable if this customer has stayed before.",
                        )
                        previous_cancellations = gr.Number(
                            label="Previous cancellations", value=0, minimum=0, maximum=20
                        )
                        previous_bookings_not_canceled = gr.Number(
                            label="Previous bookings not canceled", value=0, minimum=0, maximum=50
                        )
                    with gr.Accordion("Advanced identifiers (optional)", open=False):
                        agent = gr.Textbox(label="Agent", value="UNKNOWN")
                        company = gr.Textbox(label="Company", value="UNKNOWN")

                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    flag_btn = gr.Button("Flag")
                    reset_btn = gr.Button("Reset")

            with gr.Column(scale=1, elem_id="result-col", elem_classes=["result-panel"]):
                gr.Markdown("## Decision center")
                risk_card = gr.HTML(value=_idle_risk_card(), elem_id="risk-card")
                result = gr.Markdown(value=_idle_summary(), elem_id="result-summary")
                decision_notes = gr.Markdown(value=_idle_decision_notes(), elem_id="decision-notes")
                with gr.Accordion("Details (JSON)", open=False):
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

        def _predict(*vals):
            try:
                payload = dict(zip(inputs.keys(), vals))
                summary, details_json, risk_html, decision_md, record = _predict_output(payload)
                if record is not None:
                    _log_case(record, summary, flagged=False)
                return summary, details_json, risk_html, decision_md
            except Exception as exc:
                logger.exception("Prediction handler failed")
                return _error_output(f"Prediction failed: {exc}", exc)

        def _flag(*vals):
            try:
                payload = dict(zip(inputs.keys(), vals))
                summary, details_json, risk_html, decision_md, record = _predict_output(payload)
                if record is not None:
                    _log_case(record, summary, flagged=True)
                return summary, details_json, risk_html, decision_md
            except Exception as exc:
                logger.exception("Flag handler failed")
                return _error_output(f"Prediction failed: {exc}", exc)

        def _set_loading():
            return (
                _loading_summary(),
                "",
                _risk_meter_html(None, label="Unavailable", note="Scoring in progress..."),
                _loading_decision_notes(),
                gr.update(interactive=False),
            )

        def _set_ready():
            return gr.update(interactive=True)

        _RESET_DEFAULTS: dict[str, Any] = {
            "hotel": "City Hotel",
            "lead_time": 30,
            "stays_in_weekend_nights": 0,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "UNKNOWN",
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
            "adr": 100,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
        }

        reset_outputs = list(inputs.values()) + [details, result, risk_card, decision_notes]

        def _reset():
            vals = [_RESET_DEFAULTS.get(k, None) for k in inputs]
            vals[list(inputs.keys()).index("arrival_date")] = _default_arrival_date()
            vals.extend(["", _idle_summary(), _idle_risk_card(), _idle_decision_notes()])
            return vals

        predict_btn.click(
            _set_loading,
            outputs=[result, details, risk_card, decision_notes, predict_btn],
            queue=False,
        ).then(
            _predict,
            inputs=list(inputs.values()),
            outputs=[result, details, risk_card, decision_notes],
            show_progress="full",
        ).then(
            _set_ready,
            outputs=predict_btn,
        )

        flag_btn.click(
            _set_loading,
            outputs=[result, details, risk_card, decision_notes, flag_btn],
            queue=False,
        ).then(
            _flag,
            inputs=list(inputs.values()),
            outputs=[result, details, risk_card, decision_notes],
            show_progress="full",
        ).then(
            _set_ready,
            outputs=flag_btn,
        )
        reset_btn.click(
            _reset,
            outputs=reset_outputs,
        )

    return demo
