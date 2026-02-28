"""Gradio UI for hotel booking cancellation prediction."""

from __future__ import annotations

import csv
import json
import logging
import os
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
from src.serving.inference import load_artifacts, predict_proba

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGED_PATH = PROJECT_ROOT / ".gradio" / "flagged" / "predictions.csv"
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"

_ARTIFACTS = None


def _get_artifacts():
    global _ARTIFACTS
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


def _format_prediction_output(
    prob: float,
    thr_f1: float,
    thr_hp: float,
    timestamp_utc: str,
    model_utc: str | None,
    record: Dict[str, Any],
) -> tuple[str, str]:
    risk_label = _risk_bucket(prob)
    pct = prob * 100.0
    label_f1 = "Will Cancel" if prob >= thr_f1 else "Not Canceled"
    label_hp = "Will Cancel" if prob >= thr_hp else "Not Canceled"

    lines = [
        f"Cancellation risk: {pct:.1f}% ({risk_label})",
        f"Decision (Max-F1): {label_f1} (threshold={thr_f1:.3f})",
        f"Decision (High-precision): {label_hp} (threshold={thr_hp:.3f})",
    ]
    if label_f1 != label_hp:
        lines.append("Policies disagree; high-precision is stricter.")
    summary = "\n".join(lines)

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
    return summary, details_json


def _error_output(message: str, exc: Exception | None = None) -> tuple[str, str]:
    details = {"status": "error", "message": message}
    if exc is not None:
        details["exception"] = repr(exc)
        details["traceback"] = traceback.format_exc()
    return message, json.dumps(details, indent=2, sort_keys=True)


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


def _predict_output(values: Dict[str, Any]) -> tuple[str, str, Dict[str, Any] | None]:
    try:
        record = _validated_record(values)
    except ValidationError as exc:
        message, details_json = _error_output("Input error: " + _format_validation_error(exc))
        return message, details_json, None

    try:
        artifacts = _get_artifacts()
        prob = float(predict_proba(record, artifacts)[0][0])
        thresholds = artifacts.thresholds or {}
        thr_f1 = float(thresholds.get("max_f1", {}).get("threshold", 0.5))
        thr_hp = float(thresholds.get("high_precision", {}).get("threshold", 0.5))
        timestamp_utc = _format_utc(datetime.now(timezone.utc))
        model_ts = _model_timestamp_utc() or "unknown"
        summary, details_json = _format_prediction_output(
            prob, thr_f1, thr_hp, timestamp_utc, model_ts, record
        )
        return summary, details_json, record
    except Exception as exc:
        logger.exception("Prediction failed")
        message, details_json = _error_output(f"Prediction failed: {exc}", exc)
        return message, details_json, None


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
/* Base */
html, body { height: 100%; margin: 0; }

/* Background layer */
#app-bg{
  position: fixed;
  inset: 0;
  z-index: 0;
  background: url('/static/background.png') center/cover no-repeat fixed;
  pointer-events: none;

  /* reduce busy background */
  filter: saturate(0.95) contrast(0.98) brightness(0.90);
}

/* Better overlay: vignette + blur so text/logo in the bg won't compete */
#app-bg::after{
  content:"";
  position:absolute;
  inset:0;
  background:
    radial-gradient(1200px 800px at 50% 0%,
      rgba(255,255,255,0.08),
      rgba(0,0,0,0.35) 65%,
      rgba(0,0,0,0.55) 100%),
    radial-gradient(900px 700px at 50% 85%,
      rgba(0,0,0,0.25),
      rgba(0,0,0,0.55) 100%);
  backdrop-filter: blur(3px);
  -webkit-backdrop-filter: blur(3px);
}

/* Foreground container */
.gradio-container{
  background: transparent !important;
  position: relative;
  z-index: 1;
  min-height: 100vh;
  padding: 28px 18px;
  color: rgba(255,255,255,0.92);
}

/* Main surface (glass) */
.gradio-container > .wrap,
.gradio-container .main{
  max-width: 1100px;
  margin: 0 auto;
  background: rgba(18,18,22,0.70) !important;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  box-shadow: 0 14px 40px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 18px;
}

/* Blocks/cards */
.gradio-container .block,
.gradio-container .gr-block,
.gradio-container .gr-box{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
}

/* Labels/headings */
label, .gradio-container h1, .gradio-container h2, .gradio-container h3{
  color: rgba(255,255,255,0.94) !important;
}
.gradio-container label{
  font-weight: 600 !important;
  letter-spacing: 0.2px;
}

/* Inputs */
.gradio-container input,
.gradio-container textarea,
.gradio-container select{
  background: rgba(0,0,0,0.28) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  color: rgba(255,255,255,0.92) !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus{
  outline: none !important;
  border-color: rgba(255, 60, 60, 0.65) !important;
  box-shadow: 0 0 0 3px rgba(255, 60, 60, 0.18) !important;
}

/* Make the result panel feel like a status card */
#result-col .block,
#result-col .gr-box{
  background: rgba(255,255,255,0.06) !important;
}

#result-output textarea{
  font-size: 1rem !important;
  line-height: 1.45 !important;
  color: rgba(255,255,255,0.98) !important;
  background: rgba(0,0,0,0.35) !important;
}

#result-details textarea{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace !important;
  font-size: 0.85rem !important;
  line-height: 1.35 !important;
  color: rgba(255,255,255,0.9) !important;
  background: rgba(0,0,0,0.35) !important;
}

/* Sticky result on desktop (optional but very useful) */
@media (min-width: 900px){
  #result-col{ position: sticky; top: 18px; align-self: flex-start; }
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Hotel Booking Cancellation Prediction") as demo:
        gr.HTML('<div id="app-bg"></div>')
        gr.Markdown("# Hotel Booking Cancellation Prediction")
        gr.Markdown("**UI version: FORM_V2**")
        gr.Markdown("Note: Only booking-time fields are used by the model.")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Booking basics")
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

                gr.Markdown("### Timing")
                lead_time = gr.Number(label="Lead time", value=30, minimum=0)
                arrival_date = gr.DateTime(
                    label="Arrival date",
                    value=_default_arrival_date(),
                    include_time=False,
                    type="datetime",
                    info="Week number is derived automatically.",
                )

                gr.Markdown("### Stay & guests")
                stays_in_weekend_nights = gr.Number(label="Weekend nights", value=0, minimum=0)
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

                gr.Markdown("### Pricing & policy")
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
                adr = gr.Number(label="ADR (optional)", value=100, minimum=0, maximum=2000)
                required_car_parking_spaces = gr.Number(
                    label="Required car parking spaces", value=0, minimum=0, maximum=10
                )
                total_of_special_requests = gr.Number(
                    label="Total special requests", value=0, minimum=0, maximum=10
                )

                gr.Markdown("### History (if known)")
                with gr.Row():
                    is_repeated_guest = gr.Radio(
                        label="Is repeated guest",
                        choices=[("No", 0), ("Yes", 1)],
                        value=0,
                    )
                    previous_cancellations = gr.Number(
                        label="Previous cancellations", value=0, minimum=0, maximum=20
                    )
                    previous_bookings_not_canceled = gr.Number(
                        label="Previous bookings not canceled", value=0, minimum=0, maximum=50
                    )

                with gr.Accordion("Advanced (optional)", open=False):
                    agent = gr.Textbox(label="Agent", value="UNKNOWN")
                    company = gr.Textbox(label="Company", value="UNKNOWN")

                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary")
                    reset_btn = gr.Button("Reset")
                    flag_btn = gr.Button("Flag")

            with gr.Column(scale=1, elem_id="result-col"):
                gr.Markdown("### Result")
                result = gr.Textbox(
                    label="Output",
                    value="",
                    interactive=False,
                    lines=6,
                    max_lines=8,
                    elem_id="result-output",
                )
                with gr.Accordion("Copy details (JSON)", open=False):
                    details = gr.Textbox(
                        label="Raw JSON",
                        value="",
                        interactive=False,
                        lines=8,
                        max_lines=12,
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
                summary, details_json, record = _predict_output(payload)
                if record is not None:
                    _log_case(record, summary, flagged=False)
                return summary, details_json
            except Exception as exc:
                logger.exception("Prediction handler failed")
                return _error_output(f"Prediction failed: {exc}", exc)

        def _flag(*vals):
            try:
                payload = dict(zip(inputs.keys(), vals))
                summary, details_json, record = _predict_output(payload)
                if record is not None:
                    _log_case(record, summary, flagged=True)
                return summary, details_json
            except Exception as exc:
                logger.exception("Flag handler failed")
                return _error_output(f"Prediction failed: {exc}", exc)

        def _set_loading():
            return "Running prediction...", "", gr.update(interactive=False)

        def _set_ready():
            return gr.update(interactive=True)

        def _reset():
            return (
                "City Hotel",
                30,
                _default_arrival_date(),
                0,
                2,
                2,
                0,
                0,
                "BB",
                "UNKNOWN",
                "Online TA",
                "TA/TO",
                0,
                0,
                0,
                "A",
                "No Deposit",
                "UNKNOWN",
                "UNKNOWN",
                "Transient",
                100,
                0,
                0,
                "",
                "",
            )

        predict_btn.click(
            _set_loading,
            outputs=[result, details, predict_btn],
            queue=False,
        ).then(
            _predict,
            inputs=list(inputs.values()),
            outputs=[result, details],
            show_progress="full",
        ).then(
            _set_ready,
            outputs=predict_btn,
        )

        flag_btn.click(
            _set_loading,
            outputs=[result, details, flag_btn],
            queue=False,
        ).then(
            _flag,
            inputs=list(inputs.values()),
            outputs=[result, details],
            show_progress="full",
        ).then(
            _set_ready,
            outputs=flag_btn,
        )
        reset_btn.click(
            _reset,
            outputs=[
                hotel,
                lead_time,
                arrival_date,
                stays_in_weekend_nights,
                stays_in_week_nights,
                adults,
                children,
                babies,
                meal,
                country,
                market_segment,
                distribution_channel,
                is_repeated_guest,
                previous_cancellations,
                previous_bookings_not_canceled,
                reserved_room_type,
                deposit_type,
                agent,
                company,
                customer_type,
                adr,
                required_car_parking_spaces,
                total_of_special_requests,
                result,
                details,
            ],
        )

    return demo
