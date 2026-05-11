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
from pathlib import Path
from typing import Any

import gradio as gr
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

BACKGROUND_CSS = """
.result-card { padding: 14px 18px; border-radius: 12px;
               background: #f8fafc; border: 1px solid #e2e8f0; }
.result-good { color: #16a34a; font-weight: 700; }
.result-warn { color: #d97706; font-weight: 700; }
.result-bad  { color: #dc2626; font-weight: 700; }
.help-note   { background: #fef9c3; padding: 12px 14px;
               border-left: 4px solid #ca8a04; border-radius: 6px;
               margin: 10px 0; }
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
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            mf = data.get("max_f1", {})
            roc, pr, f1 = mf.get("roc_auc"), mf.get("pr_auc"), mf.get("f1")
            if roc is not None and pr is not None and f1 is not None:
                return (
                    f"**Model performance** — ROC-AUC {float(roc):.3f} · "
                    f"PR-AUC {float(pr):.3f} · F1 {float(f1):.3f}"
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("hero_metrics_load_failed error=%s", exc)
    return "**Model performance** — metrics not available (run `python scripts/train.py`)."


# ---------------------------------------------------------------------------
# Example scenarios for the "Try Examples" tab
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, dict[str, Any]] = {
    "high_risk": {
        "label": "🔴 High cancellation risk",
        "hint": "Long lead time, group booking, prior cancellation, no deposit.",
        "values": {
            "hotel": "City Hotel",
            "arrival_date": date(2025, 10, 15),
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
            "is_repeated_guest": False,
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
            "arrival_date": date(2025, 9, 15),
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
            "is_repeated_guest": False,
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
            "arrival_date": date(2025, 8, 5),
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
            "is_repeated_guest": True,
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
    "is_repeated_guest",
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


def _build_booking(values: dict[str, Any]) -> BookingRequest:
    payload: dict[str, Any] = {
        "hotel": values["hotel"],
        "lead_time": int(values["lead_time"]),
        "arrival_date": _coerce_date(values["arrival_date"]),
        "stays_in_weekend_nights": int(values["weekend_nights"]),
        "stays_in_week_nights": int(values["week_nights"]),
        "adults": max(1, int(values["adults"])),
        "children": int(values["children"]),
        "babies": int(values["babies"]),
        "meal": values["meal"] or None,
        "country": (str(values["country"]).strip().upper() or None) if values["country"] else None,
        "market_segment": values["market_segment"] or None,
        "distribution_channel": values["distribution_channel"] or None,
        "is_repeated_guest": int(bool(values["is_repeated_guest"])),
        "previous_cancellations": int(values["previous_cancellations"]),
        "previous_bookings_not_canceled": int(values["previous_bookings_not_canceled"]),
        "reserved_room_type": values["reserved_room_type"] or None,
        "deposit_type": values["deposit_type"] or None,
        "agent": "Direct",
        "customer_type": values["customer_type"] or None,
        "adr": float(values["adr"]),
        "required_car_parking_spaces": int(values["parking"]),
        "total_of_special_requests": int(values["special_requests"]),
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


def _format_top_features(features: list[dict[str, object]]) -> str:
    if not features:
        return "_Feature attributions not available for this prediction._"
    lines = ["**Top contributing features:**"]
    for item in features:
        feat = item.get("feature", "?")
        val = item.get("value")
        contrib_raw = item.get("contribution", 0.0)
        try:
            contrib = float(contrib_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            contrib = 0.0
        direction = "↑ cancel" if contrib > 0 else "↓ stay"
        lines.append(f"- `{feat}` = `{val}` ({direction}, {contrib:+.3f})")
    return "\n".join(lines)


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

    verdict_f1 = "❗ Likely to cancel" if prob >= thr_f1 else "✅ Likely to stay"
    verdict_hp = "❗ Cancel (high-confidence)" if prob >= thr_hp else "✅ Stay (high-confidence)"
    verdict_cost = "⚠ Flag for outreach" if prob >= thr_cost else "✅ No outreach needed"

    headline = (
        f"### Predicted cancellation risk: " f"<span class='{css}'>{pct_str}</span> · **{band}**"
    )

    parts: list[str] = [
        f"**Balanced policy @ {thr_f1:.2f}:** {verdict_f1}",
        f"**High-precision policy @ {thr_hp:.2f}:** {verdict_hp}",
        f"**Cost-optimal policy @ {thr_cost:.2f}:** {verdict_cost}",
        "",
    ]
    if prob <= 0.0:
        parts.append(_why_zero_note())

    top = explain_prediction(feature_df, artifacts, top_n=5)
    parts.append(_format_top_features(top))

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


def _set_predicting_state() -> tuple[Any, str, str, str]:
    """First step of the prediction click chain — disable the button."""
    return (
        gr.update(value="⏳ Predicting...", interactive=False),
        "_Scoring booking — calibrator and SHAP explainer running..._",
        "",
        "",
    )


def _set_ready_state() -> Any:
    """Final step of the prediction click chain — re-enable the button."""
    return gr.update(value="Predict", interactive=True)


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
    with gr.Blocks(title="Booking Cancellation Predictor") as ui:
        gr.Markdown("# 🏨 Hotel Booking Cancellation Predictor")
        gr.Markdown(_hero_metrics_line())

        d = _form_defaults()

        with gr.Tab("Predict"):
            with gr.Row():
                # ---------- Input column ----------
                with gr.Column(scale=3):
                    gr.Markdown("### Booking details")

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
                        is_repeated_guest = gr.Checkbox(
                            label="Repeated guest",
                            value=d["is_repeated_guest"],
                        )
                        previous_cancellations = gr.Number(
                            label="Previous cancellations",
                            value=d["previous_cancellations"],
                            precision=0,
                            minimum=0,
                            info=(
                                "Prior cancellations by this guest. "
                                "Even 1 dramatically increases risk."
                            ),
                        )
                        previous_bookings_not_canceled = gr.Number(
                            label="Previous bookings (not cancelled)",
                            value=d["previous_bookings_not_canceled"],
                            precision=0,
                            minimum=0,
                        )
                        parking = gr.Number(
                            label="Parking spaces requested",
                            value=d["parking"],
                            precision=0,
                            minimum=0,
                        )

                    with gr.Row():
                        predict_btn = gr.Button("Predict", variant="primary", size="lg", scale=3)
                        reset_btn = gr.Button("Reset", variant="secondary", size="lg", scale=1)

                # ---------- Output column ----------
                with gr.Column(scale=2):
                    gr.Markdown("### Result")
                    headline_out = gr.Markdown(
                        "_Run a prediction to see results._",
                        elem_classes=["result-card"],
                    )
                    details_out = gr.Markdown("")
                    with gr.Accordion("Raw response (JSON)", open=False):
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
                is_repeated_guest,
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

        with gr.Tab("Try Examples"):
            gr.Markdown("### Pre-loaded scenarios")
            gr.Markdown(
                "Click any button to fill the form on the **Predict** tab, "
                "then switch to that tab and press **Predict**."
            )
            for key, ex in EXAMPLES.items():
                with gr.Row():
                    btn = gr.Button(ex["label"], size="lg")
                    gr.Markdown(ex["hint"])
                    btn.click(
                        fn=lambda k=key: _populate_from_example(k),
                        outputs=input_components,
                    )

        with gr.Tab("Help & Troubleshooting"):
            gr.Markdown(HELP_MARKDOWN)

        # R7 — on page load, honour the ?demo=1 query param.
        ui.load(fn=_on_page_load, outputs=input_components)

    return ui
