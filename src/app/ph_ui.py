"""Gradio UI for the PH (Philippine dataset) sub-study.

Minimal form-driven UI for the PH cancellation model. Designed to be
launched either standalone (``python src/app/ph_ui.py``) on port 7861 or
mounted inside the FastAPI app at ``/ui``.

This UI is deliberately simpler than the Portugal Gradio UI in src/app/ui.py:
- No cohort distribution stats
- No example bookings dropdown
- No PowerBI export hook

Every /predict still appends to ``data/predictions/ph_predictions.sqlite``
via the FastAPI BackgroundTask wired in ``ph_main.py``. A prominent banner
explains the Philippine dataset's structural properties so a defense
audience reads the test metrics correctly.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.app.ph_schemas import PHBookingRequest
from src.config import (
    PH_DATA_PATH,
    PH_REPORTS_DIR,
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
)
from src.serving.inference_ph import (
    explain_ph_prediction,
    get_cached_ph_artifacts,
    predict_ph,
)

logger = logging.getLogger(__name__)

DATASET_BANNER_HTML = """
<div style="
    border-left: 6px solid #2563eb;
    background: #dbeafe;
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 14px;
    font-family: ui-sans-serif, system-ui, sans-serif;
">
    <strong style="font-size: 1.05rem; color: #1e3a8a;">
        Philippine resort sub-study — real-data demonstration server
    </strong>
    <p style="margin: 8px 0 0; color: #1e3a8a; line-height: 1.45;">
        This server runs the PH cancellation model trained on the
        <strong>real Punta Villa Resort PMS export (193 booking records,
        2022-2025)</strong>. The dataset is small (n_test ≈ 20 rows) and
        bootstrap CIs on PR-AUC span roughly ±15 percentage points, so
        treat the displayed metrics as directional rather than as
        production-grade headlines.
    </p>
</div>
"""


def _load_categorical_choices(*column_aliases: str, fallback: list[str]) -> list[str]:
    """Read distinct values for a column from test predictions or raw CSV.

    Tries each candidate column name in turn (the cleaned report uses
    the project-canonical name, the raw CSV uses the PMS export name).
    """
    candidates: list[Path] = [
        PH_REPORTS_DIR / "ph_test_predictions.csv",
        PH_DATA_PATH,
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:  # pragma: no cover — defensive
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
    "Room_Type_Reserved",
    fallback=["Standard Room", "De Luxe Room", "Group Room", "Presidential Room"],
)
_DEPOSIT_TYPES = _load_categorical_choices(
    "deposit_type",
    "Deposit_Type",
    fallback=["No Deposit", "Partial", "Non-Refundable"],
)


def _format_pct(prob: float) -> str:
    return f"{prob * 100:.1f}%"


def _risk_band(prob: float) -> tuple[str, str, str]:
    """Return (label, color, recommendation) for the risk tier."""
    if prob >= RISK_TIER_HIGH_THRESHOLD:
        return "HIGH", "#dc2626", "Contact guest to confirm; consider overbooking buffer."
    if prob >= RISK_TIER_MEDIUM_THRESHOLD:
        return "MEDIUM", "#d97706", "Monitor; gentle reminder email a week before arrival."
    return "LOW", "#16a34a", "Standard handling."


def _format_top_features(top: list[dict[str, Any]]) -> str:
    if not top:
        return "<em>No feature contributions returned (SHAP unavailable).</em>"
    rows = []
    for entry in top:
        feature = entry.get("feature", "?")
        value = entry.get("value", "?")
        contrib = entry.get("contribution", 0.0)
        direction = "↑ toward cancel" if (contrib or 0.0) > 0 else "↓ toward stay"
        color = "#dc2626" if (contrib or 0.0) > 0 else "#16a34a"
        rows.append(
            f"<tr>"
            f"<td style='padding:6px 10px'><code>{feature}</code></td>"
            f"<td style='padding:6px 10px'>{value}</td>"
            f"<td style='padding:6px 10px;color:{color};font-weight:600'>"
            f"{contrib:+.4f} <span style='opacity:0.7'>({direction})</span>"
            f"</td>"
            f"</tr>"
        )
    body = "".join(rows)
    return (
        "<table style='border-collapse:collapse;margin-top:8px'>"
        "<thead><tr style='background:#f3f4f6'>"
        "<th style='padding:6px 10px;text-align:left'>Feature</th>"
        "<th style='padding:6px 10px;text-align:left'>Value</th>"
        "<th style='padding:6px 10px;text-align:left'>SHAP contribution</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _predict_via_ui(
    lead_time: float,
    arrival_date_str: str,
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
    """Glue the Gradio inputs to the predict_ph() pipeline."""
    try:
        artifacts = get_cached_ph_artifacts()
    except FileNotFoundError as exc:
        msg = (
            f"<div style='color:#dc2626'>PH artifacts unavailable: {exc}<br>"
            "Run <code>python scripts/train_ph.py</code> first.</div>"
        )
        return msg, "", ""

    try:
        arrival = date.fromisoformat(arrival_date_str)
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
    except (ValueError, TypeError) as exc:
        return f"<div style='color:#dc2626'>Invalid input: {exc}</div>", "", ""

    prob, feature_df = predict_ph(booking.to_inference_dict(), artifacts)
    thresholds = artifacts.thresholds or {}
    thr_f1 = float(thresholds.get("max_f1", {}).get("threshold", 0.5))
    thr_hp = float(thresholds.get("high_precision", {}).get("threshold", 0.9))
    band_label, band_color, recommendation = _risk_band(prob)
    top = explain_ph_prediction(feature_df, artifacts, top_n=5)

    summary = f"""
    <div style="font-family:ui-sans-serif,system-ui,sans-serif">
        <div style="font-size:2.1rem;font-weight:700;color:{band_color}">
            {_format_pct(prob)} cancel probability
        </div>
        <div style="margin-top:4px;font-size:1.1rem">
            Risk tier:
            <strong style="color:{band_color}">{band_label}</strong>
        </div>
        <div style="margin-top:8px;color:#374151">
            Recommendation: {recommendation}
        </div>
    </div>
    """

    thresholds_block = f"""
    <table style="border-collapse:collapse;margin-top:8px;font-family:ui-sans-serif,system-ui,sans-serif">
        <tr style="background:#f3f4f6">
            <th style="padding:6px 10px;text-align:left">Policy</th>
            <th style="padding:6px 10px;text-align:left">Threshold</th>
            <th style="padding:6px 10px;text-align:left">Decision</th>
        </tr>
        <tr>
            <td style="padding:6px 10px">max_f1</td>
            <td style="padding:6px 10px">{thr_f1:.3f}</td>
            <td style="padding:6px 10px;font-weight:600">
                {'FLAG as cancel' if prob >= thr_f1 else 'KEEP as stay'}
            </td>
        </tr>
        <tr>
            <td style="padding:6px 10px">high_precision</td>
            <td style="padding:6px 10px">{thr_hp:.3f}</td>
            <td style="padding:6px 10px;font-weight:600">
                {'FLAG as cancel' if prob >= thr_hp else 'KEEP as stay'}
            </td>
        </tr>
    </table>
    """

    return summary, thresholds_block, _format_top_features(top)


def build_ph_ui() -> gr.Blocks:
    """Construct the PH Gradio Blocks UI."""
    with gr.Blocks(title="PH Cancellation — Philippine Resort Sub-Study") as demo:
        gr.HTML(DATASET_BANNER_HTML)
        gr.Markdown(
            "# PH Cancellation Predictor — Philippine Resort Sub-Study\n\n"
            "Enter a booking below and the model returns a cancellation "
            "probability, a risk tier (low/medium/high), and which features "
            "pushed the prediction in which direction. Backed by the real "
            "Punta Villa Resort PMS export (193 bookings, 2022-2025)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Booking details")
                lead_time = gr.Number(label="Lead time (days)", value=30, precision=0, minimum=0)
                arrival_date_str = gr.Textbox(
                    label="Arrival date (YYYY-MM-DD)",
                    value="2025-12-15",
                )
                with gr.Row():
                    weekend_nights = gr.Number(
                        label="Weekend nights", value=1, precision=0, minimum=0
                    )
                    week_nights = gr.Number(label="Week nights", value=2, precision=0, minimum=0)
                with gr.Row():
                    adults = gr.Number(label="Adults", value=2, precision=0, minimum=1)
                    children = gr.Number(label="Children", value=0, precision=0, minimum=0)
                    babies = gr.Number(label="Babies", value=0, precision=0, minimum=0)
                adr = gr.Number(label="ADR (PHP)", value=3500.0, minimum=0)
                reserved_room_type = gr.Dropdown(
                    label="Reserved room type",
                    choices=_ROOM_TYPES,
                    value=_ROOM_TYPES[0] if _ROOM_TYPES else None,
                )
                deposit_type = gr.Dropdown(
                    label="Deposit policy",
                    choices=_DEPOSIT_TYPES,
                    value=_DEPOSIT_TYPES[0] if _DEPOSIT_TYPES else None,
                )
                total_of_special_requests = gr.Number(
                    label="Special requests",
                    value=0,
                    precision=0,
                    minimum=0,
                )
                submit = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Result")
                result_summary = gr.HTML()
                gr.Markdown("### Threshold policies")
                thresholds_block = gr.HTML()
                gr.Markdown("### Top contributing features")
                features_block = gr.HTML()

        submit.click(
            fn=_predict_via_ui,
            inputs=[
                lead_time,
                arrival_date_str,
                weekend_nights,
                week_nights,
                adults,
                children,
                babies,
                adr,
                reserved_room_type,
                deposit_type,
                total_of_special_requests,
            ],
            outputs=[result_summary, thresholds_block, features_block],
        )

        gr.Markdown(
            "---\n"
            "**How to read these results**: the model was trained on 193 "
            "real bookings (n_test ≈ 20); bootstrap 95 % CIs on PR-AUC span "
            "roughly ±15 percentage points. Treat the probability as a "
            "directional signal rather than a calibrated production "
            "prediction. The top contributing features below show which "
            "fields of *this specific booking* pushed the score up or down."
        )

    return demo


def main() -> None:  # pragma: no cover — manual launch only
    """Launch the PH UI standalone on port 7861."""
    logging.basicConfig(level=logging.INFO)
    demo = build_ph_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        css=".gradio-container {max-width: 1100px; margin: auto}",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
