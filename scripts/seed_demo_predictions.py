"""Seed the prediction log with 30 varied scenarios for the Power BI demo.

Useful when:
   * You've just reset the DB (`python scripts/export_predictions.py --reset`)
   * You want a fresh, diverse dataset for the Power BI dashboard

The 30 scenarios are designed to populate every cell of the
customer_type × market_segment heatmap, span the agent dropdown, mix
risk tiers (low/medium/high), and cover the deposit_type extremes
(No Deposit / Non Refund / Refundable).

Calls log_prediction() directly (not the HTTP endpoint), so the server
does NOT need to be running. The CSV is refreshed at the end.

Usage:
    python scripts/seed_demo_predictions.py

After running, point Power BI at data/predictions/predictions_live.csv
and you'll have 30 rows spanning every interesting axis.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Allow `python scripts/seed_demo_predictions.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.schemas import BookingRequest  # noqa: E402
from src.config import (  # noqa: E402
    RISK_TIER_HIGH_THRESHOLD,
    RISK_TIER_MEDIUM_THRESHOLD,
)
from src.serving.inference import (  # noqa: E402
    explain_prediction,
    get_cached_artifacts,
    predict_proba,
)
from src.serving.prediction_log import (  # noqa: E402
    export_to_csv,
    log_prediction,
)
from src.utils.thresholds import resolve_thresholds  # noqa: E402


def _base() -> dict[str, Any]:
    """Common defaults that every scenario starts from."""
    return {
        "hotel": "City Hotel",
        "lead_time": 60,
        "arrival_date_year": 2026,
        "arrival_date_month": "August",
        "arrival_date_week_number": 32,
        "arrival_date_day_of_month": 12,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "deposit_type": "No Deposit",
        "agent": "9",
        "company": None,
        "customer_type": "Transient",
        "adr": 100.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
    }


def _scenario(overrides: dict[str, Any]) -> dict[str, Any]:
    rec = _base()
    rec.update(overrides)
    # Vary the arrival date by lead_time so the timestamps look realistic
    arrival = date.today() + timedelta(days=rec["lead_time"])
    rec["arrival_date_year"] = arrival.year
    rec["arrival_date_month"] = arrival.strftime("%B")
    rec["arrival_date_week_number"] = int(arrival.isocalendar().week)
    rec["arrival_date_day_of_month"] = arrival.day
    return rec


# Thirty varied scenarios, organised by likely risk tier
SCENARIOS: list[dict[str, Any]] = [
    # === 6× low-risk Direct bookings ===
    {
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "agent": "0",
        "lead_time": 7,
        "country": "PRT",
        "adr": 110.0,
        "total_of_special_requests": 2,
    },
    {
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "agent": "0",
        "lead_time": 14,
        "country": "ESP",
        "adr": 95.0,
        "required_car_parking_spaces": 1,
        "total_of_special_requests": 1,
    },
    {
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "agent": "0",
        "lead_time": 21,
        "country": "FRA",
        "adr": 130.0,
        "hotel": "Resort Hotel",
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 1,
    },
    {
        "market_segment": "Corporate",
        "distribution_channel": "Corporate",
        "agent": "0",
        "customer_type": "Contract",
        "lead_time": 14,
        "country": "DEU",
        "adr": 75.0,
        "previous_bookings_not_canceled": 8,
        "is_repeated_guest": 1,
    },
    {
        "market_segment": "Corporate",
        "distribution_channel": "Corporate",
        "agent": "0",
        "customer_type": "Contract",
        "lead_time": 21,
        "country": "GBR",
        "adr": 85.0,
        "previous_bookings_not_canceled": 12,
        "is_repeated_guest": 1,
    },
    {
        "market_segment": "Complementary",
        "distribution_channel": "Direct",
        "agent": "0",
        "lead_time": 5,
        "country": "PRT",
        "adr": 0.0,
        "previous_bookings_not_canceled": 3,
        "is_repeated_guest": 1,
    },
    # === 8× medium-risk Online TA bookings (the dataset baseline) ===
    {"market_segment": "Online TA", "agent": "9", "lead_time": 60, "country": "GBR", "adr": 95.0},
    {
        "market_segment": "Online TA",
        "agent": "9",
        "lead_time": 90,
        "country": "ITA",
        "adr": 110.0,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 3,
    },
    {
        "market_segment": "Online TA",
        "agent": "240",
        "lead_time": 75,
        "country": "USA",
        "adr": 145.0,
        "hotel": "Resort Hotel",
    },
    {
        "market_segment": "Online TA",
        "agent": "240",
        "lead_time": 120,
        "country": "DEU",
        "adr": 88.0,
        "children": 1,
    },
    {
        "market_segment": "Online TA",
        "agent": "14",
        "lead_time": 45,
        "country": "BRA",
        "adr": 125.0,
        "total_of_special_requests": 1,
    },
    {
        "market_segment": "Online TA",
        "agent": "250",
        "lead_time": 100,
        "country": "FRA",
        "adr": 102.0,
        "adults": 1,
        "customer_type": "Transient",
    },
    {
        "market_segment": "Offline TA/TO",
        "agent": "6",
        "lead_time": 150,
        "country": "IRL",
        "adr": 89.0,
        "distribution_channel": "TA/TO",
    },
    {
        "market_segment": "Offline TA/TO",
        "agent": "7",
        "lead_time": 130,
        "country": "ESP",
        "adr": 80.0,
        "distribution_channel": "TA/TO",
        "customer_type": "Transient-Party",
    },
    # === 6× high-risk Groups bookings (the long-lead-time canceller pattern) ===
    {
        "market_segment": "Groups",
        "distribution_channel": "TA/TO",
        "agent": "1",
        "lead_time": 200,
        "country": "GBR",
        "adr": 70.0,
        "customer_type": "Transient",
    },
    {
        "market_segment": "Groups",
        "distribution_channel": "TA/TO",
        "agent": "1",
        "lead_time": 220,
        "country": "PRT",
        "adr": 65.0,
        "customer_type": "Transient",
        "adults": 1,
    },
    {
        "market_segment": "Groups",
        "distribution_channel": "TA/TO",
        "agent": "9",
        "lead_time": 180,
        "country": "ESP",
        "adr": 75.0,
        "stays_in_week_nights": 4,
    },
    {
        "market_segment": "Groups",
        "distribution_channel": "TA/TO",
        "agent": "3",
        "lead_time": 160,
        "country": "FRA",
        "adr": 82.0,
        "previous_cancellations": 1,
    },
    {
        "market_segment": "Groups",
        "distribution_channel": "TA/TO",
        "agent": "37",
        "lead_time": 175,
        "country": "ITA",
        "adr": 78.0,
        "customer_type": "Transient-Party",
    },
    {
        "market_segment": "Groups",
        "distribution_channel": "TA/TO",
        "agent": "1",
        "lead_time": 240,
        "country": "DEU",
        "adr": 60.0,
        "adults": 1,
        "customer_type": "Transient",
    },
    # === 4× Non Refund bookings (deposit-pattern outlier — ~99% cancel) ===
    {
        "deposit_type": "Non Refund",
        "market_segment": "Online TA",
        "agent": "9",
        "lead_time": 180,
        "country": "PRT",
        "adr": 90.0,
    },
    {
        "deposit_type": "Non Refund",
        "market_segment": "Offline TA/TO",
        "agent": "6",
        "lead_time": 220,
        "country": "GBR",
        "adr": 85.0,
        "distribution_channel": "TA/TO",
    },
    {
        "deposit_type": "Non Refund",
        "market_segment": "Groups",
        "agent": "1",
        "lead_time": 200,
        "country": "ESP",
        "adr": 70.0,
    },
    {
        "deposit_type": "Non Refund",
        "market_segment": "Online TA",
        "agent": "240",
        "lead_time": 250,
        "country": "FRA",
        "adr": 95.0,
    },
    # === 3× Refundable bookings (rare segment) ===
    {
        "deposit_type": "Refundable",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "agent": "0",
        "lead_time": 90,
        "country": "PRT",
        "adr": 120.0,
    },
    {
        "deposit_type": "Refundable",
        "market_segment": "Corporate",
        "agent": "0",
        "lead_time": 30,
        "country": "DEU",
        "adr": 85.0,
        "customer_type": "Contract",
    },
    {
        "deposit_type": "Refundable",
        "market_segment": "Online TA",
        "agent": "9",
        "lead_time": 60,
        "country": "GBR",
        "adr": 100.0,
    },
    # === 3× Aviation (airline crew layover, near-100% show rate) ===
    {
        "market_segment": "Aviation",
        "distribution_channel": "Corporate",
        "agent": "0",
        "customer_type": "Contract",
        "lead_time": 4,
        "country": "PRT",
        "adr": 100.0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 1,
        "adults": 1,
    },
    {
        "market_segment": "Aviation",
        "distribution_channel": "Corporate",
        "agent": "0",
        "customer_type": "Transient",
        "lead_time": 2,
        "country": "USA",
        "adr": 95.0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 1,
        "adults": 1,
    },
    {
        "market_segment": "Aviation",
        "distribution_channel": "Corporate",
        "agent": "0",
        "customer_type": "Transient",
        "lead_time": 7,
        "country": "GBR",
        "adr": 105.0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 1,
        "adults": 1,
    },
]


def main() -> int:
    artifacts = get_cached_artifacts()
    thresholds, _, _, _ = resolve_thresholds(artifacts.thresholds or {})
    thr_f1 = thresholds["max_f1"]
    thr_hp = thresholds["high_precision"]
    thr_cost = thresholds["cost_sensitive"]

    print(f"Seeding {len(SCENARIOS)} demo predictions...")
    for i, overrides in enumerate(SCENARIOS, 1):
        rec = _scenario(overrides)
        booking = BookingRequest.model_validate(rec)
        record = booking.model_dump(exclude={"arrival_date"})
        probs, feature_df = predict_proba(record, artifacts)
        prob = float(probs[0])
        top = explain_prediction(feature_df, artifacts, top_n=5)

        if prob >= RISK_TIER_HIGH_THRESHOLD:
            risk_tier = "high"
        elif prob >= RISK_TIER_MEDIUM_THRESHOLD:
            risk_tier = "medium"
        else:
            risk_tier = "low"

        response = {
            "probability": prob,
            "label_high_precision": int(prob >= thr_hp),
            "label_max_f1": int(prob >= thr_f1),
            "label_cost_sensitive": int(prob >= thr_cost),
            "risk_tier": risk_tier,
            "threshold_high_precision": thr_hp,
            "threshold_max_f1": thr_f1,
            "threshold_cost_sensitive": thr_cost,
            "cost_threshold_source": "seed",
            "cost_threshold_fallback_used": False,
            "alerts": [],
            "top_features": top,
        }
        log_prediction(booking.model_dump(mode="json"), response)
        seg = rec["market_segment"]
        dep = rec["deposit_type"]
        print(
            f"  {i:2d}. {seg:15s} {dep:11s} lead={rec['lead_time']:3d}d "
            f"agent={rec['agent']:4s}  ->  prob={prob:.3f}  ({risk_tier})"
        )

    n = export_to_csv()
    print(f"\nExported {n} rows to data/predictions/predictions_live.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
