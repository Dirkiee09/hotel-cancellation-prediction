"""Send sample predictions to the running server and display results.

Usage:
    1. Start the server:  python demo/start_server.py
    2. In another terminal: python demo/sample_requests.py

Sends 4 contrasting booking scenarios and prints a comparison table.
Useful as a live demo during the thesis defense.
"""

from __future__ import annotations

import json
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_URL = "http://localhost:8000"

# ── Sample bookings: 4 scenarios from low to high risk ───────────────────

SCENARIOS: list[dict] = [
    {
        "name": "Last-minute direct (low risk)",
        "payload": {
            "hotel": "Resort Hotel",
            "lead_time": 3,
            "adults": 2,
            "children": 1,
            "babies": 0,
            "adr": 110.0,
            "arrival_date_year": 2025,
            "arrival_date_month": "June",
            "arrival_date_week_number": 24,
            "arrival_date_day_of_month": 12,
            "stays_in_weekend_nights": 2,
            "stays_in_week_nights": 3,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "agent": "NULL",
            "company": "NULL",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 2,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 1,
        },
    },
    {
        "name": "Online TA summer (medium risk)",
        "payload": {
            "hotel": "City Hotel",
            "lead_time": 120,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "adr": 85.0,
            "arrival_date_year": 2025,
            "arrival_date_month": "August",
            "arrival_date_week_number": 33,
            "arrival_date_day_of_month": 14,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 3,
            "meal": "HB",
            "country": "ESP",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "agent": "240",
            "company": "NULL",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 1,
        },
    },
    {
        "name": "Group far out (high risk)",
        "payload": {
            "hotel": "City Hotel",
            "lead_time": 250,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "adr": 55.0,
            "arrival_date_year": 2025,
            "arrival_date_month": "October",
            "arrival_date_week_number": 42,
            "arrival_date_day_of_month": 15,
            "stays_in_weekend_nights": 0,
            "stays_in_week_nights": 2,
            "meal": "SC",
            "country": "PRT",
            "market_segment": "Groups",
            "distribution_channel": "TA/TO",
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient-Party",
            "agent": "9",
            "company": "NULL",
            "is_repeated_guest": 0,
            "previous_cancellations": 1,
            "previous_bookings_not_canceled": 0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
        },
    },
    {
        "name": "Repeat corporate (very low risk)",
        "payload": {
            "hotel": "City Hotel",
            "lead_time": 14,
            "adults": 1,
            "children": 0,
            "babies": 0,
            "adr": 95.0,
            "arrival_date_year": 2025,
            "arrival_date_month": "March",
            "arrival_date_week_number": 12,
            "arrival_date_day_of_month": 17,
            "stays_in_weekend_nights": 0,
            "stays_in_week_nights": 4,
            "meal": "BB",
            "country": "GBR",
            "market_segment": "Corporate",
            "distribution_channel": "Corporate",
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "agent": "NULL",
            "company": "223",
            "is_repeated_guest": 1,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 8,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 1,
        },
    },
]


def _open_local(req: Request | str, timeout: float):
    """urlopen restricted to the hardcoded localhost BASE_URL (http only)."""
    url = req.full_url if isinstance(req, Request) else req
    if not url.startswith(BASE_URL):
        raise ValueError(f"Refusing non-local URL: {url}")
    return urlopen(req, timeout=timeout)  # noqa: S310  # nosec B310 - scheme checked above


def _post(url: str, data: dict) -> dict:
    req = Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with _open_local(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> None:
    # Check server is running
    try:
        with _open_local(f"{BASE_URL}/healthz", timeout=5):
            pass
    except (URLError, ConnectionError, OSError):
        print("ERROR: Server is not running.")
        print("Start it first:  python demo/start_server.py")
        sys.exit(1)

    # Print model info
    with _open_local(f"{BASE_URL}/model-info", timeout=10) as resp:
        info = json.loads(resp.read())
    print(
        f"Model: {info['model_type']}  |  Features: {info['feature_count']}  |  "
        f"Calibrated: {info['has_calibrator']}"
    )
    print(
        f"Thresholds: max_f1={info['thresholds']['max_f1']:.2f}  "
        f"high_precision={info['thresholds']['high_precision']:.2f}  "
        f"cost_sensitive={info['thresholds']['cost_sensitive']:.2f}"
    )
    print()

    # Run scenarios
    print(f"{'Scenario':<38} {'Prob':>6} {'Risk':<8} {'F1':>4} {'HP':>4} {'Cost':>4}  Top driver")
    print("-" * 95)

    for scenario in SCENARIOS:
        result = _post(f"{BASE_URL}/predict", scenario["payload"])
        prob = result["probability"]
        tier = result["risk_tier"]
        f1 = result["label_max_f1"]
        hp = result["label_high_precision"]
        cost = result["label_cost_sensitive"]

        top = ""
        if result.get("top_features"):
            feat = result["top_features"][0]
            direction = "+" if feat["contribution"] > 0 else "-"
            top = f"{feat['feature']} ({direction})"

        tier_color = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[31m"}.get(tier, "")
        reset = "\033[0m"

        print(
            f"  {scenario['name']:<36} {prob:>5.1%} {tier_color}{tier:<8}{reset} "
            f"{'Y' if f1 else 'N':>4} {'Y' if hp else 'N':>4} {'Y' if cost else 'N':>4}  {top}"
        )

    print()
    print("F1/HP/Cost columns: Y = predicted to cancel under that policy, N = keep")
    print(f"\nAPI docs:  {BASE_URL}/docs")
    print(f"Gradio UI: {BASE_URL}/ui")


if __name__ == "__main__":
    main()
