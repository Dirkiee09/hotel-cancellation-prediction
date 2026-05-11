"""Pre-demo readiness check — one command to verify everything is good to go.

Runs four progressive checks:
    1. Required artifacts present on disk
    2. Model + calibrator load without errors
    3. Predict returns sensible probabilities on canonical High/Low scenarios
    4. If a server is running on :8000, /healthz responds (optional — non-fatal)

Exit code 0 when checks 1-3 all pass.  Check 4 is informational only.

Usage:
    python scripts/demo_check.py
    make demo-check
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Plain ASCII status markers — Windows terminals can choke on Unicode.
# B105 suppressed below: bandit flags PASS as a "hardcoded password" purely
# from the variable name; it is a UI status marker, not a credential.
PASS = "[ OK ]"  # nosec B105
FAIL = "[FAIL]"
WARN = "[WARN]"

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "artifacts/best_model.pkl",
    "artifacts/probability_calibrator.pkl",
    "artifacts/thresholds.json",
    "artifacts/feature_columns.json",
    "artifacts/model_metadata.json",
    "reports/metrics.json",
)


HIGH_RISK_PAYLOAD: dict[str, Any] = {
    "hotel": "City Hotel",
    "lead_time": 200,
    "arrival_date": date(2025, 10, 15),
    "stays_in_weekend_nights": 0,
    "stays_in_week_nights": 3,
    "adults": 1,
    "children": 0,
    "babies": 0,
    "country": "PRT",
    "meal": "BB",
    "market_segment": "Groups",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 1,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "agent": "Direct",
    "customer_type": "Transient",
    "adr": 80.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
}

LOW_RISK_PAYLOAD: dict[str, Any] = {
    "hotel": "Resort Hotel",
    "lead_time": 5,
    "arrival_date": date(2025, 8, 5),
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "country": "PRT",
    "meal": "BB",
    "market_segment": "Direct",
    "distribution_channel": "Direct",
    "is_repeated_guest": 1,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 10,
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "agent": "Direct",
    "customer_type": "Transient",
    "adr": 90.0,
    "required_car_parking_spaces": 1,
    "total_of_special_requests": 2,
}


def _enrich_arrival_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Add the discrete arrival_date_* fields that BookingRequest expects."""
    arrival = payload["arrival_date"]
    months = [
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
    enriched = dict(payload)
    enriched["arrival_date_year"] = arrival.year
    enriched["arrival_date_month"] = months[arrival.month - 1]
    enriched["arrival_date_week_number"] = int(arrival.isocalendar().week)
    enriched["arrival_date_day_of_month"] = arrival.day
    return enriched


def check_artifacts() -> bool:
    print("\n[1/4] Required artifacts present")
    missing: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        path = ROOT / rel
        if path.exists():
            print(f"  {PASS} {rel}")
        else:
            print(f"  {FAIL} {rel}")
            missing.append(rel)
    if missing:
        print("\n  Fix: run `python scripts/train.py` to regenerate artifacts.")
        return False
    return True


def check_model_loads() -> bool:
    print("\n[2/4] Model + calibrator load")
    try:
        from src.serving.inference import load_artifacts  # noqa: PLC0415
    except ImportError as exc:
        print(f"  {FAIL} import failure: {exc}")
        return False

    try:
        artifacts = load_artifacts()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"  {FAIL} load_artifacts(): {exc}")
        return False

    n_features = len(artifacts.feature_columns)
    has_cal = artifacts.calibrator is not None
    has_thr = bool(artifacts.thresholds)
    has_meta = bool(artifacts.metadata)
    print(f"  {PASS} pipeline loaded ({n_features} features)")
    print(f"  {PASS if has_cal  else FAIL} calibrator present")
    print(f"  {PASS if has_thr  else FAIL} thresholds present")
    print(f"  {PASS if has_meta else FAIL} metadata present")
    return has_cal and has_thr and has_meta


def check_predictions() -> bool:
    print("\n[3/4] Predictions match expected ranges")
    from src.serving.inference import load_artifacts, predict_proba  # noqa: PLC0415

    artifacts = load_artifacts()
    scenarios: list[tuple[str, dict[str, Any], Callable[[float], bool], str]] = [
        ("high-risk", HIGH_RISK_PAYLOAD, lambda p: p >= 0.70, "expected >=70%"),
        ("low-risk", LOW_RISK_PAYLOAD, lambda p: p < 0.10, "expected <10%"),
    ]
    all_ok = True
    for name, payload, predicate, expectation in scenarios:
        enriched = _enrich_arrival_fields(payload)
        try:
            probs, _ = predict_proba(enriched, artifacts)
            prob = float(probs[0])
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            print(f"  {FAIL} {name}: prediction raised {exc}")
            all_ok = False
            continue
        status = PASS if predicate(prob) else FAIL
        print(f"  {status} {name}: {prob * 100:6.2f}% ({expectation})")
        if not predicate(prob):
            all_ok = False
    return all_ok


def check_live_server(port: int = 8000) -> bool:
    print(f"\n[4/4] Live server on :{port} (optional)")
    try:
        # B310 suppressed below: hardcoded localhost URL, no user-controlled scheme.
        url = f"http://127.0.0.1:{port}/healthz"
        with urllib.request.urlopen(url, timeout=2) as resp:  # nosec B310
            data = json.load(resp)
        if data.get("status") == "ok":
            print(f"  {PASS} /healthz returned ok (service={data.get('service')})")
            return True
        print(f"  {WARN} /healthz unexpected payload: {data}")
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
        print(f"  {WARN} no server responding on :{port}")
        print(f"          start one with: python demo/start_server.py --port {port}")
        return True


def main() -> int:
    print("Demo readiness check")
    print("=" * 50)

    artifacts_ok = check_artifacts()
    if not artifacts_ok:
        print("\nResult: NOT READY (missing artifacts)")
        return 1

    model_ok = check_model_loads()
    if not model_ok:
        print("\nResult: NOT READY (model failed to load)")
        return 1

    preds_ok = check_predictions()
    check_live_server()

    if preds_ok:
        print("\nResult: READY")
        return 0
    print("\nResult: NOT READY (predictions outside expected range)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
