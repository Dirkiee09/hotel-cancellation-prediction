"""Validate training artifacts are aligned with serving expectations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, BOOKING_TIME_FEATURES, LEAKAGE_COLS, MODEL_SELECTION_POLICY
from src.serving.inference import load_artifacts, predict_proba
from src.utils import configure_logging

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_thresholds(thresholds: dict[str, Any]) -> None:
    for policy in ("high_precision", "max_f1", "cost_sensitive"):
        policy_payload = thresholds.get(policy)
        if not isinstance(policy_payload, dict):
            raise ValueError(f"Missing threshold policy: {policy}")
        threshold = float(policy_payload.get("threshold", -1.0))
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"Invalid threshold for {policy}: {threshold}")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_hashes(artifacts_dir: Path, metadata: dict[str, Any]) -> None:
    hashes_path = artifacts_dir / "hashes.json"
    if not hashes_path.exists():
        raise FileNotFoundError("Missing hashes.json in artifacts directory")

    hashes_payload = _load_json(hashes_path)
    files = hashes_payload.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("hashes.json missing non-empty 'files' section")

    for filename, expected_hash in files.items():
        target = artifacts_dir / filename
        if not target.exists():
            raise FileNotFoundError(f"Hash target missing: {target}")
        actual_hash = _sha256_file(target)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Artifact hash mismatch for {filename}: expected={expected_hash} actual={actual_hash}"
            )

    metadata_lineage = metadata.get("lineage", {})
    metadata_hashes = metadata_lineage.get("artifacts", {})
    metadata_files = metadata_hashes.get("files")
    if metadata_files is not None and metadata_files != files:
        raise ValueError("model_metadata lineage hashes do not match hashes.json")


def _smoke_predict(artifacts_dir: Path) -> None:
    artifacts = load_artifacts(artifacts_dir)
    row = {
        "hotel": "City Hotel",
        "lead_time": 30,
        "arrival_date_year": 2017,
        "arrival_date_month": "July",
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 1,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "UNKNOWN",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
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
    probs, _ = predict_proba(row, artifacts)
    prob = float(probs[0])
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"Predicted probability out of range: {prob}")


def validate_artifacts(artifacts_dir: Path) -> None:
    required = [
        "best_model.pkl",
        "probability_calibrator.pkl",
        "feature_columns.json",
        "thresholds.json",
        "hashes.json",
    ]
    missing = [name for name in required if not (artifacts_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    feature_payload = _load_json(artifacts_dir / "feature_columns.json")
    features = feature_payload.get("features")
    if features != BOOKING_TIME_FEATURES:
        raise ValueError("feature_columns.json does not match BOOKING_TIME_FEATURES")

    leaking = sorted(set(features).intersection(LEAKAGE_COLS))
    if leaking:
        raise ValueError(f"Leakage columns found in features: {leaking}")

    thresholds = _load_json(artifacts_dir / "thresholds.json")
    _validate_thresholds(thresholds)

    metadata_path = artifacts_dir / "model_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        policy = metadata.get("model_selection_policy")
        if policy != MODEL_SELECTION_POLICY:
            raise ValueError(
                f"model_metadata.json policy mismatch expected={MODEL_SELECTION_POLICY} actual={policy}"
            )
    _validate_hashes(artifacts_dir, metadata)

    _smoke_predict(artifacts_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate artifact + serving consistency.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory containing model artifacts.",
    )
    args = parser.parse_args()

    configure_logging()
    artifacts_dir = args.artifacts_dir.resolve()
    validate_artifacts(artifacts_dir)
    logger.info("artifact_consistency_ok artifacts_dir=%s", artifacts_dir)


if __name__ == "__main__":
    main()
