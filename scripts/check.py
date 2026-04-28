"""Unified quality-gate runner.

Subcommands:
    artifacts   — Validate artifact integrity, hashes, and smoke prediction
    metrics     — Enforce metric quality gates from reports/metrics.json
    sync        — Verify thresholds are consistent across artifacts and reports
    fairness    — Run hyperparameter fairness audit (LightGBM vs XGBoost budget check)
    all         — Run artifacts, metrics, sync, and fairness in sequence
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# artifacts — validate artifact integrity and smoke prediction
# ---------------------------------------------------------------------------


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
                f"Artifact hash mismatch for {filename}: "
                f"expected={expected_hash} actual={actual_hash}"
            )

    metadata_lineage = metadata.get("lineage", {})
    metadata_hashes = metadata_lineage.get("artifacts", {})
    metadata_files = metadata_hashes.get("files")
    if metadata_files is not None and metadata_files != files:
        raise ValueError("model_metadata lineage hashes do not match hashes.json")


def _smoke_predict(artifacts_dir: Path) -> None:
    from src.serving.inference import load_artifacts, predict_proba

    artifacts = load_artifacts(artifacts_dir)
    row = {
        "hotel": "UNKNOWN",
        "lead_time": 30,
        "arrival_date_year": 2024,
        "arrival_date_month": "January",
        "arrival_date_week_number": 1,
        "arrival_date_day_of_month": 1,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "UNKNOWN",
        "country": "UNKNOWN",
        "market_segment": "UNKNOWN",
        "distribution_channel": "UNKNOWN",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "UNKNOWN",
        "deposit_type": "UNKNOWN",
        "agent": "UNKNOWN",
        "company": "UNKNOWN",
        "customer_type": "UNKNOWN",
        "adr": 100.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
    }
    probs, _ = predict_proba(row, artifacts)
    prob = float(probs[0])
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"Predicted probability out of range: {prob}")


def run_artifacts(artifacts_dir: Path | None = None) -> None:
    """Validate training artifacts are aligned with serving expectations."""
    from src.config import (
        ARTIFACTS_DIR,
        BOOKING_TIME_FEATURES,
        LEAKAGE_COLS,
        MODEL_SELECTION_POLICY,
    )

    if artifacts_dir is None:
        artifacts_dir = ARTIFACTS_DIR
    artifacts_dir = artifacts_dir.resolve()

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
                f"model_metadata.json policy mismatch "
                f"expected={MODEL_SELECTION_POLICY} actual={policy}"
            )
    _validate_hashes(artifacts_dir, metadata)
    _smoke_predict(artifacts_dir)
    logger.info("artifact_consistency_ok artifacts_dir=%s", artifacts_dir)


# ---------------------------------------------------------------------------
# metrics — enforce metric quality gates
# ---------------------------------------------------------------------------


def _check_gate(metrics: dict, policy_name: str, gate_key: str, threshold: float) -> str | None:
    metric_name = gate_key.replace("_min", "")
    policy_metrics = metrics.get(policy_name, {})
    if metric_name not in policy_metrics:
        return f"{policy_name}.{metric_name} missing from metrics payload"
    actual = float(policy_metrics[metric_name])
    if actual < threshold:
        return (
            f"{policy_name}.{metric_name}={actual:.6f} is below required minimum "
            f"{threshold:.6f}"
        )
    return None


def _check_global_metrics(metrics: dict) -> list[str]:
    from src.config import METRIC_GATES

    failures: list[str] = []
    for policy_name, gate in METRIC_GATES.items():
        if policy_name not in metrics:
            failures.append(f"Policy '{policy_name}' missing from metrics payload")
            continue
        for gate_key, threshold in gate.items():
            failure = _check_gate(metrics, policy_name, gate_key, float(threshold))
            if failure:
                failures.append(failure)
    return failures


def _check_segment_metrics(segment_payload: dict) -> list[str]:
    from src.config import SEGMENT_METRIC_GATES

    failures: list[str] = []
    segment_cfg = cast(dict[str, Any], SEGMENT_METRIC_GATES)
    expected_policy = str(segment_cfg["policy"])
    expected_dimensions = set(cast(dict[str, int], segment_cfg["dimensions"]).keys())
    metric_gates = cast(dict[str, float], segment_cfg["metrics"])

    payload_policy = str(segment_payload.get("policy"))
    if payload_policy != expected_policy:
        failures.append(
            f"segment_metrics policy mismatch "
            f"expected={expected_policy} actual={payload_policy}"
        )

    rows = segment_payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        failures.append("segment_metrics rows missing or empty")
        return failures

    gated_rows = [row for row in rows if bool(row.get("gated"))]
    if not gated_rows:
        failures.append("No gated segment rows available for validation")
        return failures

    present_dimensions = {str(row.get("dimension")) for row in gated_rows}
    missing_dimensions = sorted(expected_dimensions - present_dimensions)
    if missing_dimensions:
        failures.append(f"Gated rows missing expected dimensions: {missing_dimensions}")

    for row in gated_rows:
        dimension = str(row.get("dimension"))
        segment = str(row.get("segment"))
        for gate_key, threshold in metric_gates.items():
            metric_name = gate_key.replace("_min", "")
            raw_value = row.get(metric_name)
            if raw_value is None:
                failures.append(f"{dimension}.{segment}.{metric_name} missing")
                continue
            actual = float(raw_value)
            if actual < float(threshold):
                failures.append(
                    f"{dimension}.{segment}.{metric_name}={actual:.6f} is below "
                    f"required minimum {float(threshold):.6f}"
                )
    return failures


def run_metrics(
    metrics_path: Path | None = None,
    segment_metrics_path: Path | None = None,
) -> None:
    """Enforce minimum model quality thresholds from reports/metrics.json."""
    from src.config import REPORTS_DIR

    if metrics_path is None:
        metrics_path = REPORTS_DIR / "metrics.json"
    if segment_metrics_path is None:
        segment_metrics_path = REPORTS_DIR / "segment_metrics.json"

    metrics_path = metrics_path.resolve()
    segment_metrics_path = segment_metrics_path.resolve()

    metrics = _load_json(metrics_path)
    segment_payload = _load_json(segment_metrics_path)
    failures = _check_global_metrics(metrics) + _check_segment_metrics(segment_payload)

    if failures:
        joined = "\n - ".join(failures)
        raise AssertionError(f"Metric gate failed:\n - {joined}")

    print(f"Metric gate passed: {metrics_path} + {segment_metrics_path}")


# ---------------------------------------------------------------------------
# sync — cross-artifact threshold consistency
# ---------------------------------------------------------------------------

_TOLERANCE = 1e-6


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= _TOLERANCE


def _load_csv_row(path: Path, model_name: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("model", "").strip().lower() == model_name.lower():
                return row
    raise ValueError(f"Model '{model_name}' not found in {path}")


def _run_sync_check(
    artifacts_dir: Path,
    reports_dir: Path,
) -> list[str]:
    """Return a list of mismatch error strings (empty = all consistent)."""
    errors: list[str] = []

    thresholds_path = artifacts_dir / "thresholds.json"
    if not thresholds_path.exists():
        return [f"MISSING: {thresholds_path} — run `make train` first"]
    raw = _load_json(thresholds_path)

    def _get_thr(policy: str) -> float | None:
        payload = raw.get(policy)
        if isinstance(payload, dict):
            val = payload.get("threshold")
            if isinstance(val, int | float):
                return float(val)
        return None

    canon_f1 = _get_thr("max_f1")
    canon_hp = _get_thr("high_precision")
    canon_cost = _get_thr("cost_sensitive")

    if canon_f1 is None or canon_hp is None:
        errors.append("artifacts/thresholds.json: missing max_f1 or high_precision threshold")
        return errors

    # Cross-check thesis summary
    summary_path = reports_dir / "thesis" / "model_family_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        thesis_f1 = summary.get("max_f1_threshold")
        thesis_cost = summary.get("cost_sensitive_threshold")

        if thesis_f1 is not None and not _near(float(thesis_f1), canon_f1):
            errors.append(
                f"max_f1 threshold mismatch: "
                f"artifacts={canon_f1} vs thesis/model_family_summary={thesis_f1}"
            )
        if canon_cost is not None and thesis_cost is not None:
            if not _near(float(thesis_cost), canon_cost):
                errors.append(
                    f"cost_sensitive threshold mismatch: "
                    f"artifacts={canon_cost} vs thesis/model_family_summary={thesis_cost}"
                )
    else:
        logger.warning("sync_check: %s not found — skipping thesis snapshot check", summary_path)

    # Cross-check key metrics between metrics.json and thesis summary
    metrics_path = reports_dir / "metrics.json"
    if metrics_path.exists() and summary_path.exists():
        metrics = _load_json(metrics_path)
        summary = _load_json(summary_path)
        for metric_key, summary_key in [("roc_auc", "roc_auc"), ("pr_auc", "pr_auc")]:
            metrics_val = metrics.get("max_f1", {}).get(metric_key)
            summary_val = summary.get(f"test_{summary_key}") or summary.get(summary_key)
            if metrics_val is not None and summary_val is not None:
                if not _near(float(metrics_val), float(summary_val)):
                    errors.append(
                        f"{metric_key} mismatch: reports/metrics.json={metrics_val} "
                        f"vs thesis/model_family_summary={summary_val}"
                    )

    # Cross-check benchmark CSV
    bench_path = reports_dir / "benchmarks" / "07_thresholds_per_model.csv"
    if bench_path.exists():
        _meta_path = artifacts_dir / "model_metadata.json"
        _champion_name = "lightgbm"
        if _meta_path.exists():
            try:
                _meta = _load_json(_meta_path)
                _champion_name = str(
                    _meta.get("model_type", _meta.get("champion_family", "lightgbm"))
                ).lower()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "sync_check: could not read champion name from %s (%s), "
                    "falling back to '%s'",
                    _meta_path,
                    exc,
                    _champion_name,
                )
        try:
            row = _load_csv_row(bench_path, _champion_name)
        except ValueError as exc:
            errors.append(f"benchmarks/07_thresholds_per_model.csv: {exc}")
            row = {}

        if row:
            bench_f1 = row.get("threshold_max_f1")
            bench_hp = row.get("threshold_high_precision")
            bench_cost = row.get("threshold_cost_sensitive")

            if bench_f1 is not None and not _near(float(bench_f1), canon_f1):
                errors.append(
                    f"max_f1 threshold mismatch: artifacts={canon_f1} vs benchmarks/07={bench_f1}"
                )
            if bench_hp is not None and not _near(float(bench_hp), canon_hp):
                errors.append(
                    f"high_precision threshold mismatch: "
                    f"artifacts={canon_hp} vs benchmarks/07={bench_hp}"
                )
            if canon_cost is not None and bench_cost is not None:
                if not _near(float(bench_cost), canon_cost):
                    errors.append(
                        f"cost_sensitive threshold mismatch: "
                        f"artifacts={canon_cost} vs benchmarks/07={bench_cost}"
                    )
    else:
        logger.warning("sync_check: %s not found — skipping benchmark check", bench_path)

    return errors


def run_sync(
    artifacts_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> None:
    """Verify thresholds are consistent across artifacts, thesis, and benchmarks."""
    from src.config import ARTIFACTS_DIR, REPORTS_DIR

    if artifacts_dir is None:
        artifacts_dir = ARTIFACTS_DIR
    if reports_dir is None:
        reports_dir = REPORTS_DIR

    errors = _run_sync_check(
        artifacts_dir=artifacts_dir.resolve(),
        reports_dir=reports_dir.resolve(),
    )

    if errors:
        logger.error("sync_check FAILED — %d mismatch(es):", len(errors))
        for msg in errors:
            logger.error("  ✗ %s", msg)
        logger.error(
            "Run `make train && make benchmark && make thesis` "
            "to regenerate reports from current artifacts."
        )
        sys.exit(1)

    logger.info(
        "sync_check OK — thresholds consistent across artifacts, thesis snapshot, " "and benchmarks"
    )


# ---------------------------------------------------------------------------
# fairness — hyperparameter fairness audit
# ---------------------------------------------------------------------------


def run_fairness() -> None:
    """Test whether LightGBM > XGBoost is due to algorithm quality or capacity budget."""
    import joblib
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.config import ARTIFACTS_DIR
    from src.data.load import load_raw_data
    from src.features.build import build_preprocessor, split_time_aware
    from src.models.train import train_xgb
    from src.utils.validate_data import clean_raw

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore", message="X does not have valid feature names", category=UserWarning
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

    print("Loading and splitting data...")
    raw = load_raw_data()
    cleaned, _ = clean_raw(raw)
    train_df, val_df, test_df = split_time_aware(cleaned)

    TGT = "is_canceled"
    X_tr, y_tr = train_df.drop(columns=[TGT]), train_df[TGT].to_numpy()
    X_va, y_va = val_df.drop(columns=[TGT]), val_df[TGT].to_numpy()
    X_te, y_te = test_df.drop(columns=[TGT]), test_df[TGT].to_numpy()

    prep = build_preprocessor()
    X_tr_t = prep.fit_transform(X_tr)
    X_va_t = prep.transform(X_va)
    X_te_t = prep.transform(X_te)
    print(f"  Train {X_tr_t.shape} | Val {X_va_t.shape} | Test {X_te_t.shape}\n")

    def calibrated_eval(model: Any) -> tuple[float, float]:
        cal = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(X_va_t)[:, 1], y_va)
        probs = np.clip(cal.predict(model.predict_proba(X_te_t)[:, 1]), 0.0, 1.0)
        return float(average_precision_score(y_te, probs)), float(roc_auc_score(y_te, probs))

    print("[1/3] XGBoost — default     (n_est=100, depth=5,  lr=0.10)")
    t0 = time.perf_counter()
    m1 = train_xgb(
        X_tr_t,
        y_tr,
        X_va_t,
        y_va,
        params={"n_estimators": 100, "max_depth": 5, "learning_rate": 0.10},
    )
    pr1, roc1 = calibrated_eval(m1)
    print(f"  PR-AUC={pr1:.4f}  ROC-AUC={roc1:.4f}  ({time.perf_counter()-t0:.1f}s)\n")

    print("[2/3] XGBoost — matched     (n_est=300, depth=7,  lr=0.05)")
    t0 = time.perf_counter()
    m2 = train_xgb(
        X_tr_t,
        y_tr,
        X_va_t,
        y_va,
        params={"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05},
    )
    pr2, roc2 = calibrated_eval(m2)
    print(f"  PR-AUC={pr2:.4f}  ROC-AUC={roc2:.4f}  ({time.perf_counter()-t0:.1f}s)\n")

    print("[3/3] LightGBM — champion   (n_est=300, depth=7,  lr=0.05)  [artifact]")
    pipeline = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
    calibrator = joblib.load(ARTIFACTS_DIR / "probability_calibrator.pkl")
    probs_lgbm = np.clip(calibrator.predict(pipeline.predict_proba(X_te)[:, 1]), 0.0, 1.0)
    pr_lgbm = float(average_precision_score(y_te, probs_lgbm))
    roc_lgbm = float(roc_auc_score(y_te, probs_lgbm))
    print(f"  PR-AUC={pr_lgbm:.4f}  ROC-AUC={roc_lgbm:.4f}\n")

    print("=" * 68)
    print(f"  {'Model':<42} {'PR-AUC':>7}   {'ROC-AUC':>7}")
    print(f"  {'LightGBM champion':<42} {pr_lgbm:>7.4f}   {roc_lgbm:>7.4f}")
    print(f"  {'XGBoost matched  (n300/d7/lr0.05)':<42} {pr2:>7.4f}   {roc2:>7.4f}")
    print(f"  {'XGBoost default  (n100/d5/lr0.10)':<42} {pr1:>7.4f}   {roc1:>7.4f}")
    print("=" * 68)

    delta_pr = pr_lgbm - pr2
    delta_roc = roc_lgbm - roc2
    capacity_gain = pr2 - pr1
    total_gap = pr_lgbm - pr1

    print(
        f"\n  Delta (LightGBM vs XGBoost-matched):   "
        f"PR-AUC {delta_pr:+.4f}   ROC-AUC {delta_roc:+.4f}"
    )
    print(
        f"  Capacity effect on XGBoost:            "
        f"PR-AUC {capacity_gain:+.4f}   (matched minus default)"
    )

    if total_gap != 0:
        pct_capacity = 100.0 * capacity_gain / total_gap
        pct_algo = 100.0 - pct_capacity
        print(
            f"\n  Gap decomposition  "
            f"(LightGBM default vs XGBoost default = {total_gap:+.4f} PR-AUC):"
        )
        print(f"    - Capacity accounts for: {pct_capacity:.0f}%")
        print(f"    - Algorithm quality    : {pct_algo:.0f}%")

    print()
    if delta_pr > 0:
        print("  VERDICT: LightGBM STILL LEADS at equal hyperparameter budget.")
        print("  The champion ranking is NOT an artifact of the capacity gap.")
        print("  Leaf-wise growth + histogram binning explain the residual advantage.")
    else:
        print("  VERDICT: XGBoost matches or surpasses LightGBM at equal capacity.")
        print("  The original margin was driven by budget, not algorithm quality.")
        print("  Review champion selection — ranking may not be robust.")
    print()


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------


def main() -> None:
    from src.utils import configure_logging

    parser = argparse.ArgumentParser(
        description="Unified quality-gate runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("artifacts", help="Validate artifact integrity and smoke prediction")
    sub.add_parser("metrics", help="Enforce metric quality gates")
    sub.add_parser("sync", help="Verify threshold consistency across files")
    sub.add_parser("fairness", help="Hyperparameter fairness audit (LightGBM vs XGBoost)")
    sub.add_parser("all", help="Run all checks in sequence")

    args = parser.parse_args()
    configure_logging()

    if args.command == "all":
        run_artifacts()
        run_metrics()
        run_sync()
        run_fairness()
    elif args.command == "artifacts":
        run_artifacts()
    elif args.command == "metrics":
        run_metrics()
    elif args.command == "sync":
        run_sync()
    elif args.command == "fairness":
        run_fairness()


if __name__ == "__main__":
    main()
