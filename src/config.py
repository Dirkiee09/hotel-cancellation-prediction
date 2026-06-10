"""Project-wide configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COL = "is_canceled"
MODEL_SELECTION_POLICY = "champion_challenger_rolling_pr_auc_v1"

BOOKING_TIME_FEATURES = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    # arrival_date_week_number is deliberately excluded: the raw dataset's week
    # numbering disagrees with ISO-8601 for ~54% of dates, so serving-side
    # derivation would feed the model a different distribution than training
    # (training/serving skew). Seasonality is covered by month + month_sin/cos.
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "had_company",
    "total_stay",
    "total_guests",
    "adr_per_person",
    "is_weekend_heavy",
    "revenue_at_risk",
    "month_sin",
    "month_cos",
    "is_late_window",
]

LEAKAGE_COLS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
]

RANDOM_STATE = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
if TRAIN_RATIO + VAL_RATIO >= 1.0:
    raise ValueError("TRAIN_RATIO + VAL_RATIO must be < 1.0")

THRESHOLD_STEP = 0.01  # sweep grid step (100 thresholds: 0.00–0.99)
MIN_POSITIVE_RATE = 0.05
MIN_RECALL_FOR_HIGH_PRECISION = 0.20
# Cost model: FP cost = intervention (SMS + staff review), FN cost = ADR × lost nights.
FP_INTERVENTION_COST = 15.0  # EUR per false-positive intervention (industry range: 10–20)
FN_RECOVERY_NIGHTS = 1.0  # nights re-sold after cancellation (conservative: 1 of ~2.5 avg)
# Risk-tier display bands for the UI / API. Static values, not data-driven from the
# threshold sweep. When swapping datasets (see "Swapping Datasets" in CLAUDE.md), retune
# these alongside the cost constants — the calibrated probability distribution shifts.
RISK_TIER_MEDIUM_THRESHOLD = 0.40
RISK_TIER_HIGH_THRESHOLD = 0.70
if RISK_TIER_MEDIUM_THRESHOLD >= RISK_TIER_HIGH_THRESHOLD:
    raise ValueError("RISK_TIER_MEDIUM_THRESHOLD must be < RISK_TIER_HIGH_THRESHOLD")
LATE_WINDOW_MAX_LEAD_DAYS = 3
ADR_MAX_VALID = 50_000.0  # outlier ceiling; adjust per currency (~1000 EUR, ~50000 PHP)

# Fallback when threshold sweep produces no valid result
DEFAULT_FALLBACK_THRESHOLD = 0.5

# Rounds without val-metric improvement before boosting stops when an eval_set
# is supplied to train_xgb/train_lgbm (the use_early_stopping=True path).
EARLY_STOPPING_ROUNDS = 50

# Probability calibration controls
CALIBRATION_METHOD = "isotonic"
CALIBRATION_ECE_BINS = 10

# Champion/challenger model selection controls
ROLLING_SELECTION_CUTOFF_FRACS = [0.60, 0.70, 0.80]
ROLLING_SELECTION_VAL_RATIO = 0.10
ROLLING_SELECTION_MIN_TRAIN_ROWS = 1500
ROLLING_SELECTION_MIN_VAL_ROWS = 500

# Metric quality gates: tightened to (observed - 0.02) per CLAUDE.md
# guidance after the Portugal champion was confirmed. The 0.02 buffer
# absorbs normal retraining noise while still catching real regressions.
# Observed values (reports/metrics.json, chronological test set, 2026-06-10):
#   max_f1: ROC=0.8634, PR=0.7590, F1=0.7356, Recall=0.8946
#   high_precision: Precision=1.000, Recall=0.0945
METRIC_GATES = {
    "max_f1": {
        "roc_auc_min": 0.84,
        "pr_auc_min": 0.74,
        "f1_min": 0.71,
        "recall_min": 0.82,
    },
    "high_precision": {
        "precision_min": 0.98,
        "recall_min": 0.07,
    },
}

# Segment-level gate controls
SEGMENT_METRIC_GATES = {
    "policy": "max_f1",
    "min_rows": 500,
    "dimensions": {
        "hotel": 10,
        "market_segment": 10,
        "distribution_channel": 10,
        "arrival_date_month": 12,
    },
    "metrics": {
        "roc_auc_min": 0.75,
        "pr_auc_min": 0.42,
        "f1_min": 0.43,
    },
}

# Prediction-log persistence (Power BI integration).
# Every successful /predict call appends one row to PREDICTION_LOG_DB. The
# export script materialises it as a CSV that Power BI Desktop can read with
# no ODBC drivers required. Both files are git-ignored; the DB is the source
# of truth and the CSV is regenerated on demand. Lives under data/predictions/
# (a dedicated subfolder co-located with the source dataset for easy access)
# rather than reports/, which is cluttered with training-time artifacts.
PREDICTION_LOG_DB = PROJECT_ROOT / "data" / "predictions" / "predictions.sqlite"
PREDICTION_LOG_CSV = PROJECT_ROOT / "data" / "predictions" / "predictions_live.csv"

# PH sub-study prediction log — sits alongside Portugal's so a single
# data/predictions/ folder feeds two parallel Power BI dashboards.
PH_PREDICTION_LOG_DB = PROJECT_ROOT / "data" / "predictions" / "ph_predictions.sqlite"
PH_PREDICTION_LOG_CSV = PROJECT_ROOT / "data" / "predictions" / "ph_predictions_live.csv"

# Reproducibility controls
REPRO_TOLERANCE = 1e-6  # Relaxed for cross-platform reproducibility

# Thesis analysis controls
BOOTSTRAP_N_ITERATIONS = 2000
BOOTSTRAP_ALPHA = 0.05
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT_SECONDS = 600
EXPANDING_WINDOW_N_SPLITS = 5
TEMPORAL_STABILITY_BUCKETS = 6
LEARNING_CURVE_FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


# ── Philippine transferability probe (appendix-grade, not in CI) ─────
# Side-experiment that re-fits the methodology on the real Punta Villa
# Resort PMS export (193 bookings, 2022-2025). The real PMS export
# includes deposit_type and total_of_special_requests — two top-10
# Portugal SHAP features the earlier synthetic dataset deliberately
# lacked — so the feature set matches Portugal more closely now.
# See CLAUDE.md "PH Sub-Study" for scope and caveats. These constants
# are NEVER imported by the Portugal pipeline.

PH_DATA_PATH = PROJECT_ROOT / "data" / "Punta_Villa_Resort_PH_Dataset.csv"
PH_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "ph"
PH_REPORTS_DIR = PROJECT_ROOT / "reports" / "ph"
PH_TARGET_COL = "is_canceled"

# PH raw columns -> project-canonical names so downstream feature
# engineering (add_derived_booking_features) can be reused.
PH_COLUMN_RENAMES = {
    "Lead_Time_Days": "lead_time",
    "Weekend_Nights": "stays_in_weekend_nights",
    "Week_Nights": "stays_in_week_nights",
    "Adults": "adults",
    "Children": "children",
    "Babies": "babies",
    "ADR_Rate": "adr",
    "Room_Type": "reserved_room_type",
    "Deposit_Type": "deposit_type",
    "Special_Requests": "total_of_special_requests",
}

# Real-data feature list — the real PMS export ships with deposit_type
# and total_of_special_requests, closing the previous feature gap to
# Portugal. country/market_segment/agent/customer_type/
# previous_cancellations are still absent.
PH_BOOKING_TIME_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "adr",
    "reserved_room_type",
    "deposit_type",
    "total_of_special_requests",
    "month_sin",
    "month_cos",
    "total_stay",
    "total_guests",
    "adr_per_person",
    "is_weekend_heavy",
    "revenue_at_risk",
    "is_late_window",
]

PH_CATEGORICAL_COLS = ["reserved_room_type", "deposit_type"]
PH_NUMERIC_COLS = sorted(c for c in PH_BOOKING_TIME_FEATURES if c not in PH_CATEGORICAL_COLS)

# Directional floors only — at n_test ≈ 30, bootstrap CIs span ±15 pp.
# These are not regression-detection gates like METRIC_GATES; they
# document what would count as "the methodology generalises" if hit.
PH_METRIC_GATES = {
    "max_f1": {
        "roc_auc_min": 0.55,
        "pr_auc_min": 0.30,
        "f1_min": 0.30,
        "recall_min": 0.30,
    },
}
