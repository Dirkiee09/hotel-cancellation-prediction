# CLAUDE.md — Hotel Booking Cancellation Prediction

This file is read automatically by Claude Code at the start of every session.
It defines the project structure, conventions, quality gates, and the scope of permitted autonomous changes.

---

## Project Overview

End-to-end ML thesis project: predict hotel booking cancellations at **booking time** using gradient-boosted trees.
The pipeline covers data loading → feature engineering → model selection → calibration → threshold optimization →
serving (FastAPI + Gradio) → evaluation → thesis reporting.

**Champion model**: LightGBM (selected by rolling-origin PR-AUC).
**Target**: `is_canceled` (binary, ~37% positive rate).
**Prediction point**: booking time only — no post-booking leakage features.
**Dataset**: `data/hotel_bookings.csv` — 119k rows, 32 raw features, 49 engineered model features.

---

## Codebase Health (last verified)

| Check | Status | Detail |
|-------|--------|--------|
| `pytest` | ✅ 130/130 | 88.33% coverage (gate: ≥80%) |
| `ruff check` | ✅ clean | E, F, I rules |
| `ruff format` | ✅ clean | line length 100, 75 files |
| `mypy` | ✅ clean | 0 errors (40 source files checked) |
| `scripts/check.py all` | ✅ pass | artifacts + metrics + sync + fairness gates |
| `scripts/` | ✅ 4/4 scripts | train, benchmark, check, notebooks |
| `notebooks/` | ✅ 21/21 notebooks | 10 Portugal + 11 Philippine, all with cached outputs |
| FastAPI Portugal | ✅ verified | `/healthz`, `/model-info`, `/predict` (with SHAP + ADR) on :8000 |
| FastAPI Philippine | ✅ verified | `/healthz`, `/model-info`, `/predict` (with SHAP + caveat) on :8001 |

---

## Data Split Design

**Thesis-level description (80% train / 20% holdout)** — the headline split used in thesis text and figures.

**Internal mechanics (80 / 10 / 10)** — what the code actually implements:

```
119k rows, sorted chronologically by arrival_date
├── Train set     80%   (~95k rows)   — model fitting, rolling-origin selection
├── Val set       10%   (~12k rows)   — calibration, threshold selection (never seen during training)
└── Test set      10%   (~12k rows)   — final reported metrics (touched once, at the very end)
```

**Why the three-way split is required** (do not collapse to pure 80/20 without approval):
- Isotonic calibration is fitted on the val set → leakage if fitted on test
- Threshold sweep (max_f1, high_precision, cost_sensitive) runs on val set → leakage if run on test
- Rolling-origin champion/challenger selection uses val-like windows within train
- The test set is reserved exclusively for the final, unbiased performance numbers in the thesis

**Reporting convention**: In thesis writing, val + test = "20% holdout". The internal 10/10 split is
methodology detail that belongs in the Appendix, not the main text.

---

## Repository Layout

```
data/
  hotel_bookings.csv          # Raw dataset (119k rows, 32 features)

src/
  config.py                   # ALL constants live here — edit here first
  data/load.py                # Raw CSV loader
  features/build.py           # Feature engineering + preprocessor + split
  models/
    train.py                  # LightGBM / XGBoost / GradientBoosting training helpers
    metrics.py                # Classification metrics (ECE, ROC-AUC, PR-AUC, F1)
    baselines.py              # Dummy + logistic + Decision Tree + Naive Bayes baselines
    tuning.py                 # Optuna hyperparameter search
  pipelines/train.py          # Master training pipeline (the main entry point)
  serving/inference.py        # Prediction contract (InferenceInput → output dict)
  utils/
    thresholds.py             # Threshold selection + resolve_thresholds()
    business.py               # Revenue-at-risk, cost math, risk tiers
    validate_data.py          # Raw data validation (schema, leakage, bounds)
    reproducibility.py        # Seed helpers
    logging_utils.py          # Structured logging
  eval/
    notebook_utils.py         # ALL notebook helpers (plots, context loaders, SHAP)
    thesis.py                 # Thesis analysis runner (SHAP, CI, ablation, Optuna)
    benchmark.py              # Multi-model benchmark (16 CSV tables)
    statistical.py            # Bootstrap CI + paired two-sided significance tests
    repro.py                  # Reproducibility checker (5k-row subset hash)
    verify.py                 # Post-train verification + GB vs XGBoost comparison
  app/
    main.py                   # FastAPI endpoints (thread-safe artifact caching)
    ui.py                     # Gradio UI (adults ≥ 1, datetime-aware)
    schemas.py                # Pydantic BookingRequest with field coercion/validation

scripts/                      # CLI entry points (thin wrappers over src/)
  train.py                    # make train | --verify | --verify-only | --thesis | --repro
  benchmark.py                # make benchmark (16 CSV tables)
  check.py                    # make check (subcommands: artifacts, metrics, sync, fairness, all)
  notebooks.py                # headless notebook execution (all 10 notebooks)

tests/                        # pytest suite (91 tests, ≥80% coverage required)
  test_preprocessing.py
  test_split_and_leakage.py
  test_training_pipeline.py
  test_inference_contract.py
  test_integration_train_serve.py
  test_baselines.py
  test_business.py
  test_metrics.py
  test_model_benchmark.py
  test_schemas.py
  test_statistical.py
  test_thresholds.py
  test_tuning.py
  test_validate_data.py

notebooks/                    # 10 Jupyter notebooks (load artifacts, no retraining)
  01_eda.ipynb                # Exploratory data analysis (26 cells)
  02_modeling.ipynb           # Model selection & rolling-origin evaluation (21 cells)
  03_deep_analysis.ipynb      # Calibration, SHAP, CV, ablation, baselines (51 cells)
  04_adr_forecasting.ipynb    # ADR time-series + regression (23 cells)
  05_explainability.ipynb     # SHAP beeswarm, segment insights, cost analysis (29 cells)
  06_business_analytics.ipynb # Revenue management dashboard (20 cells)
  07_model_selection.ipynb    # Comprehensive model selection & fairness (37 cells)
  08_model_monitoring.ipynb   # Production monitoring template (21 cells)
  09_model_comparison.ipynb   # Cross-model comparison & stability analysis (31 cells)
  10_sensitivity_analysis.ipynb # Cost sensitivity, data hunger, threshold trade-offs (15 cells)

artifacts/                    # Trained model artifacts (git-ignored)
reports/                      # Metrics, benchmark tables, thesis reports (git-ignored)
  benchmarks/                 # 16 CSV benchmark tables
  thesis/                     # Thesis analysis JSONs + figures
  figures/thesis/             # Publication figures (PNG + PDF)
```

---

## Development Commands

```bash
# Full local setup
python -m pip install -e . -r requirements.txt

# Daily workflow
make lint                     # ruff check .
make format                   # ruff format --check .
make typecheck                # mypy src/
make test                     # pytest (91 tests, coverage ≥80%)
make security                 # bandit
make train                    # Run full training pipeline
make eval                     # Post-train verification (existing artifacts)
make benchmark                # Multi-model benchmark (~5 min)
make thesis                   # Full thesis analysis (SHAP, CI, Optuna)
make check                    # All quality gates (artifacts, metrics, sync, fairness)
make full-pipeline            # train → eval → benchmark → check
make clean                    # Remove caches and build artifacts

# Serving
uvicorn src.app.main:app --reload   # FastAPI at localhost:8000
python src/app/ui.py                # Gradio UI (standalone)
```

---

## CI Pipeline (GitHub Actions)

**`quality` job** (runs on every push/PR):
1. `pip install -e . -r requirements.txt && pip check`
2. ruff check + ruff format check
3. mypy
4. pytest (≥80% coverage)
5. `scripts/train.py` — full train must succeed
6. `scripts/check.py artifacts` — model + calibrator + feature columns load and round-trip
7. `scripts/check.py metrics` — metric gates must pass
8. `scripts/check.py sync` — cross-artifact threshold consistency
9. `scripts/check.py fairness` — matched-capacity XGBoost vs LightGBM (champion must lead at equal hyperparameter budget)
10. bandit security scan
11. pip-audit dependency vulnerability scan
12. On failure, uploads `artifacts/` + `reports/` as a 7-day debug artifact

**`thesis-analysis` job** (runs only on push to remote `Main` — capital M because of the GitHub default-branch rename; gating on `refs/heads/main` lowercase would silently never fire — needs `quality`):
1. `scripts/train.py --thesis` — full train + thesis analysis (SHAP, CI, ablation, Optuna)
2. `scripts/benchmark.py` — 16 CSV benchmark tables
3. `scripts/check.py sync` — post-benchmark consistency
4. Uploads `reports/thesis/` as a GitHub Actions artifact

**Critical**: Every PR must keep CI green. Never bypass hooks or skip coverage.

---

## Key Constants (`src/config.py`)

All project-wide numbers live here. Edit `config.py` first, never hardcode values elsewhere.

| Constant | Value | Purpose |
|----------|-------|---------|
| `TRAIN_RATIO` | 0.80 | Chronological train split (80% of rows) |
| `VAL_RATIO` | 0.10 | Validation slice — calibration + threshold selection |
| `RANDOM_STATE` | 42 | All random seeds (models, bootstrap, Optuna) |
| `TARGET_COL` | `"is_canceled"` | Binary target column |
| `FP_INTERVENTION_COST` | 15.0 EUR | False positive cost for cost-sensitive threshold |
| `FN_RECOVERY_NIGHTS` | 1.0 | Nights recovered after late cancellation (FN cost proxy) |
| `RISK_TIER_MEDIUM_THRESHOLD` | 0.40 | P(cancel) boundary Low → Medium |
| `RISK_TIER_HIGH_THRESHOLD` | 0.70 | P(cancel) boundary Medium → High |
| `ADR_MAX_VALID` | 50,000.0 | Currency-agnostic ceiling for valid ADR |
| `REPRO_TOLERANCE` | 1e-6 | Cross-platform hash tolerance for reproducibility checks |
| `BOOTSTRAP_N_ITERATIONS` | 2000 | Bootstrap resamples for confidence intervals |
| `CALIBRATION_METHOD` | `"isotonic"` | Probability calibration method |
| `THRESHOLD_STEP` | 0.01 | Grid step for threshold sweep (uses np.linspace) |

**Metric quality gates** (enforced by `scripts/check.py metrics`, must pass or CI fails):
- `max_f1` policy: ROC-AUC ≥ 0.70, PR-AUC ≥ 0.50, F1 ≥ 0.50, Recall ≥ 0.50
- `high_precision` policy: Precision ≥ 0.90, Recall ≥ 0.05

These are universal floor values for plug-and-play dataset support. Tighten to `(observed - 0.02)` after training on your dataset.

---

## Pipeline Flow (detailed)

```
data/hotel_bookings.csv  (119k rows, 32 raw features)
  │
  ├─▶ load_raw_data()                    [src/data/load.py]
  │     └── reads CSV, basic dtype coercion
  │
  ├─▶ clean_raw()                        [src/utils/validate_data.py]
  │     ├── fill children NaN → 0
  │     ├── encode agent=0 as "Direct"
  │     ├── add_derived_booking_features() → had_company, total_stay,
  │     │   total_guests, adr_per_person, revenue_at_risk,
  │     │   month_sin, month_cos, is_late_window, is_weekend_heavy
  │     └── drop rows: negative ADR, ADR ≥ 1000, zero/null guests
  │
  ├─▶ validate_raw()                     [src/utils/validate_data.py]
  │     └── schema checks, target binary, no negative numerics
  │
  ├─▶ add_arrival_date() → sort chronologically
  │
  ├─▶ drop LEAKAGE_COLS                  [reservation_status, assigned_room_type, …]
  │
  ├─▶ split_time_aware()                 [src/features/build.py]
  │     ├── Train   80%  (~95k rows) ──────────────────────────────┐
  │     ├── Val     10%  (~12k rows)  ← calibration + thresholds  │
  │     └── Test    10%  (~12k rows)  ← final reported metrics     │
  │           ↑ reported as "80% train / 20% holdout" in thesis    │
  │                                                                 │
  ├─▶ build_preprocessor()              [src/features/build.py]    │
  │     ├── categorical: SimpleImputer("UNKNOWN") → cast_to_str    │
  │     │   → OneHotEncoder(min_freq=0.01, handle_unknown="ignore") │
  │     └── numeric: SimpleImputer(strategy="median")              │
  │                                                                 │
  ├─▶ rolling-origin champion/challenger selection                  │
  │     ├── cutoffs: [60%, 70%, 80%] of train                      │
  │     ├── candidates: LightGBM, XGBoost, GradientBoosting        │
  │     └── primary metric: PR-AUC → champion = LightGBM ──────────┘
  │
  ├─▶ isotonic calibration              [sklearn.isotonic.IsotonicRegression]
  │     └── fitted on val set predictions → saved to artifacts/probability_calibrator.pkl
  │
  ├─▶ threshold selection               [src/utils/thresholds.py]
  │     ├── max_f1            (F1-maximising, ~0.35)
  │     ├── high_precision    (Precision ≥ 0.98, ~0.80)
  │     └── cost_sensitive    (min FP×15€ + FN×revenue_at_risk)
  │           └── sweep runs on val set → saved to artifacts/thresholds.json
  │
  ├─▶ save artifacts/                   [artifacts/*.pkl, *.json, *.csv]
  ├─▶ save reports/                     [metrics.json, segment_metrics.csv, …]
  │
  ├─▶ scripts/train.py --verify-only     [src/eval/verify.py]
  │     └── post-train sanity checks → model_verification_report.md
  │
  └─▶ FastAPI (src/app/main.py)  serves /predict, /model-info, /healthz
        └── Gradio (src/app/ui.py) wraps the same inference logic
```

---

## Architecture: Serving Layer

```
BookingRequest (Pydantic)
  ├── field coercion: agent/company → str, arrival_date components validated
  ├── adr validated: 0 ≤ adr < 1000 EUR
  └── adults ≥ 1 (schema + UI both enforce)

predict_proba() [src/serving/inference.py]
  ├── dict → DataFrame → ensure_model_features() → feature engineering
  ├── model.predict_proba() OR preprocessor.transform() + model.predict_proba()
  ├── isotonic calibrator.predict() → clipped to [0, 1]
  └── returns (probabilities, feature_df)

/predict endpoint [src/app/main.py]
  ├── artifacts loaded once at startup with threading.Lock() double-checked locking
  ├── resolve_thresholds() → policy thresholds with cost-sensitive fallback
  ├── assign risk tier: low / medium / high
  ├── PredictionResponse: probability, 3 binary labels, risk_tier, alerts
  └── BackgroundTasks → log_prediction() writes one row to predictions.sqlite

Thread safety: _ARTIFACTS singleton protected by _ARTIFACTS_LOCK
  → safe for multi-worker uvicorn deployments

Prediction audit log [src/serving/prediction_log.py]
  ├── SQLite at data/predictions/predictions.sqlite — one row per /predict call
  ├── 43-column schema: timestamp_utc + every BookingRequest field +
  │   every PredictionResponse field + top_features (JSON) +
  │   predicted_adr + adr_residual (live ADR forecast columns)
  ├── Schema migration: init_db() runs PRAGMA-driven idempotent ALTER TABLE
  │   to add new columns to pre-existing DBs (see _NULLABLE_ADDITIONS)
  ├── Non-raising: BackgroundTask path, errors logged at WARNING only
  └── Exported via `make export-predictions` → data/predictions/predictions_live.csv
        ↓ Power BI Desktop reads the CSV (no ODBC driver required)
```

### Live ADR forecast (per /predict)

Every successful /predict (HTTP or Gradio) now calls the ADR regressor
alongside the cancellation classifier and writes two extra columns to the
prediction log:

| Field | Meaning |
|-------|---------|
| `predicted_adr` | What the ADR model expects this booking to charge (EUR) |
| `adr_residual`  | `entered_adr - predicted_adr` — positive means priced above the model's expectation |

**Caveat to flag in thesis writeup**: the ADR regressor was trained with four
post-booking features (`is_canceled`, `assigned_room_type`, `booking_changes`,
`days_in_waiting_list`) that aren't known at booking time. Live inference
passes placeholders (`0` / `reserved_room_type` / `0` / `0`), so live
`predicted_adr` is slightly less accurate than the published test-set
RMSE = 44.31. The methodologically clean fix is retraining on booking-time
features only; the live-integration story is the value today.

Loading is best-effort: if `artifacts/adr_regressor.pkl` is absent, both
fields default to `None` and the cancellation /predict path is unaffected.

### Power BI dashboard setup (60-second recipe)

1. Run the server: `python demo/start_server.py`
2. Make a few predictions through the Gradio UI (or hit `/predict` directly)
3. Run `make export-predictions` (or `python scripts/export_predictions.py`)
4. In Power BI Desktop → **Home > Get Data > Text/CSV** → pick `data/predictions/predictions_live.csv` → Load
5. After every new batch of predictions, re-run the export script and click **Refresh** in Power BI

The SQLite file is the source of truth. The CSV is regenerated on every export. Both are git-ignored.

---

## Architecture: Baseline Models (Thesis Comparison)

Four baselines are implemented in `src/models/baselines.py`:

| Model | Purpose | Key params |
|-------|---------|------------|
| `DummyClassifier` | Lower bound — majority-class predictor | strategy="most_frequent" |
| `LogisticRegression` | Linear interpretable baseline | lbfgs, max_iter=2000 |
| `DecisionTreeClassifier` | Visualisable tree for thesis figure | max_depth=5, class_weight="balanced" |
| `GaussianNB` | Zero-assumption probabilistic baseline | default params |

The complexity ladder (Dummy → LR → DT → NB → LightGBM) shows the value of each additional
modelling assumption. DT is shallow enough to be printed in full in the thesis appendix.

---

## Code Conventions

### Style
- **Formatter**: `ruff format`, line length 100, LF line endings
- **Linter**: `ruff check` with E, F, I rules (E501 excluded — formatter handles it)
- **Types**: `mypy` with `check_untyped_defs=True`; all public functions need type annotations
- **Python**: 3.11+, use `from __future__ import annotations` in new files

### Architecture rules
- **No sys.path hacks**: Package installed editable (`pip install -e .`). Use `from src.xxx import yyy`.
- **All constants in `src/config.py`**: Never hardcode thresholds, ratios, or cost figures inline.
- **No leakage**: Never use `LEAKAGE_COLS` in model training. `BOOKING_TIME_FEATURES` is canonical.
- **Thread safety**: Artifact caching in `main.py` and `ui.py` uses double-checked locking. Maintain.
- **Notebooks use `notebook_utils.py`**: Never duplicate plot logic in notebooks inline.
- **No model retraining in notebooks**: Notebooks load pre-computed artifacts only.
- **JSON serialization**: Always run `_sanitise_for_json()` before `json.dumps()` — replaces NaN/Inf with `None`.
- **Threshold grids**: Use `np.linspace` (not `np.arange`) for float precision in sweep grids.

### Testing
- Tests live in `tests/`, one file per module area
- `conftest.py` provides shared fixtures (synthetic DataFrames, mock artifacts)
- Use `pytest.approx()` for float comparisons
- Coverage must stay ≥ 80% (enforced by CI)
- Excluded from coverage: `src/app/ui.py`, `src/eval/verify.py`, `src/eval/thesis.py`,
  `src/eval/repro.py`, `src/models/tuning.py`, `src/eval/notebook_utils.py`

### Notebooks
- Every section alternates: markdown header → code cell → markdown insight
- Always use `setup_plotting()` for consistent serif-font publication style
- Save figures: `save_thesis_figure(fig, fig_no, stem, FIG_DIR)` → `reports/figures/thesis/`
- Display tables: `df.style.format(...).set_caption(...)` — never `print(df)`
- Column names in CSV reports use title-case-with-spaces (e.g. `"Test RMSE"`, `"Test R2"`)
- Notebooks are excluded from ruff and mypy

---

## Artifact Locations

| Artifact | Path |
|---------|------|
| Champion model pipeline | `artifacts/best_model.pkl` |
| Isotonic calibrator | `artifacts/probability_calibrator.pkl` |
| Decision thresholds | `artifacts/thresholds.json` |
| Threshold sweep CSV | `artifacts/threshold_sweep.csv` |
| Feature column list | `artifacts/feature_columns.json` |
| Model metadata | `artifacts/model_metadata.json` + `model_verification_report.md` |
| ADR regression | `artifacts/adr_regressor.pkl` + `adr_regressor_metadata.pkl` |
| ADR neural network | `artifacts/adr_regressor_nn.keras` |
| ADR time series | `artifacts/adr_timeseries_data.pkl` + `adr_timeseries_metadata.pkl` |
| Thesis DT baseline | `artifacts/thesis_baseline_dt.pkl` (written by `make thesis`) |
| Benchmark tables | `reports/benchmarks/01_*.csv` … `16_*.csv` |
| Thesis reports | `reports/thesis/*.json` |
| Regression results | `reports/regression_results.csv` (columns: Model, Train RMSE, Val RMSE, Test RMSE, …) |
| Publication figures | `reports/figures/thesis/fig_NN_*.{png,pdf}` |
| Test predictions | `reports/test_predictions_for_powerbi.csv` (written by `scripts/train.py` only — single source of truth) |
| Live prediction log | `data/predictions/predictions.sqlite` (one row per `/predict` call) + `data/predictions/predictions_live.csv` (PowerBI-ready export) |
| Segment metrics | `reports/segment_metrics.csv` |
| Champion summary | `reports/champion_summary.json` (champion family, runner-up, PR-AUC gap, selected_at) |

### Artifact Contract (producers → consumers)

Quick reference for who reads what — useful when modifying training or inference. Every
artifact has exactly one writer; multiple consumers are normal.

| Artifact | Writer | Consumers |
|----------|--------|-----------|
| `artifacts/best_model.pkl` | `src/pipelines/train.py` | `src/serving/inference.py` (`load_artifacts`), notebooks via `load_main_context()` |
| `artifacts/probability_calibrator.pkl` | `src/pipelines/train.py` | `src/serving/inference.py` (`predict_proba`) |
| `artifacts/thresholds.json` | `src/pipelines/train.py` | `src/utils/thresholds.py` (`resolve_thresholds`), API `/predict` and `/model-info`, notebooks |
| `artifacts/feature_columns.json` | `src/pipelines/train.py` | `src/serving/inference.py`, `scripts/check.py artifacts` |
| `artifacts/model_metadata.json` | `src/pipelines/train.py` | `/model-info` (lineage), `scripts/check.py artifacts`, notebooks |
| `artifacts/threshold_sweep.csv` | `src/pipelines/train.py` | Notebook 02 plots |
| `artifacts/cost_threshold_sweep.csv` | `src/pipelines/train.py` | Notebook 10 sensitivity sweep |
| `artifacts/hashes.json` | `src/pipelines/train.py` | `scripts/check.py artifacts` (lineage verification) |
| `reports/metrics.json` | `src/pipelines/train.py` | UI hero banner, notebooks, `scripts/check.py metrics` |
| `reports/champion_summary.json` | `src/pipelines/train.py` | Thesis writeup (decision log) |
| `reports/model_selection_summary.json` | `src/pipelines/train.py` | Thesis Notebook 02, `scripts/check.py sync` |
| `reports/segment_metrics.csv` / `.json` | `src/pipelines/train.py` | Notebooks 05/07, `scripts/check.py fairness` |
| `reports/test_predictions_for_powerbi.csv` | `src/pipelines/train.py` | Notebook 08 monitoring, PowerBI |
| `data/predictions/predictions.sqlite` | FastAPI `/predict` (via BackgroundTasks → `src/serving/prediction_log.py`) + Gradio `predict_one` (inline). Both paths also compute `predicted_adr` + `adr_residual` via `predict_adr()` when the ADR regressor is loaded. | `scripts/export_predictions.py` |
| `data/predictions/predictions_live.csv` | `scripts/export_predictions.py` (auto-refreshed after every prediction). Includes live `predicted_adr` and `adr_residual` columns. | Power BI Desktop dashboard — Page 4 (Revenue) AND Page 5 (live ADR) |
| `data/predictions/drift_metrics.csv` | `scripts/compute_live_drift.py` (compares live vs `reports/test_predictions_for_powerbi.csv` via `src/utils/drift.py`) | Power BI Page 8 (monitoring) |
| `reports/adr_test_predictions.csv` | `scripts/export_adr_predictions.py` (loads `artifacts/adr_regressor.pkl`, predicts on the test split defined by `artifacts/adr_regressor_metadata.pkl`) | Power BI Page 5 (ADR Forecasting): scatter, histogram, monthly line |
| `reports/adr_segment_performance.csv` | `scripts/export_adr_predictions.py` (aggregates RMSE/MAE per hotel × room_type, min 50 rows) | Power BI Page 5 (ADR Forecasting): segment RMSE heatmap |
| `reports/benchmarks/01_*.csv` … `16_*.csv` | `scripts/benchmark.py` | Notebook 07, `scripts/check.py sync` |
| `reports/thesis/*.json` | `src/eval/thesis.py` | Notebooks 03/05, `scripts/check.py sync` |

**Threshold cross-check**: `scripts/check.py sync` verifies the same `max_f1` / `high_precision`
/ `cost_sensitive` thresholds appear in `artifacts/thresholds.json`,
`reports/thesis/model_family_summary.json`, and `reports/benchmarks/07_thresholds_per_model.csv`
within `REPRO_TOLERANCE = 1e-6`. If those drift, retrain.

**Note on `reports/threshold_summary.json`**: this is a redundant copy of
`artifacts/thresholds.json` kept for the thesis report. `artifacts/thresholds.json` is the
canonical source; consumers should prefer it.

---

## Permitted Autonomous Improvements

Claude is authorized to make the following changes **without asking for confirmation**:

### Always allowed
- Fix lint errors, format violations, mypy errors
- Fix failing tests (without weakening assertions)
- Add docstrings or comments to existing functions
- Fix bugs in existing logic when the intent is clear
- Add `# type: ignore` with explanation when mypy false-positive is confirmed
- Extend existing notebooks with new cells using pre-computed artifacts
- Add new utility functions to `src/eval/notebook_utils.py`
- Add new sections to `CLAUDE.md`
- Update `reports/` and `artifacts/` paths in documentation
- Add new notebooks to `notebooks/`

### Allowed with brief explanation to user
- Add new test cases to existing test files
- Add new config constants to `src/config.py` with a comment
- Add new columns to existing report CSVs
- Add new plotting/analysis functions to `notebook_utils.py`
- Create new notebooks (numbered sequentially: `07_*.ipynb`)
- Add new `make` targets to `Makefile`
- Extend `src/eval/thesis.py` with new analysis functions

### Requires user approval
- Change existing metric quality gates in `METRIC_GATES`
- Change `TRAIN_RATIO`, `VAL_RATIO`, or `RANDOM_STATE`
- Modify the training pipeline (`src/pipelines/train.py`) logic
- Change the feature list (`BOOKING_TIME_FEATURES` or `LEAKAGE_COLS`)
- Modify the FastAPI schema or endpoint contract
- Change the champion model family or selection policy
- Pin, upgrade, or remove packages in `requirements.txt`
- Modify CI workflow (`.github/workflows/ci.yml`)
- Delete existing files
- Make git commits or push to remote

### Never do (without explicit instruction)
- Lower coverage thresholds
- Add `--no-verify` to git commands
- Disable mypy checks with broad `type: ignore` without explanation
- Hardcode constants that belong in `config.py`
- Add `sys.path` manipulation (editable install handles imports)
- Retrain models inside notebooks
- Use leakage columns in any model input
- Use `np.arange` for float threshold grids (use `np.linspace` instead)
- Collapse the 80/10/10 split to 80/20 without user approval

---

## Improvement Backlog (ready to implement)

| ID | Item | Artifact / Notes |
|----|------|-----------------|
| IMP-01 | Learning curve section in Notebook 03 | **DONE** |
| IMP-02 | Expanding-window CV section in Notebook 03 | **DONE** |
| IMP-03 | Feature ablation section in Notebook 03 | **DONE** |
| IMP-04 | Segment cost analysis in Notebook 05 | **DONE** |
| IMP-05 | Notebook 06 — Business Analytics | **DONE** |
| IMP-06 | `plot_shap_bar`, `plot_shap_beeswarm`, `plot_segment_heatmap` | **DONE** |
| IMP-07 | Segment-specific threshold recommendations | `reports/segment_metrics.csv` + cost model — **DONE** |
| IMP-08 | Notebook 08 — Model Monitoring template | Drift detection from `test_predictions_for_powerbi.csv` baseline — **DONE** |
| IMP-09 | `had_company` binary feature | Already in `BOOKING_TIME_FEATURES` ✓ |
| IMP-10 | Baseline comparison visualisation in Notebook 03 (section 3.19) | **DONE** |
| IMP-11 | `model_family_summary.json` table in Notebook 02 | `reports/thesis/model_family_summary.json` — **DONE** |

---

## Swapping Datasets (Plug-and-Play)

The pipeline is designed to work with **any hotel booking CSV** that shares the same 32 column names.
To use a different dataset (e.g., Philippine hotels):

### Step 1: Replace the CSV
Replace `data/hotel_bookings.csv` with your new CSV. Same column names required.

### Step 2: Update currency-specific constants in `src/config.py`
| Constant | Default | What to change |
|----------|---------|----------------|
| `ADR_MAX_VALID` | 50,000 | Max valid room rate in your currency (rows above this are dropped) |
| `FP_INTERVENTION_COST` | 15.0 | Cost per false positive in your currency (e.g., EUR→PHP: 15→900) |
| `FN_RECOVERY_NIGHTS` | 1.0 | Keep at 1.0 unless you have market-specific data |

### Step 3: Retrain
```bash
make train        # Retrain on new data — all features, calibration, thresholds auto-adapt
make eval         # Post-train verification (smoke test uses generic values)
```

### Step 4: Tighten metric gates (optional but recommended)
After training, check `reports/metrics.json` for observed performance, then update
`METRIC_GATES` in `src/config.py` to `(observed_value - 0.02)` for regression detection.

### What auto-adapts (no changes needed)
- UI dropdown choices (read from CSV at startup)
- Feature engineering (operates on column names, not values)
- OneHotEncoder categories (learned from training data, `handle_unknown="ignore"`)
- Threshold sweep (data-driven on validation set)
- Isotonic calibration (fitted on validation set probabilities)
- Country list in UI (already dynamic)
- Smoke test (uses generic UNKNOWN values)
- Hero banner metrics (read from model metadata)

### What needs manual update per dataset
- `ADR_MAX_VALID` and `FP_INTERVENTION_COST` in `config.py` (currency-dependent)
- `METRIC_GATES` in `config.py` (after first training run)
- Notebook markdown narrative (search for "Portugal", "EUR", "PRT" in notebook cells)

---

## PH Sub-Study — Philippine Resort Dataset (Notebooks Under `notebooks/ph/`)

A parallel sub-study at `scripts/train_ph.py` re-fits the methodology on the
**real Punta Villa Resort PMS export**
(`data/Punta_Villa_Resort_PH_Dataset.csv`, 193 real bookings, 2022-2025).
The real PMS export ships with `deposit_type` and `total_of_special_requests`
— two top-10 Portugal SHAP features the earlier exploratory dataset
deliberately lacked — so the PH feature menu now closely mirrors Portugal's.
**This is NOT part of CI, `make`, or the Portugal pipeline.** It runs as a
separate manual command and produces a 6-notebook PH suite under
`notebooks/ph/`.

### Why it exists
The thesis claims rest on methodology (rolling-origin selection, isotonic
calibration, cost-sensitive thresholds). The PH sub-study tests how that
methodology behaves on a smaller, geographically different dataset — and
surfaces two methodological contributions: (1) a generic pre-flight check
that flags datasets organized around recurring booking archetypes (where
chronological splitting would leak twins across the train/test boundary),
and (2) a feature-availability mapping that bounds what a reduced-PMS-schema
operator can credibly model. The pre-flight check runs on the real export
and confirms the methodology operates honestly — the diagnostic does NOT
fire, so reported metrics are honest small-sample estimates.

### What runs it
```bash
python scripts/train_ph.py        # regenerate artifacts/ph/ + reports/ph/
# then execute the PH notebook suite manually or via:
for nb in notebooks/ph/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" --output "$nb"
done
```

### Live PH server (parallel to Portugal's :8000 / :7860)

A standalone FastAPI + Gradio server lives at `src/app/ph_main.py`. Two ways to launch:

```bash
# Recommended — bullet-proof launcher (artifact check, readiness poll, browser open)
python demo/start_server.py          # Portugal at :8000
python demo/start_server_ph.py       # PH at :8001 (runs side-by-side)

# Or invoke uvicorn directly
uvicorn src.app.main:app --port 8000        # Portugal
uvicorn src.app.ph_main:app --port 8001     # PH
```

Each launcher: verifies trained artifacts exist before spawning, refuses to
start if the port is taken, polls `/healthz` until the model loads, opens
the browser to `/ui`, and tails the uvicorn log on failure. PH log goes to
`.gradio/uvicorn_ph.log`; Portugal log goes to `.gradio/uvicorn.log`.

The PH server is intentionally simpler than the Portugal one:
- SQLite prediction logging + auto-CSV export available (parallel to Portugal,
  writes to `data/predictions/ph_predictions.sqlite` and `ph_predictions_live.csv`)
- No ADR forecasting (PH has no ADR regressor trained)
- No `cost_sensitive` threshold policy (n_val ≈ 19 is too small to fit a reliable cost curve)
- Prominent small-sample caveat banner in `/`, `/model-info`, `/predict` alerts, and Gradio UI

The PH and Portugal servers share zero mutable state: each caches its own
artifact singleton (`_CACHED_PH_ARTIFACTS` vs `_CACHED_ARTIFACTS`), so they can
run side-by-side for a defense demo.

### What it produces

| Path | Content |
|---|---|
| `artifacts/ph/ph_model.pkl` | sklearn Pipeline (preprocessor + LightGBM) |
| `artifacts/ph/ph_calibrator.pkl` | Isotonic calibrator fit on val set |
| `artifacts/ph/ph_thresholds.json` | max_f1 + high_precision thresholds |
| `artifacts/ph/ph_feature_columns.json` | 18-feature list (incl. deposit_type, total_of_special_requests) |
| `artifacts/ph/ph_model_metadata.json` | Lineage + cleaning + caveats |
| `artifacts/ph/cost_threshold_sweep.csv` | FP-cost sensitivity grid |
| `reports/ph/ph_transferability.json` | Test metrics + `dataset_diagnostics` block |
| `reports/ph/ph_test_predictions.csv` | Per-row predictions for the notebooks |
| `reports/ph/ph_threshold_sweep.csv` | Validation threshold sweep |
| `reports/ph/champion_summary.json` | Selection lineage for the PH champion |
| `reports/ph/baseline_comparison.json` | Dummy/LR/DT/GaussianNB vs LightGBM |
| `reports/ph/learning_curves.json` | Train/val PR-AUC at 10/25/50/75/100% |
| `reports/ph/expanding_window_cv.json` | 3-fold expanding-window CV |
| `reports/ph/shap_analysis.json` | TreeSHAP top features |
| `reports/ph/shap_feature_importance.csv` | Per-raw-feature mean(|SHAP|) |
| `reports/ph/shap_summary_plot.png` | SHAP beeswarm |
| `reports/ph/model_family_comparison.json` | 3-way calibrated metrics + paired bootstrap deltas (NB 07/09) |
| `reports/ph/model_family_predictions.csv` | Per-row test predictions for LGBM/XGB/GB (NB 09) |
| `artifacts/ph/ph_adr_regressor.pkl` | HistGradientBoosting ADR regressor (NB 04) |
| `reports/ph/ph_adr_regressor_metrics.json` | ADR regressor train/val/test RMSE/MAE/R² (NB 04) |
| `reports/ph/ph_adr_test_predictions.csv` | Per-row ADR predictions + residuals (NB 04) |

### Notebook suite (`notebooks/ph/`)

Non-contiguous numbering deliberately matches Portugal numbering so the prof can
mentally pair "Portugal 03 ↔ PH 03".

| Notebook | Status | Notes |
|---|---|---|
| `01_eda.ipynb` | NEW | PH categorical structure, pre-flight duplicate-cluster diagnostic, deposit/special-requests distributions |
| `02_modeling.ipynb` | NEW | Champion summary, ROC/PR, calibration, confusion matrix, threshold sweep, feature importance |
| `03_deep_analysis.ipynb` | NEW (light) | Cost curve, learning curves, expanding-window CV, baseline comparison |
| `04_adr_forecasting.ipynb` | NEW | Tabular ADR regressor (no time-series — N too small), feature importance, residual analysis |
| `05_explainability.ipynb` | NEW | SHAP feature importance, beeswarm, 3 individual examples, Portugal vs PH SHAP comparison |
| `06_business_analytics.ipynb` | NEW | Cancellation rate / revenue exposure by deposit, room type, lead-time band; monthly revenue-at-risk |
| `07_model_selection.ipynb` | NEW | 3-way calibrated comparison (LGBM/XGB/GB), bootstrap CIs, paired delta forest |
| `08_model_monitoring.ipynb` | NEW | Runnable monitoring template — baseline + live `/predict` log, PSI drift, risk-tier mix |
| `09_model_comparison.ipynb` | NEW | Per-row probability spread, family-disagreement table, mean-of-3 ensemble vs champion |
| `10_sensitivity_analysis.ipynb` | NEW | Cost sensitivity, data hunger, threshold policy trade-offs |
| `11_transferability.ipynb` | NEW (real-data reframe) | Pre-flight diagnostic outcome + real-data metrics + defense framing |
| `README.md` | NEW | Suite overview + small-N caveat narrative |

### All Portugal notebooks now have a PH counterpart

The PH suite mirrors Portugal 01 through 10 plus the methodology
contribution in 11. Each PH notebook reports the result of running the
same methodology on n ≈ 193 real PMS rows; small-N caveats (±15-30 pp
bootstrap CIs, family CIs that overlap, overfit ADR regressor) are
called out in each notebook rather than hidden by omission.

### Headline real-data findings (2026-05-20 retrain on real PMS export)
| Metric | Real PH (193) | Portugal (119k) |
|---|---|---|
| Train / Val / Test | 154 / 19 / 20 | 95k / 12k / 12k |
| Test ROC-AUC | ≈ 0.61 | 0.86 |
| Test PR-AUC | ≈ 0.54 | 0.76 |
| Top SHAP feature | `deposit_type` | `lead_time` |
| Duplicate rate (pre-flight) | **~0 %** — diagnostic does NOT fire | ~0 % |

Bootstrap 95 % CIs on PR-AUC span roughly ±15 percentage points at n_test = 20,
so the PH metrics are *directional*, not headlines. The PR-AUC gap to Portugal
is the expected cost of (a) ~500× fewer training rows, (b) a narrower feature
menu (still no country/market_segment/agent/customer_type/previous_cancellations),
and (c) different geography/property type.

### Defense-grade framing
The sub-study is reported in the thesis as a methodology-survives-transfer
result with two contributions:

> *"We re-ran the methodology on a 193-row real-data Philippine resort sub-study.
> The generic pre-flight duplicate-cluster diagnostic (`duplicate_rate ≥ 0.30`
> AND `clusters_with_consistent_labels_pct ≥ 0.90`) ran on the new export
> and did not fire — the methodology operates honestly on this data, and
> reported test metrics measure generalization rather than memorization
> across chronological twins. The PR-AUC gap to Portugal (0.54 vs 0.76) is
> the expected cost of ~500× less training data and a narrower feature menu.
> The defensible claim is 'same methodology, weaker model' — exactly what a
> transferability probe should produce when the destination dataset is
> genuinely smaller and feature-poorer."*

### What is deliberately NOT done
- The PH model is **not** added to `make benchmark`, `make thesis`, or CI
- `scripts/check.py` does not validate `artifacts/ph/` or `reports/ph/`
- The Portugal pipeline never imports `src/data/load_ph.py`
- The PH metric gates (`PH_METRIC_GATES` in `config.py`) are directional,
  not regression-detection gates
- No `Makefile` target — invocation stays manual via `python scripts/train_ph.py`

### Tied artifacts
- `notebooks/ph/*.ipynb` — 6-notebook PH suite (described above)
- `notebooks/ph/README.md` — suite overview
- `tests/test_load_ph.py` — 9 smoke tests for the data layer
- `src/data/load_ph.py` — thin loader
- `src/utils/validate_data.py::clean_raw_ph` — normalisation step
- `src/eval/notebook_utils.py::load_ph_context` — PH-side context loader
- `src/serving/inference_ph.py` — PH-side prediction (parallel to `inference.py`)
- `src/app/ph_schemas.py` — `PHBookingRequest` Pydantic model (10 raw fields incl. deposit_type + special_requests)
- `src/app/ph_main.py` — FastAPI server (port 8001)
- `src/app/ph_ui.py` — Gradio UI (mounted at `/ui` and standalone on 7861)
- `src/config.py::PH_*` constants — feature list, target column, paths
- `reports/figures/thesis/ph/` — PH-specific figures (PNG + PDF)
