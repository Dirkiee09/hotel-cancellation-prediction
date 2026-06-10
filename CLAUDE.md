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
| `pytest` | ✅ 114/114 | 89.87% coverage (gate: ≥80%) |
| `ruff check` | ✅ clean | E, F, I rules |
| `ruff format` | ✅ clean | line length 100 |
| `mypy` | ✅ clean | 0 errors (48 source files checked) |
| `src/` imports | ✅ 15/15 modules | all importable |
| `scripts/` | ✅ 4/4 scripts | train, benchmark, check, notebooks |
| `notebooks/` | ✅ 10/10 notebooks | 01_eda through 10_sensitivity_analysis, all with cached outputs |

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
  adapt_dataset.py            # plug-and-play adapter for new hotel CSVs (e.g. PH dataset)

demo/                         # Local prediction app launchers (defense / day-to-day use)
  start_server.py             # make demo — FastAPI + Gradio UI at localhost:8000, opens browser
  quick_train.py              # fast 10k-row smoke-train when artifacts are missing
  sample_requests.py          # send 4 contrasting bookings to the running server

docs/
  provenance.md               # maps local ph/cv/adr outputs to their producers on `master`

tests/                        # pytest suite (114 tests, ≥80% coverage required)
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
make test                     # pytest (114 tests, coverage ≥80%)
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
6. `scripts/check.py metrics` — metric gates must pass
7. `scripts/check.py sync` — cross-artifact threshold consistency
8. bandit security scan
9. pip-audit dependency vulnerability scan

**`thesis-analysis` job** (runs only on push to `main`, needs `quality`):
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
| `EARLY_STOPPING_ROUNDS` | 50 | Early-stopping patience when an eval_set is passed to train_xgb/train_lgbm |

**Feature-list note**: `arrival_date_week_number` is intentionally NOT a model feature.
The raw dataset's week numbering disagrees with ISO-8601 for ~54% of dates, so deriving
it at serving time would create training/serving skew. The API still accepts the field
(informational only); seasonality is captured by month one-hot + `month_sin`/`month_cos`.

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
  │     ├── fill agent NaN → 0 (kept as category "0" ≈ direct booking)
  │     ├── add_derived_booking_features() → had_company, total_stay,
  │     │   total_guests, adr_per_person, revenue_at_risk,
  │     │   month_sin, month_cos, is_late_window, is_weekend_heavy
  │     └── drop rows: negative ADR, ADR ≥ ADR_MAX_VALID (50,000), zero/null guests
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
  │     ├── candidates: LightGBM, XGBoost (matched capacity:        │
  │     │   300 trees / depth 7 / lr 0.05 / subsample 0.8) +        │
  │     │   GradientBoosting (classical reference, 100/5/0.1 —      │
  │     │   exact-greedy sklearn GB is too slow to match)           │
  │     ├── no class weighting for ANY candidate (symmetric, and    │
  │     │   identical to benchmark specs → sync check holds;        │
  │     │   imbalance handled by calibration + threshold policies)  │
  │     └── primary metric: PR-AUC ─────────────────────────────────┘
  │
  ├─▶ isotonic calibration              [sklearn.isotonic.IsotonicRegression]
  │     └── fitted on val set predictions → saved to artifacts/probability_calibrator.pkl
  │
  ├─▶ threshold selection               [src/utils/thresholds.py]
  │     ├── max_f1            (F1-maximising, ~0.40)
  │     ├── high_precision    (max precision s.t. recall ≥ 0.20; floor is often
  │     │                      unsatisfiable → logged fallback, observed recall ~0.09)
  │     └── cost_sensitive    (min FP×15€ + FN×revenue_at_risk; summary reports
  │           │                no-model, threshold-0.5 AND intervene-all baselines)
  │           └── sweep runs on val set → saved to artifacts/thresholds.json
  │               (test-set evaluation of the selected threshold lands in
  │                metrics.json → cost_thresholding_test; H2/H4 in
  │                hypothesis_summary.json are judged on TEST, never val)
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
  ├── adr validated: 0 ≤ adr ≤ ADR_MAX_VALID (50,000, currency-agnostic)
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
  └── PredictionResponse: probability, 3 binary labels, risk_tier, alerts

Thread safety: _ARTIFACTS singleton protected by _ARTIFACTS_LOCK
  → safe for multi-worker uvicorn deployments
```

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
| `reports/benchmarks/01_*.csv` … `16_*.csv` | `scripts/benchmark.py` | Notebook 07, `scripts/check.py sync` |
| `reports/thesis/*.json` | `src/eval/thesis.py` | Notebooks 03/05, `scripts/check.py sync` |

**Threshold cross-check**: `scripts/check.py sync` verifies the same `max_f1` / `high_precision`
/ `cost_sensitive` thresholds appear in `artifacts/thresholds.json`,
`reports/thesis/model_family_summary.json`, and `reports/benchmarks/07_thresholds_per_model.csv`
within `REPRO_TOLERANCE = 1e-6`. If those drift, retrain.

**Note**: the former `reports/threshold_summary.json` (a redundant copy of
`artifacts/thresholds.json`) was removed in the 2026-06 cleanup; `artifacts/thresholds.json`
is the single canonical source.

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
