# Hotel Booking Cancellation Prediction

Predicts whether a hotel booking will be cancelled, **using only information known at
the moment the booking is made**. The repository contains the full pipeline: data
cleaning, feature engineering, chronological model selection, probability
calibration, cost-sensitive thresholding, a serving layer (FastAPI + Gradio), and an
evaluation suite.

The design constraint that shapes everything else is the prediction point. A booking
is scored the day it arrives, so no post-booking field (`reservation_status`,
`reservation_status_date`, `assigned_room_type`, `booking_changes`,
`days_in_waiting_list`) may enter the model. Those columns are dropped explicitly
rather than merely unused.

---

## ▶ Just want to run the demo?

> **Click the green “Code → Download ZIP” button.** The trained model is
> committed, so that download is a working demo — nothing to train.
>
> Unzip it somewhere simple like `C:\hotel-demo`, read **`START_HERE.txt`**,
> then double-click **`START DEMO.bat`**.
>
> You need **Python 3.11 or 3.12**. Newer versions have no matching wheels for
> `numpy` / `scikit-learn` / `lightgbm` and the install will fail — check with
> `python --version` first. Setup is three commands and takes about five
> minutes; `START_HERE.txt` walks through them.

## Where to start

| If you are… | Read this |
|---|---|
| Presenting the work | [`demo/PRESENTATION_GUIDE.md`](demo/PRESENTATION_GUIDE.md), then [`demo/HANDOVER.md`](demo/HANDOVER.md) |
| Running the demo | `START_HERE.txt` in the project folder, or [Setup](#setup) below |
| Reviewing the method | [Results](#results-at-a-glance) → [How the pipeline works](#how-the-pipeline-works) → [Quality gates](#quality-gates) |
| Changing the code | [Repository map](#repository-map) → [Key configuration](#key-configuration) |
| Building the dashboard | [`docs/powerbi_dashboard_guide.md`](docs/powerbi_dashboard_guide.md) |
| Checking what is reproducible | [`docs/provenance.md`](docs/provenance.md) |

---

## Results at a glance

Chronological out-of-time test set, champion model (LightGBM), `max_f1` policy:

| Metric | Per-record | De-duplicated |
|---|---|---|
| ROC-AUC | 0.863 | 0.848 |
| PR-AUC | 0.759 | 0.703 |
| F1 | 0.736 | 0.707 |
| Recall | 0.895 | 0.852 |
| Expected calibration error | 0.031 | 0.018 |
| Test rows | 11,922 | 8,724 |

Two streams are reported because 26.8% of rows are exact duplicates (block
bookings). The de-duplicated stream is the conservative estimate; the per-record
stream treats every row as a separately cancellable reservation. Duplicates never
cross the train/test boundary, so the gap reflects re-weighting, not leakage.

Cost-sensitive thresholding reduces total modelled cost by **31.2%** against
contacting every guest (€84,045 → €57,815, de-duplicated test set). That figure
assumes a contact costs €15 only when the guest would not have cancelled, and that
contacting a guest who would have cancelled saves the booking. Neither assumption
is measured; the saving in euros is far less sensitive to them than the percentage.

Every figure above is written by a script listed in
[`docs/provenance.md`](docs/provenance.md). Retraining the committed code from
scratch on 12 October 2026 reproduced the model file byte for byte and every
reported value exactly.

## The dataset

`data/hotel_bookings.csv` is the public **hotel booking demand** dataset: 119,390
bookings across two Portuguese hotels — one city hotel and one resort hotel — arriving
between July 2015 and August 2017, with 32 columns per booking. Portugal is the largest
source market (48,590 bookings), followed by the United Kingdom and France.

It is redistributed here unmodified, for reproducibility. Original source:

> Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets.
> *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

Cleaning applied by this pipeline (null fills, outlier and zero-guest row drops) reduces
it to 119,209 rows before splitting; see "How the pipeline works" below. Please cite the
original authors for any reuse, and check the article for its licensing terms.

## Repository map

```
data/
  hotel_bookings.csv        Raw dataset (119,390 rows, 32 columns)

src/
  config.py                 Every project-wide constant — change values here, not inline
  data/load.py              CSV loader
  features/build.py         Feature engineering, preprocessor, chronological split
  models/
    train.py                LightGBM / XGBoost / GradientBoosting fitting helpers
    metrics.py              ROC-AUC, PR-AUC, F1, expected calibration error
    baselines.py            Dummy, logistic, decision tree, naive Bayes
    tuning.py               Optuna hyperparameter search
  pipelines/train.py        Master training pipeline
  utils/
    thresholds.py           Threshold selection and resolution
    business.py             Revenue-at-risk, cost model, risk tiers
    validate_data.py        Schema, leakage and bounds validation
    drift.py                Population-stability drift metrics
  eval/
    benchmark.py            Multi-model benchmark (16 tables)
    statistical.py          Bootstrap confidence intervals, paired significance tests
    thesis.py               SHAP, ablation, confidence intervals
    verify.py               Post-training sanity checks
    notebook_utils.py       Shared notebook plotting and context loaders
  serving/
    inference.py            Prediction contract, SHAP explanation, ADR forecast
    prediction_log.py       SQLite audit log of every prediction
  app/
    main.py                 FastAPI endpoints
    ui.py                   Gradio interface
    schemas.py              Pydantic request validation

scripts/                    Command-line entry points (thin wrappers over src/)
  training/                 train.py, benchmark.py, stratified_cv.py, dedup_sensitivity.py
  utils/                    check.py, export_predictions.py, compute_live_drift.py
  diagrams/                 Figure generators
  demo/                     Demo readiness check, prediction seeding

demo/
  start_server.py           Launcher: verifies artifacts, polls /healthz, opens browser
  quick_train.py            10k-row smoke train when artifacts are missing
  sample_requests.py        Sends contrasting bookings to a running server

tests/                      153 tests, coverage gate ≥80%
notebooks/                  Analysis notebooks (load artifacts, never retrain)
docs/                       Power BI build guide, output provenance

artifacts/                  Trained model the app serves (committed; rebuilt by make train)
reports/                    Metrics, benchmark tables, figures (committed subset; see docs/provenance.md)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e . -r requirements.txt
```

**Use Python 3.11 or 3.12.** Every dependency in `requirements.txt` is pinned to an exact
version, and those pins are verified on 3.11 (CI) and 3.12 (local). Newer Python releases
may not have matching wheels for `numpy`, `scikit-learn` or `lightgbm`, which is the most
common reason setup fails on a fresh machine. Check with `python --version` first.

The package installs in editable mode, so imports are `from src.x import y` with no path
manipulation.

### The trained model is included

`artifacts/` and the reports the app reads are committed, so a clone or a Download ZIP
can serve predictions straight away. To rebuild the model yourself:

```bash
python scripts/training/train.py
```

…which is deterministic (every seed is fixed at 42) and takes about ten minutes. On the
same Python and library versions it reproduces the committed model byte for byte; the
versions are recorded in `artifacts/model_metadata.json`. The committed results were
produced on Windows with Python 3.12.10.

`python demo/quick_train.py` trains on 10k rows in under a minute if you only want to
check the interface. Do not present with that model — its calibration differs from the
reported figures.

## Commands

| Command | Purpose |
|---------|---------|
| `make train` | Full training pipeline |
| `make train DATA_PATH=path/to.csv` | Train on a different CSV |
| `make train MAX_ROWS=10000` | Fast smoke train |
| `make eval` | Post-training verification report |
| `make benchmark` | 16 benchmark tables in `reports/benchmarks/` |
| `make check` | Quality gates: artifacts, metrics, sync, fairness |
| `make full-pipeline` | train → eval → benchmark → check |
| `make lint typecheck test` | ruff, mypy, pytest |
| `make security` | bandit scan |
| `python demo/start_server.py` | Serve the API + UI at localhost:8000 |
| `START DEMO.bat` | Windows: double-click to do the same, venv resolved automatically |

`make help` lists every target. On Windows without `make`, call the scripts directly:
`python scripts/training/train.py`, `python scripts/utils/check.py all`.

## How the pipeline works

```
data/hotel_bookings.csv  (119,390 rows, 32 columns)
  │
  ├─ clean_raw()            fill children/agent nulls, derive booking-time features,
  │                         drop negative ADR, ADR ≥ 50,000, zero-guest rows
  │                         → 119,209 rows
  ├─ validate_raw()         schema, binary target, no negative numerics
  ├─ sort by arrival date   chronological order, then drop leakage columns
  │
  ├─ split_time_aware()     train 80% / val 10% / test 10%, chronological
  │                         val  → calibration + threshold selection
  │                         test → final reported metrics, touched once
  │
  ├─ build_preprocessor()   categorical: impute "UNKNOWN" → one-hot (min_freq 1%)
  │                         numeric:     impute median
  │                         33 features → 93 columns
  │
  ├─ rolling-origin selection
  │     cutoffs at 60/70/80% of train+val; candidates LightGBM, XGBoost (matched
  │     capacity: 300 trees, depth 7, lr 0.05) and GradientBoosting; no class
  │     weighting for any candidate; selection metric PR-AUC
  │
  ├─ isotonic calibration   fitted on val only
  │
  ├─ threshold selection    max_f1 · high_precision · cost_sensitive
  │                         swept on val, evaluated on test
  │
  └─ FastAPI + Gradio serve /predict, /model-info, /healthz, /ui
```

### Why the split is three-way

The write-up describes an 80/20 split; the code implements 80/10/10. The middle slice
is necessary: isotonic calibration and the threshold sweep both need data the model
never trained on, and fitting them on the test set would leak. The test decile is
reserved for final numbers and read once.

### Calibration behaves like a step function

Isotonic regression is piecewise-constant. Its end bins collapse to exactly 0.0 or
1.0 when every validation booking in a bin shared one outcome — about 16% of test
bookings. The stored probability is left untouched, but the interface reports anything
beyond `PROBABILITY_DISPLAY_FLOOR` / `CEILING` as `<1%` / `>99%`, because a
calibrated model asserting certainty is an artefact of the calibrator, not a claim the
model supports.

## Key configuration

All constants live in `src/config.py`.

| Constant | Value | Purpose |
|----------|-------|---------|
| `TRAIN_RATIO` / `VAL_RATIO` | 0.80 / 0.10 | Chronological split sizes |
| `RANDOM_STATE` | 42 | Every seed: models, bootstrap, Optuna |
| `FP_INTERVENTION_COST` | 15.0 | Cost of one unnecessary guest contact |
| `FN_RECOVERY_NIGHTS` | 1.0 | Nights re-sold after a late cancellation |
| `RISK_TIER_MEDIUM_THRESHOLD` | 0.40 | Low → Medium tier boundary |
| `RISK_TIER_HIGH_THRESHOLD` | 0.70 | Medium → High tier boundary |
| `ADR_MAX_VALID` | 50,000 | Outlier ceiling; currency-dependent |
| `CALIBRATION_METHOD` | isotonic | Probability calibration |
| `BOOTSTRAP_N_ITERATIONS` | 2000 | Resamples for confidence intervals |

`arrival_date_week_number` is deliberately **not** a model feature: the raw dataset's
week numbering disagrees with ISO-8601 for roughly 54% of dates, so deriving it at
serving time would feed the model a different distribution than it trained on. The API
still accepts the field as informational. Seasonality is carried by month one-hot plus
`month_sin` / `month_cos`.

## Serving

The server refuses to start without artifacts; the committed ones are enough.

| Endpoint | Purpose |
|---|---|
| `GET /` | Liveness |
| `GET /healthz` | Readiness; confirms artifacts loaded |
| `GET /model-info` | Thresholds, feature count, lineage hash |
| `POST /predict` | Calibrated probability, risk tier, SHAP drivers, ADR forecast |
| `GET /ui` | Gradio interface |

Artifacts are cached in a thread-safe singleton with double-checked locking, so the app
is safe under multi-worker uvicorn.

### Live ADR forecast

Every successful prediction also runs the average-daily-rate regressor and records two
extra fields:

| Field | Meaning |
|---|---|
| `predicted_adr` | Rate the model expects this booking to charge |
| `adr_residual` | `entered_adr − predicted_adr`; positive means priced above expectation |

**Known limitation.** The ADR regressor was trained with four post-booking features
(`is_canceled`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`) that
are unknown at booking time. Live inference substitutes neutral placeholders, so live
`predicted_adr` is less accurate than the published test RMSE of 44.31. Retraining on
booking-time features only is the clean fix. Loading is best-effort: if the regressor
is absent both fields are `None` and cancellation prediction is unaffected.

## Quality gates

```bash
make lint typecheck test    # ruff, mypy, pytest (coverage ≥80%)
make security               # bandit
make check                  # artifacts, metrics, sync, fairness
```

`make check` runs four gates:

- **artifacts** — model, calibrator and feature columns load and round-trip
- **metrics** — headline metrics clear the floors in `METRIC_GATES`
- **sync** — the same thresholds appear in `artifacts/thresholds.json`, the thesis
  snapshot and the benchmark tables, within `1e-6`
- **fairness** — XGBoost is refitted at matched capacity and compared to the champion,
  so the ranking cannot be an artefact of an unequal hyperparameter budget

CI (`.github/workflows/ci.yml`) runs lint, types, tests and security scans on every
push. It checks that the committed results agree with each other, retrains on Linux
with Python 3.11, and runs `check.py reproduce`, which prints how far each headline
figure moved from the committed value and fails beyond a stated tolerance. The
comparison is posted as an annotation on the run page.

Optional pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## Reproducibility

- Fixed seed (`RANDOM_STATE = 42`) across models, bootstrap and Optuna
- Chronological split — no shuffling, no random folds in the headline result
- Calibration fitted on the validation slice only
- SHA-256 artifact hashes written to `artifacts/hashes.json`
- Git commit SHA embedded in `artifacts/model_metadata.json`

Known gap: the ADR regressor artifacts have no versioned producer script. See
[`docs/provenance.md`](docs/provenance.md).

## Power BI dashboard

Each prediction appends a row to `data/predictions/predictions.sqlite` and refreshes
`data/predictions/predictions_live.csv`, which Power BI Desktop reads directly — no
ODBC driver needed.

```
/predict  or  Gradio Predict
   ↓
data/predictions/predictions.sqlite      (one row per prediction)
   ↓
data/predictions/predictions_live.csv    (Power BI reads this)
```

1. `python demo/start_server.py`
2. Make some predictions, or seed 30 scenarios with
   `python scripts/demo/seed_demo_predictions.py`
3. Power BI Desktop → **Get Data → Text/CSV** → `predictions_live.csv`
4. Run `python scripts/utils/compute_live_drift.py` to refresh drift metrics

The full page-by-page build walkthrough is in
[`docs/powerbi_dashboard_guide.md`](docs/powerbi_dashboard_guide.md).

## Swapping datasets

The pipeline accepts any hotel booking CSV sharing the same 32 column names.

1. Replace `data/hotel_bookings.csv`.
2. Retune the currency-dependent constants in `src/config.py` — `ADR_MAX_VALID`
   (outlier ceiling) and `FP_INTERVENTION_COST` (contact cost). Keep
   `FN_RECOVERY_NIGHTS` at 1.0 unless you have market-specific evidence.
3. `make train && make eval`.
4. Optionally tighten `METRIC_GATES` to the observed values minus 0.02, so later
   retraining catches regressions.

Risk-tier bands (`RISK_TIER_MEDIUM_THRESHOLD`, `RISK_TIER_HIGH_THRESHOLD`) are static
display values, not data-driven. Retune them alongside the cost constants, because the
calibrated probability distribution shifts with the dataset.

## Philippine sub-study

A parallel study re-fits the same methodology on a real Philippine resort PMS export
(193 bookings, 2022–2025) to probe whether the approach transfers to a small property
in a different market.

```bash
python scripts/training/train_ph.py
```

This is **deliberately outside** CI, `make`, and the Portugal pipeline — it runs as a
separate manual command and writes to `artifacts/ph/` and `reports/ph/`. Notebooks
live in `notebooks/ph/`.

Two caveats matter. The source PMS data is private and is **not** in this repository,
so reproduction needs the original export. And at roughly 20 test bookings, bootstrap
confidence intervals on PR-AUC span about ±15 percentage points — these results are
directional, not headline figures.

## Docker

```bash
docker compose build
docker compose run --rm train
docker compose up api
```

`train` mounts local `data/`, `artifacts/` and `reports/` so outputs persist. `api`
serves on `http://localhost:8000` and reads artifacts from the local directory. For a
smoke train, override the command:

```bash
docker compose run --rm train python scripts/training/train.py --max-rows 10000
```

## Generated files

| Directory | In git? | Lifecycle |
|-----------|---------|-----------|
| `artifacts/` | Yes, except `artifacts/ph/` | Regenerated by `make train` |
| `reports/` | The files the app and the paper rely on | Regenerated by `make train`, `make thesis`, `make benchmark` and the scripts in [`docs/provenance.md`](docs/provenance.md) |
| `data/predictions/` | No | Append-only during serving |

Anything under `artifacts/ph/` or a row-level file under `reports/ph/` comes from the
private Philippine export and is never committed.
