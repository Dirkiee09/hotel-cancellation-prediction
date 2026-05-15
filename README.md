# Hotel Booking Cancellation

End-to-end, reproducible ML pipeline for hotel booking cancellation prediction with FastAPI + Gradio serving.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e . -r requirements.txt
```

## Quick Start
```bash
make train          # Train model end-to-end
make eval           # Post-training verification
make check          # Validate artifacts, metrics, sync, fairness
make full-pipeline  # All of the above in one command
```

Use `make help` to see all available targets.

> **Windows users**: If `make` is not available, run scripts directly:
> `python scripts/train.py`, `python scripts/train.py --verify-only`, etc.

## Project Entry Points

| Command | Purpose |
|---------|---------|
| `make train` | Full training pipeline |
| `make train DATA_PATH=path/to/data.csv` | Train on a different CSV |
| `make train MAX_ROWS=10000` | Fast smoke-train (10k rows) |
| `make eval` | Post-training verification report |
| `make benchmark` | Generate 16 benchmark CSV tables |
| `make thesis` | Full thesis analysis (SHAP, CI, Optuna) |
| `make check` | All quality gates (artifacts, metrics, sync, fairness) |
| `make full-pipeline` | train → eval → benchmark → check |
| `make export-predictions` | Export `predictions.sqlite` → `predictions_live.csv` for Power BI |
| `python demo/start_server.py` | Bullet-proof launcher: verifies artifacts + polls /healthz + opens browser |
| `python scripts/seed_demo_predictions.py` | Seed 30 varied prediction scenarios into the live log (for dashboard demos) |
| `python scripts/compute_live_drift.py` | Compute live-vs-baseline PSI drift metrics for the Power BI monitoring page |

## Data
- Raw dataset: `data/hotel_bookings.csv`
- Target: `is_canceled`
- Training uses booking-time features only (`src/config.py`), with explicit leakage exclusion for post-outcome fields.

## Navigation
See `notebooks/README.md` for notebook purposes, run order, and required artifacts.

## Reproducible Training
```bash
make train
# or with a custom dataset:
make train DATA_PATH=data/new_export.csv
```
Generated outputs:
- `artifacts/best_model.pkl`
- `artifacts/probability_calibrator.pkl`
- `artifacts/feature_columns.json`
- `artifacts/thresholds.json`
- `artifacts/model_metadata.json`
- `artifacts/hashes.json`

Determinism controls:
- Fixed random seed (`RANDOM_STATE = 42` in `src/config.py`)
- Time-aware 80/10/10 chronological split
- Rolling-origin champion selection (LightGBM vs XGBoost vs GradientBoosting)
- Isotonic calibration fitted on val set only
- SHA-256 artifact hashes persisted in `artifacts/hashes.json`
- Git commit SHA embedded in `artifacts/model_metadata.json`

## Serving
Artifacts must exist before predicting (run `make train` first).

Endpoints:
- `GET /` liveness
- `GET /healthz` readiness (checks artifacts loaded)
- `POST /predict` prediction API
- `GET /ui` Gradio interface

## Power BI Dashboard

Every successful `/predict` call (HTTP or Gradio UI) appends one row to a
SQLite audit log and refreshes a Power BI-friendly CSV. The dashboard
visualises live predictions alongside the held-out test-set baseline.

**Pipeline**:
```
FastAPI /predict   or   Gradio "Predict" button
     ↓ (auto)
data/predictions/predictions.sqlite   (one row per prediction)
     ↓ (auto)
data/predictions/predictions_live.csv (Power BI consumes this)
```

**60-second setup**:
1. `python demo/start_server.py` — start the server
2. Make ~30 predictions through the Gradio UI (or run `python scripts/seed_demo_predictions.py` for 30 pre-built scenarios)
3. Open Power BI Desktop → **Home > Get Data > Text/CSV** → pick `data/predictions/predictions_live.csv` → **Load**
4. After each new prediction, click **Refresh** in Power BI — new rows appear automatically

**Dashboard data sources** (each is a separate Power BI table):

| Path | Role |
|------|------|
| `data/predictions/predictions_live.csv` | Live operational predictions (auto-refreshed) |
| `data/predictions/drift_metrics.csv` | PSI drift metrics from `compute_live_drift.py` |
| `reports/test_predictions_for_powerbi.csv` | 11,922 holdout predictions with `is_canceled` outcomes (for calibration + backtest visuals) |
| `reports/segment_metrics.csv` | Per-segment ROC-AUC + thresholds (for fairness matrix) |
| `reports/thesis/shap_feature_importance.csv` | Global SHAP feature ranking (for explainability page) |
| `reports/metrics.json` | Champion model headline metrics (for KPI cards) |

**Recommended pages** (see `C:\Users\dirkv\.claude\plans\misty-hugging-wirth.md` or the dashboard guide for full specs):
1. **Risk Overview** — KPI cards, risk-tier donut, probability histogram
2. **Action List** — filterable table of high-risk predictions with SHAP explanations
3. **Risk Patterns** — customer_type × market_segment heatmap, cancel rate by deposit/agent
4. **Policy Comparison** — counts flagged by each of the 3 threshold policies
5. **Explainability** — global SHAP bar + per-prediction feature contributions
6. **Business Impact** — revenue at risk by tier/segment/country
7. **Trustworthiness** — calibration reliability scatter + per-segment ROC-AUC matrix
8. **Monitoring** — PSI heatmap, drift zone bar (`drift_metrics.csv`)

**Refreshing drift metrics**: run `python scripts/compute_live_drift.py` before opening Power BI to update the monitoring page.

## Quality Gates
```bash
make lint typecheck test    # ruff + mypy + pytest (≥80% coverage)
make security               # bandit security scan
make check                  # artifacts + metrics + sync + fairness
```

Pre-commit hooks:
```bash
pre-commit install
pre-commit run --all-files
```

CI runs all of the above via `.github/workflows/ci.yml` (Python 3.11). Local development on Python 3.12+ is also supported.

## Docker
```bash
docker compose build
docker compose run --rm train
docker compose up api
```

Notes:
- Install Docker Desktop or Docker Engine before using these commands.
- `train` mounts local `data/`, `artifacts/`, and `reports/` so generated model files persist on your machine.
- `api` serves the FastAPI + Gradio app on `http://localhost:8000` and reads model artifacts from the local `artifacts/` directory.
- For a quick smoke-train you can override the command, for example:

```bash
docker compose run --rm train python scripts/train.py --max-rows 10000
```

## Generated Files
Artifacts and reports are generated outputs and git-ignored by default.

Three directories carry runtime state:

| Directory | Lifecycle | Examples |
|-----------|-----------|----------|
| `artifacts/` | `make train` regenerates everything from scratch | model pipeline, calibrator, thresholds, feature columns |
| `reports/` | `make train` / `make benchmark` / `make thesis` regenerate | metrics JSON, 16 benchmark CSVs, thesis figures, segment metrics |
| `data/predictions/` | Append-only during serving; `--reset` to wipe | SQLite audit log + Power BI CSV + drift metrics |
