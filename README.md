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
make artifact-check # Validate artifacts
make sync-check     # Verify threshold consistency
make full-pipeline  # All of the above in one command
```

Use `make help` to see all available targets.

## Project Entry Points

| Command | Purpose |
|---------|---------|
| `make train` | Full training pipeline |
| `make train DATA_PATH=path/to/data.csv` | Train on a different CSV |
| `make train MAX_ROWS=10000` | Fast smoke-train (10k rows) |
| `make eval` | Post-training verification report |
| `make benchmark` | Generate 16 benchmark CSV tables |
| `make full-pipeline` | train → eval → benchmark → artifact-check → sync-check |
| `make run-notebooks` | Execute all 8 notebooks headlessly |
| `make fairness-check` | Hyperparameter fairness audit |
| `uvicorn src.app.main:app --host 0.0.0.0 --port 8000` | Start API + Gradio UI |

## Data
- Raw dataset: `data/hotel_bookings.csv`
- Target: `is_canceled`
- Training uses booking-time features only (`src/config.py`), with explicit leakage exclusion for post-outcome fields.

## Navigation
See `PROJECT_MAP.md` for a full guide to folders, files, and "where to edit" for each common task.
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

## Quality Gates
```bash
make lint typecheck test    # ruff + mypy + pytest (≥80% coverage)
make security deps-audit    # bandit + pip-audit
make metrics-gate           # ROC-AUC ≥ 0.84, PR-AUC ≥ 0.74, F1 ≥ 0.70
make sync-check             # thresholds consistent across artifacts and reports
```

Pre-commit hooks:
```bash
pre-commit install
pre-commit run --all-files
```

CI runs all of the above via `.github/workflows/ci.yml`.

## Docker
```bash
docker build -t hotel-cancellation-app .
docker run -p 8000:8000 hotel-cancellation-app
```

## Generated Files
Artifacts and reports are generated outputs and git-ignored by default.
