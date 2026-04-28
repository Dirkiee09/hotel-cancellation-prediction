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
| `uvicorn src.app.main:app --host 0.0.0.0 --port 8000` | Start API + Gradio UI |

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
