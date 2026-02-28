# Hotel Booking Cancellation

End-to-end, reproducible ML pipeline for hotel booking cancellation prediction with FastAPI + Gradio serving.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project Entry Points
- Training: `python scripts/train.py`
- Multi-model benchmark tables: `python scripts/benchmark.py`
- Reproducibility check: `python scripts/repro.py --max-rows 5000`
- Model verification report: `python scripts/verify.py`
- Artifact/serving consistency check: `python scripts/check_artifacts.py`
- Serving API + UI: `python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000`
- Local server helper (PowerShell): `powershell -File scripts/server.ps1`

## Data
- Raw dataset: `data/hotel_bookings.csv`
- Target: `is_canceled`
- Training uses booking-time features only (`src/config.py`), with explicit leakage exclusion for post-outcome fields.

## Reproducible Training
```bash
python scripts/train.py
```
Generated outputs:
- `artifacts/best_model.pkl`
- `artifacts/probability_calibrator.pkl`
- `artifacts/feature_columns.json`
- `artifacts/thresholds.json`
- `artifacts/model_metadata.json`
- `artifacts/hashes.json`
- `artifacts/threshold_sweep.csv`
- `reports/metrics.json`
- `reports/calibration_metrics.json`
- `reports/segment_metrics.json`
- `reports/segment_metrics.csv`
- `reports/model_selection_summary.json`
- `reports/model_selection_rolling.csv`
- `reports/confusion_matrix_*.csv`
- `reports/threshold_summary.json`
- `reports/benchmarks/*.csv` and `reports/benchmarks/*.md` (model comparison tables)

Determinism controls:
- Central seed utility: `src/utils/reproducibility.py`
- Fixed random seed in config (`RANDOM_STATE`)
- Time-aware split (`split_time_aware`) for train/val/test
- Deterministic champion/challenger selection (Gradient Boosting vs XGBoost) on rolling time splits
- Deterministic model selection policy tag in metadata: `MODEL_SELECTION_POLICY`
- Persisted probability calibration artifact applied during serving (`probability_calibrator.pkl`)
- Artifact/data/source hashes persisted in metadata and `artifacts/hashes.json`

## Serving
Artifacts must exist before predicting.

Endpoints:
- `GET /` liveness
- `GET /healthz` readiness (artifact load)
- `POST /predict` prediction API
- `GET /ui` Gradio interface

## Quality Gates
Run locally:
```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
python scripts/check_metrics.py  # global + segment-level gates
python -m bandit -q -r src scripts -s B101 -x scripts/test_*.py,tests
python -m pip_audit -r requirements.txt --no-deps --disable-pip
```

Pre-commit:
```bash
pre-commit install
pre-commit run --all-files
```

CI runs the same checks in `.github/workflows/ci.yml`.

## Model Governance
- Model card: `MODEL_CARD.md`
- Reproducibility check module: `src/eval/repro.py`
- Artifact consistency check script: `scripts/check_artifacts.py`

## Docker
```bash
docker build -t hotel-cancellation-app .
docker run -p 8000:8000 hotel-cancellation-app
```

## Generated Files
Artifacts and reports are generated outputs and ignored by default via `.gitignore`.
