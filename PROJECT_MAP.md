# Project Map — Hotel Booking Cancellation Prediction

Quick reference for navigating and contributing to this project.

---

## Folder Overview

| Folder | Purpose |
|--------|---------|
| `src/` | All importable Python source code |
| `scripts/` | Thin CLI entry points — one command per job |
| `notebooks/` | Analysis and reporting notebooks (load artifacts, no retraining) |
| `data/` | Raw dataset (`hotel_bookings.csv`, 119k rows, git-ignored) |
| `artifacts/` | Trained model files produced by `make train` (git-ignored) |
| `reports/` | Generated metrics, benchmark tables, figures (git-ignored) |
| `tests/` | pytest suite — 38 tests, ≥80% coverage enforced by CI |
| `.github/workflows/` | CI pipeline — quality gates + thesis analysis |

---

## Where to Edit — Common Tasks

| Task | File(s) to change |
|------|------------------|
| Change a constant (threshold, cost, ratio) | `src/config.py` — all constants live here |
| Edit the Gradio UI | `src/app/ui.py` |
| Edit the FastAPI endpoints | `src/app/main.py` + `src/app/schemas.py` |
| Change feature engineering | `src/features/build.py` |
| Change model training logic | `src/models/train.py` + `src/pipelines/train.py` |
| Change threshold selection | `src/utils/thresholds.py` |
| Add a notebook plot / helper | `src/eval/notebook_utils.py` |
| Add a thesis analysis step | `src/eval/thesis.py` |
| Add a benchmark table | `src/eval/benchmark.py` |
| Change data validation rules | `src/utils/validate_data.py` |
| Change serving / inference | `src/serving/inference.py` |

---

## Key Source Modules

```
src/
  config.py               ← ALL constants (edit here first, never hardcode)
  data/load.py            ← Raw CSV loader
  features/build.py       ← Feature engineering, preprocessor, train/val/test split
  models/
    train.py              ← LightGBM / XGBoost / GradientBoosting training
    metrics.py            ← ROC-AUC, PR-AUC, F1, ECE
    baselines.py          ← Dummy, LR, DT, NB baselines for thesis comparison
  pipelines/train.py      ← Master pipeline (called by scripts/train.py)
  serving/inference.py    ← predict_proba(), load_artifacts()
  utils/
    thresholds.py         ← Threshold sweep + resolve_thresholds()
    validate_data.py      ← Data contract checks before training
    business.py           ← Revenue-at-risk, cost math, risk tiers
  eval/
    notebook_utils.py     ← All notebook helpers (plots, SHAP, context loaders)
    thesis.py             ← SHAP, CI, ablation, Optuna analysis
    benchmark.py          ← 16 benchmark CSV tables
    statistical.py        ← Bootstrap CI + significance tests
  app/
    main.py               ← FastAPI: /predict, /model-info, /healthz
    ui.py                 ← Gradio UI (hotel operations dashboard)
    schemas.py            ← Pydantic BookingRequest / PredictionResponse
```

---

## Scripts Reference

| Command | What it does |
|---------|-------------|
| `make train` | Full training pipeline → saves `artifacts/` + `reports/` |
| `make train DATA_PATH=path/to/data.csv` | Train on a different CSV file |
| `make train MAX_ROWS=10000` | Quick smoke-train on first 10k rows |
| `make eval` | Post-training verification report |
| `make benchmark` | Generate 16 CSV benchmark tables |
| `make artifact-check` | Validate artifact hashes + smoke prediction |
| `make sync-check` | Verify thresholds match across artifacts and reports |
| `make full-pipeline` | Chain: train → eval → benchmark → artifact-check → sync-check |
| `make thesis-analysis-fast` | Thesis analysis (skips Optuna + SHAP, faster iteration) |
| `make test` | pytest suite with coverage |
| `make lint` / `make typecheck` | ruff + mypy |
| `make help` | Print all targets with descriptions |

---

## Artifact Files (produced by `make train`)

| File | Contents |
|------|---------|
| `artifacts/best_model.pkl` | Trained sklearn Pipeline (preprocessor + LightGBM) |
| `artifacts/probability_calibrator.pkl` | Isotonic calibrator fitted on val set |
| `artifacts/thresholds.json` | max_f1, high_precision, cost_sensitive thresholds |
| `artifacts/feature_columns.json` | Canonical 49-feature list |
| `artifacts/model_metadata.json` | Git SHA, timestamps, data stats, model selection summary |
| `artifacts/hashes.json` | SHA-256 checksums of all artifact files |

---

## Data Split Design

```
119k rows (chronological by arrival_date)
├── Train  80%  (~95k)   model fitting + rolling-origin selection
├── Val    10%  (~12k)   calibration + threshold selection
└── Test   10%  (~12k)   final reported metrics (touched once)
```

Reported as "80% train / 20% holdout" in thesis text.
The 10/10 val/test split is methodology detail (Appendix).

---

## Reproducibility Checklist

Before sharing results or submitting the thesis:

- [ ] `make full-pipeline` exits 0
- [ ] `make test` shows 38 passed, ≥80% coverage
- [ ] `make repro-check` shows hash match
- [ ] `make sync-check` shows thresholds consistent
- [ ] `git status` in `artifacts/` shows no uncommitted drift
