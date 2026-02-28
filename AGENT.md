# CODEX

This file defines the default operating rules for Codex in this repository.
Primary goal: keep the thesis project simple, reproducible, and auditable.

## Thesis Readiness Rules

1. Prefer one-path workflows over multiple parallel implementations.
2. Do not add new analysis scripts when an existing pipeline/module can be extended.
3. Every thesis claim must be backed by files under `reports/` or `artifacts/`.
4. Never hardcode a thesis "champion" model that disagrees with training selection outputs.
5. Optional dependencies (`optuna`, `shap`) must fail gracefully and never crash the CLI.
6. Reproducibility is required: same seed and same inputs must reproduce metrics within tolerance.

## Golden Path Commands

Use only these for thesis verification:

```bash
# 1) Install dependencies
python -m pip install -r requirements.txt

# 2) Train production pipeline
python scripts/train.py

# 3) Run thesis analysis (full)
python scripts/thesis.py

# 4) Fast thesis run (skip expensive optional steps)
python scripts/thesis.py --skip-tuning --skip-shap

# 5) Reproducibility and gates
python scripts/repro.py --max-rows 5000
python scripts/check_metrics.py
python scripts/check_artifacts.py
python -m pytest
python -m mypy
python -m ruff check .
python -m ruff format --check .
```

## Required Thesis Outputs

The following must exist after a full run:

- `reports/metrics.json`
- `reports/model_selection_summary.json`
- `reports/calibration_metrics.json`
- `reports/segment_metrics.json`
- `reports/thesis/baseline_comparison.json`
- `reports/thesis/confidence_intervals.json`
- `reports/thesis/expanding_window_cv.json`
- `reports/thesis/learning_curves.json`
- `reports/thesis/temporal_stability.json`
- `artifacts/model_metadata.json`
- `artifacts/hashes.json`

## Simplification Plan (Keep / Update / Delete)

### Keep

- `scripts/train.py`
- `scripts/thesis.py`
- `scripts/repro.py`
- `scripts/check_metrics.py`
- `scripts/check_artifacts.py`

### Update

- `src/eval/thesis_analysis.py`
  - Use the selected model family from training policy, not a fixed model.
  - Catch missing optional dependency errors where they actually occur.
- `src/eval/statistical.py`
  - Replace current paired bootstrap p-value logic with a valid null-based method.
- `src/eval/verify_models.py`
  - Remove stale references to old notebook filenames.
- `README.md` and `MODEL_CARD.md`
  - Add thesis command and expected thesis artifact list.

### Delete or Archive (to reduce thesis noise)

- Archive `src/eval/verify_models.py` and `scripts/verify.py` if not required by thesis deliverables.
- Keep one canonical cancellation notebook path only; remove duplicate variants after final migration.
- Remove any old scripts that duplicate pytest-covered checks.

## Reproducibility and Governance

- Commit and tag thesis milestones (no "thesis-ready" claim without git history).
- Keep all dependency versions pinned in requirement files.
- Treat warnings that affect scientific validity as blockers (for example, statistical test errors or convergence failures in baseline comparisons).

## Architecture Guardrails

- Training and serving must share identical feature transformations via persisted preprocessing artifacts.
- Use booking-time features only and keep leakage columns excluded.
- Record thresholds, calibration, and lineage hashes on every training run.
