---
name: retrain-and-verify
description: Full retrain + verification cycle (train → eval → benchmark → check sync → demo_check). Use after dataset changes, config changes, or before a thesis demo. User invokes via /retrain-and-verify.
disable-model-invocation: true
---

# Retrain & Verify

This skill runs the complete model pipeline and verifies the result is
demo-ready. It is **user-only** (the model cannot invoke it autonomously)
because it overwrites all artifacts and takes ~10 minutes of CPU.

## When to use

- Dataset changed (new CSV in `data/`)
- Config changed (`src/config.py` thresholds, costs, ratios)
- Before a thesis demo or panel presentation
- After bumping any dependency that touches modeling (lightgbm, xgboost,
  scikit-learn, shap, optuna)

## Procedure

Run each step in order. **Stop on first failure** and surface the failing
command to the user — do not try to repair silently.

### 1. Train the production pipeline
```bash
python scripts/train.py
```
Expected: `reports/metrics.json` updated, all `artifacts/*.pkl` written,
`reports/champion_summary.json` written.

### 2. Post-train verification
```bash
python scripts/train.py --verify-only
```
Expected: `artifacts/model_verification_report.md` clean, no leakage warnings.

### 3. Benchmark all candidate models
```bash
python scripts/benchmark.py
```
Expected: 16 CSV tables under `reports/benchmarks/`.

### 4. Cross-artifact sync check
```bash
python scripts/check.py sync
```
Expected: thresholds in `artifacts/thresholds.json`,
`reports/thesis/model_family_summary.json`, and
`reports/benchmarks/07_thresholds_per_model.csv` agree within
`REPRO_TOLERANCE = 1e-6`.

### 5. Metric gates
```bash
python scripts/check.py metrics
```
Expected: ROC-AUC ≥ 0.70, PR-AUC ≥ 0.50, F1 ≥ 0.50, Recall ≥ 0.50 (max_f1
policy); Precision ≥ 0.90, Recall ≥ 0.05 (high_precision policy).

### 6. Demo readiness
```bash
python scripts/demo_check.py
```
Expected: high-risk scenario ≥ 70%, low-risk scenario < 10%. Live server
check is informational only.

## Output

After all six steps pass, report:

- New ROC-AUC / PR-AUC / F1 (from `reports/metrics.json`)
- New thresholds (from `artifacts/thresholds.json`)
- Champion family + runner-up (from `reports/champion_summary.json`)
- Demo check scenarios (from step 6 stdout)

## Failure handling

- **Step 1 fails**: most likely data validation. Check
  `src/utils/validate_data.py` and the input CSV's schema.
- **Step 4 fails (sync drift)**: retrain produced inconsistent thresholds.
  Usually means a fold or seed changed. Investigate `src/pipelines/train.py`.
- **Step 5 fails (metric gates)**: model regressed. Compare new
  `reports/metrics.json` to git history.
- **Step 6 fails (demo check)**: predictions outside expected ranges.
  Calibrator may have collapsed — inspect
  `artifacts/probability_calibrator.pkl`.
