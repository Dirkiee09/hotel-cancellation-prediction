# `scripts/` — Command-Line Entry Points

This directory holds the runnable scripts for the project, grouped by purpose.
**The model/feature/serving logic itself lives in `src/`** — these scripts are thin
entry points over it. Run everything with the project virtual-env Python from the
repo root (e.g. `.venv/Scripts/python.exe scripts/training/train.py`), or use the
`make` targets below.

★ = **essential** (part of the daily pipeline, the `make` workflow, or CI).

---

## Quick start — the commands you actually need

| `make` target | Script | What it does |
|---|---|---|
| `make train` | ★ `training/train.py` | Train the full pipeline end-to-end → `artifacts/` + `reports/` |
| `make benchmark` | ★ `training/benchmark.py` | Generate the 16 benchmark tables → `reports/benchmarks/` |
| `make thesis` | ★ `training/train.py --thesis` | Train + thesis analysis (SHAP, bootstrap CIs) |
| `make check` | ★ `utils/check.py all` | Run all quality gates (artifacts, metrics, sync, fairness) |
| — | ★ `utils/check.py reproduce --baseline-dir DIR` | Compare a fresh retrain with a snapshot of the committed `metrics.json` and `thresholds.json` (CI runs this) |
| `make eval` | ★ `training/train.py --verify-only` | Post-train verification on existing artifacts |
| `make export-predictions` | `utils/export_predictions.py` | Prediction log → Power BI CSV |
| `make export-adr` | `utils/export_adr_predictions.py` | ADR test predictions → Power BI |
| `make demo` | `../demo/start_server.py` | Launch the FastAPI + Gradio prediction app |

---

## `training/` — model training & analysis

| Script | ★ | Purpose |
|---|---|---|
| `train.py` | ★ | End-to-end training pipeline (`--thesis`, `--verify`, `--verify-only`, `--repro` flags). The main entry point. |
| `benchmark.py` | ★ | Thin wrapper over `src/eval/benchmark.py` → the 16 benchmark CSV tables. |
| `stratified_cv.py` | ★ | Stratified 10-fold CV across 7 algorithms → `reports/cv/` (produces thesis Table 4.1). |
| `dedup_sensitivity.py` | ★ | Duplicate-row sensitivity retrain → `reports/dedup_sensitivity.json` + persists the de-dup run's metrics/thresholds/predictions/segments under `reports/dedup/` (the manuscript's PRIMARY evaluation stream). |
| `feature_ablation.py` |  | Deposit-type robustness ablation (retrain on de-dup stream with `deposit_type` neutralized) → `reports/deposit_ablation.json` + `reports/thesis/deposit_interactions.csv` (panel revision point 11). |
| `revision_diagnostics.py` |  | Extended bootstrap CIs, calibration diagnostics (Brier/log-loss/slope/intercept, by-segment) and 150-scenario cost grid for both streams → `reports/thesis/confidence_intervals_extended.json`, `calibration_diagnostics.json`, `calibration_by_segment.csv`, `cost_scenarios.csv` (panel points 5-7). |
| `train_ph.py` |  | Philippine transferability sub-study (Punta Villa Resort). NOT part of CI or the Portugal pipeline. |

## `utils/` — quality, exports & operations

| Script | ★ | Purpose |
|---|---|---|
| `check.py` | ★ | Unified quality-gate runner: `artifacts`, `metrics`, `sync`, `fairness`, `reproduce`, or `all`. |
| `notebooks.py` | ★ | Clear / execute / validate the analysis notebooks deterministically. |
| `export_predictions.py` |  | Export `predictions.sqlite` → `predictions_live.csv` for Power BI. |
| `export_adr_predictions.py` |  | Export ADR regressor test predictions + segment RMSE for Power BI. |
| `compute_live_drift.py` |  | Compute drift metrics for the Power BI monitoring page. |
| `adapt_dataset.py` |  | Plug-and-play adapter to map a new hotel CSV onto the project schema. |

## `diagrams/` — thesis figure generators (matplotlib)

| Script | Produces |
|---|---|
| `create_conceptual_framework_diagram.py` | Figure 1.2 — Sense→Seize→Transform framework |
| `create_conceptual_systems_diagram.py` | Figure 4.8 — Conceptual systems positioning |
| `create_deployment_diagram.py` | Figure 4.9 — Technical serving architecture |
| `rebuild_shap_summary_plot.py` | SHAP summary plot with raw feature names |

*(The remaining Chapter-4 results figures — calibration reliability, risk-tier overview,
cost ladder, SHAP beeswarm — are produced by the notebooks.)*

## `demo/` — local demo helpers

| Script | Purpose |
|---|---|
| `demo_check.py` | One-command pre-demo readiness check. |
| `seed_demo_predictions.py` | Seed the prediction log with varied scenarios for the Power BI demo. |

## `presentation/` — defense aids

| Script | Purpose |
|---|---|
| `show_benchmark_table.py` | Print the benchmark results table to the terminal. |
| `show_defense_tables.py` | Print the key defense tables to the terminal. |
