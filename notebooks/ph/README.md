# PH Sub-Study — Philippine Resort Dataset Notebook Suite

This folder contains the **6-notebook Philippine sub-study** that mirrors the
Portugal main study at `notebooks/01_*` through `notebooks/10_*`. The numbering
is intentionally non-contiguous (01, 02, 03, 05, 10, 11) so that the prof can
mentally pair "Portugal 02 ↔ PH 02" rather than translating renumbered files.

## Dataset framing — read this first

The PH dataset (`data/Punta_Villa_Resort_2022_2024.csv`, 300 rows, 2022–2024)
is the **Philippine resort dataset** — booking records from Punta Villa Resort
across three years. The dataset is organized around a small set of recurring
booking archetypes: 77% of post-engineering feature vectors are shared by
another row, and 100% of those duplicate clusters share a single label. This
is a **structural property of the data**, not a quality concern.

The most striking finding in the sub-study — perfect test-set scores
(PR-AUC = 1.000) — is *not* evidence that the methodology generalises
perfectly. It reflects the chronological-twin effect under this cluster
structure: the test set is dominated by twins of training rows, so the model
classifies correctly by memorizing the archetype space, not by generalising
to unseen customers. Notebook 11 (`11_transferability.ipynb`) characterizes
this in detail.

## The 6 notebooks

| File | Purpose | Mirrors Portugal |
|---|---|---|
| `01_eda.ipynb` | Exploratory data analysis (dataset size, target, temporal coverage, room-type mix, leakage check) | `notebooks/01_eda.ipynb` |
| `02_modeling.ipynb` | Champion summary, ROC/PR curves, calibration, confusion matrix, threshold sweep, feature importance, memorization signature | `notebooks/02_modeling.ipynb` |
| `03_deep_analysis.ipynb` | (Light) cost-curve analysis, learning curves, expanding-window CV, baseline comparison | `notebooks/03_deep_analysis.ipynb` |
| `05_explainability.ipynb` | TreeSHAP feature importance, beeswarm, 3 individual explanations, cross-dataset SHAP comparison | `notebooks/05_explainability.ipynb` |
| `10_sensitivity_analysis.ipynb` | Cost sensitivity sweep, data hunger curve, threshold policy trade-offs | `notebooks/10_sensitivity_analysis.ipynb` |
| `11_transferability.ipynb` | Dataset cluster characterization + chronological-twin effect + thesis framing | (no Portugal equivalent — this is the new methodology contribution) |

## Why some Portugal notebooks are NOT mirrored

| Missing | Reason |
|---|---|
| `04_adr_forecasting.ipynb` | 300 rows over 3 years is too thin for the ADR time-series story; the seasonality plots would be all noise |
| `06_business_analytics.ipynb` | Requires segment breakdowns (country, market segment, customer type) that PH cannot provide |
| `07_model_selection.ipynb` | Comprehensive model selection on 300 rows gives bootstrap CIs ±15pp wide — not statistically defensible |
| `08_model_monitoring.ipynb` | Requires production deployment data (live `/predict` log) that doesn't exist for PH |
| `09_model_comparison.ipynb` | Same sample-size constraint as 07 |

These omissions are part of the thesis story: a methodology is only as
portable as the features its target PMS records *and* the dataset size the
target operator can collect.

## How to regenerate

```bash
# 1. Re-train and produce all artifacts (artifacts/ph/ + reports/ph/)
python scripts/train_ph.py

# 2. Headless-execute every notebook in this folder
for nb in notebooks/ph/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" --output "$nb"
done
```

The figures are saved to `reports/figures/thesis/ph/` (parallel to Portugal's
`reports/figures/thesis/`).

## Prerequisite artifacts

Each notebook calls `load_ph_context()` from `src/eval/notebook_utils.py`,
which expects the following to exist:

```
artifacts/ph/
  ph_model.pkl
  ph_calibrator.pkl
  ph_thresholds.json
  ph_feature_columns.json
  ph_model_metadata.json
  cost_threshold_sweep.csv

reports/ph/
  ph_transferability.json
  ph_test_predictions.csv
  ph_threshold_sweep.csv
  champion_summary.json
  baseline_comparison.json
  learning_curves.json
  expanding_window_cv.json
  shap_analysis.json
  shap_feature_importance.csv
  shap_summary_plot.png
```

All of these are produced by `scripts/train_ph.py` in one run.

## Relationship to the Portugal main study

The PH sub-study **does not replace** Portugal as the primary thesis study.
Portugal remains the headline result (119k real bookings, 49 features,
rolling-origin selection, bootstrap CIs, etc.). PH is the appendix-grade
sub-study that surfaces two methodology contributions:

1. **Pre-flight check for archetype-organized data** — `duplicate_rate ≥ 0.30`
   AND `clusters_with_consistent_labels_pct ≥ 0.90` together flag a dataset
   organized around recurring booking archetypes before any model metric is
   reported.
2. **Feature-availability constraint** — 7 of Portugal's top-10 SHAP-ranked
   features (deposit type, country, market segment, agent, customer type,
   total special requests, previous cancellations) are not in the PH PMS
   export. A model is only as portable as the columns the target PMS records.

See `CLAUDE.md` § "PH Sub-Study — Philippine Resort Dataset" for the full project context.
