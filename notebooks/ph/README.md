# PH Sub-Study — Real Punta Villa Resort Notebook Suite

This folder contains the **11-notebook Philippine sub-study** that mirrors
the Portugal main study at `notebooks/01_*` through `notebooks/10_*`, plus a
methodology-contribution notebook 11. Every Portugal notebook now has a PH
counterpart so the prof can pair them mentally (Portugal 02 ↔ PH 02 etc.).

## Dataset framing — read this first

The PH dataset is the **real Punta Villa Resort PMS export**
(`data/Punta_Villa_Resort_PH_Dataset.csv`, 193 bookings, 2022-2025). Every
row is one historical booking — there is no synthetic generation. The
cancellation rate is 15.0 % (29/193) and the data ships with
**`deposit_type` and `total_of_special_requests`** — two top-10 Portugal
SHAP features that the earlier exploratory dataset deliberately lacked. The
PH model therefore has a feature menu that closely mirrors Portugal's.

The pre-flight **duplicate-cluster diagnostic** (Notebook 11) runs on the
new data and does *not* fire: post-engineering feature vectors are
essentially all unique, so chronological splitting carries no twin-leakage
risk. Reported test metrics measure honest small-sample generalization,
not memorization.

## Headline real-data findings

| Aspect | Real PH (193 rows) | Portugal (119k rows) |
|---|---|---|
| Train / Val / Test | 154 / 19 / 20 | 95k / 12k / 12k |
| Test ROC-AUC | ≈ 0.61 | 0.86 |
| Test PR-AUC | ≈ 0.54 | 0.76 |
| Cancellation rate | 15.0 % | 37.0 % |
| Top SHAP feature | `deposit_type` | `lead_time` |
| 3-way family comparison | CIs overlap totally — not statistically conclusive | LightGBM significantly best |
| ADR regressor test R² | -0.97 (overfits at n=154) | +0.78 |

Bootstrap 95 % CIs span roughly ±15-30 percentage points at n_test = 20,
so PH metrics are **directional**, not headlines. The PR-AUC gap to
Portugal is the expected cost of (a) ~500× fewer training rows, (b) a
narrower feature menu, and (c) different geography/property type.

## The 11 notebooks

| File | Purpose | Mirrors Portugal |
|---|---|---|
| `01_eda.ipynb` | EDA + duplicate-cluster pre-flight check + deposit/special-requests distributions | `notebooks/01_eda.ipynb` |
| `02_modeling.ipynb` | Champion summary, ROC/PR, calibration, confusion matrix, threshold sweep, feature importance | `notebooks/02_modeling.ipynb` |
| `03_deep_analysis.ipynb` | Cost-curve, learning curves, expanding-window CV, baseline comparison (Dummy/LR/DT/NB vs LightGBM) | `notebooks/03_deep_analysis.ipynb` |
| `04_adr_forecasting.ipynb` | **Tabular** ADR regression (no time series — too few rows/month); feature importance; residual analysis | `notebooks/04_adr_forecasting.ipynb` |
| `05_explainability.ipynb` | TreeSHAP feature importance + beeswarm + 3 individual explanations + Portugal-vs-PH SHAP comparison | `notebooks/05_explainability.ipynb` |
| `06_business_analytics.ipynb` | Cancellation rate / revenue exposure by deposit, room type, lead-time band; monthly revenue-at-risk timeline | `notebooks/06_business_analytics.ipynb` |
| `07_model_selection.ipynb` | 3-way calibrated comparison (LightGBM/XGBoost/GradientBoosting) with bootstrap CIs and paired delta forest | `notebooks/07_model_selection.ipynb` |
| `08_model_monitoring.ipynb` | Runnable monitoring template — baseline (test) + live `/predict` log, PSI drift, risk-tier mix | `notebooks/08_model_monitoring.ipynb` |
| `09_model_comparison.ipynb` | Per-row family disagreement, top-spread rows, mean-of-3 ensemble vs champion | `notebooks/09_model_comparison.ipynb` |
| `10_sensitivity_analysis.ipynb` | Cost sensitivity sweep, data hunger curve, threshold policy trade-offs | `notebooks/10_sensitivity_analysis.ipynb` |
| `11_transferability.ipynb` | Pre-flight diagnostic outcome + real-data metrics + defense framing | (no Portugal equivalent — methodology contribution) |

## Honest small-N caveats per notebook

Each notebook documents the small-sample caveat that applies to its
specific analysis. The summary:

- **04**: ADR regressor overfits (train R² 0.87, test R² -0.97). Use as a
  directional feature-importance explainer, not a forecast.
- **06**: Per-segment cancellation rates have very wide uncertainty bands
  at 1-5 bookings per segment cell.
- **07/09**: 3-way family CIs overlap; the LightGBM advantage is not
  statistically significant. Selection is by point-estimate parity +
  parallel-to-Portugal lineage + Occam's razor under indistinguishability.
- **08**: The live log is sparse (smoke-test rows only); charts demonstrate
  monitoring infrastructure rather than real drift findings.
- **10**: Data-hunger curve doesn't saturate — defensible argument for
  collecting more bookings before relying on the model in production.

## How to regenerate

```bash
# 1. Re-train and produce all artifacts (artifacts/ph/ + reports/ph/)
python scripts/train_ph.py

# 2. Headless-execute every notebook in this folder
for nb in notebooks/ph/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" --output "$nb"
done
```

The figures land in `reports/figures/thesis/ph/` (parallel to Portugal's
`reports/figures/thesis/`).

## Relationship to the Portugal main study

The PH sub-study **does not replace** Portugal as the primary thesis
study. Portugal remains the headline result (119k real bookings, 49
features, rolling-origin selection, bootstrap CIs, etc.). PH is the
real-data transferability probe that surfaces two methodology
contributions:

1. **Pre-flight duplicate-cluster diagnostic** that flags datasets
   organized around recurring booking archetypes (where chronological
   splitting would leak twins). Ran on the real PMS export and did
   NOT fire — methodology proceeds honestly.
2. **Feature-availability mapping**: of Portugal's top-10 SHAP features,
   the PMS export captures `deposit_type`, `total_of_special_requests`,
   `adr`, `lead_time` and date derivatives, but NOT `country`,
   `market_segment`, `customer_type`, `agent`, or
   `previous_cancellations`. A reduced-feature model is the most this
   PMS schema can produce — a useful bound for other small properties
   with similar PMS exports.

See `CLAUDE.md` § "PH Sub-Study — Philippine Resort Dataset" for the full
project context.
