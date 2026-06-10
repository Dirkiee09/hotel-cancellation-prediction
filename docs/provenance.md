# Output Provenance — Where Local Artifacts Come From

This branch (`Main_Project`) is a clean rewrite of the training/serving pipeline.
Several output sets sitting in the local `artifacts/` and `reports/` directories
were produced by code that lives on the **`master` branch** (= `origin/Main`),
not on this branch. This file maps every such output to its producer so the
results remain reproducible and defensible.

> Recover any producer with:
> `git checkout master -- <path>` (then review its imports — some need the
> PH support layer listed below).

| Local output | Producer (on `master`) | Commit | Notes |
|---|---|---|---|
| `artifacts/ph/*` (PH model, calibrator, thresholds, ADR regressor) | `scripts/train_ph.py` | `8c03059` | Needs the PH support layer: `src/data/load_ph.py`, `PH_*` constants in `src/config.py`, `clean_raw_ph`/`validate_raw_ph` in `src/utils/validate_data.py`. **The PH PMS dataset is private and is NOT in the repository** — reproduction requires the original CSV. |
| `reports/ph/*` (transferability, SHAP, baselines, learning curves) | `scripts/train_ph.py` | `8c03059` | Same requirements as above. **Aggregated result JSONs are committed on this branch** (2026-06-10) so Notebook 11 renders everywhere; row-level prediction CSVs stay local (private PMS rows). |
| `reports/cv/*` (Portugal + Philippine stratified 10-fold, 7 algorithms) | `scripts/stratified_cv.py` | `351a75c` | Portugal half runs from this branch's data; PH half needs the private dataset. **Summary + fold-level metric files are committed on this branch** and rendered by `notebooks/11_transferability_ph.ipynb`. |
| `data/predictions/` (deleted in cleanup, 2026-06-10) | `scripts/seed_demo_predictions.py`, `scripts/compute_live_drift.py`, `scripts/export_predictions.py` | `0cf4b4a` | Power BI / live-demo seeding. Regenerable at will; now gitignored. |
| Chapter IV tables & paper figures | `docs/thesis_drafts/` on `master` | various | Includes the test-set threshold-policy table referenced by older `metrics.json` notes. |

## Unresolved: Portugal ADR artifacts

`artifacts/adr_regressor.pkl`, `adr_regressor_preprocessor.pkl`,
`adr_regressor_y_scaler.pkl`, `adr_regressor_nn.keras`, and
`adr_timeseries_*.pkl` have **no versioned producer on any branch** (verified
2026-06-10: `git grep` across all refs finds only consumers). They were trained
in an unversioned session. TensorFlow/Keras is not in `requirements.txt`.

Consequences and options:

1. **Before the defense**, either write `scripts/train_adr.py` reproducing them
   (the PH equivalent inside `train_ph.py` on `master` is a good template), or
2. present the ADR chapter results as exploratory with this limitation stated,
   or
3. drop the Keras NN comparison (the sklearn regressor is the one cited in
   `reports/regression_results.csv`) so the missing TensorFlow dependency
   disappears.

## Branch topology (for whoever reads this later)

- `Main_Project` (this branch): clean pipeline rewrite — training, serving,
  evaluation, 10 notebooks, CI. Unrelated git history to `master`.
- `master` / `origin/Main`: full thesis lineage — everything above **plus** the
  PH application stack (`src/app/ph_*`, `src/serving/inference_ph.py`), PH
  notebook suite (`notebooks/ph/`), defense runbook, and `docs/thesis_drafts/`.
