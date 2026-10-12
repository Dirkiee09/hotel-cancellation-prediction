# Output provenance

Every result file in `artifacts/` and `reports/` that is committed to this
repository is listed here with the script that writes it. If a number in the
paper or the slides cannot be traced to a row in this table, treat it as
unverified.

## Committed results and their producers

| Committed output | Producer | Reproducible from this repository? |
|---|---|---|
| `artifacts/` (model, calibrator, thresholds, feature list, metadata, hashes) | `scripts/training/train.py` | Yes |
| `reports/metrics.json`, `champion_summary.json`, `segment_metrics.*`, `hypothesis_summary.json` | `scripts/training/train.py` | Yes |
| `reports/thesis/` except the files listed separately below | `scripts/training/train.py --thesis` | Yes |
| `reports/thesis/optuna_tuning.json` | `scripts/training/train.py --thesis` (without `--skip-tuning`) | Yes; Optuna stops at a 10-minute budget, so trial counts can vary with machine speed |
| `reports/thesis/calibration_diagnostics.json`, `confidence_intervals_extended.json`, `calibration_by_segment.csv`, `cost_scenarios.csv` | `scripts/training/revision_diagnostics.py` | Yes; reads saved test predictions, no retraining |
| `reports/thesis/deposit_interactions.csv`, `reports/deposit_ablation.json` | `scripts/training/feature_ablation.py` | Yes |
| `reports/dedup/`, `reports/dedup_sensitivity.json` | `scripts/training/dedup_sensitivity.py` | Yes |
| `reports/benchmarks/*.csv`, `*.md` | `scripts/training/benchmark.py` | Yes |
| `reports/cv/portugal_stratified_10fold_*` | `scripts/training/stratified_cv.py` | Yes |
| `reports/cv/philippine_stratified_10fold_*` | `scripts/training/stratified_cv.py` | No; needs the private PH export |
| `reports/ph/*.json`, `reports/ph/ph_threshold_sweep.csv` | `scripts/training/train_ph.py` | No; needs the private PH export |
| `reports/figures/essential/` | `scripts/make_essential_figures.py`, removed in June 2026 (commit `a215fe0`) | Not as committed; recover the script from git history |
| `artifacts/adr_*` | None; see the open gap below | No |

Run the producers in this order, because later ones read earlier outputs:
`train.py --thesis`, then `dedup_sensitivity.py`, then `benchmark.py`,
`revision_diagnostics.py` and `feature_ablation.py`.

## How exact the reproduction is

On 12 October 2026 the committed code was extracted to an empty folder and every
Portugal producer above was run from scratch on the machine that made the
committed results (Windows 11, Python 3.12.10). The retrained model file had the
same SHA-256 as the committed one, and all 43 report files matched value for
value.

That exactness holds for the same environment. `artifacts/model_metadata.json`
records the Python, operating-system and library versions under
`lineage.environment`, and CI enforces it: a Windows / Python 3.12.10 job retrains
and requires every headline figure to match (`check.py reproduce --mode exact`).

On a different platform the figures drift. The first Linux / Python 3.11 retrain
in CI (12 October 2026, per-record stream) gave:

| | Committed (Windows) | Linux CI |
|---|---|---|
| Selected model | LightGBM | XGBoost |
| ROC-AUC | 0.8634 | 0.8645 |
| PR-AUC | 0.7590 | 0.7623 |
| Recall at the max-F1 threshold | 0.895 | 0.860 |
| Cost-sensitive threshold | 0.06 | 0.05 |
| Cost cut against contacting everyone | 36.1% | 34.3% |

The model family changes because LightGBM and XGBoost are 0.0009 apart in
rolling-origin PR-AUC, well inside the 0.032 across-fold standard deviation. Every
CI run posts this comparison as an annotation (`check.py reproduce --mode drift`).

## The Philippine dataset is not in this repository

`data/Punta_Villa_Resort_PH_Dataset.csv` is a real property-management-system export
and is deliberately untracked. Anything in the PH sub-study that depends on it cannot
be regenerated from a clone alone. Reproducing it requires:

- the original CSV placed at that path
- `src/data/load_ph.py`, the `PH_*` constants in `src/config.py`, and
  `clean_raw_ph` / `validate_raw_ph` in `src/utils/validate_data.py` (all present here)

The Philippine results are committed as **aggregated JSON only** so the PH notebooks
render for anyone who clones the repository. Row-level prediction files and the
PH model in `artifacts/ph/` are not committed, because they contain or were fitted
on real reservation records.

## Open gap: the ADR regressor has no producer script

These artifacts have **no versioned script that creates them**:

```
artifacts/adr_regressor.pkl
artifacts/adr_regressor_preprocessor.pkl
artifacts/adr_regressor_y_scaler.pkl
artifacts/adr_regressor_nn.keras
artifacts/adr_timeseries_data.pkl
artifacts/adr_timeseries_metadata.pkl
```

Searching the repository finds only consumers, never a producer. They were trained in
an unversioned session, and TensorFlow/Keras, needed to load the `.keras` model, is
not in `requirements.txt`. The sklearn regressor loads and serves fine; it is the one
cited in results. None of the paper's figures depend on it.

This matters because `make train` does not rebuild these files. Deleting them cannot
be undone by retraining.

Three ways to close it, in descending order of rigour:

1. **Write the producer.** Add `scripts/training/train_adr.py` that fits and saves the
   regressor from `data/hotel_bookings.csv`. The ADR section inside
   `scripts/training/train_ph.py` is a working template. This also fixes the
   booking-time feature problem described under "Live ADR forecast" in the README.
2. **State the limitation.** Present the ADR results as exploratory and say plainly
   that those artifacts predate version control.
3. **Drop the neural-network comparison.** Keep only the sklearn regressor, which
   removes the undeclared TensorFlow dependency entirely.

Option 1 is the only one that makes the ADR results independently reproducible.
