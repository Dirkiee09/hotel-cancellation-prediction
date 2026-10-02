# Output provenance

Most of `artifacts/` and `reports/` is generated and git-ignored: `make train`
rebuilds it from `data/hotel_bookings.csv`. A small number of result files are
**committed on purpose**, because the code that produced them needs data this
repository does not contain. This file records where those came from, so the
results stay reproducible and defensible.

## Committed results and their producers

| Committed output | Producer | Reproducible here? |
|---|---|---|
| `reports/cv/portugal_stratified_10fold_*` | `scripts/training/stratified_cv.py` | Yes — runs from `data/hotel_bookings.csv` |
| `reports/cv/philippine_stratified_10fold_*` | `scripts/training/stratified_cv.py` | No — needs the private PH export |
| `reports/ph/*.json`, `reports/ph/ph_threshold_sweep.csv` | `scripts/training/train_ph.py` | No — needs the private PH export |
| `reports/dedup_sensitivity.json` | `scripts/training/dedup_sensitivity.py` | Yes — runs from `data/hotel_bookings.csv` |

The Philippine results are committed as **aggregated JSON only** so the PH notebooks
render for anyone who clones the repository. Row-level prediction files are not
committed, because they contain real reservation records.

### The Philippine dataset is not in this repository

`data/Punta_Villa_Resort_PH_Dataset.csv` is a real property-management-system export
and is deliberately untracked. Anything in the PH sub-study that depends on it cannot
be regenerated from a clone alone. Reproducing it requires:

- the original CSV placed at that path
- `src/data/load_ph.py`, the `PH_*` constants in `src/config.py`, and
  `clean_raw_ph` / `validate_raw_ph` in `src/utils/validate_data.py` (all present here)

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
an unversioned session, and TensorFlow/Keras — needed to load the `.keras` model — is
not in `requirements.txt`. The sklearn regressor loads and serves fine; it is the one
cited in results.

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
