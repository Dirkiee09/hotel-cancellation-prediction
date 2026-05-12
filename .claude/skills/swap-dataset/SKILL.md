---
name: swap-dataset
description: Replace the input CSV with a new hotel-bookings dataset (same 32 columns) and walk the user through the currency-specific config.py edits, then retrain and verify. User invokes via /swap-dataset <path-to-new-csv>.
disable-model-invocation: true
---

# Swap Dataset

Plug-and-play dataset replacement for the hotel-cancellation pipeline.
This skill is **user-only** because it overwrites `data/hotel_bookings.csv`
and `src/config.py`.

## Argument

Path to a new CSV with the **same 32 column names** as the canonical
`data/hotel_bookings.csv`. The 32 expected columns are documented in
`src/utils/validate_data.py`.

## Procedure

### 1. Schema check
Read the new CSV's header row. Compare against the expected 32 columns.
If any are missing or mis-named, **stop and surface the diff to the user.**

```bash
python -c "import pandas as pd; print(list(pd.read_csv('<new_csv>', nrows=0).columns))"
```

### 2. Backup the current dataset
```bash
cp data/hotel_bookings.csv data/hotel_bookings.csv.bak
```

### 3. Move the new CSV in
```bash
cp <new_csv> data/hotel_bookings.csv
```

### 4. Update `src/config.py`
Ask the user three questions:

1. **What currency does the new dataset use?** (e.g., PHP, USD, EUR)
2. **What's the maximum valid ADR (average daily rate) in that currency?**
   Defaults to 50,000 (currency-agnostic ceiling). Update `ADR_MAX_VALID`.
3. **What's the cost of a false-positive intervention in that currency?**
   The EUR default is 15.0. For PHP this might be 900 (rough EUR→PHP).
   Update `FP_INTERVENTION_COST`.

Use the Edit tool to update these two constants in `src/config.py`. Preserve
all surrounding comments.

### 5. Retrain on the new data
```bash
python scripts/train.py
```

### 6. Demo-check the new model
```bash
python scripts/demo_check.py
```

If the high-risk scenario doesn't hit ≥ 70% or low-risk doesn't hit < 10%,
note that the new dataset's signal may differ — but this is **informational**,
not a failure.

### 7. Surface new headline metrics
Read `reports/metrics.json` and tell the user:

- ROC-AUC, PR-AUC, F1, Recall on the test set
- Recommend updating `METRIC_GATES` in `src/config.py` to
  `(observed - 0.02)` for regression detection on subsequent retrains.

### 8. (Optional) Update notebook narrative
Many notebook markdown cells reference "Portugal", "EUR", or "PRT". If the
user wants the notebooks to read correctly for the new market, surface
which cells need attention:

```bash
grep -rn -E "(Portugal|EUR|PRT)" notebooks/ | head -20
```

Do **not** auto-edit notebook prose — the user owns thesis writing.

## Rules

- Never delete the `.bak` backup automatically.
- Never modify the column names — feature engineering depends on them.
- If retraining fails at step 5, restore the backup:
  `cp data/hotel_bookings.csv.bak data/hotel_bookings.csv`
- Surface every config.py edit as a diff so the user can review.
