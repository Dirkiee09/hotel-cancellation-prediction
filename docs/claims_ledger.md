# Claims ledger

Every number in the ICBIM 2026 paper (*A Strategic Business Intelligence Approach to
Predicting Hotel Booking Cancellations*, BM4117) and in the conference deck, with the
committed file it comes from and how to check it. If a figure is not in this table,
treat it as unverified.

Checked on 12–13 October 2026 against a retrain of the committed code from scratch:
the model file was byte-identical and every value below matched. The paper is final,
so where its wording is loose the right-hand column says so and
[`demo/HANDOVER.md`](../demo/HANDOVER.md) §5 holds the prepared answer.

**Streams.** *Per-record* = every row, the model the demo serves
(`reports/metrics.json`, 119,209 rows, test 11,922). *Primary* = exact duplicates
removed, the paper's headline (`reports/dedup/metrics.json`, 87,229 rows, test 8,724).

**Producers.** Regenerate everything with, in order:

```bash
python scripts/training/train.py --thesis --skip-tuning   # artifacts/, reports/metrics.json, reports/thesis/
python scripts/training/dedup_sensitivity.py              # reports/dedup/, reports/dedup_sensitivity.json
python scripts/training/benchmark.py                      # reports/benchmarks/  (about 25 min)
python scripts/training/revision_diagnostics.py           # calibration diagnostics, extended CIs, 150 cost scenarios
python scripts/training/feature_ablation.py               # reports/deposit_ablation.json
python scripts/training/field_ablation.py                 # reports/field_ablation.json
python scripts/training/cost_conventions.py               # reports/thesis/cost_conventions.csv
python scripts/training/stratified_cv.py                  # reports/cv/  (shuffled 10-fold)
```

`preds` below means `pandas.read_csv("reports/dedup/test_predictions_for_powerbi.csv")`.

## Data and design

| Claim | Source | How to check | Note |
|---|---|---|---|
| 119,390 bookings, 32 columns, 1 Jul 2015 – 31 Aug 2017 | `data/hotel_bookings.csv` | Row count; arrival year/month/day columns | |
| 119,209 after cleaning (181 dropped) | `reports/metrics.json` | `data_cleaning.rows_dropped_total` = 181 (1 negative rate, 180 zero guests) | |
| 87,229 after removing exact duplicates | `reports/dedup/metrics.json` | Training + validation + test rows | The pipeline de-duplicates the raw file (87,396) and then cleans; cleaning first gives 87,222 |
| About 27% exact duplicates | `reports/dedup_sensitivity.json` | `duplicate_share` = 0.268 | |
| Duplicates never cross the train–test boundary | Recomputed from the split | No identical feature row appears in both train and test, either stream | |
| Five post-booking fields excluded | `src/config.py` | `LEAKAGE_COLS` | Parking requests may also be recorded at check-in; see `field_ablation.json` below |
| 33 variables → 93 columns | `artifacts/feature_columns.json`, `reports/thesis/shap_analysis.json` | `n_features` = 93 | |
| Week number disagrees with ISO for about half the dates | Recomputed | 53.7% | |
| Primary split: train to 2 May, validation to 2 Jul, test to 31 Aug 2017 | `reports/dedup/test_predictions_for_powerbi.csv` | Arrival dates of the test rows | |
| Cancellation rate 0.371 per-record, 0.275 primary | Recomputed | Mean of `is_canceled` after cleaning | |
| Primary test set: 8,724 bookings, 3,121 cancelled | `preds` | Row count; sum of `is_canceled` | 0.358 is what random guessing scores on PR area |
| Mean stay 3.6 nights | Recomputed | 3.63 on the primary stream | |
| Rolling-origin cutoffs 0.60 / 0.70 / 0.80 | `reports/model_selection_summary.json` | `window.cutoff_fracs`; fractions of train + validation | |
| XGBoost and LightGBM at 300 trees, depth 7, rate 0.05 | `reports/benchmarks/02_model_specs.csv` | `params_json` | LightGBM's row subsampling is inactive and its trees stop at 31 leaves |
| Python 3.11 | `artifacts/model_metadata.json` | `lineage.environment.python` = 3.12.10 | The paper's version is the CI version; the reported run used 3.12.10 |

## Results

| Claim | Source | How to check | Note |
|---|---|---|---|
| Shuffled 10-fold LightGBM ROC area 0.947 | `reports/cv/portugal_stratified_10fold_summary.json` | `ranking[0].roc_auc_mean` | Optimistic by design |
| Table II (six models, ROC, PR, 95% CI, F1) | `reports/benchmarks/03_…`, `05_…`, `13_bootstrap_confidence_intervals.csv` | Per-model rows | Per-record stream |
| Tree → logistic +0.226; logistic → best boosted +0.023 | `reports/benchmarks/03_holdout_probability_metrics.csv` | Differences of `pr_auc` | |
| Rolling-origin PR area 0.869 / 0.868 / 0.867, SD ≈ 0.032 | `reports/model_selection_summary.json`, `reports/benchmarks/11_rolling_origin_summary.csv` | `rolling_pr_auc_mean`, `pr_auc_std` | Per-record. Primary stream: 0.714 / 0.711 / 0.708. On Linux the same code selects XGBoost; see `docs/provenance.md` |
| Table III, primary column | `reports/dedup/metrics.json` | `max_f1.*`, `calibration.test.ece_*`, `cost_thresholding_test` | |
| Table III, per-record column | `reports/metrics.json` | Same fields | |
| F1 thresholds 0.34 / 0.41 | `reports/dedup/thresholds.json`, `artifacts/thresholds.json` | `max_f1.threshold` | |
| Paired bootstrap: vs tree 0.249, vs logistic 0.023, XGBoost−LightGBM 0.0036 (p 0.010), F1 0.0087 (p 0.002), ROC p 0.183 | `reports/benchmarks/14_paired_significance_vs_champion.csv` | Rows with `champion_model` = xgboost | |
| LightGBM vs logistic regression, like for like | `reports/metrics.json`, `reports/dedup/metrics.json` | `h2_baseline_comparison`: +0.019 per-record, +0.026 primary, p < 0.001 | Not in the paper; the paper reports XGBoost vs logistic |
| Training time 2.7 / 7.5 / 60.6 s | `reports/benchmarks/15_training_inference_cost.csv` | `fit_seconds` | Single thread, one machine |
| Calibration error 0.035 → 0.018; Brier 0.152; log loss 0.448; slope 0.98; intercept −0.02 | `reports/dedup/metrics.json`, `reports/thesis/calibration_diagnostics.json` | `deduplicated.*` | |
| Observed cancel rate 0.140 / 0.573 / 0.789 by tier | `preds` | Mean of `is_canceled` for probability < 0.40, 0.40–0.70, ≥ 0.70 | |
| Deposit type, then country, then agent | `reports/thesis/shap_feature_importance.csv` | Sum `mean_abs_shap` over each feature's one-hot columns: 1.116, 1.085, 0.926 | Per-record model. Narrowly ahead, not "by a wide margin" |
| Withholding deposit type: 0.703 → 0.702 | `reports/deposit_ablation.json` | `ablated_dedup.pr_auc` | The ablated run selected XGBoost |
| Without parking / without country | `reports/field_ablation.json` | `variants.*.max_f1.pr_auc` | Not in the paper. Without parking: 0.674, cost cut 25.4%. Without parking or special requests: 0.620, 23.7%. Without country: 0.648, 13.6% |
| Table IV (no action, 0.50, contact everyone, cost-sensitive) | `reports/dedup/metrics.json` | `cost_thresholding_test`; false positives and negatives at 0.50 from `preds` | |
| At τ = 0.03: recall 99.5%, precision 46.8%, 6,639 contacted, 17 missed | `preds` | Rows with `cancel_probability` ≥ 0.03 | |
| Cost cut 31.2% (primary), 36.1% (per-record) | `reports/dedup/metrics.json`, `reports/metrics.json` | `(intervene_all_cost − total_cost) / intervene_all_cost` | |
| The 31.2% under stricter pricing of a contact | `reports/thesis/cost_conventions.csv` | `saving_pct`: 20.2 if every contact costs €15; 2.7 if a contact saves half of would-be cancellations | Euro saving stays €26,000–30,000 |
| Ordering holds across 150 scenarios | `reports/thesis/cost_scenarios.csv` | 150 rows, all with positive `Savings Vs Intervene All` | None varies the two conventions above |
| 12.8% high tier, 31.1% medium | `preds` | Share with probability ≥ 0.70 and 0.40–0.70 | |
| The tiers read as a contact rule: contact only from 0.40 | `preds` | Rows with `cancel_probability` ≥ 0.40, priced as in `src/utils/business.py`: 3,832 contacted, 683 missed, €317,718 | Not in the paper. The 3% rule decides whether to contact; the tiers decide who acts. See `demo/HANDOVER.md` §5 |
| About 130 high-risk bookings a week, 19 a day | `preds` | 1,115 high-tier rows over the test window | The window is 61 days, not 60: 128 a week, 18 a day |
| Corporate and Groups left unscored | `reports/dedup/segment_metrics.json` | `skip_reason` = insufficient_rows | |

## Philippine pilot

| Claim | Source | How to check | Note |
|---|---|---|---|
| 193 bookings, 2022–2025, 15.0% cancelled | `reports/ph/ph_transferability.json` | `dataset_diagnostics.n_rows_total` | Raw data is private and not in this repository |
| 20 test bookings, 9 cancellations, intervals about ±15 points | `reports/ph/champion_summary.json` | `n_test`, `caveats` | |
| Deposit type leads again | `reports/ph/shap_feature_importance.csv` | First row | |
| Ran without modification | `src/utils/validate_data.py` | `clean_raw_ph` | Own cleaning function and an 18-feature set; the modelling stages were unchanged |
| Not deployment-ready | `reports/ph/ph_transferability.json` | `confusion_max_f1`: 0 of 9 test cancellations flagged | Logistic regression, decision tree and naive Bayes all scored higher |

## Deck only

| Claim | Source | Note |
|---|---|---|
| €1,851,306 cost of taking no action | `reports/dedup/metrics.json` `cost_thresholding_test.no_model_cost` | Assumes one night of each cancelled stay is resold |
| 37% of bookings cancel | `data/hotel_bookings.csv` | Raw rows; 27.5% on the primary stream |
| 0.848 / 0.018 / −31.2% on the closing slide | `reports/dedup/metrics.json` | −31.2% is total modelled cost, not intervention cost |
