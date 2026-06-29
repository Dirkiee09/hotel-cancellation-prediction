# Thesis Number Corrections — pre-audit drafts vs current canonical results

**Generated 2026-06-11 from the live result files.** The Chapter IV drafts in
this folder were written against the pre-audit model (different hyperparameter
budgets, validation-set hypothesis tests, week-number feature still present).
Every number below is read directly from the current canonical JSON/CSV at
generation time — use this sheet to correct the manuscript before the defense.

## Headline test metrics (max_f1 policy) — `reports/metrics.json`

| Quantity | Drafts say | Current canonical |
|---|---|---|
| ROC-AUC | 0.864 (0.8641) | **0.8634** (cite as 0.863) |
| PR-AUC | 0.760 (0.7601) | **0.7590** (cite as 0.759) |
| F1 | 0.7346 | **0.7356** |
| Recall | 0.841 | **0.8946** |
| Precision | — | **0.6246** |

## Decision thresholds — `artifacts/thresholds.json`

| Policy | Drafts say | Current |
|---|---|---|
| max_f1 | 0.40 | **0.41** |
| high_precision | 0.98 | **0.98** |
| cost_sensitive | 0.04 | **0.06** |

## H2 — REWRITE REQUIRED (table_06, chapter IV, conclusion)

Drafts claim: "LightGBM beats every challenger ... all p < 0.001 ... Supported
(significant)". This is no longer true and contradicts the cited file
(`reports/benchmarks/14_paired_significance_vs_champion.csv`, whose reference
model is now XGBoost at matched capacity).

Current canonical verdict (`reports/hypothesis_summary.json`):
- Status: **not_supported**
- Champion test PR-AUC 0.7590 vs logistic-regression
  baseline 0.7544; delta = +0.0045,
  p = 0.177 (paired bootstrap, n=2000) — **not significant at 0.05**.
- Honest framing: GBTs and a tuned linear baseline are close in pure
  discrimination here; LightGBM's case rests on calibration quality,
  cost-policy performance, and training speed, plus winning the prespecified
  rolling-origin validation protocol (0.8693 vs XGBoost 0.8684).

## H4 — REWRITE REQUIRED (table_05, table_08)

Drafts evaluate savings on the VALIDATION set (threshold 0.04, total cost
EUR 59,160, savings EUR 1,547,510 vs no model). The de-circularised canonical
result is on the TEST set at the val-selected threshold
(`reports/metrics.json -> cost_thresholding_test`):

| Policy (test set) | Total cost (EUR) |
|---|---|
| Do nothing (no model) | 2,322,794 |
| Threshold 0.50 | 669,637 |
| Intervene on all (trivial) | 111,240 |
| Cost-sensitive (thr=0.06) | **71,136** |

Savings: **598,502** vs threshold-0.5,
**40,104 (36%)
vs intervene-on-all** (the defensible headline), 2,251,658 vs no model
(context only — do not headline this).

## H3 (partially supported — drafts already say this; keep, update values)

Current SHAP top ranks: deposit_type #1, country #2, agent #3; lead_time #7;
previous_cancellations #10. H3's claimed ordering (lead time first) is
**not supported as stated** — report features-correct/order-wrong.

## New evidence available to cite (did not exist when drafts were written)

- **Duplicate sensitivity** (`reports/dedup_sensitivity.json`): removing
  31,994 exact duplicates (26.8%) changes test
  PR-AUC 0.759 -> 0.703
  and ROC-AUC 0.863 -> 0.848;
  no train/test boundary crossing (verified) — report as a per-record vs
  per-unique-profile range in limitations.
- **Champion split decision** (benchmarks table 14): XGBoost +0.0036 PR-AUC
  (p=0.01) vs LightGBM +0.0087 F1 (p=0.002); ROC tie (p=0.183) — metric-dependent
  winners at matched capacity; selection by prespecified validation protocol.
- Calibration: test ECE 0.062 -> 0.031 after isotonic
  (`reports/calibration_metrics.json`).
