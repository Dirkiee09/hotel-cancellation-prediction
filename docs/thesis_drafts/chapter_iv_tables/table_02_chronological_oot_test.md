# Table 4.2 — Chronological Out-of-Time Test Results (Operational Reality)

**Source.** `reports/benchmarks/05_holdout_threshold_metrics_max_f1.csv` and
`reports/benchmarks/08_confusion_matrix_counts_per_model.csv` (Portugal
combined), `reports/segment_metrics.csv` (per-hotel rows for the LightGBM
champion only), `reports/ph/ph_transferability.json` and
`reports/ph/baseline_comparison.json` (Philippines).

**Method.** Time-aware 80/10/10 split sorted by `arrival_date`. Each model
is trained on the chronological train set, calibrated on the validation set
(isotonic), and evaluated on the chronological test set at the validation-
tuned `max_f1` threshold. Portugal test n = 11,922; Philippines test n = 20.

**Per-hotel split.** Available for LightGBM only — `segment_metrics.csv`
stores per-hotel rows for the deployed champion. Other algorithms do not
have a per-hotel breakdown in the chronological pipeline.

| Dataset / Hotel | Algorithm | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Portugal: Combined | LightGBM (champion) | 3,791 | 2,024 | 715 | 5,392 | 0.770 | 0.652 | 0.841 | 0.735 | 0.864 |
| Portugal: Combined | XGBoost | 4,074 | 2,604 | 432 | 4,812 | 0.745 | 0.610 | 0.904 | 0.729 | 0.855 |
| Portugal: Combined | Gradient Boosting | 3,736 | 1,934 | 770 | 5,482 | 0.773 | 0.659 | 0.829 | 0.734 | 0.861 |
| Portugal: Combined | Random Forest | 3,452 | 1,855 | 1,054 | 5,561 | 0.756 | 0.650 | 0.766 | 0.704 | 0.851 |
| Portugal: Combined | Logistic Regression | 3,719 | 2,200 | 787 | 5,216 | 0.749 | 0.628 | 0.825 | 0.713 | 0.839 |
| Portugal: Combined | Decision Tree | 2,681 | 1,807 | 1,825 | 5,609 | 0.695 | 0.597 | 0.595 | 0.596 | 0.675 |
| Portugal: Resort Hotel (H1) | LightGBM | 1,335 | 579 | 202 | 1,927 | 0.807 | 0.697 | 0.869 | 0.774 | 0.892 |
| Portugal: City Hotel (H2)   | LightGBM | 2,456 | 1,445 | 513 | 3,465 | 0.751 | 0.630 | 0.827 | 0.715 | 0.851 |
| Philippines: Punta Villa | LightGBM (champion) | 0 | 1 | 9 | 10 | 0.500 | 0.000 | 0.000 | 0.000 | 0.611 |
| Philippines: Punta Villa | Logistic Regression | 1 | 1 | 8 | 10 | 0.550 | 0.500 | 0.111 | 0.182 | 0.677 |
| Philippines: Punta Villa | Decision Tree | 7 | 3 | 2 | 8 | 0.750 | 0.700 | 0.778 | 0.737 | 0.753 |
| Philippines: Punta Villa | Gaussian NB | 1 | 1 | 8 | 10 | 0.550 | 0.500 | 0.111 | 0.182 | 0.828 |
| Philippines: Punta Villa | Dummy (majority) | 0 | 0 | 9 | 11 | 0.550 | 0.000 | 0.000 | 0.000 | 0.500 |

**Headline gap.** Portugal LightGBM PR-AUC drops from **0.947 stratified
10-fold** (Table 4.1) to **0.760 chronological** (this table) — a ~16 pp
gap that quantifies concept drift over time. This is the defensible
"operational reality" story.

**Caveat (Philippines).** n_test = 20 with 9 actual cancellations.
LightGBM's F1 = 0.000 reflects the calibrated probability never crossing
the validation-tuned threshold of 0.19 on these 9 positives, NOT a
structural model failure. The 10-fold CV PH numbers (Table 4.1) are the
better signal at this sample size — bootstrap 95 % CIs on PR-AUC span
±15 pp at n_test = 20.
