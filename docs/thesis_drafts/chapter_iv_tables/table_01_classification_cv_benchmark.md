# Table 4.1 — 10-Fold Stratified Cross-Validation Classification Benchmark

**Source.** `reports/cv/portugal_stratified_10fold_summary.json`,
`reports/cv/philippine_stratified_10fold_summary.json`, plus per-fold CSVs
in the same directory.

**Method.** `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
Probabilities are the raw (uncalibrated) model outputs at threshold 0.5.
TP/FP/FN/TN are summed across all 10 folds — because every row appears
in exactly one test fold, the column sums equal the full dataset size
(Portugal 119,209; Philippines 193). Precision / Recall / F1 / AUC are
the mean across the 10 folds.

**Per-hotel split.** Not available — the 10-fold CV was executed on the
combined Portugal dataset. Per-hotel CV would require a rerun.

| Dataset / Hotel | Algorithm | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Portugal: Combined | LightGBM | 35,221 | 6,376 | 8,978 | 68,634 | 0.871 | 0.847 | 0.797 | 0.821 | 0.947 |
| Portugal: Combined | XGBoost | 37,976 | 11,474 | 6,223 | 63,536 | 0.852 | 0.768 | 0.859 | 0.811 | 0.937 |
| Portugal: Combined | Gradient Boosting | 34,292 | 6,393 | 9,907 | 68,617 | 0.863 | 0.843 | 0.776 | 0.808 | 0.940 |
| Portugal: Combined | Logistic Regression | 30,317 | 7,425 | 13,882 | 67,585 | 0.821 | 0.803 | 0.686 | 0.740 | 0.901 |
| Portugal: Combined | Decision Tree | 34,967 | 15,442 | 9,232 | 59,568 | 0.793 | 0.694 | 0.791 | 0.739 | 0.876 |
| Portugal: Combined | Gaussian NB | 35,780 | 27,932 | 8,419 | 47,078 | 0.695 | 0.562 | 0.810 | 0.663 | 0.814 |
| Portugal: Combined | Dummy (majority) | 0 | 0 | 44,199 | 75,010 | 0.629 | 0.000 | 0.000 | 0.000 | 0.500 |
| Philippines: Punta Villa | LightGBM | 14 | 6 | 15 | 158 | 0.891 | 0.708 | 0.483 | 0.536 | 0.872 |
| Philippines: Punta Villa | XGBoost | 16 | 10 | 13 | 154 | 0.881 | 0.643 | 0.550 | 0.575 | 0.870 |
| Philippines: Punta Villa | Gradient Boosting | 14 | 4 | 15 | 160 | 0.902 | 0.683 | 0.483 | 0.537 | 0.884 |
| Philippines: Punta Villa | Logistic Regression | 12 | 4 | 17 | 160 | 0.891 | 0.775 | 0.417 | 0.515 | 0.886 |
| Philippines: Punta Villa | Decision Tree | 25 | 48 | 4 | 116 | 0.731 | 0.380 | 0.867 | 0.511 | 0.821 |
| Philippines: Punta Villa | Gaussian NB | 18 | 30 | 11 | 134 | 0.788 | 0.464 | 0.633 | 0.489 | 0.848 |
| Philippines: Punta Villa | Dummy (majority) | 0 | 0 | 29 | 164 | 0.850 | 0.000 | 0.000 | 0.000 | 0.500 |

**Caveat (Philippines).** With n_test ≈ 19 per fold, the per-fold standard
deviation on PR-AUC is roughly ±0.15–0.20. The PH numbers are directional,
not headline. The four top algorithms (XGBoost, GB, LR, LightGBM) are
statistically indistinguishable at this sample size — their confidence
intervals overlap substantially.
