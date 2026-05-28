# Table 4.6 — Hypothesis × Evidence × Verdict Matrix

**Source.** `reports/hypothesis_summary.json`,
`reports/benchmarks/14_paired_significance_vs_champion.csv`,
`reports/thesis/shap_feature_importance.csv`,
`reports/ph/shap_feature_importance.csv`,
`reports/cost_threshold_summary.json`.

**Method.** Each pre-registered hypothesis from Chapter I is mapped to its
evidence source, the statistical test applied, the key result, and the
verdict. This is the canonical Chapter IV closure table — the panel reads
this to confirm every hypothesis is accounted for.

| H | Statement | Evidence Source | Test Used | Key Statistic | Verdict |
|---|---|---|---|---|---|
| **H1** | Lead time, deposit type, and previous cancellations are significant predictors of cancellation. | `reports/thesis/shap_feature_importance.csv` | TreeSHAP top-10 ranking on the Portugal test set | All three features appear in the top-10 by mean(\|SHAP\|): `deposit_type` (rank 1, 1.150), `lead_time` (rank 7, 0.393), `previous_cancellations` (rank 10, 0.234). | **Supported** |
| **H2** | A gradient-boosted tree model outperforms baseline models (Decision Tree, Logistic Regression, Random Forest). | `reports/benchmarks/14_paired_significance_vs_champion.csv` | Paired bootstrap on PR-AUC (n = 2,000 resamples) | LightGBM beats every challenger: Δ PR-AUC = +0.2522 vs DT, +0.0213 vs RF, +0.0208 vs LR, +0.0109 vs XGB, +0.0065 vs GB. **All p < 0.001 except GB (p = 0.001).** | **Supported (significant)** |
| **H3** | Lead time has the highest SHAP importance, followed by deposit type, then previous cancellations. | `reports/thesis/shap_feature_importance.csv` | Rank-order check on TreeSHAP aggregated to raw features | Actual order: `deposit_type` (1) > `country` (2) > `agent` (3) > `required_car_parking_spaces` (4) > `total_of_special_requests` (5) > `market_segment` (6) > **`lead_time` (7)** > … > **`previous_cancellations` (10)**. The three pre-registered features all appear in the top-10 but the rank order differs. | **Partially supported** |
| **H4** | Cost-sensitive thresholding reduces expected revenue loss compared to a naïve threshold of 0.50. | `reports/cost_threshold_summary.json` | Per-policy cost computation on the Portugal validation set | Cost-sensitive total cost = €59,160.04 vs no-model cost = €1,606,669.92 vs 0.50-threshold baseline = €387,350.44. Savings = **€1,547,509.88 (95.4 %)** vs no model and **€328,190.40** vs threshold 0.50. | **Supported** |
| **H5** | The top SHAP feature on the Portugal model also ranks in the top 3 SHAP features on the Philippine model, providing cross-dataset evidence that the methodology detects a consistent cancellation driver across geographies. | Portugal: `reports/thesis/shap_feature_importance.csv`. Philippines: `reports/ph/shap_feature_importance.csv`. | Top-1 cross-dataset feature match | `deposit_type` ranks **#1 in both datasets** (Portugal mean\|SHAP\| = 1.150; PH mean\|SHAP\| = 2.323). The same operational signal — deposit policy — dominates predictions in two geographies, two property types, and two dataset sizes (119k vs 193 rows). | **Supported** |

**Summary.** Four hypotheses fully supported (H1, H2, H4, H5); one
partially supported (H3 — features correct, rank order wrong). H2 is
supported with statistical significance at p ≤ 0.001 against every
competing model family.
