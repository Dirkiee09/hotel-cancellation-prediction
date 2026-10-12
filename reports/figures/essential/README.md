# Essential Figures — Publication Set

The curated, defense-ready subset: one claim per figure, ordered by the
thesis narrative. PNG for Word, PDF for LaTeX.

These images are committed as drawn. The script that drew them,
`scripts/make_essential_figures.py`, was removed in June 2026 (commit
`a215fe0`) and can be recovered from git history; see `docs/provenance.md`.
Two images carry text that has since gone stale, noted in the table.

| Figure | Claim it defends |
|---|---|
| `E01_problem_monthly_cancellation_trend` | Cancellation is large (37%) and time-varying -> chronological evaluation is mandatory |
| `E02_feature_distributions` | Key booking-time features: skewed lead time (median 69d), unimodal ADR, short stays — the modelling context (NOTE: not split by outcome; for separation evidence cite the lead-time medians table instead) |
| `E03_model_selection_rolling_origin` | LightGBM has the highest mean PR-AUC in the prespecified rolling-origin validation, ahead of XGBoost by 0.0009 |
| `E04_model_comparison_significance` | Paired bootstrap deltas vs the top test-set model, with the validation-protocol champion highlighted — the honest H2/split-decision figure |
| `E05_speed_vs_accuracy_pareto` | LightGBM and XGBoost are the only fast, accurate models; LightGBM trains about 2.7x faster, XGBoost has slightly higher test PR-AUC (0.763 vs 0.759, p = 0.010). *Stale in the image: training times come from an earlier benchmark run (LightGBM about 1.8 s; now 2.74 s)* |
| `E06_roc_pr_curves_test` | Headline discrimination on untouched test data (per-record stream): ROC-AUC 0.863 / PR-AUC 0.759. *The image labels PR-AUC 0.760, from an earlier run* |
| `E07_calibration_before_after` | Reliability diagram raw vs isotonic-calibrated with both ECE values — probabilities are trustworthy AND the calibration step earned its keep |
| `E08_confusion_matrix_operating_point` | What the deployed max-F1 operating point does to real bookings |
| `E09_shap_global_importance` | Global drivers, one-hot columns summed: deposit type (1.117), country (1.085), agent (0.926), parking spaces, special requests; lead time ranks seventh |
| `E10_shap_beeswarm` | Direction and spread of every driver's effect |
| `E11_shap_waterfall_examples` | Single-booking explanations — the bridge to the live app |
| `E12_policy_cost_ladder_test` | De-circularised H4 on the test set: the model beats even intervene-on-everyone by ~36% |
| `E13_threshold_policy_tradeoffs` | The three deployed threshold policies as one precision/recall/F1 trade-off |
| `E14_cost_assumption_sensitivity` | The cost policy is robust to the EUR 15 FP-cost assumption |
| `E15_feature_ablation_deltas` | Dependence on the deposit-type artifact is quantified, not hidden |
| `E16_temporal_stability_expanding_cv` | Performance is stable across time windows (no lucky split) |
| `E17_transferability_rank_slope` | Algorithm rankings are similar across markets (Spearman rho = 0.71, p = 0.071, 7 algorithms, so not statistically established); boosted models take the top two places in both, and LightGBM falls to fourth in the Philippine data. *The image title says the ranking "transfers", which overstates p = 0.071* |
| `E18_transferability_fold_spread` | Why PH pilot metrics are directional: fold spread at n~19 vs n~9.5k |
| `E19_champion_split_decision` | LightGBM minus XGBoost on the test set: XGBoost ahead on PR-AUC (-0.0036), LightGBM ahead on F1 (+0.0087), ROC-AUC a tie; neither dominates |
| `E20_conceptual_systems_positioning` | Where the framework sits among related systems (conceptual diagram) |
| `E21_deployment_framework` | Technical serving architecture (conceptual diagram) |
