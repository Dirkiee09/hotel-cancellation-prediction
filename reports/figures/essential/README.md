# Essential Figures — Publication Set

The curated, defense-ready subset: one claim per figure, ordered by the
thesis narrative. Regenerate with `make essential-figures`.
PNG for Word, PDF for LaTeX.

| Figure | Claim it defends |
|---|---|
| `E01_problem_monthly_cancellation_trend` | Cancellation is large (37%) and time-varying -> chronological evaluation is mandatory |
| `E02_feature_distributions` | Key booking-time features: skewed lead time (median 69d), unimodal ADR, short stays — the modelling context (NOTE: not split by outcome; for separation evidence cite the lead-time medians table instead) |
| `E03_model_selection_rolling_origin` | LightGBM wins the prespecified rolling-origin validation protocol |
| `E04_model_comparison_significance` | Paired bootstrap deltas vs the top test-set model, with the validation-protocol champion highlighted — the honest H2/split-decision figure |
| `E05_speed_vs_accuracy_pareto` | LightGBM trains ~2.8x faster at statistically equal quality (the rational tiebreaker) |
| `E06_roc_pr_curves_test` | Headline discrimination: ROC-AUC 0.863 / PR-AUC 0.759 on untouched test data |
| `E07_calibration_before_after` | Reliability diagram raw vs isotonic-calibrated with both ECE values — probabilities are trustworthy AND the calibration step earned its keep |
| `E08_confusion_matrix_operating_point` | What the deployed max-F1 operating point does to real bookings |
| `E09_shap_global_importance` | Global drivers: deposit type, lead time, prior cancellations (H1/H3 verdict) |
| `E10_shap_beeswarm` | Direction and spread of every driver's effect |
| `E11_shap_waterfall_examples` | Single-booking explanations — the bridge to the live app |
| `E12_policy_cost_ladder_test` | De-circularised H4 on the test set: the model beats even intervene-on-everyone by ~36% |
| `E13_threshold_policy_tradeoffs` | The three deployed threshold policies as one precision/recall/F1 trade-off |
| `E14_cost_assumption_sensitivity` | The cost policy is robust to the EUR 15 FP-cost assumption |
| `E15_feature_ablation_deltas` | Dependence on the deposit-type artifact is quantified, not hidden |
| `E16_temporal_stability_expanding_cv` | Performance is stable across time windows (no lucky split) |
| `E17_transferability_rank_slope` | Algorithm rankings transfer across markets (Spearman rho = 0.71); GBTs top both |
| `E18_transferability_fold_spread` | Why PH pilot metrics are directional: fold spread at n~19 vs n~9.5k |

Source material and producers: see `docs/figure_manifest.md`. Figures
E04/E07/E12/E13 are regenerated directly from canonical artifacts by
`scripts/make_essential_figures.py` (visualization-audit fixes: wrong
champion label, missing raw-calibration curve, axis offset notation,
legend overlapping data).
