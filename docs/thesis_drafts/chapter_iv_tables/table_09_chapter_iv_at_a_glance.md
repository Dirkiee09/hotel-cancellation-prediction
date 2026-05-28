# Table 4.9 — Chapter IV at a Glance (60-Second Defense Summary)

*Single-page consolidation of every claim made in Chapter IV. Point at this when the panel says "give me the gist." Every cell is recomputable from the artifact named in the right-most column.*

| # | Claim / Question Panel Asks | Result | Source (Drill-Down Table) |
|---|---|---|---|
| **— Hypotheses (5 of 5 closed) —** | | | |
| H1 | Are lead time, deposit type, and previous cancellations significant predictors? | **Supported** — all three in top-10 SHAP (`deposit_type` #1, `lead_time` #7, `previous_cancellations` #10). | Table 4.6 → Table 4.4 |
| H2 | Do gradient-boosted trees beat baselines with statistical significance? | **Supported** — LightGBM beats every challenger at p < 0.001 (p = 0.001 for GB) on paired bootstrap PR-AUC. | Table 4.6 → benchmark CSV 14 |
| H3 | Is the SHAP rank order `lead_time` > `deposit_type` > `previous_cancellations`? | **Partially supported** — all three appear in top-10, but actual order is `deposit_type` > `country` > `agent` > … > `lead_time` (7) > `previous_cancellations` (10). | Table 4.6 → Table 4.4 |
| H4 | Does cost-sensitive thresholding reduce expected revenue loss? | **Supported** — €76,512 total cost vs €3.01 M no-model baseline → **97.5 % recovery on the test set**. | Table 4.6 → Table 4.8 |
| H5 | Does the top SHAP feature transfer across geographies (Portugal ↔ Philippines)? | **Supported** — `deposit_type` ranks **#1 in both datasets** (Portugal mean\|SHAP\| 1.150; PH mean\|SHAP\| 2.323). | Table 4.6 → Table 4.4 |
| **— Headline Performance —** | | | |
| P1 | Champion classifier on the chronological test set? | **LightGBM, calibrated.** ROC-AUC 0.864, PR-AUC 0.760, F1 0.735, Recall 0.841, Precision 0.652 (at `max_f1` = 0.40). | Table 4.2 |
| P2 | How tight are the confidence intervals around those numbers? | Bootstrap 95 % CI: PR-AUC [0.748, 0.772] (width 0.024); ROC-AUC [0.858, 0.871] (width 0.013); F1 [0.725, 0.744]. n_bootstraps = 2,000. | benchmark CSV 13 |
| P3 | ADR regression champion and accuracy? | **Gradient Boosting**, test RMSE = €44.31 / MAE = €32.24 / R² = 0.234 / MAPE = 23.5 %. Directional pricing signal, not exact prediction. | Table 4.3 |
| P4 | Is the calibration step worth the engineering effort? | **Yes.** Isotonic regression halves test-set ECE: 0.058 → 0.029. Brier improves 0.150 → 0.146. | calibration_metrics.json |
| **— Business Impact —** | | | |
| B1 | What fraction of revenue exposure does the High risk tier carry? | **52.2 %** of all realised cancellation losses (€1,571,978 / €3,014,266) sit in 26.1 % of bookings. | Table 4.7 |
| B2 | What does the recommended (cost-sensitive) policy save on the test set? | **€2,937,754** vs no model (97.5 % of theoretical maximum) — and **€329,231** vs the balanced `max_f1` threshold. | Table 4.8 |
| B3 | Cancellation rates per risk tier — is the calibration empirically honest? | Low 11.7 %, Medium 53.0 %, High 75.8 %. The model's "75 %" really means ~76 % observed cancellation rate. | Table 4.7 |
| B4 | What concrete action does each tier trigger? | Low → no action; Medium → 72 h reminder email; High → confirmation call + partial deposit. | Table 4.7 |
| **— Methodology Rigor —** | | | |
| M1 | Stratified 10-fold CV vs chronological test PR-AUC — how big is the gap? | Stratified 0.947 → Chronological 0.760 = **−18.7 pp gap**, the cost of concept drift over time. Honest reporting; not an artefact. | Table 4.1 vs Table 4.2 |
| M2 | Does the methodology transfer to a different dataset (Punta Villa Resort, 193 rows)? | Yes for the pipeline (re-runs end-to-end); directionally for the metrics (PR-AUC 0.541 chronological with ±15 pp CI at n_test = 20). The pre-flight duplicate-cluster diagnostic ran and confirmed honest evaluation. | Tables 4.1, 4.2, 4.4 |
| M3 | Cross-dataset feature consistency (Portugal ↔ Philippines)? | **`deposit_type` is the #1 SHAP feature on both datasets.** Same operational signal dominates two geographies, two property types, and two dataset sizes (119k vs 193 rows). | Table 4.4 |
| M4 | How were the hyperparameters chosen? | Rolling-origin champion/challenger selection across 3 chronological folds (60/70/80 % cutoffs), primary metric PR-AUC. LightGBM selected ahead of Gradient Boosting (Δ = +0.0065 PR-AUC) and XGBoost (Δ = +0.0109). | model_selection_summary.json |

---

## The 60-second verbal version

If the panel cuts you off and asks for the executive summary out loud, deliver these five sentences in order:

1. **Champion model:** "LightGBM with isotonic calibration, ROC-AUC 0.864 and PR-AUC 0.760 on the chronological test set, with bootstrap 95 % CI widths under 0.025."
2. **Five hypotheses, five verdicts:** "H1, H2, H4, and H5 supported; H3 partially supported — the three pre-registered SHAP features all appear in the top-10 but the rank order differs from what I predicted."
3. **Business value:** "The cost-sensitive policy recovers €2.94 M of revenue at risk on the test set — 97.5 % of the theoretical maximum."
4. **Risk concentration:** "26 % of bookings — the High risk tier — carry 52 % of realised losses, which is why the deployment focuses intervention effort on that tier."
5. **Transferability:** "The same methodology runs on a 193-row Philippine resort dataset, and `deposit_type` is the #1 driver on both — strong evidence the pipeline isn't memorising Portugal-specific quirks."

---

## How this table relates to the rest of the Chapter IV pack

```
chapter_iv_tables/
├── table_01_classification_cv_benchmark.md      (stratified 10-fold, 7 algos × 2 datasets)
├── table_02_chronological_oot_test.md           (chronological, 6 algos + per-hotel)
├── table_03_adr_regression.md                   (7 regressors)
├── table_04_shap_feature_importance.md          (top 5 raw features × 2 datasets)
├── table_05_cost_savings_by_threshold.md        (3 policies + no-model)
├── table_06_hypothesis_evidence_verdict.md      (H1–H5 closure)  ← Table 4.6 above
├── table_07_risk_tier_revenue_exposure.md       (low / medium / high tiers)  ← Table 4.7 above
├── table_08_threshold_policy_operational.md     (max_f1 / high_prec / cost_sensitive)  ← Table 4.8 above
└── table_09_chapter_iv_at_a_glance.md           (THIS file — the 60-second summary)
```

> **For the printed thesis:** Tables 4.1–4.8 each go in their own subsection of Chapter IV. Table 4.9 (this one) belongs at the **end of Chapter IV** as the closing summary, immediately before Chapter V begins. It is also the natural slide #1 of the defense deck — show this table, then drill into whichever sub-table the panel points to.
