# Paper Figures — Curated Visualizations for the Thesis

> **A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations**
>
> This folder contains the **18 figures** selected for inclusion in the printed thesis,
> organised by chapter section. Every figure here is the "center focus" of an argument
> in Chapter IV or Chapter V. Each figure ships in **both PNG and PDF** — use PNG for
> Word/Markdown previews, PDF for LaTeX submission.

---

## Quick Index

| # | Folder | Figure (filename stem) | Paper section | Hypothesis / Finding |
|---|---|---|---|---|
| 1 | `01_eda_and_data` | `fig_01_monthly_trend_with_volume_and_splits` | Ch IV Section 4.2.1 | Chronological-split honesty |
| 2 | `01_eda_and_data` | `fig_02_cancellation_rate_donut` | Ch IV Section 4.2.1 | Class balance (37 % positive) |
| 3 | `01_eda_and_data` | `fig_1.4_ph_monthly_trend` | Ch IV Section 4.2.2 | PH temporal context |
| 4 | `02_model_performance` | `fig_01_roc_pr_curves` | Ch IV Section 4.3.3 | Headline model quality |
| 5 | `02_model_performance` | `fig_72_ranked_dumbbell_model_selection` | Ch IV Section 4.3.2 | **H2** — rolling-CV winner |
| 6 | `02_model_performance` | `fig_03_normalized_confusion_matrix_max_f1` | Ch IV Section 4.3.3 | Operational error rates |
| 7 | `03_calibration_and_thresholds` | `fig_05_calibration_reliability_and_histogram` | Ch IV Section 4.3.5 | Decision-grade probabilities |
| 8 | `03_calibration_and_thresholds` | `fig_11_cost_sensitive_threshold_sweep` | Ch IV Section 4.4.2 | **H4** — €1.55M savings curve |
| 9 | `04_shap_and_interpretation` | `fig_13_shap_feature_importance_bar` | Ch IV Section 4.3.4 | **H1, H3** — feature ranking |
| 10 | `04_shap_and_interpretation` | `fig_14_shap_beeswarm` | Ch IV Section 4.3.4 | **H1, H3** — direction + magnitude |
| 11 | `05_business_value` | `fig_23_risk_tier_business_overview` | Ch IV Section 4.4.2 | Risk tiers + revenue exposure |
| 12 | `05_business_value` | `fig_24_policy_comparison_business` | Ch IV Section 4.4.1 | 3-policy business comparison |
| 13 | `05_business_value` | `fig_25_monthly_revenue_risk_timeline` | Ch IV Section 4.4.3 | Revenue-at-risk timeline |
| 14 | `06_transferability_ph` | `fig_5.4_ph_vs_pt_shap_comparison` | Ch IV Section 4.5.3 | **H5** — cross-dataset SHAP |
| 15 | `06_transferability_ph` | `fig_2.2_ph_roc_pr_curves` | Ch IV Section 4.5.2 | PH model quality |
| 16 | `06_transferability_ph` | `fig_2.7_ph_feature_importance` | Ch IV Section 4.5.2 | PH champion drivers |
| 17 | `07_methodology_contributions` | `fig_7.6_paired_delta_forest` | Ch IV Section 4.3.4 / Section 4.6 | Bootstrap-significance rigor |
| 18 | `07_methodology_contributions` | `fig_1.2_ph_cluster_size_distribution` | Ch IV Section 4.5.1 / Section 4.6 | Pre-flight diagnostic outcome |

**Total: 18 figures across 7 thematic groups, ~2-3 figures per chapter section.**

---

## Folder 01 — EDA & Data Characterization (Ch IV Section 4.2)

### Fig. 1 — Monthly Cancellation Trend with Split Boundaries (Portugal)
- **File:** `fig_01_monthly_trend_with_volume_and_splits.{png,pdf}`
- **Paper section:** Section 4.2.1 Portugal Dataset Characterisation
- **What it shows:** Monthly cancellation rate over 2015-2017 with the train / val / test
  split lines overlaid. Demonstrates the chronological 80/10/10 split visually.
- **Suggested caption:** *"Monthly booking volume and cancellation rate across the Portugal
  dataset (2015-2017), with chronological train (80 %), validation (10 %), and test (10 %)
  split boundaries marked. The validation and test windows fall after the train window in
  time, preventing temporal leakage during calibration and threshold tuning."*
- **Why it earns space:** Any defender will ask *"are you sure this isn't leaky?"* — this
  figure pre-empts that question.

### Fig. 2 — Cancellation Rate Donut
- **File:** `fig_02_cancellation_rate_donut.{png,pdf}`
- **Paper section:** Section 4.2.1
- **What it shows:** Class balance — 37 % canceled / 63 % retained on the full Portugal
  dataset.
- **Suggested caption:** *"Class balance on the Portugal dataset: 37.0 % of bookings were
  ultimately cancelled (n = 44,224 of 119,210). The class skew motivates the use of
  PR-AUC over accuracy as the primary model-selection metric."*
- **Why it earns space:** Justifies why PR-AUC (not accuracy) is the primary metric.

### Fig. 3 — Philippine Monthly Trend (PH)
- **File:** `fig_1.4_ph_monthly_trend.{png,pdf}`
- **Paper section:** Section 4.2.2 Philippine Sub-Study Characterisation
- **What it shows:** Booking volume and cancellation rate across the Punta Villa Resort
  PMS export (Dec 2022 – Dec 2025), 193 real bookings.
- **Suggested caption:** *"Monthly booking volume and cancellation rate at Punta Villa
  Resort (n = 193 bookings, 2022-2025). The overall cancellation rate of 15.0 % is markedly
  lower than the Portugal benchmark, consistent with a smaller property serving a more
  loyal Walk-In segment."*

---

## Folder 02 — Model Performance (Ch IV Section 4.3)

### Fig. 4 — ROC + PR Curves (Test Set, Portugal)
- **File:** `fig_01_roc_pr_curves.{png,pdf}`
- **Paper section:** Section 4.3.3 Held-Out Test-Set Performance
- **What it shows:** ROC and Precision-Recall curves for the champion LightGBM on the
  11,922-row test set. Headline: PR-AUC = 0.760 vs class baseline = 0.370 (≈2× lift).
- **Suggested caption:** *"ROC (left) and Precision-Recall (right) curves for the champion
  LightGBM model on the held-out 11,922-row test set. ROC-AUC = 0.864 (95 % CI [0.857,
  0.870]) and PR-AUC = 0.760 (95 % CI [0.749, 0.770]) — a 2.05× lift over the class-baseline
  precision of 0.370."*
- **Why it earns space:** The single "is this model any good?" plot.

### Fig. 5 — Rolling-Origin Model Selection Dumbbell
- **File:** `fig_72_ranked_dumbbell_model_selection.{png,pdf}`
- **Paper section:** Section 4.3.2 Portugal Model Selection
- **What it shows:** Per-model PR-AUC across the 3 rolling-origin folds (60/70/80 %
  cutoffs). LightGBM, XGBoost, and GradientBoosting cluster tightly at the top;
  LogisticRegression, RandomForest follow; DecisionTree at the bottom.
- **Suggested caption:** *"Rolling-origin cross-validation results across three chronological
  cutoffs (60 %, 70 %, 80 % of training data). Each model is plotted as a dumbbell joining
  its minimum and maximum fold PR-AUC; the marker indicates the mean. LightGBM (mean
  PR-AUC = 0.870) wins the selection by a 0.0028 margin over XGBoost."*
- **Why it earns space:** **Discharges H2.** Proves the champion was selected by evidence,
  not preference.

### Fig. 6 — Confusion Matrix at max_F1 Threshold
- **File:** `fig_03_normalized_confusion_matrix_max_f1.{png,pdf}`
- **Paper section:** Section 4.3.3
- **What it shows:** Normalised confusion matrix at the F1-maximising threshold (0.40).
- **Suggested caption:** *"Normalised confusion matrix for LightGBM at the F1-maximising
  threshold of 0.40 on the test set. Recall = 0.841, precision = 0.652 — the model
  captures 84 % of cancellations at the cost of one false alarm per ~1.5 true catches."*
- **Why it earns space:** Translates probability metrics into actionable error counts.

---

## Folder 03 — Calibration & Thresholds (Ch IV Section 4.3.5 + Section 4.4)

### Fig. 7 — Calibration Reliability + Probability Histogram
- **File:** `fig_05_calibration_reliability_and_histogram.{png,pdf}`
- **Paper section:** Section 4.3.5 Probability Calibration
- **What it shows:** Reliability diagram (predicted vs observed cancellation rate) before
  and after isotonic calibration, plus the post-calibration probability histogram.
- **Suggested caption:** *"Reliability diagram before and after isotonic calibration on
  the validation set. Expected Calibration Error (ECE) improves from 0.058 to 0.029
  (50 % reduction). The bottom panel shows the post-calibration probability distribution
  across the test set."*
- **Why it earns space:** Justifies that the probabilities are *decision-grade* — every
  downstream policy choice (risk tiers, cost-sensitive threshold) depends on this.

### Fig. 8 — Cost-Sensitive Threshold Sweep
- **File:** `fig_11_cost_sensitive_threshold_sweep.{png,pdf}`
- **Paper section:** Section 4.4.2 H4 Verdict — Cost-Sensitive Threshold
- **What it shows:** Total business cost as a function of decision threshold, with FP cost
  €15 and FN cost = revenue-at-risk per booking. The minimum sits at threshold = 0.04.
- **Suggested caption:** *"Cost-sensitive threshold sweep across the test set. Total cost
  combines FP penalty (€15 per intervention) and FN penalty (revenue-at-risk for missed
  cancellations). The cost-minimising threshold of 0.04 yields a total cost of €73,449.92,
  compared with €387,350.44 at the conventional 0.50 cut-off and €1,606,669.92 with no
  model — a 95.4 % reduction in expected revenue loss."*
- **Why it earns space:** **Discharges H4.** The €1.55M-savings number is the thesis's
  headline business finding.

---

## Folder 04 — SHAP & Interpretation (Ch IV Section 4.3.4)

### Fig. 9 — SHAP Feature Importance Bar
- **File:** `fig_13_shap_feature_importance_bar.{png,pdf}`
- **Paper section:** Section 4.3.4 Hypothesis Tests
- **What it shows:** Mean(|SHAP|) per feature across the test set, top 15 features.
- **Suggested caption:** *"Top 15 features by mean absolute SHAP contribution on the
  Portugal test set. `deposit_type`, `country`, and `lead_time` lead the ranking — the
  three predicted in H1, but in a different order than H3 (which predicted lead_time first).
  The divergence is reported as evidence of H3 being partially supported."*
- **Why it earns space:** **Discharges H1 + H3.**

### Fig. 10 — SHAP Beeswarm
- **File:** `fig_14_shap_beeswarm.{png,pdf}`
- **Paper section:** Section 4.3.4
- **What it shows:** Per-feature SHAP value distribution with point colour encoding the
  feature's raw value — adds direction + magnitude information beyond the bar chart.
- **Suggested caption:** *"SHAP beeswarm summary on the test set. Each dot represents a
  single booking's SHAP contribution for a given feature; colour encodes the feature
  value (red = high, blue = low). The directional pattern confirms intuition: long
  lead times push toward cancellation, prior successful stays push against."*
- **Why it earns space:** Where the bar chart shows *which* features matter, this shows
  *how* — essential for the discussion section.

---

## Folder 05 — Business Value (Ch IV Section 4.4)

### Fig. 11 — Risk Tier Business Overview
- **File:** `fig_23_risk_tier_business_overview.{png,pdf}`
- **Paper section:** Section 4.4.2
- **What it shows:** Test-set distribution across LOW / MEDIUM / HIGH risk tiers with
  revenue exposure and observed cancellation rate per tier.
- **Suggested caption:** *"Risk-tier composition of the test set (n = 11,922). Each tier
  is annotated with its observed cancellation rate, total revenue exposure, and recommended
  operational handling: LOW (P < 0.40) — standard; MEDIUM (0.40 ≤ P < 0.70) — reminder at
  T-7 days; HIGH (P ≥ 0.70) — deposit or confirmation call required."*
- **Why it earns space:** Translates the abstract probability into a 3-bucket policy.

### Fig. 12 — Policy Comparison Business
- **File:** `fig_24_policy_comparison_business.{png,pdf}`
- **Paper section:** Section 4.4.1 Threshold Policy Comparison
- **What it shows:** max_f1 vs high_precision vs cost_sensitive side-by-side: threshold,
  precision, recall, F1, total cost.
- **Suggested caption:** *"Three operating policies on the test set: F1-maximising
  (threshold 0.40), high-precision (0.98), and cost-sensitive (0.04). Each policy serves
  a different operational stance; the cost-sensitive policy is selected as the
  recommendation for revenue protection because of its 95.4 % reduction in expected loss."*

### Fig. 13 — Monthly Revenue-at-Risk Timeline
- **File:** `fig_25_monthly_revenue_risk_timeline.{png,pdf}`
- **Paper section:** Section 4.4.3 Power BI Dashboard
- **What it shows:** Revenue at risk per month over the test window, with risk-tier
  breakdown.
- **Suggested caption:** *"Monthly revenue at risk computed from cost-sensitive flags,
  stratified by risk tier. The timeline drives Page 4 of the Power BI dashboard and
  provides the daily-operational signal for revenue management staff."*

---

## Folder 06 — Philippine Transferability Study (Ch IV Section 4.5)

### Fig. 14 — Cross-Dataset SHAP Comparison (PT vs PH)
- **File:** `fig_5.4_ph_vs_pt_shap_comparison.{png,pdf}`
- **Paper section:** Section 4.5.3 H5 Verdict
- **What it shows:** Side-by-side mean(|SHAP|) bars from the Portugal and Philippine
  models. `deposit_type` is #1 on both.
- **Suggested caption:** *"Cross-dataset SHAP feature importance: Portugal champion (left,
  n_test = 11,922) versus Philippine champion (right, n_test = 20). `deposit_type` ranks
  #1 in both models, confirming H5. The shared dominance of a single feature across two
  geographies, sample sizes, and property types is the study's strongest evidence that the
  model captures genuine cancellation signal rather than data-specific artefacts."*
- **Why it earns space:** **Discharges H5.** This is the single most novel finding in
  the thesis — the visual that should appear in the executive summary if there is one.

### Fig. 15 — Philippine ROC + PR Curves
- **File:** `fig_2.2_ph_roc_pr_curves.{png,pdf}`
- **Paper section:** Section 4.5.2 PH Model Performance
- **What it shows:** ROC and PR curves on the 20-row PH test set; ROC-AUC ≈ 0.611,
  PR-AUC ≈ 0.542 (vs class baseline 0.150 → ~3.6× lift).
- **Suggested caption:** *"ROC (left) and Precision-Recall (right) curves on the Philippine
  test set (n = 20). The 95 % bootstrap CIs span roughly ±15 percentage points at this
  sample size — metrics are reported as directional evidence that the methodology
  transfers, not as production-grade performance numbers."*
- **Why it earns space:** Honest small-N performance reporting.

### Fig. 16 — Philippine Feature Importance
- **File:** `fig_2.7_ph_feature_importance.{png,pdf}`
- **Paper section:** Section 4.5.2
- **What it shows:** Top-feature ranking in the Philippine model. `deposit_type` clearly
  dominates.
- **Suggested caption:** *"LightGBM split-frequency importance on the Philippine sub-study
  (n_train = 154). `deposit_type` is the dominant predictor by a 27 % margin over the
  runner-up, mirroring the Portugal SHAP ranking and supporting H5."*

---

## Folder 07 — Methodology Contributions (Ch IV Section 4.6)

### Fig. 17 — Paired Bootstrap Delta Forest (Portugal)
- **File:** `fig_7.6_paired_delta_forest.{png,pdf}`
- **Paper section:** Section 4.3.4 H2 statistical rigour / Section 4.6 Methodology Contributions
- **What it shows:** ΔPR-AUC of the champion versus each challenger, with 95 % CIs from
  paired bootstrap.
- **Suggested caption:** *"Paired-bootstrap deltas (champion LightGBM minus challenger)
  in test PR-AUC, with 95 % confidence intervals. All five comparisons exclude zero,
  confirming LightGBM's PR-AUC advantage is statistically significant; the smallest
  significant gap is +0.007 over GradientBoosting (p = 0.001)."*
- **Why it earns space:** Closes the H2 statistical-rigour loop — without this, "the
  champion beats the runner-up" is only a point estimate.

### Fig. 18 — Philippine Pre-Flight Cluster Diagnostic
- **File:** `fig_1.2_ph_cluster_size_distribution.{png,pdf}`
- **Paper section:** Section 4.5.1 PH Setup / Section 4.6 Methodology Contributions
- **What it shows:** Distribution of duplicate-vector cluster sizes in the PH dataset.
  ~0 % duplicate rate confirms the methodology proceeds honestly.
- **Suggested caption:** *"Pre-flight duplicate-cluster diagnostic on the Philippine
  dataset (n = 193). The diagnostic flags datasets organised around recurring booking
  archetypes that would leak twins across chronological splits. With a duplicate rate
  near 0 %, the diagnostic does not fire — reported PH test metrics measure
  generalisation rather than memorisation across twin bookings."*
- **Why it earns space:** This is the **methodology contribution** the thesis offers
  for future small-N transferability studies — naming it visually anchors the contribution.

---

## What to Focus On — In Priority Order

If you only have time to read four figures before defense, read these:

1. **Fig. 14** (`fig_5.4_ph_vs_pt_shap_comparison`) — the novel cross-dataset finding.
2. **Fig. 8** (`fig_11_cost_sensitive_threshold_sweep`) — the €1.55M business case.
3. **Fig. 9** (`fig_13_shap_feature_importance_bar`) — the why-does-the-model-decide-this story.
4. **Fig. 4** (`fig_01_roc_pr_curves`) — the bottom-line "is it any good?" answer.

These four alone discharge H2 (model quality), H4 (savings), H5 (cross-dataset), and
provide the SHAP-driven discussion for H1 + H3.

---

## What Was Deliberately *Not* Included

For transparency, here is what was considered and cut, and the reasoning:

| Considered | Why cut |
|---|---|
| `fig_06_bootstrap_ci_forest` (PT) | Redundant with `fig_7.6_paired_delta_forest`; the paired version is the stronger argument. |
| `fig_15_shap_dependence_top3` | Interesting but too detailed for the thesis body — belongs in an appendix or notebook. |
| `fig_16_shap_waterfall_examples` | Single-booking explanations are great for the demo, distracting in the thesis. |
| Most `fig_4*` ADR figures (PT) | The ADR regressor is a secondary product; one mention + caveat is enough. |
| All `fig_47*`, `fig_48*` ADR diagnostics | Same reason — appendix material at best. |
| `fig_8*` monitoring drift figures | Best shown as Power BI screenshots, not thesis figures. |
| `fig_4.2_ph_adr_residuals` (PH) | The PH ADR overfits (R² = −0.97); a sentence is enough, no need for a figure. |
| Most `fig_9*` benchmark variants | Tier-2/3 model comparisons live in benchmark tables, not figures. |
| `fig_5.3_ph_individual_shap` | Single-booking SHAP — same reason as `fig_16` above. |
| Per-segment EDA grids | Collapsed into prose tables — figures are expensive page-space. |

The cut rule: *if removing a figure does not weaken a specific hypothesis verdict or a
defense-day question, cut it.* This left 18 figures across 7 themes, each non-redundant.

---

## Where Each Figure Lives in the Final Thesis

| Section in `complete_thesis.md` | Figures referenced |
|---|---|
| Ch IV Section 4.2 (Sense) | Fig. 1, 2, 3 |
| Ch IV Section 4.3 (Seize — modelling) | Fig. 4, 5, 6, 9, 10, 17 |
| Ch IV Section 4.3.5 (Calibration) | Fig. 7 |
| Ch IV Section 4.4 (Transform — business) | Fig. 8, 11, 12, 13 |
| Ch IV Section 4.5 (PH transferability) | Fig. 14, 15, 16, 18 |
| Ch IV Section 4.6 (Methodology contributions) | Fig. 17, 18 (cross-referenced) |
| Ch V Section 5.4 (Theoretical contributions) | Fig. 14 (cross-referenced) |
| Ch V Section 5.5 (Practical contributions) | Fig. 8, 11 (cross-referenced) |

**Total cross-references across Ch IV/V: ~22 figure citations** for 18 unique figures —
some figures (especially Fig. 8, 14, 17, 18) are appropriately cited twice because they
support both a results claim and a methodology / contribution claim.

---

## Reproducibility Notes

Each figure here is a copy of an artefact in `reports/figures/thesis/` (or `reports/figures/thesis/ph/`)
written by code in `notebooks/` and `src/eval/thesis.py`. If a number in the thesis text
disagrees with a figure caption, the **artefact is the source of truth** — re-export the
caption, never edit the figure.

To regenerate all figures from scratch:

```bash
make thesis          # regenerates Portugal figures via src/eval/thesis.py
python scripts/train_ph.py   # regenerates PH model + figures via notebooks/ph/
jupyter nbconvert --to notebook --execute notebooks/*.ipynb     # re-runs Portugal notebooks
jupyter nbconvert --to notebook --execute notebooks/ph/*.ipynb  # re-runs PH notebooks
```

After regeneration, re-run the curation copies in this folder (or simply use the
canonical paths under `reports/figures/thesis/`).
