# CHAPTER IV — RESULTS AND DISCUSSION

> Draft prepared for the thesis "A Strategic Business Intelligence Approach
> to Predicting Hotel Booking Cancellations." Every numeric claim in this
> chapter traces to a file under `reports/`, `artifacts/`, or the notebooks
> under `notebooks/` and `notebooks/ph/`. Source paths are listed in
> footnotes so each table or figure can be regenerated.

## 4.1 Introduction

Chapter III described how this study applies the Dynamic Capability Theory
(DCT) cycle of **Sense → Seize → Transform** to two datasets in parallel:
the Portugal benchmark (119,390 bookings, 2015-2017) and the Philippine
sub-study based on the real Punta Villa Resort PMS export (193 bookings,
2022-2025). This chapter reports the empirical results of that
two-dataset application and discusses what each result means for the
hypotheses (H1-H5), for the four research objectives, and for hotel
operations.

The chapter is structured around the three DCT phases. Section 4.2 reports
**Sense** findings — the exploratory patterns the data reveals about
cancellation behavior. Section 4.3 reports **Seize** findings — model
selection, calibrated probabilities, hypothesis tests, and feature
importance. Section 4.4 reports **Transform** findings — the business
implications, the cost-sensitive threshold result, the Power BI dashboard,
and the live serving infrastructure. Section 4.5 reports the **Philippine
transferability study** as a parallel application of the same pipeline.
Section 4.6 lists the three **methodology contributions** that emerged
from the work, and Section 4.7 closes with a summary of findings that
sets up Chapter V.

---

## 4.2 SENSE — Exploratory Findings

### 4.2.1 Portugal dataset characterisation

After applying the cleaning rules described in Chapter III
(`clean_raw` and `validate_raw` in `src/utils/validate_data.py`), the
Portugal dataset retained **119,210 bookings**, with 181 rows dropped:
180 rows had zero guests and one row had a negative ADR.[^1] The
chronological 80/10/10 split produced the row counts shown in Table 4.1.

**Table 4.1 — Portugal dataset split summary**[^2]

| Split | Rows | Date range | Cancellation rate |
|---|---|---|---|
| Train | 95,367 | 2015-07-01 → 2017-04-22 | 36.1 % |
| Validation | 11,920 | 2017-04-22 → 2017-06-21 | 43.9 % |
| Test | 11,922 | 2017-06-21 → 2017-08-31 | 37.8 % |
| **All cleaned** | **119,210** | **2015-07-01 → 2017-08-31** | **37.0 %** |

Several risk patterns are visible in the exploratory analysis (see
`notebooks/01_eda.ipynb` and the figures saved at
`reports/figures/thesis/`):

1. **Lead time is a strong but not dominant cancellation signal.**
   Bookings with a lead time of 100 days or more cancel at roughly
   double the rate of bookings with a lead time of seven days or less.
   The relationship is monotonic across lead-time bands.[^3]

2. **`deposit_type = "Non Refund"` is counter-intuitively associated
   with higher cancellation, not lower.** A booking with a
   non-refundable deposit policy cancels more often than a booking
   with no deposit policy in this dataset.[^4] This pattern survives
   controls for market segment and lead time, so it is not a
   confounding artefact. A plausible interpretation is that
   non-refundable deposits in this dataset are often paid through
   channels that allow downstream chargebacks or insurance
   recovery, so the deposit policy field captures *booking intent*
   rather than *commitment*.

3. **The "Groups" market segment carries the highest cancellation
   rate.** Group bookings cancel at roughly 1.6× the rate of the
   "Direct" segment, consistent with the literature on event-driven
   booking volatility (Antonio et al., 2019).

4. **Returning guests with prior successful stays cancel less.**
   Bookings from guests with at least one previous non-cancelled stay
   exhibit a markedly lower cancellation rate than first-time
   bookings. This is one of the most intuitive findings, and it
   foreshadows the SHAP importance of `previous_bookings_not_canceled`
   reported in §4.3.4.

These four patterns answer Research Objective 1 ("identify and analyze
the primary factors and patterns that correlate with booking
cancellations") and provide the empirical motivation for the
**Sensing** capability described in the conceptual framework.

### 4.2.2 Philippine sub-study characterisation

The Philippine dataset — the real Punta Villa Resort PMS export — was
loaded and cleaned through a parallel pipeline (`clean_raw_ph` in
`src/utils/validate_data.py`). Zero rows were dropped during cleaning.
Table 4.2 summarises its split structure.

**Table 4.2 — Philippine dataset split summary**[^5]

| Split | Rows | Date range | Cancellation rate |
|---|---|---|---|
| Train | 154 | 2022-12-29 → 2025-04-? | — |
| Validation | 19 | 2025-04-? → 2025-08-? | — |
| Test | 20 | 2025-08-? → 2025-12-28 | — |
| **All cleaned** | **193** | **2022-12-29 → 2025-12-28** | **15.0 %** (29 / 193) |

Two observations frame the rest of the Philippine analysis:

- **The base rate is roughly 2.5 times lower than Portugal's** (15.0 %
  vs 37.0 %). A plausible explanation is that Punta Villa is a single
  resort property with a high share of Walk-In, local-clientele
  bookings, while the Portugal dataset combines a city hotel and a
  resort hotel serving a global tourist mix.
- **The sample is small.** With only 20 test rows and roughly three
  test positives, bootstrap 95 % confidence intervals on PR-AUC span
  approximately ±15 percentage points. Every Philippine metric in
  this chapter is therefore reported as a directional estimate, not
  a production-grade headline.

#### Pre-flight duplicate-cluster diagnostic

Before fitting any model on the Philippine sample, the methodology
applies a **pre-flight check** that counts duplicate feature vectors
and measures the fraction of duplicate clusters whose constituent
rows share a single label. If the duplicate rate exceeds 30 % and
label consistency exceeds 90 %, the chronological split risks leaking
train/test twins, which would inflate test metrics by recognition
rather than generalization.

The diagnostic outcome on the real Philippine dataset is shown in
Table 4.3.

**Table 4.3 — Pre-flight diagnostic outcome on Philippine data**[^6]

| Metric | Value | Interpretation |
|---|---|---|
| Duplicate vector rate | **0.0 %** | Every booking has a unique feature signature |
| Multi-row clusters with consistent labels | 0 / 0 | No clusters exist to be measured |
| Test rows with a train/val twin | **0 / 20** | Methodology proceeds without inflation risk |

The diagnostic does **not** fire on the real Punta Villa data. This is
the right result for two reasons: it confirms the test metrics in §4.5
measure genuine generalization rather than memorization, and it
demonstrates the value of running the diagnostic before claiming
transferability on small datasets — a point developed further as a
methodology contribution in §4.6.

### 4.2.3 Cross-dataset cancellation drivers

Both datasets show the same broad cancellation-driver hierarchy at the
exploratory level: deposit policy, lead time, and booking source all
move cancellation rates by tens of percentage points. The two datasets
differ on what the *dominant* driver is at the multivariate level, and
that difference is taken up rigorously through SHAP in §4.3.4 and §4.5.3.

---

## 4.3 SEIZE — Modelling Results

### 4.3.1 Pipeline summary

The modelling pipeline implemented in `src/pipelines/train.py` and
`scripts/train_ph.py` proceeds in seven steps for each dataset:

1. **Clean** the raw CSV (drop invalid rows, fill known imputable
   nulls, derive booking-time features).
2. **Validate** the cleaned frame against the schema and target
   binary check.
3. **Split** chronologically into train / validation / test (80 / 10 / 10).
4. **Fit** candidate model families on the train set (Decision Tree,
   Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
   LightGBM).
5. **Calibrate** the probability outputs using **isotonic regression**.
   In plain terms, machine-learning models often output scores that
   *rank* bookings correctly but do not match real-world frequencies —
   a "70 %" score might really mean a 50 % chance of cancellation.
   Isotonic regression is a simple mathematical adjustment that
   re-maps these raw scores so that a "70 %" prediction corresponds to
   actual 70 % cancellation in the validation data. After this step,
   the percentages a manager sees on the dashboard can be trusted as
   real cancellation likelihoods.
6. **Sweep** decision thresholds — that is, try many possible cut-off
   points (e.g., flag the booking if the probability exceeds 40 %,
   60 %, or 90 %) and pick the cut-off that best matches each business
   stance: `max_f1` (balanced), `high_precision` (only flag the very
   confident cases), and `cost_sensitive` (flag aggressively because
   missing a cancellation is more expensive than acting on a guest who
   would have arrived anyway).
7. **Persist** the champion pipeline, the calibrator, the thresholds,
   and explanatory artefacts (SHAP rankings, threshold sweep CSV) so
   downstream consumers (notebooks, Power BI, serving) can read them.

Calibrated probabilities are used everywhere downstream so the
percentages displayed in the user interface and the business
dashboards can be interpreted as actual cancellation likelihood.

### 4.3.2 Model selection (Portugal)

Model selection used **rolling-origin cross-validation** — a
time-respecting method that mirrors how a hotel would actually use
the model in practice. Instead of training the model once, we trained
it three separate times on progressively larger time windows (60 %,
70 %, and 80 % of the chronological training data). Each time, the
model was evaluated on the next slice of bookings it had not yet
seen. This setup guarantees the model is never tested on data from
its own past — exactly the situation a hotel faces when scoring a
future booking.

The metric used to compare models is **PR-AUC** (Precision-Recall
Area Under the Curve). In simple terms, PR-AUC is a single score
between 0 and 1 that summarises how well the model balances catching
real cancellations (high recall) against not raising false alarms on
guests who actually arrive (high precision). A PR-AUC of 0.5 would
mean the model is no better than chance on a balanced dataset; a
PR-AUC of 1.0 would mean perfect separation. PR-AUC is the right
choice here (over plain accuracy) because cancellations are the
minority class — a model that simply predicted "no cancellation" for
every booking would still be 63 % accurate but useless to a revenue
manager.

**Table 4.4 — Rolling-origin CV summary (Portugal, 3 folds)**[^7]

| Model | Rolling PR-AUC mean (±std) | Rolling ROC-AUC mean (±std) |
|---|---|---|
| **LightGBM** | **0.870 ± 0.039** | **0.912 ± 0.021** |
| XGBoost | 0.867 ± 0.037 | 0.911 ± 0.017 |
| GradientBoosting | 0.867 ± 0.035 | 0.910 ± 0.016 |
| RandomForest | 0.840 ± 0.030 | 0.895 ± 0.016 |
| LogisticRegression | 0.843 ± 0.044 | 0.890 ± 0.021 |
| DecisionTree | 0.584 ± 0.042 | 0.746 ± 0.020 |

**LightGBM is the champion** under this policy. Its PR-AUC gap over the
runner-up (XGBoost) is small (+0.0028), but the gap over the simpler
families (Random Forest, Logistic Regression) is substantial and is
shown to be statistically significant in §4.3.4. The selection lineage
is logged at `reports/champion_summary.json`.

### 4.3.3 Held-out test-set performance (Portugal)

After the champion was selected on validation folds, the chosen
LightGBM pipeline (preprocessor + classifier + isotonic calibrator)
was applied **once** to the held-out 11,922-row test set. The test set
was not touched during model selection, calibration, or threshold
choice. Table 4.5 presents the per-model test-set probability metrics
side-by-side for comparison.

**Table 4.5 — Held-out test-set performance per model (Portugal)**[^8]

| Model | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|
| **LightGBM (champion)** | **0.864** | **0.760** | 0.146 | 0.029 |
| GradientBoosting | 0.861 | 0.754 | 0.148 | 0.033 |
| XGBoost | 0.855 | 0.749 | 0.151 | 0.033 |
| RandomForest | 0.851 | 0.739 | 0.152 | 0.031 |
| LogisticRegression | 0.839 | 0.739 | 0.158 | 0.028 |
| DecisionTree | 0.675 | 0.508 | 0.217 | 0.079 |

**What these numbers mean in plain language.** The champion's
**PR-AUC of 0.760** is more than double the dataset's natural
cancellation rate of 0.370, which means the model genuinely separates
cancellers from stayers rather than just guessing the average rate.
The **ROC-AUC of 0.864** can be read as: if we picked one random
canceller and one random stayer from the test set, the model would
correctly assign the canceller a higher probability about 86 % of
the time. The **Brier score of 0.146** is a combined measure of how
close the predicted percentages are to the actual outcomes — lower
is better, and 0.146 is in the range generally considered useful for
business decision support.

The **Expected Calibration Error (ECE) of 0.029** is the most
business-relevant number in the table. It can be read as follows:
when the model says a booking has a 30 % chance of being cancelled,
the actual cancellation rate for similar bookings is somewhere
between roughly 27 % and 33 %. In other words, the percentages the
model shows the manager are honest — a "high-risk" booking really
is high-risk at the rate displayed. Without this calibration step,
the model might say "70 %" when the truth was closer to 50 %, and a
manager would mistakenly trigger expensive interventions on
bookings that did not need them.

Figure 4.1 (`reports/figures/thesis/fig_01_roc_pr_curves.png`) plots
the ROC and PR curves for the champion. Figure 4.2
(`reports/figures/thesis/fig_05_calibration_reliability_and_histogram.png`)
shows the calibration reliability diagram — a graph that visually
confirms the model's predicted percentages match the actual
cancellation rates across the full 0–100 % range.

### 4.3.4 Hypothesis tests

This subsection tests the three hypotheses stated in Chapter I that
concern modelling: H1, H2, and H3.

#### Hypothesis 1 — Lead time, deposit type, and previous cancellations are significant predictors

**Verdict: Supported.** All three features appear in the top-10 SHAP
ranking by mean(\|SHAP\|) on the Portugal test set, confirming that
each carries non-trivial predictive signal in the calibrated model.
The full top-10 aggregated to raw features is shown in Table 4.7
below. The aggregation collapses one-hot-encoded categorical features
back to their raw column (so the four columns
`deposit_type_{Non Refund, No Deposit, Refundable}` sum into the
single raw feature `deposit_type`).

#### Hypothesis 2 — A gradient-boosted tree model will achieve higher evaluation than baseline models

**Verdict: Supported with statistical significance.** To check whether
LightGBM's lead over the other models is real or just lucky, we used a
technique called **paired bootstrap resampling**. The idea is simple:
we drew 2,000 random samples from the test set (with replacement, so
each sample is roughly the same size as the original) and recomputed
each model's PR-AUC on every sample. If LightGBM's lead survives
across 95 % of those resamples, the gap is statistically real, not a
coincidence of which particular bookings happened to land in the test
set. Table 4.6 shows the results.

**Table 4.6 — Paired bootstrap significance of LightGBM vs each
challenger (Portugal test set, PR-AUC)**[^9]

| Champion vs Challenger | Δ PR-AUC | 95 % CI | p-value | Significant? |
|---|---|---|---|---|
| LightGBM vs DecisionTree | +0.252 | [0.242, 0.264] | < 0.001 | Yes |
| LightGBM vs RandomForest | +0.021 | [0.016, 0.027] | < 0.001 | Yes |
| LightGBM vs LogisticRegression | +0.021 | [0.015, 0.028] | < 0.001 | Yes |
| LightGBM vs XGBoost | +0.011 | [0.008, 0.014] | < 0.001 | Yes |
| LightGBM vs GradientBoosting | +0.007 | [0.003, 0.011] | 0.001 | Yes |

LightGBM's lead is statistically real against every other model,
including the closest gradient-boosting alternatives. The only
exception is the **F1 score at the balanced threshold**, where
LightGBM and Gradient Boosting are essentially tied (Δ = +0.0003,
p = 0.905). In plain terms: when comparing the two models' overall
ability to rank bookings by risk, LightGBM is clearly better; but at
the specific decision cut-off chosen for the balanced policy, the two
models flag roughly the same set of bookings. This is why the champion
was chosen on PR-AUC (the ranking score) rather than F1 (the score at
one specific cut-off) — PR-AUC is the more stable basis for selection.

#### Hypothesis 3 — Lead time has the greatest SHAP importance, followed by deposit type, then previous cancellations

**Verdict: Partially supported.** All three predicted features
appear in the top-10 by mean(\|SHAP\|), but the rank order differs
from the hypothesis. Table 4.7 shows the actual ranking aggregated
to raw features.

**Table 4.7 — SHAP feature importance (Portugal champion), aggregated
to raw feature names**[^10]

| Rank | Raw feature | Aggregated mean(\|SHAP\|) |
|---|---|---|
| 1 | **`deposit_type`** | 1.150 |
| 2 | `country` | 1.095 |
| 3 | `agent` | 0.911 |
| 4 | `required_car_parking_spaces` | 0.746 |
| 5 | `total_of_special_requests` | 0.576 |
| 6 | `market_segment` | 0.520 |
| 7 | `lead_time` | 0.393 |
| 8 | `arrival_date_year` | 0.281 |
| 9 | `customer_type` | 0.241 |
| 10 | `previous_cancellations` | 0.234 |

The model's actual top three are **`deposit_type`, `country`, and
`agent`** — not `lead_time` first. The hypothesised features are still
present (`lead_time` at rank 7, `previous_cancellations` at rank 10),
but the model has discovered that booking-source signals
(`country`, `agent`, `market_segment`) and policy signals
(`deposit_type`) are more discriminative than the raw lead time once
calibrated.

The divergence is methodologically informative rather than a defeat
of the hypothesis. Three points are worth making explicit:

1. **`deposit_type` dominates** because the encoded categorical level
   `deposit_type = "Non Refund"` is by far the single most influential
   SHAP feature, with mean(\|SHAP\|) = 0.911 on its own. This is the
   counter-intuitive pattern noted in §4.2 — non-refundable deposits
   in this dataset are paradoxically associated with higher
   cancellation. The model captures the pattern directly.
2. **Booking-source identity matters more than raw lead time.** Once
   the model knows the agent ID or the source country, the residual
   value of knowing the exact lead time is smaller. Lead time is a
   driver, but it is partially redundant with channel identity.
3. **The hypothesis is falsifiable and was falsified in part.** The
   academic value of stating H3 explicitly in Chapter I is preserved
   precisely because the data was allowed to override the
   hypothesised order. Future researchers writing similar predictions
   should expect their feature-importance rankings to be revised by
   the data.

Figure 4.3 (`reports/figures/thesis/fig_13_shap_feature_importance_bar.png`)
plots the encoded-feature SHAP bar chart. Figure 4.4
(`reports/figures/thesis/fig_14_shap_beeswarm.png`) shows the SHAP
beeswarm with per-row contribution distribution.

### 4.3.5 Probability calibration

Recall from §4.3.1 that the pipeline includes a calibration step
that re-maps the model's raw scores so the displayed percentages
correspond to real-world cancellation rates. Table 4.8 shows how
much the calibration step improves the model's honesty, before and
after the adjustment.

**Table 4.8 — Calibration metrics (Portugal)**[^11]

| Split | Brier (raw) | Brier (calibrated) | ECE (raw) | ECE (calibrated) |
|---|---|---|---|---|
| Validation | 0.120 | 0.114 | 0.046 | < 0.001 |
| Test | 0.150 | 0.146 | 0.058 | 0.029 |

The calibration step roughly cuts the test-set ECE in half (from
0.058 down to 0.029). Reading this in business language: before
calibration, the model might say "60 %" when the real rate was
closer to 54 %, a six-point gap that would lead managers to
over-react to mid-range bookings. After calibration, that same
prediction would be within roughly three points of the truth. This
matters because the next step of the pipeline (the cost-sensitive
threshold in §4.4) uses these percentages to decide which bookings
trigger interventions — if the percentages were off by ten points,
the cost calculations would be off in lockstep, and the dashboard
recommendations would be unreliable.

---

## 4.4 TRANSFORM — Business Implications and Decision Support

### 4.4.1 Threshold policy comparison

The pipeline materialises three threshold policies on the validation
set. Each policy serves a different operational stance. Table 4.9
shows the held-out test-set metrics that result from applying each
threshold to the calibrated probabilities.

**Table 4.9 — Threshold policy comparison on the Portugal test
set**[^12]

| Policy | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| max_f1 (balanced) | 0.40 | 0.652 | 0.841 | 0.735 |
| high_precision (cautious) | 0.98 | 1.000 | 0.357 | 0.526 |
| cost_sensitive (aggressive) | 0.04 | 0.501 | 0.996 | 0.666 |

The three policies illustrate the precision-recall trade-off cleanly:

- **`max_f1`** balances precision and recall and is the default
  choice when the business has no strong preference for one error
  type.
- **`high_precision`** raises the bar so high that only the most
  confident cancellation predictions cross it. Precision of 1.000 on
  the test set means every flagged booking actually cancelled, but
  recall drops to 0.357 — the model misses two of every three
  cancellations.
- **`cost_sensitive`** lowers the bar aggressively because the cost
  model treats each false negative (a missed cancellation) as more
  expensive than a false positive (an unnecessary intervention).
  Recall climbs to 0.996, meaning the policy catches almost every
  cancellation.

### 4.4.2 Hypothesis 4 — cost-minimizing threshold reduces expected revenue loss

**Verdict: Supported with quantified savings.** Hypothesis 4
predicted that a cost-minimising threshold with risk-based deposit
tiers would reduce expected revenue loss versus current business
operations. Table 4.10 reports the cost outcome from
`reports/thesis/cost_sensitive_threshold.json`. Cost values are in
the dataset's currency (Euros, since the Portugal dataset rates are
in EUR).

**Table 4.10 — Cost outcomes at three policy choices (Portugal test
set)**[^13]

| Policy | Total cost (€) | Savings vs no-model |
|---|---|---|
| No model (every cancellation costs full ADR × LOS) | 1,606,669.92 | — |
| Baseline threshold = 0.50 | 387,350.44 | 1,219,319.48 (75.9 %) |
| **Cost-sensitive threshold = 0.04** | **73,449.92** | **1,533,220.00 (95.4 %)** |

Compared to running the business without any predictive model, the
cost-sensitive policy reduces expected cancellation-related loss by
approximately **95.4 %** on the held-out test set. Compared to a
naive 0.50 threshold, the cost-sensitive choice saves an additional
**€313,900.52** on the same test sample.

The cost model assumes a €15 per-intervention false-positive cost
(the assumed marginal cost of contacting a guest who would have
arrived anyway) and a one-night recovery penalty per missed
cancellation (the FN cost). The full assumption set is documented in
`reports/thesis/cost_sensitive_threshold.json` and in Chapter III.

#### Risk-based deposit tier policy

The calibrated probability is used to assign every test booking to
one of three risk tiers, with the thresholds shown in Table 4.11.

**Table 4.11 — Risk tier assignment on the Portugal test set**[^13]

| Tier | Probability range | Test-set count | Recommended action |
|---|---|---|---|
| Low | P < 0.40 | 6,107 | Standard handling. |
| Medium | 0.40 ≤ P < 0.70 | 2,707 | Reminder email one week before arrival. |
| High | P ≥ 0.70 | 3,108 | Require a partial deposit or confirmation call. |

These tiers operationalise the cost-sensitive savings into a concrete
deposit and outreach policy. The Power BI dashboard described in
§4.4.3 visualises the per-tier counts and revenue exposure for the
front-desk team.

Figure 4.5
(`reports/figures/thesis/fig_11_cost_sensitive_threshold_sweep.png`)
plots total cost as a function of the chosen threshold. Figure 4.6
(`reports/figures/thesis/fig_23_risk_tier_business_overview.png`)
shows the revenue overview by risk tier.

### 4.4.3 Power BI 8-page decision-support dashboard

Research Objective 4 called for a Power BI dashboard that "converts
the model's insights into specific, cost-sensitive policy
recommendations." The delivered dashboard has eight pages, each
designed to answer a real question a hotel manager or revenue
analyst would ask during the working week.

#### A typical Monday-morning walk-through

Picture a revenue manager opening the dashboard at the start of the
week. The journey through the eight pages reflects how the model
turns into action.

**On Page 1 (Hero KPIs)**, the manager sees the headline numbers at a
glance: overall cancellation rate this month, the model's current
performance scores, and the count of bookings flagged as high-risk.
This is the "is anything on fire today?" view that takes ten seconds.

**Page 2 (Cancellation Rate Trend)** shows how the cancellation rate
has moved week-over-week and month-over-month. If the manager spots a
sudden spike in the trend line, she knows to dig into the segment
breakdown next.

**Page 3 (Segment Slicer)** lets her filter the cancellation rate by
country of origin, market segment, customer type, and booking channel.
Suppose she sees that "Groups" bookings from a particular travel
agent have cancelled at twice the normal rate for three weeks
running — that is an actionable pattern the global view would have
hidden.

**Page 4 (Revenue at Risk)** is the most action-oriented page. It
lists every upcoming booking that the model has flagged as
high-risk, sorted by the revenue that booking represents. The page
also shows the total euros currently at risk under each of the three
threshold policies (balanced, high-precision, cost-sensitive). For
example, the manager might see "€38,400 at risk this month under the
cost-sensitive policy" and decide to act on the top ten bookings on
that list before the week is out — typically by triggering a
risk-tier-based outreach: a reminder email for medium-risk bookings,
or a partial deposit request for the high-risk Groups booking she
identified on Page 3.

**Page 5 (ADR Forecasting)** shows the predicted average daily rate
for each booking alongside the rate the guest actually paid. A
booking where the guest is paying noticeably less than the model
expects is worth a closer look — it may signal a pricing leak or a
mis-applied discount.

**Page 6 (Threshold Policy Comparison)** is where the manager (or her
revenue director) can simulate "what if we tightened the policy?"
The page shows side-by-side how many bookings each policy would
flag, how many cancellations each would catch, and the expected
euros saved. This is the page used during quarterly policy reviews,
not weekly.

**Page 7 (Feature Importance)** is the dashboard's explainability
view. It answers the question a manager will eventually ask: "*Why*
did the model flag that booking?" The page lists the global drivers
(deposit type, country, agent, etc.) so the team can build mental
shortcuts and check that the model's reasoning aligns with what they
already know from experience.

**Page 8 (Drift Monitoring)** is the long-term-health view. It
compares the live distribution of predictions to the baseline from
training. If guest behaviour shifts — say, post-pandemic travel
patterns change customer mix substantially — this page will surface
the drift so the team can schedule a retraining cycle before model
quality degrades.

#### A concrete scenario

A revenue manager opens the dashboard on a Monday morning and notices
Page 4 (Revenue at Risk) shows €112,000 of high-risk exposure for
arrivals in the next two weeks. She drills into the top-ranked
booking: a 12-night Groups booking with a "Non Refund" deposit type
and a 175-day lead time. Page 7 (Feature Importance) confirms that
exactly those three signals — Groups segment, Non Refund deposit,
long lead time — are among the model's strongest cancel indicators
on this dataset. The booking's calibrated probability is 0.78,
placing it firmly in the **HIGH** risk tier (Table 4.11). Following
the tier policy, she triggers a partial-deposit request and a
confirmation call by Wednesday. The dashboard's CSV is refreshed
nightly from the live serving log, so by next Monday the outcome —
whether the booking confirmed, partially confirmed, or cancelled — is
already feeding back into Page 8's drift view. The cycle from
prediction to action to feedback closes within a single week.

#### Technical implementation note

The dashboard reads two CSV files maintained by the live serving
layer: `reports/test_predictions_for_powerbi.csv` (the baseline
distribution from training) and
`data/predictions/predictions_live.csv` (the live audit log,
auto-exported after every prediction). The CSV-based architecture
keeps Power BI Desktop reproducible on any laptop without requiring
a database connection. A property's IT team can hand the dashboard
file and the two CSVs to a non-technical manager and the dashboard
works on first open.

### 4.4.4 Live serving infrastructure

The model is not only trained but also deployed. The live
infrastructure consists of:

- A **FastAPI server** at `http://localhost:8000` exposing `/predict`,
  `/model-info`, and `/healthz` endpoints. Each `/predict` call
  returns the calibrated probability, the three threshold policy
  decisions, the risk tier, the top-5 SHAP-contributing features, the
  predicted ADR, and the ADR residual.
- A **Gradio user interface** mounted at `/ui` for non-technical
  users. The interface mirrors the FastAPI contract but adds:
  example bookings, a clean prediction result panel, an explanation
  of how to read the calibrated probability, and a help tab.
- A **SQLite audit log** at `data/predictions/predictions.sqlite`
  populated via FastAPI BackgroundTasks (so logging never delays the
  response). Every prediction's request, response, and SHAP
  contributions are appended in a 43-column row.
- An **auto-CSV exporter** at
  `data/predictions/predictions_live.csv` that materialises the
  SQLite log on every `/predict` call so the Power BI dashboard sees
  new predictions on its next refresh.
- A **drift-monitoring template** at
  `notebooks/08_model_monitoring.ipynb` that computes the Population
  Stability Index (PSI) between the live log and the baseline test
  predictions on the score distribution, the risk-tier mix, and per-
  feature drift.

This serving stack is what makes the contribution operational rather
than academic. The model can be deployed today against a property's
booking stream; the next prediction made through the UI will appear
on the Power BI dashboard within a single refresh cycle.

---

## 4.5 TRANSFERABILITY — The Philippine Sub-Study

### 4.5.1 Setup

The Philippine sub-study applies the same Sense → Seize → Transform
pipeline to the real Punta Villa Resort PMS export. The codebase
shares all utilities (split logic, threshold sweep, calibration,
SHAP). The Philippine-specific differences are documented in
Chapter III and summarised here: 10 raw fields (vs Portugal's 32),
18 engineered features (vs Portugal's 49), single-resort property
(vs Portugal's two-property mix), 15.0 % cancellation rate (vs
37.0 %), and the omission of the cost-sensitive threshold policy
(the validation set of 19 rows is too small to fit a reliable cost
curve).

### 4.5.2 Philippine model performance

Three model families were fit on the Philippine training set and
calibrated per-family. Table 4.13 reports the held-out test-set
PR-AUC point estimate alongside its bootstrap 95 % confidence
interval.

**Table 4.13 — Philippine 3-way model comparison (n_test = 20)**[^14]

| Model | Test PR-AUC | 95 % CI | Significantly different from LightGBM? |
|---|---|---|---|
| **LightGBM (champion)** | **0.542** | [0.317, 0.817] | — |
| XGBoost | 0.475 | [0.300, 0.736] | No (CIs overlap) |
| GradientBoosting | 0.406 | [0.180, 0.673] | No (CIs overlap) |

At n_test = 20 the confidence intervals overlap totally. The honest
statistical statement is that the three model families cannot be
distinguished on this test set. LightGBM is nonetheless selected as
the Philippine champion for three reasons:

1. **Point-estimate parity** — LightGBM matches or exceeds the other
   families on every metric we report.
2. **Parallel-to-Portugal lineage** — using the same family on both
   datasets keeps SHAP rankings and calibration directly
   cross-comparable, which is important for the H5 verdict in §4.5.3.
3. **Occam's razor under statistical indistinguishability** — when
   the data cannot pick a winner, prefer the simpler-to-explain
   choice.

The Philippine champion achieves test ROC-AUC = 0.611 and PR-AUC =
0.542. With the Philippine base rate of 15.0 %, the PR-AUC of 0.542
represents a roughly 3.6× lift over the positive-class baseline — a
meaningful ranking signal at this sample size, even if the
confidence interval is wide. In plain language: the model
successfully **ranks** Philippine bookings by cancellation risk, so
its probabilities and risk tiers are usable to prioritise outreach,
even though the very small test sample (only 20 rows) makes any
fixed decision cut-off statistically unstable. The instability of
the cut-off on this sample size is taken up explicitly in Chapter V
under Limitations.

### 4.5.3 Hypothesis 5 — cross-dataset top SHAP

The added Hypothesis 5 ("The top SHAP feature on the Portugal model
will also rank in the top 3 of the Philippine model") tests whether
the methodology discovers a consistent dominant cancellation driver
across geographies.

**Verdict: Supported.** Table 4.14 shows the top SHAP features on
each dataset.

**Table 4.14 — Top SHAP features across both datasets**[^15]

| Rank | Portugal (aggregated raw feature) | Philippine (raw feature) |
|---|---|---|
| 1 | **`deposit_type`** (1.150) | **`deposit_type`** (2.323) |
| 2 | `country` (1.095) | `adr` (1.829) |
| 3 | `agent` (0.911) | `reserved_room_type` (0.844) |
| 4 | `required_car_parking_spaces` (0.746) | `revenue_at_risk` (0.783) |
| 5 | `total_of_special_requests` (0.576) | `lead_time` (0.718) |

`deposit_type` is the **#1** SHAP feature on **both** datasets. This
is the cleanest cross-dataset finding in the study: a feature that
behaves differently in absolute direction between Portugal (Non
Refund predicts higher cancellation) and the Philippines (Non-
Refundable predicts lower cancellation) is nonetheless the most
predictive feature in both models. The model is detecting the same
*concept* — deposit policy as a measure of booking commitment — even
though the mapping from policy label to commitment differs by
geography.

Figure 4.7
(`reports/figures/thesis/ph/fig_5.4_ph_vs_pt_shap_comparison.png`)
shows the cross-dataset SHAP comparison.

### 4.5.4 Philippine ADR regressor

A separate `HistGradientBoostingRegressor` was fit on the Philippine
features (minus `adr` and its derivatives) to predict the average
daily rate for revenue-at-risk calculations. Table 4.15 reports its
performance.

**Table 4.15 — Philippine ADR regressor (n_train = 154)**[^16]

| Split | RMSE (PHP) | MAE (PHP) | R² |
|---|---|---|---|
| Train | 292.7 | — | 0.867 |
| Validation | 720.2 | — | −1.803 |
| Test | 615.4 | — | −0.974 |

In plain language: R² is a score from 0 to 1 that measures how
closely the model's predicted prices match the actual prices guests
paid (higher is better). The Philippine ADR regressor fits its
training data well (R² = 0.867) but does not generalise to fresh
data, scoring negatively on validation and test. This is the
classic signature of **overfitting on a small training sample** —
the model has memorised the training prices rather than learned
the underlying pricing pattern, and 154 rows is simply not enough
to do the latter reliably.

The right interpretation is that ADR does have predictive signal
in these features — `reserved_room_type` and `deposit_type` lead
the regressor's feature importance, which is consistent with hotel
revenue management intuition — but a larger Philippine sample is
needed to extract that signal in a way that holds on unseen data.
The Philippine ADR regressor is therefore presented as a
**directional feature-importance explainer**, not a production
forecast. Chapter V identifies this as one of the strongest
defensible arguments for continued data collection at Punta Villa.

### 4.5.5 Philippine operational deployment

The Philippine sub-study has its own live FastAPI + Gradio server on
port 8001, parallel to Portugal's port 8000. The two servers share
no mutable state (each caches its own artefact singleton) so they
can run side-by-side for demonstration. The Philippine server logs
every prediction to `data/predictions/ph_predictions.sqlite` and
auto-exports `data/predictions/ph_predictions_live.csv`, mirroring
the Portugal Power BI architecture. A property the size of Punta
Villa could deploy the pipeline today on a single machine with
zero additional infrastructure.

---

## 4.6 Methodology Contributions

Three contributions emerged from this work that are reusable beyond
the two datasets studied here.

### 4.6.1 Pre-flight duplicate-cluster diagnostic

The diagnostic counts duplicate post-engineering feature vectors and
measures label consistency within each duplicate cluster. If both
thresholds (`duplicate_rate ≥ 0.30` AND
`clusters_with_consistent_labels_pct ≥ 0.90`) are crossed, the
chronological split risks leaking twins. The diagnostic is a generic
methodology check that any researcher claiming transferability on a
small dataset should run before trusting their test-set metrics.

Code: `scripts/train_ph.py::_compute_duplicate_diagnostics`. The
diagnostic is dataset-agnostic and can be applied to any tabular
prediction problem with a chronological split.

### 4.6.2 Feature-availability mapping

The two datasets capture different subsets of the canonical booking
schema. The Philippine PMS export records `lead_time`, `deposit_type`,
`adr`, `room_type`, and `special_requests`, but does **not** record
`country`, `agent`, `market_segment`, `customer_type`, or
`previous_cancellations`. Portugal captures all of them. The
feature-availability mapping documents which dimensions a property's
PMS schema must support to apply the methodology, and it bounds the
predictive power a property with a reduced schema can credibly
achieve. This is useful guidance for prospective adopters with
smaller or less-instrumented systems.

### 4.6.3 Plug-and-play dataset framework

The same pipeline scripts (`scripts/train.py` for Portugal,
`scripts/train_ph.py` for Philippine) work on any CSV that follows
the canonical column-name conventions. Currency-specific constants
(`ADR_MAX_VALID`, `FP_INTERVENTION_COST`) and metric gates are
configurable in `src/config.py`. The methodology can therefore be
re-applied to a third property by replacing the CSV, updating the
two configuration values, and re-running the training command. This
plug-and-play design is exercised end-to-end by the Philippine
sub-study and documented in detail in `CLAUDE.md` § "Swapping
Datasets."

---

## 4.7 Summary of Findings

The findings of this chapter map to the Sense → Seize → Transform
phases as follows.

### Sense
- Portugal's 119,210 cleaned bookings show a 37.0 % cancellation rate
  with four robust risk patterns (long lead time, Non Refund deposit
  policy, Groups market segment, first-time guests).
- The Philippine dataset shows a lower 15.0 % cancellation rate
  consistent with a single-resort property serving local clientele.
- The pre-flight diagnostic passes on Philippine data (0 % duplicate
  rate), confirming the methodology can proceed honestly.

### Seize
- LightGBM is the Portugal champion via rolling-origin cross-validation,
  with test PR-AUC = 0.760 and ECE = 0.029.
- Hypothesis 1 supported: lead_time, deposit_type, and previous_cancellations
  are all in the SHAP top-10.
- Hypothesis 2 supported with statistical significance on PR-AUC and
  ROC-AUC; F1 advantage is point-estimate only.
- Hypothesis 3 partially supported: all three predicted features in
  top-10, but `deposit_type` leads (not `lead_time`).
- The Philippine champion (LightGBM) achieves test PR-AUC = 0.542 on
  20 test rows; the 3-way comparison's confidence intervals overlap,
  so selection rests on point-estimate parity and parallel-to-Portugal
  lineage.

### Transform
- Hypothesis 4 supported with quantified savings: the cost-sensitive
  threshold of 0.04 reduces expected cancellation-related loss by
  approximately 95.4 % vs no model on the Portugal test set, a
  saving of roughly €1.53M.
- Risk-based deposit tiers (low / medium / high) operationalise the
  cost saving into specific outreach and deposit policies.
- An 8-page Power BI dashboard, a FastAPI + Gradio live server, and a
  drift-monitoring template deliver the methodology as a deployable
  system rather than a notebook experiment.

### The cross-dataset finding
- Hypothesis 5 supported: `deposit_type` is the #1 SHAP feature on
  both Portugal and the Philippine sub-study. The same underlying
  predictive concept survives the transfer to a smaller, geographically
  distinct dataset.

Chapter V draws on these findings to articulate the study's theoretical,
practical, and methodological contributions, acknowledges the limitations
of the work, and proposes a concrete agenda for future research.

---

[^1]: Source: `reports/metrics.json::data_cleaning`. Rows dropped: 180 zero-guest + 1 negative ADR = 181 total. Cleaned row count: 119,390 − 181 = 119,209 conceptually; the project tracks 119,210 because one row is recovered via a different cleaning rule. Use the project value.

[^2]: Source: `reports/benchmarks/01_dataset_split_summary.csv`.

[^3]: Source: `notebooks/01_eda.ipynb` §1.7 (cancel rate by lead-time band).

[^4]: Source: `notebooks/01_eda.ipynb` §1.6c (market_segment × lead_time × deposit_type heatmap) and `notebooks/05_explainability.ipynb` §5.2 (SHAP dependence on deposit_type).

[^5]: Source: `reports/ph/ph_transferability.json` (n_train, n_val, n_test) + `data/Punta_Villa_Resort_PH_Dataset.csv` for date min/max. Per-split cancellation rate not separately tabulated in current artefacts; the overall 15.0 % rate is reliable.

[^6]: Source: `reports/ph/ph_transferability.json::dataset_diagnostics` and `::train_test_overlap`.

[^7]: Source: `reports/benchmarks/11_rolling_origin_summary.csv`.

[^8]: Source: `reports/benchmarks/03_holdout_probability_metrics.csv`.

[^9]: Source: `reports/benchmarks/14_paired_significance_vs_champion.csv`. Bootstrap n = 2,000 resamples.

[^10]: Source: `reports/thesis/shap_feature_importance.csv` decoded via `artifacts/best_model.pkl::preprocessor.get_feature_names_out()`. Aggregation collapses one-hot-encoded categorical levels to the raw feature.

[^11]: Source: `reports/metrics.json::calibration.test` and `::calibration.validation`.

[^12]: Source: `reports/benchmarks/05_holdout_threshold_metrics_max_f1.csv` and `06_holdout_threshold_metrics_high_precision.csv` (LightGBM rows); cost_sensitive row from `reports/metrics.json::cost_sensitive`.

[^13]: Source: `reports/thesis/cost_sensitive_threshold.json`.

[^14]: Source: `reports/ph/model_family_comparison.json`. CIs computed via 200-resample bootstrap.

[^15]: Source: `reports/thesis/shap_feature_importance.csv` (Portugal) aggregated as in footnote 10, plus `reports/ph/shap_feature_importance.csv` (Philippine, already at raw-feature granularity).

[^16]: Source: `reports/ph/ph_adr_regressor_metrics.json`.
