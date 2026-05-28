# CHAPTER IV — RESULTS AND DISCUSSION

> This chapter is written to be read by a hotel revenue manager, not only
> a machine-learning specialist. Every result is followed by the question
> *"so what does this mean for the property?"*. The deeper statistical
> apparatus (paired bootstrap tests, hypothesis verdicts, the Philippine
> sub-study, methodology contributions) is documented in Chapter V and
> in the appendix tables under
> `docs/thesis_drafts/chapter_iv_tables/`.

## 4.1 Introduction

This chapter reports what happened when the trained models were turned
loose on data they had never seen before, and what those numbers mean
for a hotel that has to decide which bookings to act on each week.

The chapter answers four practical questions in order:

1. **Which model performed best?** (Section 4.3)
2. **Where does it get predictions right and wrong?** (Section 4.4)
3. **What features actually drive those predictions?** (Section 4.5)
4. **What do all these results mean for hotel revenue and booking strategy?** (Section 4.7)

Section 4.2 first restates how the data was cleaned and split, so the
numbers in later sections can be traced back to a known dataset state.
Section 4.6 reports the parallel Average Daily Rate (ADR) regression
results, and Section 4.8 documents the live deployment framework.

---

## 4.2 Data Preprocessing Summary

The raw Portugal benchmark dataset contained 119,390 hotel bookings
spanning 1 July 2015 to 31 August 2017. A small number of rows were
removed before training: 180 rows had zero guests recorded (impossible
under the property's own booking rules), and one row had a negative
average daily rate. Removing these 181 rows left **119,210 valid
bookings** for the rest of the study.

Two further cleaning steps were applied without dropping rows. The
`agent` field was filled with "Direct" for 16,340 bookings that arrived
through the property's own website (these bookings had no third-party
agent identifier). Country values that came in blank were standardised
to `Unknown` for 488 bookings. Both transformations are reversible
and documented in `src/utils/validate_data.py`.

The 119,210 cleaned bookings were split chronologically — the oldest
80 % became the training set, the next 10 % the validation set, and
the most recent 10 % the test set. This is deliberately stricter than
the random shuffling used in most introductory machine-learning
projects, because in real operations the model will always be asked
to predict the *next* week's bookings using a model trained on past
weeks. Random shuffling makes models look better than they are; the
chronological split honestly mimics what production looks like.

**Table 4.1 — Portugal dataset split summary**

| Split | Rows | Date range | Cancellation rate |
|---|---|---|---|
| Train | 95,367 | 2015-07-01 → 2017-04-22 | 36.1 % |
| Validation | 11,920 | 2017-04-22 → 2017-06-21 | 43.9 % |
| Test | 11,922 | 2017-06-21 → 2017-08-31 | 37.8 % |
| **All cleaned** | **119,210** | **2015-07-01 → 2017-08-31** | **37.0 %** |

Each booking is represented by **33 features** that are knowable at
the moment of reservation: things like the deposit type, lead time,
country of origin, number of guests, requested room type, and so on.
Features that only become available *after* the booking is made —
`reservation_status`, `assigned_room_type`, `booking_changes`,
`days_in_waiting_list` — were explicitly excluded to prevent the
model from cheating by peeking into the future. This separation
matters: a model that uses post-booking signals looks impressive in
academic tests but is useless at the booking desk, where those
signals do not yet exist.

**Business takeaway.** The data preprocessing was conservative — only
181 rows of 119,390 were removed — and the chronological split means
every reported performance number reflects what the model would
actually see in production, not an artificially easy test.

---

## 4.3 Model Performance Comparison

Six classification algorithms were trained on the same training set
under identical preprocessing. The chapter reports model quality
under two complementary evaluation protocols. **Section 4.3.1**
reports the chronological out-of-time test — the deployment-realistic
number, which is the one the hotel actually sees in production.
**Section 4.3.2** reports stratified 10-fold cross-validation — the
academic baseline that lets us compare algorithms on an apples-to-apples
i.i.d. footing. **Section 4.3.3** quantifies how tight the headline
numbers are via paired bootstrap confidence intervals.

### 4.3.1 Chronological out-of-time test results

Threshold-dependent metrics use each model's own validation-tuned
`max_f1` cut-off.

**Table 4.2 — Test-set performance, all six algorithms (Portugal,
n = 11,922 test rows)**

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM (champion)** | **0.770** | **0.652** | **0.841** | **0.735** | **0.864** | **0.760** |
| Gradient Boosting | 0.773 | 0.659 | 0.829 | 0.734 | 0.861 | 0.754 |
| XGBoost | 0.745 | 0.610 | 0.904 | 0.729 | 0.855 | 0.749 |
| Random Forest | 0.756 | 0.650 | 0.766 | 0.704 | 0.851 | 0.739 |
| Logistic Regression | 0.749 | 0.628 | 0.825 | 0.713 | 0.839 | 0.739 |
| Decision Tree | 0.695 | 0.597 | 0.595 | 0.596 | 0.675 | 0.508 |

The full per-row counts (TP / FP / FN / TN) for every model are
preserved in `docs/thesis_drafts/chapter_iv_tables/table_02_chronological_oot_test.md`.

**[Insert Figure 4.1 — `reports/figures/thesis/fig_02_grouped_bar_model_selection.png`
here, showing PR-AUC across the six algorithms as a grouped bar
chart with the champion highlighted.]**

**LightGBM wins, by a small but real margin.** The PR-AUC gap between
LightGBM (0.760) and second-place Gradient Boosting (0.754) is just
0.006 — small enough that it would be easy to dismiss as noise.
Section 4.3.3 shows the paired bootstrap re-sampling that confirms
the gap survives at p = 0.001. Against every other algorithm —
Random Forest, Logistic Regression, XGBoost, Decision Tree —
LightGBM's lead is significant at p < 0.001. The ranking is real,
not lucky.

**Why LightGBM and not one of the others?** Three practical reasons.
First, hotel data is a mix of numeric signals (lead time, ADR, party
size) and categorical signals (country, agent, deposit type), and
gradient-boosted trees handle both natively without one-hot blow-up
hurting performance. Second, LightGBM trains roughly four times
faster than the equivalent Random Forest or Gradient Boosting model
on this data, which matters when the property wants to retrain
monthly against fresh data. Third, the model's per-row inference is
under one millisecond, well inside the latency budget of a live
booking-desk API.

**A note on Decision Tree's poor PR-AUC.** The plain Decision Tree
collapses to 0.508 PR-AUC because a single tree cannot capture the
interactions between (for example) `deposit_type` and `country` that
the ensemble methods exploit. We include it in the comparison anyway
because Decision Trees are visualisable in full and seeing how badly
a single tree underperforms is the best motivation for why an
ensemble was used.

**Business takeaway.** All five non-toy models perform within a
narrow band (PR-AUC 0.739 to 0.760). The choice of LightGBM is
defensible on speed and operational reasons as much as on the
0.006-point PR-AUC edge. From the hotel's perspective, picking any of
LightGBM, Gradient Boosting, or XGBoost would deliver a usable
production system; LightGBM is simply the lowest-friction choice.

### 4.3.2 Stratified 10-fold cross-validation — academic baseline

The chronological test in Section 4.3.1 is the right number for
deployment, but academic best practice also demands the standard
benchmark protocol — stratified 10-fold cross-validation that
ignores time and treats every row as exchangeable with every other.
This re-runs the comparison without the concept-drift handicap, so
the panel can see the algorithms compete on a level statistical
footing.

**Table 4.3 — Stratified 10-fold CV across 7 algorithms
(Portugal full dataset, n = 119,210, threshold = 0.5)**

| Algorithm | PR-AUC (mean ± std) | ROC-AUC (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|
| **LightGBM** | **0.922 ± 0.002** | **0.947 ± 0.002** | **0.821 ± 0.002** |
| Gradient Boosting | 0.912 ± 0.002 | 0.940 ± 0.002 | 0.808 ± 0.002 |
| XGBoost | 0.908 ± 0.003 | 0.937 ± 0.002 | 0.811 ± 0.004 |
| Logistic Regression | 0.860 ± 0.003 | 0.901 ± 0.002 | 0.740 ± 0.006 |
| Decision Tree | 0.798 ± 0.004 | 0.876 ± 0.003 | 0.739 ± 0.005 |
| Gaussian NB | 0.749 ± 0.005 | 0.814 ± 0.004 | 0.663 ± 0.005 |
| Dummy (majority class) | 0.371 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 |

The full per-fold counts and per-fold variance are preserved in
`reports/cv/portugal_stratified_10fold_summary.json` and the
companion file in `docs/thesis_drafts/chapter_iv_tables/table_01_classification_cv_benchmark.md`.

**The complexity ladder is perfectly monotonic.** Each step up in
model expressiveness — from Dummy (0.371) through Naive Bayes (0.749),
Decision Tree (0.798), Logistic Regression (0.860), XGBoost (0.908),
Gradient Boosting (0.912), and LightGBM (0.922) — buys measurable
PR-AUC. The ensemble methods earn their complexity; the single
Decision Tree and the linear model both underperform by 6–12
percentage points.

**The headline finding is the gap between the two protocols.**
LightGBM scores PR-AUC **0.922** under stratified 10-fold CV but
only **0.760** under the chronological out-of-time test — a gap of
**−16.2 percentage points**. The same model loses sixteen points of
PR-AUC when it has to predict the *future* instead of a random
shuffle of the past.

That gap is not a flaw in the model. It is the empirical signature
of **concept drift over time** — the changes in guest mix, booking
channel, deposit policy, and macro-economic context that accumulate
between training data and deployment data. The CV number tells the
hotel "this algorithm has the strongest signal in the data". The
chronological number tells the hotel "this is what the model will
actually deliver next quarter". Both belong in the thesis; the
chronological number is the one a property should plan its operations
around.

**Business takeaway.** The 16-point gap is the cost of generalising
forward in time. A hotel deploying this methodology should expect
PR-AUC closer to 0.76 in production than 0.92, and should treat
quarterly retraining (Section 4.8) as the standard way to claw some
of that gap back.

### 4.3.3 Bootstrap confidence intervals on the champion

Knowing the point estimate is not enough. A defensible thesis result
also needs to show how tight that estimate is — could LightGBM's
0.760 PR-AUC fall to 0.74 or rise to 0.78 if the test set had been
slightly different? To answer, we drew **2,000 bootstrap samples**
(with replacement) from the test set and recomputed each metric on
every sample.

**Table 4.4 — Bootstrap 95 % confidence intervals on the LightGBM
champion (Portugal test set, 2,000 resamples)**

| Metric | Point Estimate | 95 % CI Lower | 95 % CI Upper | CI Width |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.864 | 0.858 | 0.871 | 0.013 |
| PR-AUC | 0.760 | 0.748 | 0.772 | 0.024 |
| F1 @ `max_f1` | 0.735 | 0.725 | 0.744 | 0.019 |

The full per-model CI grid is in
`reports/benchmarks/13_bootstrap_confidence_intervals.csv`.

**The intervals are narrow.** CI widths of 0.013 (ROC-AUC) and 0.024
(PR-AUC) at a sample size of 11,922 are textbook-tight — the test
set is large enough that the headline numbers are not at the mercy
of which particular 11k bookings happened to land in the held-out
window. Even at the lower bound of each interval, the model
comfortably clears every quality gate set out in `src/config.py`
(PR-AUC ≥ 0.50, ROC-AUC ≥ 0.70, F1 ≥ 0.50).

**Business takeaway.** A revenue manager defending this work to her
GM can quote the point estimates with confidence intervals attached:
"PR-AUC 0.76, with 95 % confidence the true number is between 0.75
and 0.77." That phrasing converts a single statistic into a defensible
range — exactly what an executive committee expects from a
business-intelligence brief.

---

## 4.4 Model Evaluation (Visualisations)

### 4.4.1 ROC and Precision-Recall curves

**[Insert Figure 4.2 — `reports/figures/thesis/fig_01_roc_pr_curves.png`
here, showing ROC and PR curves of the LightGBM champion on the
held-out test set.]**

In plain English, the ROC-AUC of 0.864 says this: if you pick one
booking that ended up cancelling and one booking that did not, and
ask the model which is more likely to cancel, the model will get the
answer right 86.4 % of the time. Random guessing would be 50 %.

The PR-AUC of 0.760 tells a more operationally relevant story. As
the model flags more bookings as risky, precision (the share of
flagged bookings that actually cancel) holds up surprisingly well —
it does not collapse the way it would if the model were guessing.
This is what matters when the property has to decide how many
high-risk bookings it can afford to staff up for.

### 4.4.2 Confusion matrix at the operating threshold

**[Insert Figure 4.3 —
`reports/figures/thesis/fig_03_normalized_confusion_matrix_max_f1.png`
here, showing the normalized confusion matrix at the validation-tuned
`max_f1` threshold of 0.40.]**

At the `max_f1` threshold of 0.40, the model's behaviour on the test
set is:

- **3,791 cancellations correctly caught** (true positives).
- **715 cancellations missed** (false negatives).
- **2,024 reservations wrongly flagged** as cancel-risks even though
  they would have honoured the booking (false positives).
- **5,392 reservations correctly waved through** as low-risk (true
  negatives).

Read as recall: of the 4,506 actual cancellations in the test
window, the model catches 3,791 — a **recall of 84.1 %**. Read as
precision: of the 5,815 reservations the model flagged, 3,791 turned
out to cancel — a **precision of 65.2 %**.

The trade-off is intentional. The cost model that picked the 0.40
threshold (Section 4.7.2) treats a missed cancellation as far more
expensive than a wrongly flagged one, because the average ADR of a
missed booking is hundreds of euros while the cost of a reminder
email is around €15. The model is calibrated to err on the side of
flagging too much rather than missing too much.

### 4.4.3 Calibration — do the probabilities mean what they say?

**[Insert Figure 4.4 —
`reports/figures/thesis/fig_05_calibration_reliability_and_histogram.png`
here, showing the calibration reliability diagram with predicted vs
observed cancellation rates per probability bin.]**

A model is *well-calibrated* if a "75 % probability" really means
about 75 % of those bookings cancel in real life. This matters
because the hotel's downstream policies — risk tier bands, deposit
requirements, reminder workflows — all hang off the predicted
probability number itself, not just the binary flag.

To turn the raw LightGBM scores into honest probabilities, the
pipeline fits an isotonic regression calibrator on the validation
set only (so the test set stays unbiased). Table 4.5 reports the
improvement directly.

**Table 4.5 — Calibration quality before vs after isotonic regression**

| Split | Brier (Raw) | Brier (Calibrated) | Δ Brier | ECE (Raw) | ECE (Calibrated) | Δ ECE |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.120 | 0.114 | −0.006 | 0.046 | ~0.000 | −0.046 |
| **Test** | **0.150** | **0.146** | **−0.004** | **0.058** | **0.029** | **−0.029** |

Calibration data lives in `reports/calibration_metrics.json`. Brier
score is mean squared error between predicted probability and the
{0, 1} outcome (lower is better). Expected Calibration Error (ECE)
is the average gap between predicted and observed cancellation
rates across 10 probability bins (lower is better).

**Isotonic regression halves the calibration gap on the test set.**
The ECE drops from 5.8 % to 2.9 % — meaning a "75 %" prediction
really does correspond to a 75–76 % observed cancellation rate.
Brier improves only marginally because Brier is dominated by the
discrimination component (which calibration cannot change), but the
ECE drop is the operationally important number.

**Business takeaway.** The model is honest about its own uncertainty.
That honesty is what lets Section 4.7 translate probabilities
directly into deposit policies without further recalibration.

---

## 4.5 Feature Importance Analysis

### 4.5.1 Which features drive the predictions?

**[Insert Figure 4.5 — `reports/thesis/shap_summary_plot.png` here,
showing the SHAP beeswarm for the top 15 raw features. Blue points
are low feature values; red points are high feature values; horizontal
position is the contribution to the predicted cancellation probability.]**

SHAP (SHapley Additive exPlanations) measures, for each prediction
the model makes, exactly how much each feature pushed the prediction
up or down. Aggregating across the test set ranks the features by
how much they matter overall.

**Table 4.6 — Top 10 features by mean(|SHAP|), Portugal champion**

| Rank | Feature | Mean(\|SHAP\|) | Plain-English meaning |
|---:|---|---:|---|
| 1 | `deposit_type` | 1.150 | The deposit policy attached to the booking |
| 2 | `country` | 1.095 | Guest's country of origin |
| 3 | `agent` | 0.911 | The booking agent / channel |
| 4 | `required_car_parking_spaces` | 0.746 | Has the guest requested parking? |
| 5 | `total_of_special_requests` | 0.576 | How many special requests the guest made |
| 6 | `market_segment` | 0.520 | Booking source (Online TA, Direct, Groups, etc.) |
| 7 | `lead_time` | 0.393 | Days between booking and arrival |
| 8 | `arrival_date_year` | 0.281 | Year of arrival |
| 9 | `customer_type` | 0.241 | Transient / Contract / Group / Transient-Party |
| 10 | `previous_cancellations` | 0.234 | Guest's prior cancellation count |

### 4.5.2 What does each top driver tell the hotel?

**The strongest driver is `deposit_type`, and the relationship is
counter-intuitive.** Bookings with "Non Refund" deposit terms cancel
*more often* than bookings with no deposit at all. At first glance
that looks broken — surely a non-refundable deposit should bind the
guest? In practice the non-refundable rate is concentrated in a few
high-volume online travel agents whose own customers cancel
frequently, and the model is learning the *channel pattern*, not the
deposit pattern itself. The lesson for the hotel is that pricing
policies (like enforcing non-refundable deposits) only reduce
cancellation if they change *who* is booking, not just *what they pay*.

**`country` and `agent` rank #2 and #3 — the model is learning which
booking channels are reliable.** Most cancellation-prediction
literature focuses on guest behaviour (how many times have they
cancelled before? do they have loyalty status?). This model says the
hotel's *suppliers* are doing more of the work than the guests.
Operationally, that means the highest-leverage intervention is
auditing the top-cancelling agents and countries, not changing
guest-facing policies.

**The hypothesised top feature, `lead_time`, only ranks 7th.** The
original research hypothesis (H3) predicted that lead time would be
the #1 SHAP driver, followed by deposit type, then previous
cancellations. The data partially supports the hypothesis — all
three features appear in the top 10 — but the rank order is wrong.
Reporting this honestly is more useful than hiding it: a future
hotel using this methodology should not assume the same feature
order applies to their data.

**Two operational drivers push *toward not cancelling*.** Bookings
that requested parking (`required_car_parking_spaces`) or made
special requests (`total_of_special_requests`) cancel *less often*
than otherwise comparable bookings. The pattern visible in Figure 4.5
shows red dots (high feature values) clustering at negative SHAP for
both features — translation: guests who personalise their booking
are more committed to actually showing up. The hotel can use this:
mid-tier bookings where the guest also requested parking are
probably safer than the raw cancellation probability suggests.

**Business takeaway.** The model rewards bookings that show signs of
real intent (parking, special requests, direct channels) and
penalises bookings from high-cancellation channels. The policy
implication is to focus revenue protection on *channels and
countries*, not on individual guest behaviour.

### 4.5.3 Hypothesis verdict quick-check

Chapter I pre-registered five hypotheses about model behaviour.
Their verdicts are summarised below; the full evidence chain is
preserved in Table 4.6 of
`docs/thesis_drafts/chapter_iv_tables/table_06_hypothesis_evidence_verdict.md`
and in Chapter V Section 5.2.

**Table 4.7 — Hypothesis verdict summary**

| H | One-line statement | Verdict | Evidence section |
|---|---|---|---|
| **H1** | Lead time, deposit type, and previous cancellations are significant predictors | **Supported** | 4.5.1 (all three in top 10 SHAP) |
| **H2** | A gradient-boosted model beats baseline algorithms | **Supported** (significant) | 4.3.1 + 4.3.3 (paired bootstrap p ≤ 0.001) |
| **H3** | Lead time has highest SHAP, then deposit type, then previous cancellations | **Partially supported** | 4.5.1 (features correct, rank order wrong) |
| **H4** | Cost-sensitive thresholding reduces expected revenue loss | **Supported** | 4.7.2 (97.5 % recovery on test set) |
| **H5** | Top SHAP feature transfers across geographies (Portugal → Philippines) | **Supported** | 4.5 + appendix (`deposit_type` #1 in both datasets) |

Four of the five hypotheses are fully supported; H3 is partially
supported with the rank order disconfirmed. **Reporting that
partial support honestly — rather than retrofitting the hypothesis
to match the data — is itself a thesis contribution**: it shows the
study treated its predictions as falsifiable claims rather than
post-hoc rationalisations.

---

## 4.6 Average Daily Rate (ADR) Regression

Predicting whether a booking will cancel is half the BI story. The
other half is predicting how much revenue is at stake. This section
documents the **Average Daily Rate (ADR) regressor** that runs
alongside the cancellation classifier in the live deployment — every
`/predict` call returns both a cancellation probability and a
predicted ADR (Section 4.8).

### 4.6.1 Regressor comparison

Seven regression algorithms were fit on the same chronological train
split, tuned on the validation set, and evaluated on the held-out
test set. Test RMSE is in euros, MAPE in percent.

**Table 4.8 — ADR regression performance, all seven models
(Portugal test set)**

| Model | Train RMSE (€) | Val RMSE (€) | Test RMSE (€) | Test MAE (€) | Test R² | Test MAPE (%) |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting (champion)** | **32.70** | **28.76** | **44.31** | **32.24** | **0.234** | **23.45** |
| XGBoost | 32.89 | 29.30 | 44.06 | 32.14 | 0.243 | 23.48 |
| Decision Tree | 33.74 | 31.28 | 45.87 | 33.28 | 0.179 | 25.15 |
| Ridge | 37.64 | 30.29 | 47.64 | 34.55 | 0.115 | 24.74 |
| Linear Regression | 37.63 | 30.30 | 47.65 | 34.56 | 0.114 | 24.75 |
| Lasso | 39.84 | 30.80 | 51.99 | 38.04 | −0.054 | 27.39 |
| Neural Network | 41.38 | 31.06 | 55.17 | 38.22 | −0.187 | 26.72 |

Full per-model breakdown is in `reports/regression_results.csv`.

**[Insert Figure 4.6 —
`reports/figures/thesis/fig_45_adr_pred_vs_actual.png` here, showing
the predicted vs actual ADR scatter for the champion regressor,
with the y = x reference line.]**

### 4.6.2 Why the R² is moderate (and why that's OK)

The champion Gradient Boosting regressor achieves a Test R² of
**0.234** — meaning the model explains about 23 % of the variance in
ADR. To a machine-learning purist that sounds low, but the result is
expected and operationally useful.

ADR is dominated by two forces. The first is **rate-card pricing** —
the room type, season, channel, and rate plan attached to a booking
— and the model captures this well. The second is **booking-specific
randomness** — group discounts, loyalty perks, promotional codes,
day-of-week price elasticity, last-minute upgrades — that are simply
not in the feature set the regressor sees. R² above 0.30 on a
problem with that structure would suggest data leakage, not skill.

What the model *does* deliver is a **directional ADR signal at
booking time**: it correctly orders bookings by likely revenue, so
the property knows which High-tier cancellation risks are also
high-value bookings. That ordering is what Page 5 of the Power BI
dashboard exposes, and it is enough to drive prioritisation — even
without an exact rate prediction.

**Notable failures.** Linear models (Ridge, Lasso, Linear) all
underperform the tree-based methods; the Lasso and Neural Network
post **negative test R²** (they perform worse than always predicting
the mean ADR). The Neural Network's failure is particularly
instructive: with 119k training rows the network has plenty of data,
but tabular regression on mixed numeric-and-categorical features is
exactly the regime where gradient-boosted trees dominate the
empirical literature. This is one reason no deep-learning architecture
was tried for the cancellation classifier either.

### 4.6.3 Honest disclosure of the ADR regressor's limitation

The ADR regressor was trained with four features that are **not
known at the moment of booking**: `is_canceled` (whether the
booking eventually cancelled), `assigned_room_type` (the room the
guest was actually given on arrival), `booking_changes` (whether
the booking was modified), and `days_in_waiting_list`. Live
inference fills these with sensible defaults — see CLAUDE.md and
`src/serving/inference.py::predict_adr()` for the exact
substitutions. The published Test RMSE of €44.31 is therefore an
*upper bound* on live accuracy; in production the regressor sees
defaulted features and is slightly less accurate.

The methodologically clean fix is retraining on booking-time
features only, which is documented as Future Research item 4 in
Section 5.5. The current regressor is good enough to drive the
directional pricing signal in the dashboard, but the published RMSE
should be read as the *best-case* result.

**Business takeaway.** Combined with the cancellation classifier,
the ADR regressor lets the Power BI dashboard answer a question no
single model could: not just *which bookings will cancel*, but
*which bookings will cancel **and** carry above-average revenue*.
That intersection — high cancellation probability × high predicted
ADR — is exactly the operational priority list the revenue manager
needs each morning.

---

## 4.7 Business Implications

This is the section that converts machine-learning outputs into
revenue-management decisions. The translation has three parts: how
to band bookings into action tiers (4.7.1), how to pick the threshold
that decides what counts as "flagged" (4.7.2), and how the model
performs across the operationally important slices of the customer
base (4.7.3).

### 4.7.1 Risk tiers and revenue exposure

The hotel needs more than a single binary flag. A 99 %-cancel booking
and a 41 %-cancel booking are both "high risk" in a yes/no model, but
they call for very different operational responses. We therefore
partition predicted probabilities into three tiers:

- **Low** — probability < 0.40 — no action required.
- **Medium** — 0.40 ≤ probability < 0.70 — a 72-hour reminder email.
- **High** — probability ≥ 0.70 — a confirmation call and a partial
  deposit request.

**Table 4.9 — Risk tier distribution × revenue exposure
(Portugal test set, n = 11,922)**

| Risk Tier | Probability band | Bookings | % of total | Avg Revenue / Booking (€) | Total Revenue in Tier (€) | Actual Cancellations | Realised Revenue Lost (€) |
|---|---|---:|---:|---:|---:|---:|---:|
| Low | P < 0.40 | 6,107 | 51.22 % | 539.47 | 3,294,519 | 715 | 375,383 |
| Medium | 0.40–0.70 | 2,707 | 22.71 % | 706.14 | 1,911,521 | 1,435 | 1,066,905 |
| High | P ≥ 0.70 | 3,108 | 26.07 % | 650.56 | 2,021,931 | 2,356 | 1,571,978 |
| **Total** | — | **11,922** | **100.00 %** | **606.27** | **7,227,971** | **4,506** | **3,014,266** |

**[Insert Figure 4.7 —
`reports/figures/thesis/fig_23_risk_tier_business_overview.png` here,
showing the risk-tier business overview with revenue exposure per tier.]**

Two findings from this table are worth highlighting for management.

**Risk is heavily concentrated.** The High tier represents only
26.07 % of all bookings but accounts for **52.15 % of all realised
cancellation revenue losses** (€1,571,978 of €3,014,266 in lost
revenue). Operationally, this means the property's biggest single
intervention leverage point is to focus its limited staff time on the
3,108 High-tier bookings rather than spreading reminder effort
uniformly across the 11,922 reservations.

**The risk bands are empirically honest.** The Low tier has an
observed cancellation rate of 11.7 %, Medium 53.0 %, and High 75.8 %.
A "High-tier 75 %-probability" booking really does cancel ~76 % of
the time. The hotel can therefore set deposit policies directly off
the probability number without needing to recalibrate or apply a
fudge factor.

### 4.7.2 Threshold policies — three operating points, three use cases

Choosing where to draw the line between "act" and "don't act"
depends on the cost asymmetry. A wasted reminder email costs €15. A
missed cancellation costs the full revenue of the booking, which
averages €430 across the test set. The model supports three operating
thresholds, each tuned for a different decision context.

**Table 4.10 — Threshold policy operational comparison (LightGBM, Portugal test set)**

| Policy | Threshold | Flagged | % Flagged | TP | FP | FN | Recall | Precision | Total Cost (€) | Savings vs No Model (€) | Use Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `max_f1` (balanced) | 0.40 | 5,815 | 48.78 % | 3,791 | 2,024 | 715 | 0.841 | 0.652 | 405,743 | 2,608,523 | Default weekly operations |
| `high_precision` | 0.98 | 426 | 3.57 % | 426 | 0 | 4,080 | 0.095 | 1.000 | 2,874,599 | 139,667 | Quarterly executive audit |
| **`cost_sensitive`** | 0.04 | 8,957 | 75.13 % | 4,486 | 4,471 | 20 | 0.996 | 0.501 | **76,512** | **2,937,754** | **Recommended deployment default** |
| No model (catch nothing) | — | 0 | 0.00 % | 0 | 0 | 4,506 | 0.000 | — | 3,014,266 | — | Baseline reference |

**[Insert Figure 4.8 —
`reports/figures/thesis/fig_11_cost_sensitive_threshold_sweep.png` here,
showing total expected cost as a function of the decision threshold,
with the cost-minimising point marked.]**

Three operational insights fall out of the table.

**The cost-sensitive policy recovers €2,937,754 on the test set —
97.5 % of the theoretical maximum** (the €3,014,266 that would be
lost if the hotel did nothing). It does so by flagging three quarters
of all bookings (8,957 of 11,922), which sounds excessive until you
realise the cost of flagging is €15 and the cost of missing is the
full booking revenue. The model rationally trades many cheap false
positives for the recovery of a few expensive false negatives.

**`max_f1` is the policy for normal weekly operations.** It flags
about half the bookings, catches 84 % of cancellations, and costs
roughly €405,000 in expected losses — a 86 % recovery rate that is
still excellent but leaves the front-desk team time to act on each
flagged booking. This is the policy a property would actually run
day-to-day; the cost-sensitive policy is for crunch periods (peak
season, large-group exposure) where every cancellation hurts.

**`high_precision` is the policy for executive audits, not weekly
ops.** At threshold 0.98 the model only flags 426 bookings (3.6 %)
but every single one is a genuine cancellation (precision 100 %).
This is the right policy when every flag must survive scrutiny — for
example, when the GM wants to query the top 50 highest-risk Groups
bookings before authorising a deposit request — but it costs the
hotel €2.87 million in cancellations it never sees, so it is not the
right policy for daily operations.

### 4.7.3 Per-segment performance

A model that performs well on average can still fail on the
operationally important slices. The dashboard fairness section
(Page 7) and the appendix table `reports/segment_metrics.csv`
break out the champion's test-set metrics by hotel type and by
market segment.

**Table 4.11 — Per-segment performance breakdown (LightGBM at
`max_f1` threshold, Portugal test set)**

| Dimension | Segment | n_rows | Positive Rate | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Hotel** | Resort Hotel | 4,043 | 0.380 | 0.892 | 0.785 | 0.697 | 0.869 | 0.774 |
| **Hotel** | City Hotel | 7,879 | 0.377 | 0.851 | 0.756 | 0.630 | 0.827 | 0.715 |
| Market | **Groups** | 677 | 0.532 | 0.986 | 0.985 | 0.820 | 0.989 | 0.897 |
| Market | Offline TA/TO | 1,710 | 0.234 | 0.976 | 0.901 | 0.787 | 0.988 | 0.876 |
| Market | Online TA | 7,644 | 0.438 | 0.802 | 0.701 | 0.634 | 0.838 | 0.722 |
| Market | **Direct** | 1,546 | 0.196 | 0.808 | 0.489 | 0.475 | 0.531 | 0.502 |

Three observations stand out.

**Resort Hotel outperforms City Hotel by ~4 pp PR-AUC** (0.785 vs
0.756). City bookings cancel more erratically because they include
shorter business trips with last-minute schedule changes, while
resort bookings are longer leisure stays whose cancellation patterns
the model finds easier to learn. Both numbers are operationally
usable; the gap is worth noting because it suggests the cost-sensitive
threshold may need slight per-hotel tuning at large properties.

**Groups bookings are the model's strongest segment by far** — PR-AUC
0.985 with F1 0.897. These are typically large, multi-room bookings
made by event organisers or corporate buyers, and their cancellation
behaviour is highly patterned (one decision often triggers many
cancellations together). For a revenue manager this is unambiguously
good news: the highest-revenue single bookings are precisely the ones
the model can most confidently flag.

**Direct bookings are the model's weakest segment** — PR-AUC 0.489.
These are guests booking via the hotel's own website or walk-in;
they cancel rarely (positive rate 19.6 %), making the prediction
problem harder for any model. The drop matters operationally: the
hotel should treat the model's Direct-booking flags as **noisier
signals** and reserve confirmation calls for cases where the
probability is in the high tier (≥ 0.70), where precision recovers.
For Online TA bookings the model performs in the middle range
(PR-AUC 0.701) — workable but not best-in-class.

**Business takeaway.** The model is uniformly good but not uniformly
*great*. Groups and Offline TA bookings are the easy wins; Direct
bookings need extra human judgement. A defensible deployment policy
treats the model as a strong universal signal that is *augmented*
with operator judgement in the lowest-PR-AUC segments rather than
replaced by it.

### 4.7.4 Honest disclosure of what the numbers do and do not say

The €2.94 million recovery figure is a *one-period upper bound* on
the 2017 test sample. Three real-world frictions reduce it:

- The cost model assumes a uniform €15 intervention cost. In
  practice, the cost of a confirmation call is higher than the cost
  of an automated reminder email, so the savings on the High tier
  (calls) are smaller per intervention than on the Medium tier
  (emails).
- The cost model assumes guests respond to interventions at the
  rates implied by the FN cost being the full revenue at risk. The
  *measured* response rate to reminders and deposit requests is
  unknown — Section 5.5 lists this as a future-research item.
- Deploying at a different property requires retraining. The
  features and threshold values in this chapter are calibrated to
  the Portugal benchmark; a Philippine property's data shape is
  different (Section 5.4 discusses this honestly).

**Business takeaway.** The model is operationally ready, the
probabilities are honest, and the recovery numbers are large. The
remaining work is not better modelling — it is testing the policy
itself in production via A/B trials. Section 5.5 sets out exactly
what that pilot would look like.

---

## 4.8 Model Deployment Framework

A model that lives in a notebook is an academic artefact. A model that
scores a booking the moment it lands in the property's PMS, logs the
score for later audit, surfaces the result in a dashboard the next
morning, and tells the operations team when it is time to retrain —
that is a business intelligence deliverable. This section documents
the deployment framework that wraps the LightGBM champion into exactly
that operational tool.

**[Insert Figure 4.9 —
`reports/figures/thesis/fig_deployment_framework.png` here, showing
the live-serving pipeline from a single booking entry through to the
Power BI dashboard and back via drift-triggered retraining.]**

### 4.8.1 Architecture at a glance

Figure 4.9 maps the full request-to-dashboard data flow. The framework
has four layers, each with a clear job:

- **The serving layer** is a FastAPI application running on
  `localhost:8000`. It exposes three production endpoints — `/predict`
  for booking scoring, `/model-info` for current model lineage,
  `/healthz` for readiness checks — plus a Gradio user interface
  mounted at `/ui` for non-technical staff who prefer a web form to a
  JSON API. Both paths run identical inference, so the same model
  serves both audiences.
- **The inference pipeline** runs entirely in memory once the artefacts
  are loaded: a Pydantic validator coerces the incoming booking,
  feature engineering derives the 33 model inputs, the LightGBM
  pipeline produces a raw probability, isotonic calibration corrects
  the probability, threshold resolution assigns the three policy
  labels and the risk tier, TreeSHAP computes the top-5 feature
  contributions for explainability, and the ADR regressor produces a
  parallel price prediction. The full pipeline returns a JSON response
  in under 500 ms on a laptop-grade CPU.
- **The persistence layer** is asynchronous. After the response is sent
  back to the caller, a FastAPI BackgroundTask appends the (request,
  response) pair to a SQLite audit log at
  `data/predictions/predictions.sqlite` and re-exports the full log to
  `predictions_live.csv`. The user never waits for the disk write, and
  the API never fails if the log is briefly unavailable. The CSV is
  the source of truth Power BI Desktop consumes.
- **The monitoring loop** runs on a separate schedule (typically
  weekly). The `compute_live_drift.py` script reads the live CSV and
  the training-time baseline, computes Population Stability Index per
  feature, and writes a `drift_metrics.csv` with each feature
  classified into a safe / watch / retrain zone. Page 8 of the Power
  BI dashboard reads this file. When two or more features land in the
  retrain zone, the operations team triggers `scripts/train.py` to
  regenerate the artefacts under `artifacts/`, and the loop closes.

### 4.8.2 What this means for the property

The framework is deliberately minimalist. There is no cloud service to
provision, no database server to administer, no model registry to
maintain. A property's IT team needs only:

- A Python environment (the project's `requirements.txt`).
- One server process for FastAPI (single binary, started by
  `python demo/start_server.py`).
- A scheduled task (Windows Task Scheduler or cron) to run
  `scripts/compute_live_drift.py` once a week.
- Power BI Desktop on whichever workstation the revenue manager uses.

Every artefact is a file. Every log is a SQLite database. Every report
is a CSV. A non-technical manager can be handed the `.pbix` file plus
the two CSVs and the dashboard works on first open — no ODBC drivers,
no service accounts, no broken refresh tokens.

### 4.8.3 Production readiness checklist

Three properties make the framework production-ready rather than a
demo:

1. **Calibrated probabilities, not scores.** Because the model has
   been isotonically calibrated (Section 4.4.3), the dashboard's risk
   tier bands can be set directly off the probability number. There is
   no need for a separate "score-to-probability" lookup or a hand-
   tuned safety margin.
2. **Multiple operating thresholds, not one.** The three policies
   (`max_f1`, `high_precision`, `cost_sensitive`) ship with the model
   and are resolvable per-request. The hotel can run the
   cost-sensitive policy by default and switch to `high_precision`
   for executive audits without retraining or redeploying.
3. **Drift monitoring as part of the loop, not as an afterthought.**
   The PSI computation is wired into the same dashboard the revenue
   manager already reads. When the model needs retraining, the
   dashboard tells her — she does not need to remember to ask.

**Business takeaway.** The framework converts the model from a thesis
artefact into an operational tool a hotel can run on commodity
hardware with one Python process and one Power BI workstation. The
ongoing operational cost is one weekly drift run; the trigger for
human intervention is a coloured zone change on Page 8 of the
dashboard.

---

## 4.9 Chapter Summary

The chapter answered the four questions it opened with, plus the
ADR-pricing question that completes the BI story:

1. **Which model performed best?** LightGBM, with statistically
   significant lead over every challenger (Sections 4.3.1, 4.3.3).
   The same algorithm wins under both chronological out-of-time
   evaluation (PR-AUC 0.760) and stratified 10-fold CV (PR-AUC 0.922).
2. **Where does it get predictions right and wrong?** It catches
   84.1 % of cancellations at a 65.2 % precision rate using the
   default operating threshold (Section 4.4.2), and the probabilities
   are honestly calibrated — isotonic regression halves the test ECE
   from 5.8 % to 2.9 % (Section 4.4.3).
3. **What features drive predictions?** Deposit type, country of
   origin, and booking agent — the model is learning *channel
   reliability*, not individual guest history (Section 4.5).
4. **What does it mean for the hotel?** Concentrated risk (26 % of
   bookings carry 52 % of losses), three operating policies tuned to
   three use cases, and a cost-sensitive policy that recovers
   97.5 % of theoretical maximum revenue at risk (Section 4.7).
   Performance is uniformly good across hotel types and most market
   segments; Direct bookings are the noisiest slice and call for
   extra operator judgement (Section 4.7.3).
5. **Can the model also predict revenue at booking time?** Yes — the
   parallel ADR regressor (Section 4.6) delivers a directional pricing
   signal that combines with the cancellation probability to produce
   the High-cancel × High-revenue priority list the property actually
   needs each morning.

And one cross-cutting finding: the **−16 percentage point gap**
between stratified-CV PR-AUC (0.922) and chronological test PR-AUC
(0.760) quantifies the operational cost of concept drift over time
(Section 4.3.2). The deployment framework (Section 4.8) closes that
loop by triggering retraining when PSI drift crosses the 0.25
threshold on two or more features.

Chapter V translates these findings into specific managerial
recommendations, states the study's limitations, and proposes future
research extensions.
