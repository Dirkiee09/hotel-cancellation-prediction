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
4. **What do all these results mean for hotel revenue and booking strategy?** (Section 4.6)

Section 4.2 first restates how the data was cleaned and split, so the
numbers in later sections can be traced back to a known dataset state.

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
under identical preprocessing and evaluated on the same held-out test
set. Threshold-dependent metrics use each model's own validation-tuned
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
0.006 — small enough that it would be easy to dismiss as noise. We
re-sampled the test set 2,000 times using paired bootstrap resampling
to check, and the gap survives at p = 0.001 (see
`reports/benchmarks/14_paired_significance_vs_champion.csv`). Against
every other algorithm — Random Forest, Logistic Regression, XGBoost,
Decision Tree — LightGBM's lead is significant at p < 0.001. The
ranking is real, not lucky.

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
threshold (Section 4.6.2) treats a missed cancellation as far more
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

After isotonic regression calibration (fitted on the validation set
only), the model's predicted probabilities track observed
cancellation rates closely across the full 0–100 % range. The
Expected Calibration Error on the test set is 2.9 % — a "75 %"
prediction really does correspond to a 75–76 % observed cancellation
rate. The hotel can therefore treat the predicted number as a
trustworthy probability, not a black-box score.

**Business takeaway.** The model is honest about its own uncertainty.
That honesty is what lets Section 4.6 translate probabilities
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

**Table 4.3 — Top 10 features by mean(|SHAP|), Portugal champion**

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

---

## 4.6 Business Implications

This is the section that converts machine-learning outputs into
revenue-management decisions. The translation has two parts: how to
band bookings into action tiers (4.6.1), and how to pick the
threshold that decides what counts as "flagged" (4.6.2).

### 4.6.1 Risk tiers and revenue exposure

The hotel needs more than a single binary flag. A 99 %-cancel booking
and a 41 %-cancel booking are both "high risk" in a yes/no model, but
they call for very different operational responses. We therefore
partition predicted probabilities into three tiers:

- **Low** — probability < 0.40 — no action required.
- **Medium** — 0.40 ≤ probability < 0.70 — a 72-hour reminder email.
- **High** — probability ≥ 0.70 — a confirmation call and a partial
  deposit request.

**Table 4.4 — Risk tier distribution × revenue exposure
(Portugal test set, n = 11,922)**

| Risk Tier | Probability band | Bookings | % of total | Avg Revenue / Booking (€) | Total Revenue in Tier (€) | Actual Cancellations | Realised Revenue Lost (€) |
|---|---|---:|---:|---:|---:|---:|---:|
| Low | P < 0.40 | 6,107 | 51.22 % | 539.47 | 3,294,519 | 715 | 375,383 |
| Medium | 0.40–0.70 | 2,707 | 22.71 % | 706.14 | 1,911,521 | 1,435 | 1,066,905 |
| High | P ≥ 0.70 | 3,108 | 26.07 % | 650.56 | 2,021,931 | 2,356 | 1,571,978 |
| **Total** | — | **11,922** | **100.00 %** | **606.27** | **7,227,971** | **4,506** | **3,014,266** |

**[Insert Figure 4.6 —
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

### 4.6.2 Threshold policies — three operating points, three use cases

Choosing where to draw the line between "act" and "don't act"
depends on the cost asymmetry. A wasted reminder email costs €15. A
missed cancellation costs the full revenue of the booking, which
averages €430 across the test set. The model supports three operating
thresholds, each tuned for a different decision context.

**Table 4.5 — Threshold policy operational comparison (LightGBM, Portugal test set)**

| Policy | Threshold | Flagged | % Flagged | TP | FP | FN | Recall | Precision | Total Cost (€) | Savings vs No Model (€) | Use Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `max_f1` (balanced) | 0.40 | 5,815 | 48.78 % | 3,791 | 2,024 | 715 | 0.841 | 0.652 | 405,743 | 2,608,523 | Default weekly operations |
| `high_precision` | 0.98 | 426 | 3.57 % | 426 | 0 | 4,080 | 0.095 | 1.000 | 2,874,599 | 139,667 | Quarterly executive audit |
| **`cost_sensitive`** | 0.04 | 8,957 | 75.13 % | 4,486 | 4,471 | 20 | 0.996 | 0.501 | **76,512** | **2,937,754** | **Recommended deployment default** |
| No model (catch nothing) | — | 0 | 0.00 % | 0 | 0 | 4,506 | 0.000 | — | 3,014,266 | — | Baseline reference |

**[Insert Figure 4.7 —
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

### 4.6.3 Honest disclosure of what the numbers do and do not say

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

## 4.7 Chapter Summary

The chapter answered the four questions it opened with:

1. **Which model performed best?** LightGBM, with statistically
   significant lead over every challenger.
2. **Where does it get predictions right and wrong?** It catches
   84.1 % of cancellations at a 65.2 % precision rate using the
   default operating threshold. The probabilities are honestly
   calibrated.
3. **What features drive predictions?** Deposit type, country of
   origin, and booking agent — the model is learning *channel
   reliability*, not individual guest history.
4. **What does it mean for the hotel?** Concentrated risk (26 % of
   bookings carry 52 % of losses), three operating policies tuned to
   three use cases, and a cost-sensitive policy that recovers
   97.5 % of theoretical maximum revenue at risk.

Chapter V translates these findings into specific managerial
recommendations, states the study's limitations, and proposes future
research extensions.
