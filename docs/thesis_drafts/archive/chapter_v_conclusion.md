# CHAPTER V — CONCLUSION AND RECOMMENDATIONS

> This chapter summarises what the study found, what those findings
> mean for hotel managers in practice, where the work is limited, and
> what future research should add. It is deliberately written in plain
> language so a non-technical reader can act on it without having to
> re-read Chapter IV.

## 5.1 Summary of the Study

Hotel cancellations are expensive. On the Portugal benchmark used in
this study, **€3.01 million of room revenue** was lost to cancelled
bookings across just the 2017 test window — money that walks out the
door before the guest ever arrives. The question the study set out to
answer was simple: *can we tell, at the moment a booking is made,
which ones are likely to cancel — and use that information to act
before the loss happens?*

The approach was machine learning, but the test was operational. Six
algorithms — LightGBM, XGBoost, Gradient Boosting, Random Forest,
Logistic Regression, and a baseline Decision Tree — were trained on
**119,210 cleaned bookings** under a strict chronological split
(oldest 80 % for training, next 10 % for validation, most recent 10 %
held out for testing). Each model's predicted probabilities were
calibrated using isotonic regression so that a "75 % probability"
really means about 75 % of those bookings cancel in real life. The
chosen operating threshold was tuned to minimise the *cost* of wrong
decisions, not just statistical error: missing a cancellation costs
the full revenue of the booking, while flagging one in error costs
about €15 for an automated reminder.

The headline result is that **LightGBM with cost-sensitive
thresholding recovers 97.5 % of the theoretical maximum revenue at
risk** on the test set — €2.94 million of €3.01 million — and the
predictions are honest enough to be used directly as the basis for
deposit and reminder policies. The model is already wired up to a
live FastAPI + Gradio booking-desk interface and feeds an eight-page
Power BI decision-support dashboard, so the operational delivery is
not theoretical.

---

## 5.2 Key Findings

Five findings stand out from Chapter IV.

**1. LightGBM is the best performer, and the lead is statistically
real.** LightGBM achieves a ROC-AUC of 0.864, a PR-AUC of 0.760, and
an F1 score of 0.735 on the chronological out-of-time test set. Paired
bootstrap resampling (2,000 resamples) confirms the lead is
statistically significant against every competing algorithm — p < 0.001
against Random Forest, Logistic Regression, XGBoost, and Decision Tree,
and p = 0.001 against the closest challenger, Gradient Boosting. The
ranking is not a fluke.

**2. The strongest predictor is `deposit_type`, not `lead_time` as
hypothesised.** The original Chapter I hypothesis predicted that
booking-to-arrival lead time would dominate the SHAP ranking, followed
by deposit type, then guest history. The data partially supports this
— all three appear in the top 10 — but the actual rank order is
`deposit_type` (#1), `country` (#2), `agent` (#3), with `lead_time`
only at rank #7. Hotels using this methodology should *not* assume
their own data will reproduce the lead-time-first ranking. The
counter-intuitive direction also matters: bookings with non-refundable
deposits cancel *more* often, not less, because the non-refundable
rate is concentrated in channels whose customers cancel frequently
regardless of the deposit policy.

**3. The model is well-calibrated.** After isotonic calibration, the
Expected Calibration Error on the test set is 2.9 %. A "75 %
probability" booking really does cancel about 75–76 % of the time. The
operational consequence is that deposit policies can be set directly
off the probability number without further adjustment — there is no
need to add a safety margin or apply a fudge factor.

**4. Cancellation risk is heavily concentrated.** The High risk tier
(probability ≥ 0.70) represents only 26.07 % of test-set bookings but
accounts for 52.15 % of realised cancellation revenue losses
(€1.57 million of €3.01 million). The implication is direct:
intervention effort should be tiered, not blanket. The 3,108 High-tier
bookings in the test sample are the single highest-leverage operational
target.

**5. Cost-sensitive thresholding pays its keep.** Under the
cost-sensitive operating policy (threshold = 0.04), the model recovers
**97.5 % of the theoretical maximum revenue at risk** — €2,937,754 of
€3,014,266 — at the cost of about €67,000 in false-positive
interventions. Even under the more conservative `max_f1` policy used
for default weekly operations, the model still saves €2.61 million.
The model is not just academically accurate; it pays for itself
operationally several times over.

---

## 5.3 Managerial Implications

This section is the most important part of the chapter. It translates
findings into six concrete actions a hotel revenue manager can put on
their Monday-morning checklist.

**Recommendation 1 — Adopt the risk-tier-based operational policy.**
Bucket every new booking into Low (probability < 0.40), Medium
(0.40–0.70), or High (≥ 0.70). On the Portugal test sample these tiers
contained 51 %, 23 %, and 26 % of all bookings respectively. The
Power BI dashboard auto-refreshes these counts every week from the
live prediction log. Treat the three bands as policy tiers, not just
analytic categories: each one triggers a different action (see
recommendations 3 and 4).

**Recommendation 2 — Tighten policy by booking source, not by guest
history.** The top three SHAP drivers (`deposit_type`, `country`,
`agent`) are all *channel* signals. The hotel's biggest leverage is
not changing what individual guests pay; it is auditing which agents
and which countries cancel most often, and renegotiating commission
structures conditional on cancellation rates. The model can produce a
sorted list of top-cancelling agents per quarter from
`reports/segment_metrics.csv`. Build the quarterly review meeting
agenda around that list.

**Recommendation 3 — Run a 72-hour reminder workflow on Medium-tier
bookings.** At €15 per intervention, automated reminder emails are
the cheapest layer of the policy stack and address the largest single
slice of revenue at risk in absolute terms (2,707 bookings,
€1.07 million in realised losses on the test sample). The reminder is
operationally light — it can be a templated email sent from the PMS
72 hours before arrival — and is the highest return-on-effort
recommendation in this chapter.

**Recommendation 4 — Reserve confirmation calls and partial deposit
requests for the High tier.** The High tier (3,108 bookings on the
test sample) carried 75.8 % observed cancellation rate, so a manual
intervention here is justified by the hit rate. The intervention is
more expensive than a reminder email (staff time + risk of irritating
the guest) but the High tier's revenue concentration means it earns
back its cost many times over. Front-desk staff should treat the
High-tier list as a "call before Wednesday" workflow.

**Recommendation 5 — Use the live FastAPI + Gradio system as a
frontline tool.** Every booking entered through the existing PMS can
be scored in under 500 ms against the deployed champion model. The
Gradio UI at `localhost:8000/ui` exposes the same model in a form a
non-technical agent can use directly, complete with the predicted
probability, the risk tier, and a top-5 SHAP explanation of *why* the
model flagged this particular booking. The audit log feeds the Power
BI dashboard, so every prediction also becomes part of the
property's ongoing operational record.

**Recommendation 6 — Treat the dashboard's PSI drift page as the
retraining trigger.** Production models silently degrade as customer
behaviour shifts. The dashboard's monitoring page (Page 8) computes
the Population Stability Index for each feature against the training
baseline. When two or more features cross PSI = 0.25, schedule a
retraining cycle. Without this trigger, last quarter's model will
quietly drift below the recovery numbers reported in this thesis,
and the hotel will not notice until it is too late.

---

## 5.4 Limitations of the Study

Honest reporting of what the study did *not* do is as important as
reporting what it did.

**Single benchmark dataset.** The headline numbers in Chapter IV come
from one Portugal property (technically two — City Hotel and Resort
Hotel — within the same dataset) across one geographic region in one
pre-pandemic era (2015–2017). A small Philippine sub-study at Punta
Villa Resort (n = 193 bookings, 2022–2025) was run alongside the main
study as a transferability probe, but the test sample of only 20
bookings produced bootstrap 95 % confidence intervals of roughly
± 15 percentage points on PR-AUC — directionally useful but not
headline-grade. The pre-flight duplicate-cluster diagnostic ran
cleanly on the Philippine export and confirmed the methodology
operates honestly on that data; the *metrics*, however, should be
read as suggestive rather than definitive at that sample size.

**No external context features.** The model uses only the booking's
own data: lead time, deposit, country, agent, requested room,
party composition. It does not see weather forecasts, local event
calendars, airline cancellation feeds, currency-rate movements, or
news of strikes. All of these plausibly affect cancellation
behaviour, especially for international leisure travel, and could
add meaningful predictive power. Section 5.5 lists this as the
first future-research direction.

**Chronological split assumes stationarity within the test period.**
The 2017 test window covers roughly two months. Cancellation
patterns over longer horizons (years, post-pandemic shifts) are
documented in the dashboard's drift monitoring page but were not
modelled directly. A hotel deploying this model in 2025 against
2024 data should validate the metrics on their own holdout, not
assume the 0.864 ROC-AUC will transfer unchanged.

**Cost model is a single-point estimate.** The cost-sensitive
threshold was tuned against a €15 false-positive cost and a
false-negative cost equal to the booking's revenue at risk. The
sensitivity analysis in Notebook 10 shows the policy ranking
(`cost_sensitive` < `max_f1` < `high_precision`) is robust across a
4× perturbation of the false-positive cost, but the *absolute* total
cost figures are obviously sensitive to the assumption. Property-
specific calibration is recommended before deployment.

**ADR regressor uses post-booking features at training time.** The
accompanying Average Daily Rate (ADR) regression model was trained
with four features (`is_canceled`, `assigned_room_type`,
`booking_changes`, `days_in_waiting_list`) that are not known at
booking time. Live inference fills these with sensible defaults, so
live `predicted_adr` values are slightly less accurate than the
test-set RMSE of €44.31 reported in the appendix. The
methodologically clean fix is retraining on booking-time features
only — flagged in Section 5.5.

**No A/B testing of the intervention policies themselves.** The
€2.94 million recovery figure is an *upper bound* — it assumes that
when the hotel reminds, calls, or asks for a deposit, the guest
responds at the rate implied by the cost model. The *measured*
response rate is unknown until the policies are run in production.
Without A/B testing, the savings number is best treated as an
operational target, not a guaranteed outcome.

---

## 5.5 Recommendations for Future Research

Five concrete extensions would build directly on this work.

**Future Research 1 — Add external context features.** Public APIs
provide weather forecasts, local event calendars (concerts,
conferences, sports), airline schedule changes, and FX-rate
movements. Adding even a subset of these — say, a daily weather
forecast and a "local event happening within 10 km" indicator —
could plausibly add 1–3 percentage points of PR-AUC and would
specifically improve performance on leisure-travel cancellations,
which are the most weather-sensitive segment. A follow-up student
could build a feature pipeline that joins external feeds against the
booking arrival date and re-run the chronological evaluation.

**Future Research 2 — Replicate on additional Philippine properties.**
The Punta Villa Resort sub-study (n = 193) was a transferability
probe, not a headline result. Replicating the same methodology on
10–15 Philippine resorts — ideally a mix of city and beach properties
— would let the field produce region-specific headline numbers with
tight enough confidence intervals to be operationally actionable.
This is the most direct extension of the present work and is well
within a future thesis student's scope.

**Future Research 3 — A/B test the intervention policies.** Randomly
assign Medium-tier bookings to "reminder" vs "no reminder" arms over
a six-month deployment, and compare the realised cancellation rate
between arms. This converts the current upper-bound €2.94 million
figure into a measured treatment effect that survives causal
scrutiny. The same A/B framework could test the precise wording of
the reminder email, the timing (72 hours vs 48 hours), and the
deposit-request threshold for the High tier.

**Future Research 4 — Build an ADR regressor on booking-time features
only.** The current ADR regressor reaches Test R² ≈ 0.23, reflecting
both fundamental noise (rate cards change with promotions and
day-of-week pricing) and the use of post-booking features at training
time. A clean retrain on booking-time features only — combined with
external pricing context like competitor rate scrapes — would tighten
the live-time forecast and let Page 5 of the Power BI dashboard
report a true booking-time ADR signal.

**Future Research 5 — Package the methodology contributions as a
library.** Two reusable artefacts came out of this work that have
value beyond the specific cancellation problem. The first is a
**pre-flight duplicate-cluster diagnostic** that detects datasets
where chronological splitting would leak twins across the train/test
boundary (a problem the Philippine PMS export forced us to discover).
The second is a **feature-availability mapping** for reduced-PMS
schemas — a structured way to map the features a small property has
against the features a benchmark model expects, and to honestly
report what predictive power is lost when columns are missing. A
future student could package both as a standalone Python library,
publishable on PyPI as a contribution to the broader hospitality
analytics community.

---

## 5.6 Closing Statement

This study set out to show that cancellation risk is predictable at
the moment of booking with calibrated probabilities honest enough to
drive cost-sensitive action. The Portugal benchmark gave a clean,
defensible answer: yes, it is — and the recovery numbers are large
enough that the model pays for itself many times over per booking
cycle.

The operational pipeline is in place. The model is deployed behind a
live API, the predictions feed an audit log, the audit log feeds a
production-grade Power BI dashboard, and the dashboard's monitoring
page knows when to ask for a retrain. The policy recommendations in
Section 5.3 are concrete and ready to run. The €2.94 million figure
on the test set is the upper bound; what the hotel actually recovers
will depend on how well its staff execute the reminders, calls, and
deposit requests, and on how well guests respond to them.

The remaining work is not better modelling — it is replication on
additional properties, the addition of external context features,
and live A/B validation of the policy itself. Each of those is set
out concretely in Section 5.5 as a thesis-scaled research extension.
