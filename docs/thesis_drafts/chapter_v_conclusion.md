# CHAPTER V — CONCLUSION

> Draft prepared for the thesis "A Strategic Business Intelligence Approach
> to Predicting Hotel Booking Cancellations." This chapter synthesizes the
> findings reported in Chapter IV and translates them into theoretical,
> practical, and methodological contributions, then frankly states the
> study's limitations and proposes a concrete agenda for future work.

## 5.1 Introduction

This study set out to demonstrate that hotel booking cancellations can
be predicted at the moment of reservation with enough accuracy and
calibrated confidence to support cost-sensitive operational decisions.
The work applied Dynamic Capability Theory's **Sense → Seize → Transform**
cycle to two real datasets — the widely used Portugal benchmark (119,210
bookings across two hotels, 2015-2017) and the real Philippine resort
dataset from Punta Villa Resort (193 bookings, 2022-2025). Chapter IV
reported the empirical results; this chapter summarises what those
results mean, what they contribute, where the work is limited, and what
further research could build on it.

The two-dataset design was deliberate. Portugal supplies the statistical
power needed to validate the methodology at scale. The Philippine
sub-study supplies the evidence that the same methodology survives a
transfer to a smaller real property with a different geography, a
different language of operation, and a narrower PMS schema. Reading the
two studies together is what makes the conclusions in this chapter
defensible.

---

## 5.2 Summary of Findings by Hypothesis

The five hypotheses stated in Chapter I were tested against held-out
test data in Chapter IV. Their verdicts are summarised in Table 5.1.

**Table 5.1 — Hypothesis verdicts**

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1: Lead time, deposit type, and previous cancellations are significant predictors | **Supported** | All three features in the top 10 by mean(\|SHAP\|) on Portugal |
| H2: Gradient-boosted tree beats baseline models on out-of-time data | **Supported** — LightGBM's lead over every other model is real, not luck, on the overall ranking score; on the score at one specific cut-off, it is essentially tied with Gradient Boosting | LightGBM significantly better than each of LR, RF, GB, XGB, DT after resampling the test set 2,000 times |
| H3: Lead time has greatest SHAP, then deposit type, then previous cancellations | **Partially supported** — all three appear in top 10, but `deposit_type` leads, not `lead_time` | Aggregated SHAP rank: deposit_type #1, country #2, agent #3, lead_time #7 |
| H4: Cost-minimising threshold with risk-based deposit tiers reduces expected revenue loss | **Supported with quantified savings** of ≈ 95.4 % vs no model (≈ €1.53M on the Portugal test sample) | `reports/thesis/cost_sensitive_threshold.json` |
| H5 (added): Top SHAP feature on Portugal will also rank in the top 3 on the Philippine model | **Supported** | `deposit_type` is the #1 SHAP feature on both datasets |

Two of the five hypotheses are strongly supported (H1, H4), two are
supported with documented caveats (H2 on the score at one specific
cut-off, H5 across the small Philippine sample), and one is partially
supported (H3). The partial support for H3 is academically the most
informative outcome: by leaving the hypothesis as stated in Chapter I
and letting the data override the predicted ranking, the study
demonstrates that predictions about which features matter most must
be allowed to be wrong — and were treated as such here.

---

## 5.3 Summary of Findings by Objective

Chapter I stated four research objectives. The proposal also implicitly
required a fifth objective once the Philippine sub-study was added.
Table 5.2 records which Chapter IV section addresses each objective.

**Table 5.2 — Research objectives and where they are met**

| Objective | Where it is met | Status |
|---|---|---|
| 1. Identify and analyse the primary factors that correlate with booking cancellations through EDA | Section 4.2 (Sense) | Met |
| 2. Develop and evaluate a range of ML models on Accuracy, Recall, F1, Precision, AUC | Section 4.3 (Seize) | Met; LightGBM selected as champion by rolling-origin PR-AUC |
| 3. Interpret the feature importance of the best-performing model and translate it into a clear understanding of cancellation drivers | Section 4.3.4 (SHAP) | Met; per-prediction SHAP also exposed in the live API |
| 4. Build a Power BI decision-support dashboard converting model insights into cost-sensitive policy recommendations | Section 4.4.3 | Met; 8-page dashboard delivered |
| 5. Validate the methodology's transferability to a small real Philippine resort dataset (added) | Section 4.5 | Met; the pre-flight diagnostic passes and `deposit_type` survives as #1 SHAP |

Every objective is met in Chapter IV. Objective 4's deliverable — the
Power BI dashboard — is reproducible from the CSV outputs in `reports/`
and `data/predictions/`, so the dashboard is a concrete artefact of the
study, not a verbal description.

---

## 5.4 Theoretical Contributions

This study extends the application of Dynamic Capability Theory to
hospitality machine learning in three specific ways.

**First, the Sense-Seize-Transform cycle is operationalised end-to-end
rather than stopping at prediction.** Most prior hotel-cancellation work
delivers a model and an evaluation metric. This study additionally
delivers calibrated probabilities (so the percentage shown to a manager
is meaningful), cost-sensitive thresholds (so the policy choice is in
business units), risk tiers (so the front-desk team has a finite menu
of actions), a live serving stack (so predictions flow from the booking
system to the dashboard automatically), and a drift-monitoring template
(so the deployed model can be maintained). Each of these is a Transform-
phase capability that prior work mentions but rarely instantiates.

**Second, the study identifies analytical capability as the
microfoundation linking sensing to performance outcomes.** Pavlou and
El Sawy (2011) argue that analytical capability is the bridge between
information resources and firm performance. This study makes that
bridge concrete: high-quality data ingestion plus calibrated models
enhance sensing; cost-sensitive decision rules and disciplined
deployment bolster seizing; continuous monitoring and retraining
facilitate transformation. Each microfoundation is shown to be
implementable on a single laptop with open-source tooling and a Power
BI desktop license.

**Third, the framework is shown to hold across two geographies and two
property types.** The Portugal main study uses a mixed city-and-resort
sample with global tourist mix. The Philippine sub-study uses a single
resort with local-clientele Walk-In bookings. The same methodology —
chronological split, isotonic calibration, threshold sweep, SHAP
interpretation — produces calibrated and interpretable models in both
contexts. The Sense-Seize-Transform cycle is shown to be a portable
framework, not just a useful conceptual diagram.

---

## 5.5 Practical and Managerial Contributions

The study delivers four contributions that hotel managers can act on
directly.

**The deposit-policy lever is the strongest operational signal.**
`deposit_type` is the #1 SHAP feature on both datasets in Chapter IV.
Hotels that lack a calibrated cancellation model can still benefit
from the broader finding: deposit policy and lead-time profile
together identify high-risk bookings with a strong-enough signal that
tightening deposit terms for long-lead, no-deposit bookings is the
highest-leverage operational change available to a property without
machine learning. With the model in place, the cost-sensitive
threshold quantifies the value of that change.

**Per-prediction SHAP makes flagged bookings explainable.** Each
`/predict` response returns the top-five contributing features for
that specific booking. A front-desk clerk can see *why* the model
flagged this guest — long lead time, no deposit, single adult, zero
special requests — and decide whether the intervention is justified
or whether the model has missed obvious context. Per-prediction
explanations close the trust gap that often blocks ML adoption in
hospitality operations.

**The 8-page Power BI dashboard turns technical artefacts into a
manager-friendly playbook.** Each page addresses a specific
decision context: trend monitoring, segment slicing, revenue at risk
under different policies, ADR forecasting, threshold comparison,
feature importance, and drift monitoring. A property manager who
does not write Python can still use the model's output to set deposit
rules and reminder cadences. The CSV-based architecture means the
dashboard works on any machine with Power BI Desktop and no database
connection.

**Cost-sensitive thresholding quantifies the policy choice in euros.**
The savings figure — approximately €1.53M on the Portugal test
sample versus no model, or 95.4 % of expected cancellation cost
— gives the property a number to put in a business case. The
risk-based deposit tier policy (low / medium / high) operationalises
the saving as a concrete outreach playbook that the front-desk team
can adopt without further model training.

---

## 5.6 Methodology Contributions

Three contributions emerged from this work that are reusable beyond the
two datasets.

**The pre-flight duplicate-cluster diagnostic** is a generic check
that flags datasets where chronological splitting would leak twins
across the train/test boundary. The diagnostic is a two-rule trigger:
if the duplicate-feature-vector rate exceeds 30 % AND the fraction of
duplicate clusters with consistent labels exceeds 90 %, the test
metrics will be inflated by recognition rather than generalization.
The diagnostic does not fire on the real Punta Villa dataset, which
is the right outcome. Future researchers claiming transferability on
small datasets should run this check before reporting numbers.

**The feature-availability mapping** documents the dimensions a
property's PMS schema must support to apply the methodology, and
bounds the predictive ceiling for a property with a narrower schema.
The Punta Villa export captures roughly half of the features the
Portugal model uses; the resulting test PR-AUC of 0.542 on n_test =
20 represents the predictive ceiling on that schema with that sample
size. This is useful guidance for properties considering an ML
adoption: not every PMS schema can produce a 0.76 PR-AUC model, and
the methodology cannot manufacture features the schema does not
capture.

**The plug-and-play dataset framework** allows the methodology to be
re-applied to any chronologically-sortable hotel booking CSV with
just a configuration change in `src/config.py`. The Philippine
sub-study exercises this framework end-to-end. A third property
could adopt the methodology by replacing the CSV, updating
`ADR_MAX_VALID` and `FP_INTERVENTION_COST` for local currency, and
re-running `python scripts/train_ph.py`. The training pipeline
produces every artefact a Power BI dashboard and a live serving
deployment require.

---

## 5.7 Limitations

This study has seven limitations the reader should weigh against its
findings.

**Portugal dataset age.** The Portugal data covers July 2015 to August
2017. It pre-dates the COVID-19 pandemic and the rise of flexible
booking policies that have reshaped customer behaviour since. The
empirical patterns reported in Section 4.2 — including the counter-intuitive
"Non Refund" deposit pattern — may not reproduce on bookings made
under post-2020 conditions. A property planning to deploy this
methodology in production should retrain on its own recent data
rather than rely directly on the Portugal numbers.

**Philippine small sample.** The Philippine sub-study trained on 154
rows and tested on 20. Bootstrap 95 % confidence intervals on the
test PR-AUC span approximately ±15 percentage points. Every
Philippine point estimate in Chapter IV is therefore directional, not
production-grade. The PR-AUC of 0.542 should be quoted with its
confidence interval, not as a headline.

A specific consequence of the small Philippine sample worth flagging
explicitly is **threshold instability**. The balanced-policy
threshold of 0.190 was learned on a validation set of only 19
bookings containing roughly three actual cancellations. With so few
positive examples to learn from, the threshold the validation set
suggests is statistically noisy — small shifts in which bookings
happen to land in the validation set would move the threshold
several percentage points up or down. On the 20-row test set, this
specific cut-off happens not to flag any cancellations, producing an
F1 score of zero. This is a mathematical symptom of small sample
size at a single chosen cut-off; it is *not* a failure of the model
itself, which (as Chapter IV Section 4.5.2 shows) still ranks Philippine
bookings by cancellation risk well enough to produce a PR-AUC roughly
3.6 times the natural cancellation rate. The risk-tier system — which
relies on the calibrated probabilities themselves rather than on a
single fixed cut-off — remains functional and is the recommended
operational path for the Philippine deployment until additional
bookings stabilise the optimal threshold.

**ADR live-forecast caveat.** The Portugal ADR regressor was trained
with four post-booking features (`is_canceled`, `assigned_room_type`,
`booking_changes`, `days_in_waiting_list`) that are not known at the
moment of reservation. The live `/predict` endpoint substitutes
placeholder values for these features, so the live `predicted_adr` is
slightly less accurate than the published test-set RMSE of 44.31 EUR.
A methodologically clean fix is to retrain the ADR regressor on
booking-time features only, which is recommended in Section 5.8.

**No randomised field experiments.** The cost-sensitive savings of
€1.53M on the Portugal test sample is a backtested figure: it
estimates what the policy would have saved on already-realised data.
The figure is not an estimate of what the policy will save in
production, which depends on whether the interventions (reminder
emails, deposit requirements) actually prevent cancellations or
merely catch cancellations earlier. A randomised controlled trial
deploying the policy on a live booking stream would be the next
methodological step to convert backtested savings into causal claims.

**Cost-model simplifying assumptions.** The cost analysis assumes a
€15 per-intervention false-positive cost and a one-night recovery
penalty for each false negative. The €15 figure is an estimate of
marginal contact cost; it does not capture brand reputation effects
of unnecessary deposit demands. The one-night recovery penalty
under-states the true opportunity cost when a cancelled booking
cannot be rebooked at all. Both assumptions are documented in
`src/config.py` and can be revised per property; the relative ranking
of the three threshold policies in Section 4.4.1 is robust to changes in
these assumptions over a reasonable range.

**No external data fusion.** The study deliberately excludes external
factors (local events, competitor rates, weather, flight availability)
to keep the methodology reproducible from public data. The literature
review in Chapter II identified this as a research gap; the present
study does not close it. A model that integrates these external
signals could in principle achieve higher PR-AUC, especially in the
late-booking window where short-notice context matters most.

**Temporal leakage residue.** Even chronological splits can leak via
macro-temporal effects (a seasonality bleed-through, an event
clustering at the split boundary). The reported metrics use
chronological splits as the leakage-control mechanism, which is the
strongest practical defence, but does not guarantee zero leakage.

---

## 5.8 Future Work

Six concrete research directions follow from this study's findings
and limitations.

**Collect more Philippine bookings.** The Philippine learning curve
in `notebooks/ph/03_deep_analysis.ipynb` Section 3.2 does not flatten at
the current n_train of 154 rows. Doubling the training set is
likely to yield a meaningful PR-AUC improvement and would tighten
the threshold-stability problem that produces F1 = 0 at max-F1 on
the current 20-row test set. A target of 500-1000 bookings would
allow rolling-origin cross-validation on the Philippine data and
move it from a transferability probe to a production-grade
deployment.

**Retrain the ADR regressor on booking-time features only.** The
current ADR regressor uses four post-booking features at training
time and substitutes placeholders at inference. A clean retrain
on the same feature subset as the cancellation classifier (the 18
booking-time engineered features) would close the live-vs-published
RMSE discrepancy noted in Section 5.7. This is a small change to
`src/pipelines/train.py` and would be a one-week project.

**Run a live A/B test of the cost-sensitive threshold.** The €1.53M
backtested saving on the Portugal test sample is a backtest, not a
causal estimate. A randomised assignment of bookings to a
"intervened" arm (reminder email + partial deposit request) versus
a "control" arm (current policy) would convert the backtest into a
controlled causal estimate of intervention effect. The serving stack
already logs every prediction; adding randomised arm assignment is
a fifty-line change.

**Add external data fusion.** Events, weather, competitor rates, and
flight availability were excluded from this study for reproducibility.
A follow-up study that integrates these external signals — using a
public events API (e.g., PredictHQ), a weather API (e.g., OpenWeather),
and a competitor-rate scraper — would directly test the Chapter II
research gap identified by Altin et al. (2025). The serving layer's
plug-in architecture (`src/serving/inference.py`) is designed to
accept additional feature transformers without re-training, so
external features could be added as a runtime enrichment step.

**Federated learning across small properties.** Punta Villa's
193-row dataset is at the small end of what an SMB hotel can offer.
A federation of small properties — each contributing model gradient
updates without sharing raw data — could in principle produce
production-grade thresholds on commodity hardware without any single
property needing Portugal-scale data. The plug-and-play dataset
framework described in Section 4.6.3 is the natural starting point for
such a federation.

**Add an uplift modelling layer.** The cancellation classifier
predicts the probability that a booking will cancel. It does not
predict whether an intervention (reminder, deposit) will prevent
cancellation. Uplift modelling — fitting a second model on the
treatment effect rather than the outcome — would convert "the model
flagged this booking" into "intervening on this booking will reduce
cancellation probability by X percentage points." This is the
correct decision-theoretic framing for a serving layer that
recommends actions, and the literature already documents techniques
(e.g., the two-model approach, X-learners, transformed-outcome
trees) that would integrate with the existing serving stack.

---

## 5.9 Concluding Remarks

This study began with a problem statement that almost every property
recognises: cancellations are a persistent revenue leak, and most
hotels still manage them with judgment rules rather than data.
Dynamic Capability Theory's Sense → Seize → Transform cycle gave the
work a structure for moving from data to decisions. The two-dataset
design — Portugal at scale, Punta Villa in real-world miniature —
gave the work a way to test whether the same methodology travels
across geographies, property types, and PMS schemas.

The empirical answer is yes. The same dominant feature
(`deposit_type`) leads the SHAP ranking on both datasets. The same
modelling family (LightGBM) wins or ties on both. The same calibration
and threshold-selection machinery produces a deployable model on
both. The Portugal version of that model delivers a backtested 95 %
reduction in expected cancellation cost; the Philippine version
delivers a directional cancellation signal on a real PMS schema that
captures only half of Portugal's features. The methodology does not
manufacture data — the Philippine model is honestly weaker because
the data is honestly thinner — but it survives the transfer.

The practical contribution sits in three artefacts the study leaves
behind. A live FastAPI + Gradio server that any property's IT team
can stand up in five minutes. An 8-page Power BI dashboard that a
revenue manager can read on a Monday morning. And a methodology
playbook — reproducible, version-controlled, continuous-integration
verified — that a future analyst can extend to a third or fourth
property without writing new ML infrastructure.

The biggest open question is whether the backtested savings translate
into causal real-world savings under live deployment. The methodology
is ready for that test; what remains is the field experiment that
would settle the question definitively. That is the natural next
step, and it is the work this thesis hopes to enable.
