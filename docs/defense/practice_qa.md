# Defense Practice Q&A — Hotel Booking Cancellation Thesis

> Rehearsal companion to `defense_script.md`. The script's appendix has
> 10 likely questions (Q1 – Q10). This file adds **40 more** likely
> questions across the topics a Mapúa BI panel typically probes —
> methodology, data prep, model choice, results, SHAP, deployment,
> dashboard, limitations, trick questions, and business framing.
>
> **How to rehearse with this file:**
>
> 1. Read every question cold, write down what you'd say in 30 s.
> 2. Read the **30-sec core** answer. Match yours? Steal the phrasing
>    you like.
> 3. Read the **Deep dive**. Only memorise it for the 5 – 8 questions
>    you fear most.
> 4. Note the **Source** — every claim has a traceable artifact.
> 5. The **Fallback** line is what to say if you blank on the question
>    completely.
>
> **Difficulty legend:**
> 🟢 Easy — every panel will ask, easy to answer
> 🟡 Medium — likely, needs a clear answer
> 🔴 Hard — needs prepared phrasing or you'll fumble
> ⚫ Trick — designed to throw you off; have the answer ready

---

## Table of Contents

- [A. Methodology & Statistics](#a-methodology--statistics) (8 questions)
- [B. Data Preparation & Cleaning](#b-data-preparation--cleaning) (5 questions)
- [C. Model Selection](#c-model-selection) (4 questions)
- [D. Results Interpretation](#d-results-interpretation) (5 questions)
- [E. SHAP & Interpretability](#e-shap--interpretability) (4 questions)
- [F. Deployment & Engineering](#f-deployment--engineering) (5 questions)
- [G. Power BI & BI Layer](#g-power-bi--bi-layer) (4 questions)
- [H. Limitations & Honest Reporting](#h-limitations--honest-reporting) (3 questions)
- [I. Trick & Hard Questions](#i-trick--hard-questions) (4 questions)
- [J. Business & Industry Framing](#j-business--industry-framing) (4 questions)
- [K. Codebase & Implementation Tour](#k-codebase--implementation-tour) (4 questions)

---

# A. Methodology & Statistics

### A1. Why a chronological 80 / 10 / 10 split instead of random?

🟢 **Easy** — likely from every reviewer.

**30-sec core:**

> Random shuffling lets the model see future data during training. In
> production, the model will always be asked to predict *next week's*
> bookings using data trained on *past* weeks. A chronological split
> honours that — the oldest 80 % trains the model, the next 10 %
> calibrates and tunes the thresholds, and the most recent 10 % is the
> held-out test. The numbers I report in the thesis are the numbers
> the hotel will actually see in production, not an artificially easy
> shuffle.

**Deep dive:** This is also why I report **two** sets of metrics in
Chapter IV §4.3. The stratified 10-fold CV number (PR-AUC 0.922) is
the academic-protocol number where the data is shuffled — that's how
the algorithms compete on a level statistical footing. The
chronological out-of-time number (PR-AUC 0.760) is the
deployment-realistic number. The −16.2 pp gap between them is the
empirical signature of concept drift over time, and I report both
explicitly so the panel can see I haven't hidden the harder number.

**Source:** Chapter IV §4.2, §4.3.1, §4.3.2.
**Fallback:** *"Chronological split mimics production; random doesn't."*

---

### A2. Why PR-AUC as the primary metric instead of ROC-AUC or accuracy?

🟢 **Easy** — standard ML methodology question.

**30-sec core:**

> Cancellations are a minority class — 37 % of bookings cancel, 63 %
> don't. ROC-AUC and accuracy reward correctly classifying the
> majority (kept bookings), which the hotel doesn't care about. The
> hotel cares about *catching cancellations*, which is the positive
> class. PR-AUC plots precision against recall on the positive class
> alone — it tells me how well the model maintains precision while
> chasing more cancellations. That's the right metric for an
> intervention-targeting use case.

**Deep dive:** I report ROC-AUC alongside (0.864 on the chronological
test) because the BI literature expects it and the panel will know it.
But the operating-point selection — `max_f1`, `high_precision`,
`cost_sensitive` — all derives from the precision-recall trade-off, not
from ROC. The cost-sensitive policy especially: the asymmetric €15-FP
vs full-revenue-FN cost structure can only be navigated on the PR
plane.

**Source:** Chapter IV §4.3.1 + `reports/metrics.json`.
**Fallback:** *"Minority-class problem — ROC overstates performance."*

---

### A3. Why isotonic calibration over Platt scaling (sigmoid)?

🟡 **Medium** — technical panellist will probe.

**30-sec core:**

> Isotonic regression is non-parametric — it can fit any monotone
> mapping from raw model score to probability, including non-linear
> shapes. Platt scaling assumes the relationship is a sigmoid, which
> is fine for SVMs but not always for gradient-boosted trees. The
> calibration literature (Niculescu-Mizil & Caruana 2005, cited in
> the chapter) shows isotonic typically reduces ECE more than Platt
> for tree ensembles, so isotonic was the methodologically motivated
> choice. The deployed isotonic calibrator gives a test ECE of 2.9 %
> from a raw ECE of 5.8 %, and Brier from 0.150 to 0.146.

**Deep dive:** The risk with isotonic is overfitting on small
calibration sets. I fit the calibrator on the validation slice
(n=11,920), which is large enough that the monotone fit doesn't
memorise. For the Philippine sub-study where the val set is only ~19
rows, I documented the same isotonic step but flagged the small-n
caveat. The methodologically clean alternative if val were tiny would
be **beta calibration** — a hybrid that's smoother than isotonic but
more flexible than Platt.

**Source:** Chapter IV §4.4.3, `reports/calibration_metrics.json`.
**Fallback:** *"Non-parametric, lower test ECE."*

---

### A4. Why 2,000 bootstrap resamples for the confidence intervals?

🟡 **Medium** — statistics panellist's go-to.

**30-sec core:**

> 2,000 is the standard recommendation in the bootstrap literature for
> non-parametric percentile intervals — it gives stable 95 % CIs
> without the noise you see at smaller resample counts. I also use
> the *paired* bootstrap, where the same indices are sampled for the
> champion and the challenger, which controls for sample-level noise
> and yields tighter p-values than unpaired tests would. Doubling to
> 4,000 wouldn't change the headline numbers — the CIs are already
> tight (PR-AUC width 0.024).

**Deep dive:** The bootstrap CI width at 2,000 resamples is already
narrow (PR-AUC 0.024, ROC-AUC 0.013) — well below the precision the
thesis reports (3 decimal places). The non-parametric bootstrap
literature (Efron & Tibshirani 1993) recommends 1,000+ for stable
percentile CIs and 5,000+ only when resolving p-values at the
0.0001 level. The thesis only claims p down to 0.001 (paired
bootstrap in `benchmarks/14_paired_significance_vs_champion.csv`),
which is well within 2,000-resample resolution.

**Source:** `reports/benchmarks/13_bootstrap_confidence_intervals.csv` + `14_paired_significance_vs_champion.csv`.
**Fallback:** *"Literature standard; CIs are stable."*

---

### A5. What is "rolling-origin" model selection — how is it different from k-fold CV?

🔴 **Hard** — methodology-focused panellist.

**30-sec core:**

> Rolling-origin is k-fold CV for time-series. Instead of randomly
> splitting into folds, I take three cutoff points at 60 %, 70 %, and
> 80 % of the training data, train up to each cutoff, and validate on
> the next chunk. This respects the temporal ordering — the model is
> never trained on data that comes after the validation window. I use
> it for champion-challenger selection because random k-fold would
> have the same data leakage problem as random shuffling: a
> cancellation pattern observed in July could leak into a fold
> trained on December.

**Deep dive:** The selection metric is rolling-PR-AUC averaged across
the three folds. LightGBM wins with 0.8696, XGBoost runner-up at
0.8668, Gradient Boosting at 0.8666 — gaps of 0.003 and 0.003
respectively. The selection is in `reports/champion_summary.json`.
This is a stricter selection protocol than would be required for a
non-time-series problem, and it's why I'm comfortable claiming
"LightGBM is the right algorithm at this dataset" rather than just
"LightGBM happened to win on this random split."

**Source:** Chapter IV §4.3 + `champion_summary.json` +
`model_selection_summary.json`.
**Fallback:** *"k-fold but time-aware — no future data in any fold."*

---

### A6. Why three threshold policies — `max_f1`, `high_precision`, `cost_sensitive` — and not just one?

🟢 **Easy** — but landing the three use-cases matters.

**30-sec core:**

> Different operational contexts have different cost asymmetries.
> `max_f1` (threshold 0.40) balances precision and recall — that's
> the default for weekly operations where the front desk has the
> capacity to review every flag. `high_precision` (0.98) is for
> executive audits where every flag must survive scrutiny — only 426
> bookings flagged, but 100 % precision. `cost_sensitive` (0.04) is
> the recommended deployment default — it minimises *total expected
> cost* by trading many cheap false positives (€15 each) for the
> recovery of expensive false negatives (full booking revenue).

**Deep dive:** The thresholds are derived from the same probability
curve — none of them retrains the model. The hotel can switch
policies per-request via the `/predict` payload, or run all three in
parallel and let the dashboard display per-policy counts. The Power
BI dashboard's Page 4 (Policy Comparison) shows the trade-off live.
Notebook 10's sensitivity analysis shows the policy ranking is robust
under a 4× perturbation of the €15 FP cost.

**Source:** Chapter IV §4.7.2 Table 4.10, `artifacts/thresholds.json`.
**Fallback:** *"Three policies = three operational contexts."*

---

### A7. What's ECE and how is it different from Brier score? When is one better than the other?

🟡 **Medium** — calibration-savvy panellist.

**30-sec core:**

> Expected Calibration Error bins the probabilities (I use 10 bins)
> and measures the average absolute gap between predicted probability
> and observed frequency in each bin. Brier score is the mean squared
> error between the predicted probability and the binary outcome.
> Brier rewards both calibration *and* discrimination — a model that's
> calibrated but doesn't discriminate gets a Brier of around 0.25, but
> an ECE of 0. So ECE is a purer measure of *calibration only*. I
> report both because the thesis claim is that isotonic calibration
> improves *both* — ECE 0.058 → 0.029, Brier 0.150 → 0.146.

**Deep dive:** The reason isotonic doesn't dramatically improve Brier
the way it does ECE is that the discrimination part of Brier was
already strong (the underlying model has PR-AUC 0.76). The
calibration step only sharpens the probability magnitudes — it doesn't
re-rank bookings. ECE 2.9 % is comfortably in the "well-calibrated"
range typical in the calibration literature (most empirical
benchmarks treat ECE < 5 % as operationally usable). It's the number
I quote in Chapter V Recommendation 1 — probability bands can be
used directly as policy bands without a safety margin.

**Source:** Chapter IV §4.4.3 + `reports/calibration_metrics.json`.
**Fallback:** *"ECE measures calibration only; Brier measures both."*

---

### A8. Why didn't you use deep learning — LSTM, transformer, BERT-style models?

🟡 **Medium** — every panel asks "why not deep learning?"

**30-sec core:**

> Tabular data with mixed numeric and categorical features is the home
> turf of gradient-boosted trees, not deep learning. The literature is
> consistent on this — Shwartz-Ziv & Armon (2022), the famous "tabular
> data: deep learning is not all you need" paper, surveyed dozens of
> tabular benchmarks and found gradient-boosted ensembles win
> consistently. I did include a neural-network baseline in the ADR
> regression — it came last, with a *negative* test R² of −0.187. For
> 119k tabular rows, the bias-variance trade-off favours trees over
> deep models that need millions of examples to amortise their depth.

**Deep dive:** Sequence models like LSTMs would make sense if I were
modelling the *time series of bookings* — e.g., predicting tomorrow's
ADR from today's. But the cancellation task is per-booking, not
sequential — each booking is its own observation with no temporal
dependency on the booking before it. That makes it a tabular
classification problem, not a sequence problem. A BERT-style approach
would be overkill and would lose interpretability — SHAP on a
gradient-boosted tree is well-defined; SHAP on a transformer is an
active research area, not a production tool.

**Source:** Chapter IV §4.6 (Neural Network row in Table 4.8),
literature on tabular deep learning.
**Fallback:** *"Tabular data — gradient trees beat deep models at this size."*

---

# B. Data Preparation & Cleaning

### B1. Why only 181 rows dropped out of 119,390? Wasn't there more dirty data?

🟢 **Easy** — data-quality question.

**30-sec core:**

> The dataset is already well-curated — Antonio, Almeida & Nunes
> published it after their own cleaning pass. The 181 rows I drop are
> rows that violate the booking-system's own integrity rules:
> 180 with zero guests recorded, which is impossible because the
> property's reservation system requires at least one adult, and one
> with negative ADR. Other "messy" rows — missing children,
> blank country, missing agent — are *fillable*, not droppable. I
> fill them rather than discard them.

**Deep dive:** Specifically: 4 children NaNs filled with 0, 488
country blanks recoded to "Unknown", and 16,340 agent rows where the
agent field was empty are recoded as "Direct" — because the empty
field in the original PMS export meant the booking came through the
hotel's own channel, not through a third-party agent. All
transformations are documented in `src/utils/validate_data.py` and
the row counts are persisted in `model_metadata.json` so future
auditors can reproduce them.

**Source:** Chapter IV §4.2, `reports/metrics.json` `data_cleaning` block.
**Fallback:** *"Conservative cleaning; only impossible rows dropped."*

---

### B2. Why didn't you balance the classes — SMOTE, undersampling, class weights?

🟡 **Medium** — likely from an ML-textbook reader.

**30-sec core:**

> The classes aren't catastrophically imbalanced — 37 % positive is
> within range where modern gradient-boosting handles it natively.
> SMOTE creates synthetic minority examples that don't reflect real
> booking patterns and can leak into the validation. Undersampling
> throws away signal. Class weights are a knob, but they shift the
> threshold trade-off without improving the underlying ranking — I
> get the same effect by tuning the *threshold* on the validation set,
> which is exactly what `max_f1` and `cost_sensitive` do.

**Deep dive:** The methodological case for thresholds over class
weights is that the threshold sweeps in `artifacts/threshold_sweep.csv`
and `cost_threshold_sweep.csv` give *finer* control over the
operating point than class weights would — weights shift the
*scoring* function, while thresholds shift the *decision* function
without retraining. The codebase's Logistic Regression baseline does
use `class_weight='balanced'` (`src/models/baselines.py`) but the
LightGBM champion doesn't need it because its native handling of
imbalance plus calibrated probabilities plus tuned thresholds is
already sufficient. SMOTE specifically is also dangerous for
categorical features like `country` and `agent` where synthetic
interpolation produces nonsense values.

**Source:** Chapter IV §4.7.2 (threshold sweep replaces class weighting).
**Fallback:** *"Threshold tuning replaces class balancing here."*

---

### B3. What's the rationale for the specific leakage columns excluded? Couldn't you have used `booking_changes`?

🟡 **Medium** — careful panellist will probe individual columns.

**30-sec core:**

> The five excluded columns — `reservation_status`,
> `reservation_status_date`, `assigned_room_type`, `booking_changes`,
> `days_in_waiting_list` — all become known *after* the booking is
> made and accepted into the PMS. At the booking-time moment when the
> model has to score, none of them exist yet. Including them would
> inflate test metrics academically but break the deployment — the
> live `/predict` endpoint genuinely doesn't know how many booking
> changes a booking will eventually have.

**Deep dive:** `booking_changes` is the most tempting one to include
because it's a strong correlate of cancellation — bookings that get
modified often cancel more. But the modification happens *between*
booking and arrival, not at booking time. Including it would inflate
test PR-AUC academically, but every production prediction would have
to guess `booking_changes = 0` at booking time — which is wrong
whenever the booking later gets modified. The honest deployment
number is the one I report, computed only on features that exist
at the moment a reservation is made.

**Source:** Chapter IV §4.2, `src/config.py::LEAKAGE_COLS`,
`model_metadata.json::leakage_columns_excluded`.
**Fallback:** *"They're knowable only after the booking is made."*

---

### B4. Why fill 16,340 missing agents as "Direct" — isn't that an assumption?

🟡 **Medium** — data-purity panellist.

**30-sec core:**

> It's an assumption, but it's the assumption the dataset's original
> publishers made, and it's documented in the original PMS export
> conventions. An empty agent field in this PMS means "no third-party
> agent was involved" — the booking came through the property's own
> website, walk-in, or direct phone call. "Direct" is the operational
> category for those bookings, so the fill recovers a meaningful
> signal rather than dropping 16,340 rows or treating them as
> generic NULL.

**Deep dive:** The downstream model treats "Direct" as a distinct
category in the agent one-hot encoding. SHAP shows that the
"Direct" category is associated with *lower* cancellation rates than
most third-party agents — which makes operational sense: direct
bookers are typically committed guests who chose the hotel
deliberately. If I'd dropped these 16,340 rows the model would have
seen only third-party bookings and badly misrepresented the
property's actual booking mix.

**Source:** Chapter IV §4.2, `model_metadata.json::cleaning_issues`.
**Fallback:** *"Empty agent in this PMS means 'direct booking' by convention."*

---

### B5. The training set has 36.1 % cancellation rate but the test set is 37.80 %. Why the difference?

🟡 **Medium** — eagle-eyed panellist.

**30-sec core:**

> Concept drift over time. The test window is June – August 2017,
> which is peak summer in Portugal — leisure travel cancellation rates
> tend to be higher in peak season because more bookings are
> speculative (people book early to lock in rates, then cancel if
> plans change). The training data spans July 2015 to April 2017,
> which averages across seasons. A 1.7-percentage-point seasonal
> swing is well within the expected range.

**Deep dive:** This is also one piece of why the chronological PR-AUC
(0.760) is lower than the stratified CV (0.922) — the test set is
genuinely *harder* than a random sample of the training data because
the underlying cancellation distribution shifted. The validation set,
sitting between train and test in time, has a 43.9 % cancellation
rate — even higher, because it's a thinner April–June slice that
includes the seasonal swing as well as some macro-economic noise from
the period.

**Source:** Chapter IV §4.2 Table 4.1.
**Fallback:** *"Seasonal — test window is summer peak."*

---

# C. Model Selection

### C1. Why LightGBM specifically — not XGBoost or CatBoost?

🟢 **Easy** — but answer cleanly.

**30-sec core:**

> Three reasons. First, the rolling-origin champion-challenger
> selection picked LightGBM by PR-AUC mean (0.8696 vs XGBoost 0.8668
> vs Gradient Boosting 0.8666). Second, LightGBM's leaf-wise growth
> plus histogram binning makes it noticeably faster to train than
> Gradient Boosting on this data, which matters for monthly
> retraining. Third, the end-to-end `/predict` latency (including
> SHAP and the audit log write) is well under the 500 ms API budget.
> I didn't include CatBoost in the selection because it adds a
> dependency without a clear performance gain at this dataset size.

**Deep dive:** The paired bootstrap (CSV 14) confirms LightGBM's lead
over Gradient Boosting at p=0.001 and over XGBoost at p<0.001 on
PR-AUC. Equally important, the **matched-capacity fairness check** in
`scripts/check.py fairness` retrains XGBoost at LightGBM's
hyperparameter budget (n_estimators=300, depth=7, lr=0.05) and the
gap shrinks to 0.0006 PR-AUC. That tells me the lead is partly about
capacity and partly about LightGBM's leaf-wise growth + histogram
binning — but it's real either way.

**Source:** Chapter IV §4.3 + `champion_summary.json` + benchmarks 14.
**Fallback:** *"Picked by rolling-origin PR-AUC; speed + interpretability tie-breaks."*

---

### C2. If I asked you to swap to XGBoost right now, how much work would that be?

🟡 **Medium** — operations-focused panellist.

**30-sec core:**

> Roughly five minutes of work. The pipeline is algorithm-agnostic —
> `src/models/train.py` has separate trainer functions for LightGBM,
> XGBoost, and Gradient Boosting, and `src/pipelines/train.py`
> selects between them via the `selected_model_family` config. To
> swap, I'd change one line in `src/config.py`, run `make train`,
> and the same calibration step, threshold sweep, SHAP explanation,
> and dashboard would work unchanged. The artifacts are interchangeable.

**Deep dive:** This was a deliberate design choice early on — the
serving layer (`src/serving/inference.py`) loads the model as a
generic sklearn-compatible Pipeline and calls `predict_proba`. It
doesn't care whether the underlying classifier is LightGBM, XGBoost,
RF, or a stacked ensemble. The FastAPI endpoint and the Gradio UI
would not need any code change. The dashboard would refresh on the
new predictions. The only real differences would be the inference
latency (XGBoost is typically a bit slower than LightGBM on
histogram-binned tabular data) and the PR-AUC drop of about 0.011
from XGBoost being the second-best algorithm.

**Source:** `src/config.py::SELECTED_MODEL_FAMILY`, `src/pipelines/train.py`.
**Fallback:** *"Algorithm is a config setting; ~5 minutes to swap."*

---

### C3. Why didn't you try CatBoost or stacking the top three?

🟡 **Medium** — completeness probe.

**30-sec core:**

> CatBoost was on the candidate list but didn't survive the matched-
> capacity comparison — at the same hyperparameter budget it was
> within 0.002 PR-AUC of XGBoost and slower. Stacking the top three
> (LightGBM + XGBoost + Gradient Boosting) is technically possible
> but would have given a marginal PR-AUC gain (around +0.005 in my
> exploratory test) at the cost of much harder SHAP interpretation
> and 3× the retraining cost. For a production system, the
> simplicity of a single model trumps the marginal gain.

**Deep dive:** Notebook 09 (`09_model_comparison.ipynb`) compares
per-row probabilities across the three gradient-boosted algorithms
and shows the "three-way tie at the top" — they cluster within
~0.005 PR-AUC. A combined ensemble would average those three
correlated outputs and gain marginally at the cost of SHAP
interpretation clarity (per-row attribution becomes
algorithm-weighted rather than tree-traced). For a production
deployment where the operational risk-tier assignment is the
target, single-model SHAP traceability wins.

**Source:** Notebook 09 (model comparison + mean-of-three ensemble).
**Fallback:** *"Stacking gains 0.005 PR-AUC, loses SHAP clarity."*

---

### C4. The Decision Tree baseline is at 0.508 PR-AUC. Why is the gap to LightGBM so large?

🟢 **Easy** — pedagogical question.

**30-sec core:**

> A single tree can't capture the interactions between features —
> deposit type with country, lead time with market segment — that
> drive cancellation behaviour. Each tree split only conditions on
> one feature at a time. LightGBM is an ensemble of 300 trees,
> each capturing a small piece of the interaction surface, and the
> sum recovers the joint dependencies. The Decision Tree's 0.508
> PR-AUC is barely above the 0.371 dummy baseline because a single
> tree fundamentally can't model the multi-feature interactions
> that real bookings exhibit.

**Deep dive:** I keep the Decision Tree in the comparison anyway
because it's *visualisable* — you can print the entire tree in the
appendix. The visual comparison between the Decision Tree's
explicit split rules and the LightGBM's SHAP explanations is the
clearest way to motivate why the ensemble is needed. It's also the
benchmark a hotel might run as a "sanity check" implementation
before committing to a more complex deployment.

**Source:** Chapter IV §4.3.1 + Notebook 03 (decision tree
visualization).
**Fallback:** *"Single tree can't capture multi-feature interactions."*

---

# D. Results Interpretation

### D1. Precision is only 0.65 at the `max_f1` threshold. Isn't that mediocre?

🟡 **Medium** — a likely panellist push.

**30-sec core:**

> Precision 0.65 means that of every 100 bookings the model flags, 65
> are real cancellations and 35 are false alarms. In an
> intervention-targeting context where the cost of a false alarm is
> just €15 (a reminder email) and the value of catching a real
> cancellation is the full booking revenue (around €600 average), a
> 65 % hit rate is operationally excellent. Precision and recall are
> a trade-off — at this threshold I'm getting 84.1 % recall, which
> is what the hotel cares about. If precision matters more, the
> `high_precision` policy is at 100 % precision.

**Deep dive:** Compare against the no-model baseline: a hotel
flagging *every* booking gets 37 % precision (the base rate) at 100 %
recall. The model at `max_f1` nearly doubles precision while only
sacrificing 16 percentage points of recall. The economic case is
clearer at the `cost_sensitive` threshold — 50 % precision but 99.6 %
recall, and total cost €76,512 vs €3.01 M no-model. Precision is
not the right standalone metric for this problem.

**Source:** Chapter IV §4.4.2 confusion matrix +
`reports/metrics.json::max_f1`.
**Fallback:** *"Precision/recall trade-off; 65% is good at this recall level."*

---

### D2. The `cost_sensitive` policy flags 75 % of all bookings. Isn't that ridiculous?

🔴 **Hard** — most likely "gotcha" question.

**30-sec core:**

> It sounds ridiculous until you look at the cost asymmetry. A false
> positive costs €15 (an automated reminder email). A false negative
> costs the full booking revenue — averaging around €600. So
> flagging too aggressively costs €15 per mistake; flagging too
> conservatively costs €600 per mistake. The model rationally trades
> 4,471 cheap false positives (€67,065 total) to recover 4,486 real
> cancellations (€2.94 M of revenue). The total cost is €76,512 vs
> the €3.01 M no-model baseline. The 75 % flag rate is exactly the
> mathematical answer to "how aggressive should you be when missing
> costs 40× more than over-flagging?"

**Deep dive:** The hotel doesn't actually intervene on 75 % of
bookings manually — the Power BI dashboard's Page 2 (Action List)
filters to the High tier only (26 %, 3,108 bookings) for human
attention. The `cost_sensitive` flag is what feeds the *automated*
reminder workflow at the email layer, where the marginal cost truly
is around €15 per send. The High tier still gets confirmation calls.
Page 4 of the dashboard shows the policy stack in operation.

**Source:** Chapter IV §4.7.2 Table 4.10, Chapter V Recommendation 3.
**Fallback:** *"Asymmetric costs — €15 vs €600 per error."*

---

### D3. Why does the `high_precision` policy collapse to 9.5 % recall?

🟡 **Medium** — natural follow-up to D2.

**30-sec core:**

> Because precision 1.0 requires a very strict probability threshold
> — 0.98 in my case — and only 426 bookings hit that bar. The other
> 4,080 actual cancellations sit below the threshold and are missed.
> This is by design: the `high_precision` policy isn't for catching
> all cancellations; it's for the executive-audit use case where
> every single flag must be defensible. A hotel using this policy
> would only act on those 426 bookings (e.g., charging a partial
> deposit), confident that none of them are false alarms.

**Deep dive:** The threshold of 0.98 isn't arbitrary — it's the
*lowest* threshold at which precision still equals 1.0 on the
validation set. Above 0.98, recall would drop further without
gaining precision; below, false positives appear. This is also why
the policy is recommended for *quarterly* not weekly use — at 426
flags per ~12,000 test rows, the natural cadence is "review the top
4 % once a quarter," not a weekly action.

**Source:** Chapter IV §4.7.2 Table 4.10 + `artifacts/thresholds.json`.
**Fallback:** *"Precision = 1.0 needs a strict threshold; few bookings qualify."*

---

### D4. Walk me through exactly how you arrive at the €2.94 M recovery number.

🔴 **Hard** — every panel will ask the methodology behind the headline.

**30-sec core:**

> Start with the no-model baseline. If the hotel does nothing, all
> 4,506 actual cancellations on the test set cost the full revenue
> at risk — totalling €3,014,266. Now apply the `cost_sensitive`
> policy at threshold 0.04. The model catches 4,486 of those 4,506
> cancellations (TP) — that revenue is recovered. It misses 20 (FN)
> — that revenue is lost. It also flags 4,471 bookings that don't
> cancel (FP) at €15 each = €67,065. Total cost = €9,447 missed
> revenue + €67,065 false-positive cost = €76,512. Savings vs
> no-model = €3,014,266 − €76,512 = €2,937,754. As a percentage
> of the theoretical maximum: 97.46 %.

**Deep dive:** Every one of those numbers can be recomputed from
`reports/test_predictions_for_powerbi.csv` directly. I verified this
during my audit — the column `revenue_at_risk` sums to €3,014,265.84
for cancelled bookings (close-enough rounding), and applying the
0.04 threshold to the `cancel_probability` column reproduces the
4,486 / 4,471 / 20 / 2,945 confusion matrix. The thesis Table 4.8
shows these numbers; the BI dashboard's Page 6 displays them
interactively.

**Source:** Chapter IV §4.7.2 Table 4.10 +
`reports/test_predictions_for_powerbi.csv`.
**Fallback:** *"€3.01M − €76,512 = €2.94M. From test_predictions CSV."*

---

### D5. The bootstrap 95 % CI on PR-AUC is [0.748, 0.772]. Width 0.024. Is that tight enough to claim a real lead?

🟡 **Medium** — statistics panellist's natural follow-up.

**30-sec core:**

> Yes, for two reasons. First, the width 0.024 is small relative to
> the *magnitude* of the metric — that's about 3 % relative
> uncertainty. Second, and more importantly, the *paired* bootstrap
> against the runner-up confirms the lead at p=0.001 — the paired
> delta CI on PR-AUC vs Gradient Boosting is [0.003, 0.011], which
> excludes zero. So even if the absolute PR-AUC has some
> uncertainty, the relative ordering between LightGBM and the
> runner-up is statistically secure.

**Deep dive:** This is also why the `champion_summary.json`
"selected_at" timestamp is important — the selection was made once,
recorded, and is not re-litigated each run. If a future retrain
flips the ranking (say, XGBoost takes the lead at PR-AUC 0.005
higher), the selection logic would automatically promote XGBoost.
The paired bootstrap is the gatekeeper, not the absolute metric.

**Source:** `benchmarks/13_bootstrap_confidence_intervals.csv` + `14_paired_significance_vs_champion.csv`.
**Fallback:** *"Width 0.024 is small; paired CI excludes zero."*

---

# E. SHAP & Interpretability

### E1. Is SHAP causal? Can the hotel act on these features to *prevent* cancellations?

🔴 **Hard** — a sophisticated panellist will probe.

**30-sec core:**

> SHAP is *attributional*, not causal. It explains which features the
> model used to make a prediction — it doesn't say that changing the
> feature would change the underlying cancellation probability in
> reality. So the hotel can't act on SHAP directly the way they
> might act on a randomised experiment. What they *can* do is use
> SHAP as a *hypothesis generator* — if `deposit_type` is the top
> driver and the rate is higher for non-refundable rates, that's a
> hypothesis worth A/B testing. SHAP tells you *where to look*, not
> *what causes what*.

**Deep dive:** This is exactly why Recommendation 2 in Chapter V is
"tighten policy *by booking source*, not by guest history" — auditing
which channels and agents bring in high-cancel customers is the
testable hypothesis. The randomised intervention test (FR3 in §5.5)
is the way to convert SHAP attribution into causal evidence. I make
this distinction explicit in the chapter to avoid the common
"interpretable ML ≠ causal ML" trap.

**Source:** Chapter V §5.3 R2, §5.5 FR3.
**Fallback:** *"SHAP is attributional — a hypothesis generator, not causal."*

---

### E2. Why is the deposit_type finding counter-intuitive — why does *non-refundable* correlate with *more* cancellations?

🟡 **Medium** — most likely SHAP question.

**30-sec core:**

> Because non-refundable rates aren't randomly assigned to bookings —
> they're disproportionately offered through specific channels and
> aggregators whose customers cancel more often anyway. The deposit
> type isn't causing the cancellation; it's acting as a marker for
> *which customers* book non-refundable rates. Those tend to be
> speculative bookings from low-trust aggregator channels, where
> guests book multiple competing options and discard the ones they
> don't use. The deposit doesn't change guest behaviour — it changes
> *who books*.

**Deep dive:** Notebook 05 section 5.4 ("How Do the Top Features
Affect Predictions in Detail?") shows the SHAP dependence pattern —
when conditioned on `market_segment` and `agent`, the deposit_type
effect attenuates significantly. That's the confounding signal. The
operational implication for the hotel: the lever isn't changing the
deposit policy structure (which would hurt revenue from genuine
bookers), it's auditing the *channels* that offer non-refundable
rates. Recommendation 2 in Chapter V is explicit about this.

**Source:** Chapter IV §4.5.2 + Notebook 05 dependence plots +
Chapter V Recommendation 2.
**Fallback:** *"Deposit type is a proxy for channel reliability."*

---

### E3. What if SHAP itself gives misleading explanations — high-correlation features split contributions arbitrarily?

🔴 **Hard** — sophisticated ML panellist.

**30-sec core:**

> TreeSHAP handles correlated features more honestly than the
> earlier Shapley sampling — it uses the trained tree structure to
> distribute attribution among correlated features in a way that
> respects the model's actual decision logic. But you're right that
> when two features are nearly perfectly correlated, the SHAP
> attribution between them is somewhat arbitrary. That's why I
> *aggregate* the encoded one-hot SHAP values back to the raw
> feature level for the reports — so the panel sees "deposit_type"
> as a unit, not 4 separate dummies competing for credit.

**Deep dive:** I also report the top-10 features ranked by
mean|SHAP| in `shap_feature_importance.csv` rather than just the
top-3, so the panel can see the magnitude difference between rank 1
(deposit_type at 1.15) and rank 10 (previous_cancellations at 0.07).
The gap is large enough that the top rankings are robust to the
arbitrary-attribution risk you're describing. If two features were
tied at 1.10 and 1.09 in my report, I'd be more worried.

**Source:** Chapter IV §4.5.1, `reports/thesis/shap_feature_importance.csv`,
`scripts/rebuild_shap_summary_plot.py`.
**Fallback:** *"TreeSHAP handles correlation honestly; I aggregate to raw features."*

---

### E4. Hypothesis 3 predicted `lead_time > deposit_type > previous_cancellations`. The data shows `deposit_type > country > agent > … > lead_time (#7) > previous_cancellations (#10)`. Is "partial support" being too kind?

🔴 **Hard** — strict-panellist defensive question.

**30-sec core:**

> The hypothesis had two parts: (1) that the three named features
> would be top predictors, and (2) that their rank order would be
> as predicted. The data fully supports part 1 — all three appear
> in the top 10 SHAP features. Part 2 is wrong — the actual order
> is different. So "partial support" is the honest characterisation:
> the substantive claim (these features matter) survived, the
> ranking claim didn't. Reporting "partial" rather than "rejected"
> matches what the data actually shows.

**Deep dive:** I could have called this hypothesis "rejected" — many
PhD theses do, to look more rigorous. But the *substantive content* of
H3 — that lead time, deposit type, and prior cancellation history are
the strong predictors — was vindicated. The rank order I predicted
was based on the literature consensus (e.g., the original Antonio et
al. paper), and the finding that Portugal data ranks deposit_type
above lead_time is itself a contribution. So calling it "partial"
flags the rank-order surprise without throwing out the substantive
agreement.

**Source:** Chapter IV §4.5.3, Table 4.6 hypothesis verdict.
**Fallback:** *"Substantive claim supported; rank order not."*

---

# F. Deployment & Engineering

### F1. Why FastAPI instead of Flask or Django?

🟢 **Easy** — engineering panellist will ask.

**30-sec core:**

> Three reasons. First, FastAPI generates an OpenAPI schema and
> interactive `/docs` page automatically from the Pydantic models —
> the panel can hit `/docs` during the live demo and see exactly
> what the API expects. Second, it's async-native, which lets the
> `BackgroundTasks` pattern handle the SQLite log write without
> blocking the response — the user gets the prediction in under
> 500 ms even if the disk write is slow. Third, FastAPI is the
> current Python web-API standard and is what employers in BI/MLOps
> will recognise on a resume.

**Deep dive:** Flask would have worked for a synchronous, simpler API.
Django is overkill — a full ORM and admin for an inference API is
unnecessary weight. The async story matters here because the audit
log feeds the dashboard, and a synchronous Flask implementation
would either slow down `/predict` or risk losing log rows on
failure. The Pydantic schemas (in `src/app/schemas.py`) also enforce
input validation at the API boundary — invalid bookings get a 422
error with a structured response instead of a 500 crash.

**Source:** `src/app/main.py`, `src/app/schemas.py`.
**Fallback:** *"Async + Pydantic validation + auto-OpenAPI docs."*

---

### F2. Why SQLite for the audit log? Wouldn't Postgres scale better?

🟢 **Easy** — engineering question.

**30-sec core:**

> SQLite is right for the scale this thesis demonstrates — a single-
> property hotel making a few thousand bookings a month. Single-file
> database, no server process, no DBA. Power BI Desktop reads from
> the exported CSV anyway, so even Postgres wouldn't change the
> dashboard refresh path. For a multi-property enterprise rollout,
> Postgres makes sense — but the migration is a one-line connection-
> string change in `src/serving/prediction_log.py`. The schema is
> standard SQL.

**Deep dive:** The 43-column schema in
`src/serving/prediction_log.py` uses standard SQL types and
idempotent ALTER-TABLE migration logic, so the same code that runs
against SQLite would also run against Postgres or MySQL. I chose
SQLite for the thesis because it lets me hand the prof a `.zip`
file and have everything work on their laptop with zero setup. For
production at a hotel chain with 50+ properties, I'd switch to
Postgres and add a Power BI Service connection rather than Desktop.

**Source:** `src/serving/prediction_log.py`, CLAUDE.md "Live ADR
forecast" section.
**Fallback:** *"Single-property scale; one-line change to Postgres."*

---

### F3. What's the latency for a batch of 1,000 bookings — could you handle a 50,000-row daily batch?

🟡 **Medium** — scaling question.

**30-sec core:**

> Single-prediction latency is under 2 ms for the model and around
> 500 ms end-to-end including SHAP and the audit-log write. For
> batch, LightGBM's `predict_proba` is vectorised, so 1,000 rows
> take roughly the same time as 1 row — under 100 ms. A 50,000-row
> daily batch would take under 5 seconds for inference, plus the
> audit-log writes — maybe 60 seconds total including persistence.
> Easily within an overnight or per-shift batch window.

**Deep dive:** For very large batches, the bottleneck isn't the
model — it's the per-row Pydantic validation overhead and the SHAP
computation. SHAP especially: TreeSHAP is fast per row but doing it
for 50,000 rows would take a minute or two. The current `/predict`
endpoint takes one booking at a time; a batch mode that accepts a
list and runs SHAP only for the top-N high-risk rows is a
straightforward extension of the existing `predict_proba` (which
already vectorises across rows internally), but isn't implemented
in the current thesis deployment — it's a logical next step for a
multi-property rollout.

**Source:** `src/serving/inference.py::predict_proba`, `src/serving/inference.py::explain_prediction`.
**Fallback:** *"Sub-second for 1,000; SHAP is the bottleneck at scale."*

---

### F4. What if SQLite gets locked during high-traffic prediction bursts? Will `/predict` fail?

🟡 **Medium** — operations panellist.

**30-sec core:**

> No — the audit-log write happens in a FastAPI BackgroundTask, after
> the response has already gone back to the caller. If the SQLite
> write fails (transient lock, disk full, whatever), the prediction
> is still returned successfully and the failure is logged at
> WARNING level. The `/predict` endpoint never depends on the audit
> log succeeding. The trade-off is that under sustained failure
> some predictions might be missing from the log, but the operating
> system never sees a 500 error from the API.

**Deep dive:** The non-blocking pattern is built into FastAPI's
`BackgroundTasks` and the `log_prediction` function in
`src/serving/prediction_log.py` wraps the SQLite write in a
try/except that logs warnings without raising — so a transient lock
or disk failure surfaces in logs but never propagates back to the
caller. `tests/test_integration_train_serve.py` covers the
train → load → predict roundtrip but doesn't exercise the lock path
specifically. Under sustained high traffic, the right mitigation is
connection pooling via SQLAlchemy or a switch to Postgres (see F2).
For the single-property scale this thesis demonstrates, transient
locks are unlikely — SQLite handles thousands of writes per second
on commodity hardware.

**Source:** `src/serving/prediction_log.py`, `tests/test_integration_train_serve.py`.
**Fallback:** *"Background task — `/predict` never blocks on the log."*

---

### F5. How does this fit GDPR / data-privacy compliance?

🔴 **Hard** — likely from a hospitality industry panellist.

**30-sec core:**

> Two principles are in scope. First, **purpose limitation** — the
> audit log only stores fields directly used for the cancellation
> prediction; no payment info, no PII beyond what's already in the
> PMS. Second, **right to erasure** — every audit row has a
> `prediction_id` key, so deleting predictions for a specific guest
> is a single SQL DELETE. The 43-column schema doesn't store the
> guest's name or contact details; it stores the booking's
> *attributes* (country, market segment, agent), which are
> categorical and aggregable. So the log is pseudonymised by design.

**Deep dive:** The model itself never sees PII — features like
`country` are at the country-code level, `agent` is an integer ID,
not an agent name. The Power BI dashboard aggregates these into
counts and risk-tier distributions; individual guest identification
requires joining back to the PMS, which the hotel's existing
processes already control. For a strict GDPR audit, the recommended
addition would be a retention policy on the SQLite log — e.g.,
auto-purge predictions older than 90 days. This isn't implemented
in the current thesis deployment but would be a straightforward
scheduled-task addition on the same cadence as the drift
monitoring run.

**Source:** `src/serving/prediction_log.py` schema, CLAUDE.md.
**Fallback:** *"Pseudonymised by design; retention policy is a one-liner."*

---

# G. Power BI & BI Layer

### G1. Why Power BI Desktop instead of a web BI tool like Tableau or Looker?

🟢 **Easy** — BI/business panellist's go-to.

**30-sec core:**

> Three reasons. First, Power BI Desktop is free and runs on the
> revenue manager's existing Windows machine — no SaaS subscription,
> no IT-driven cloud rollout. Second, the data sources are local
> files (CSV, SQLite), which means the entire thesis stack runs
> offline — no broker, no service principal, no auth flow. Third,
> Power BI is what Philippine hotels actually use — IDeaS, Duetto,
> and other commercial RMS systems all integrate with Power BI more
> than with Tableau. The dashboard delivers in the format the user's
> existing analytics team can take over.

**Deep dive:** The deliberate choice was *Desktop* rather than
*Service* — Service requires a Microsoft 365 tenant and complicates
the deployment. Desktop reads the CSV with no auth, refreshes on a
button click, and shares as a `.pbix` file via email. For a hotel
chain that wants centralised dashboards across properties, migrating
to Service is straightforward — same DAX measures, same data model,
just a different connection.

**Source:** `docs/powerbi_dashboard_guide.md`.
**Fallback:** *"Free, local, what hospitality IT departments actually use."*

---

### G2. How does the dashboard support a non-technical revenue manager who's never seen ML before?

🟡 **Medium** — likely from BI professor.

**30-sec core:**

> The dashboard hides every ML concept behind a business label. There
> are no "logits" or "probabilities" or "F1 scores" on the
> manager-facing pages. Page 1 shows three KPI cards: "high-risk
> bookings this week," "expected revenue loss," and "interventions
> recommended." Page 2 is an action list — the manager clicks on a
> booking and sees "this booking has a 78 % chance of cancelling
> because the deposit type is non-refundable and the country is X."
> The SHAP top-5 features become plain-English reasons.

**Deep dive:** The narrative discipline is in the dashboard build
guide (`docs/powerbi_dashboard_guide.md`). Every chart has a one-
sentence caption in plain English. The "Trustworthiness" page
(Page 7) shows the calibration curve as "when the model says 75 %,
this is what actually happened" — no ECE, no Brier. The translation
of ML jargon to business language is in CLAUDE.md auto-memory
("Non-Technical Readability Pass") and applied throughout.

**Source:** `docs/powerbi_dashboard_guide.md`, Power BI dashboard
itself.
**Fallback:** *"Plain-English labels; no ML jargon."*

---

### G3. Can a manager drill into a specific high-risk booking and see why the model flagged it?

🟡 **Medium** — likely from BI/operations panellist.

**30-sec core:**

> Yes — Page 2 of the dashboard is an action list with one row per
> booking, sorted by predicted probability descending. Clicking a
> row expands the SHAP top-5 features for that specific booking —
> e.g., "country (PRT): +0.83 toward cancel", "deposit type (Non
> Refund): +0.45 toward cancel", "lead time (250 days): +0.21". The
> manager sees both the prediction *and* the explanation, in one
> click. This is what makes the model defensible to a customer if
> the hotel decides to ask for a deposit.

**Deep dive:** The SHAP per-row data is stored in the
`top_features` column of `predictions_live.csv`, as JSON. Power BI
parses the JSON via a measure and unpacks the top features into a
table visual. The data engineering is in
`src/serving/inference.py::explain_prediction` and the Power BI
binding is in `docs/powerbi_dashboard_guide.md` Page 2.

**Source:** `docs/powerbi_dashboard_guide.md` Page 2,
`src/serving/inference.py`.
**Fallback:** *"Page 2: click a booking, see SHAP top-5 reasons."*

---

### G4. Can the `.pbix` file be shared without re-running the entire pipeline?

🟢 **Easy** — practical sharing question.

**30-sec core:**

> Yes. The `.pbix` file embeds the data model and visuals; it
> connects to the CSV files via relative paths. To share, you zip
> the `.pbix` together with the CSV files (around 3 MB total) and
> hand it to the recipient. They open the `.pbix` in Power BI
> Desktop, refresh once, and the dashboard works. No service
> account, no Power BI Service tenant, no ODBC driver. This was a
> deliberate design choice — the artefact is portable.

**Deep dive:** For organisations that want centralised refresh, the
migration to Power BI Service is straightforward — change the data
source from "Local Folder" to "Azure Blob Storage" or "OneDrive for
Business" and the same `.pbix` works against the cloud-stored CSVs.
This is also why the data update path is "rerun `make
export-predictions` → CSV is regenerated → refresh in Power BI" —
the .pbix never needs editing; only the CSVs change.

**Source:** `docs/powerbi_dashboard_guide.md` "Refreshing" section.
**Fallback:** *"Zip `.pbix` + CSVs; recipient refreshes once. Portable."*

---

# H. Limitations & Honest Reporting

### H1. The ADR regressor's R² is only 0.234. Isn't that a weak model?

🟡 **Medium** — defensive limitation question.

**30-sec core:**

> R² 0.234 is moderate, but for ADR prediction it's the right number.
> ADR is dominated by *rate-card pricing* — room type, season,
> channel, rate plan — which the model captures. The remaining
> variance is from *booking-specific randomness* the model can't see:
> promotional codes, loyalty discounts, group rates, last-minute
> upgrades, dynamic pricing decisions. An R² above 0.30 on this
> structure would suggest the model has access to leakage. The
> thesis claim is **directional ADR signal at booking time** — the
> model correctly *orders* bookings by likely revenue. That's
> enough for prioritisation; you don't need an exact ADR.

**Deep dive:** The dashboard uses ADR for *risk-tier prioritisation*
— combining cancellation probability with predicted ADR to identify
the "high-cancel × high-revenue" bookings that deserve urgent
attention. For that use case, ordering matters more than absolute
accuracy. The "directional pricing signal" framing is in Chapter IV
§4.6.2 and is explicit about this. If a panellist insists on exact
ADR prediction, the right answer is "retrain on booking-time features
only and add competitor rate scrapes" — Future Research extension 4
in Chapter V.

**Source:** Chapter IV §4.6.2, Chapter V §5.5 FR4.
**Fallback:** *"Directional signal, not exact prediction; R² > 0.3 would mean leakage."*

---

### H2. The model is trained on 2015 – 2017 pre-pandemic data. Will it work in 2025?

🔴 **Hard** — every panel will ask about generalisation.

**30-sec core:**

> Honest answer: I don't know yet, and that's why the deployment
> framework includes drift monitoring. The PSI page in the dashboard
> would flag if the booking distribution has shifted significantly
> from training time — that's the alert mechanism. For a hotel
> deploying this in 2025 against 2024 data, the right protocol is to
> validate the metrics on their own 2024 holdout before relying on
> the headline numbers. The methodology is robust; the specific
> calibration is dataset-specific. Chapter V §5.4 flags this
> limitation explicitly.

**Deep dive:** The structural shift in cancellation behaviour is
expected to be towards *more* cancellations and *more* last-minute
ones. The cost-sensitive policy would adapt automatically — the
optimum threshold would move down if cancellations are more
common — but the *level* of recovery might shift. The retraining
trigger (PSI ≥ 0.25 on ≥ 2 features) is the mechanism that catches
this without manual intervention. I deliberately don't claim the
2017 numbers transfer to 2025 — I claim the *methodology* does, and
the dashboard monitors when it stops.

**Source:** Chapter V §5.4 "Single benchmark dataset" limitation +
§5.4 "Chronological split assumes stationarity."
**Fallback:** *"Don't know yet; PSI drift monitor catches the shift."*

---

### H3. Why didn't you A/B test the intervention policies in production?

🟡 **Medium** — natural follow-on.

**30-sec core:**

> Time, access, and ethics. A/B testing requires a deployed system
> running for at least three months with real bookings randomly
> assigned to "intervention" and "control" arms — that's a budget
> and partnership level beyond a thesis. The honest reporting is
> that the €2.94 M number is an *upper bound* — what's recoverable
> if guests respond to interventions at the rates the cost model
> assumes. The measured rate is unknown until the A/B trial runs.
> Future Research extension 3 in Chapter V proposes exactly this as
> the next step.

**Deep dive:** The thesis stays honest by reporting both the
*identification* number (€2.94 M, what the model can flag) and
explicitly flagging that the *recovery* number depends on guest
response rates. The cost-sensitive policy can be tuned post-A/B —
if reminders work 30 % rather than the implied 100 %, the
cost-sensitive threshold would shift, but the *ranking* of bookings
by risk wouldn't. The model itself is robust; the action layer
needs the trial.

**Source:** Chapter V §5.4 "No A/B testing," §5.5 FR3.
**Fallback:** *"Upper bound; A/B trial is Future Research extension 3."*

---

# I. Trick & Hard Questions

### I1. Aren't all your headline numbers from one dataset? How do you know they generalise?

⚫ **Trick** — the "single benchmark" attack.

**30-sec core:**

> They are from one dataset, and I've been transparent about that
> throughout the thesis. What I claim is that the *methodology*
> generalises — the chronological split, the rolling-origin model
> selection, the isotonic calibration, the cost-sensitive
> thresholding, the SHAP attribution, the dashboard deployment —
> applies to any hotel cancellation dataset. The Philippine
> sub-study at Punta Villa Resort (n=193) tested exactly this and
> the methodology produced honest results even at small scale —
> the pre-flight diagnostic ran cleanly, the chronological PR-AUC
> was 0.54, the top SHAP driver was deposit_type, just like
> Portugal. So the *methodology* survives transfer; the *headline
> numbers* are Portugal-specific and I never claim otherwise.

**Deep dive:** This is also why Chapter V Recommendation 1 talks
about "risk tiers" in operational terms rather than committing to
specific probability bands universally — the 0.40/0.70 cuts are
calibrated to Portugal. A new property would recalibrate those by
running the same threshold sweep on its own validation slice.
Future Research extension 2 proposes replicating on 10-15
Philippine properties to produce a regional headline number that's
defensible at industry scale.

**Source:** CLAUDE.md PH section, Chapter V §5.4 + §5.5 FR2.
**Fallback:** *"Methodology generalises; headlines are Portugal-specific."*

---

### I2. Why didn't you compare against commercial RMS systems like IDeaS, Duetto, or RateGain?

⚫ **Trick** — industry-adviser panellist.

**30-sec core:**

> Two reasons. First, commercial RMS systems don't publish their
> internal classifiers — they're proprietary. I can't benchmark
> against a black-box. Second, commercial RMS solves a different
> problem — *rate optimisation* for revenue maximisation, not
> *cancellation prediction* for intervention. IDeaS predicts the
> right *price* to charge; my thesis predicts the right *action* to
> take for a booking already made at a known price. Both are
> revenue-management tools but they're complementary, not
> substitutes. A hotel running IDeaS could plug my cancellation
> score into their pricing decision; the two would compose.

**Deep dive:** I do reference Antonio et al. (2017) — the source
dataset paper — as the academic precedent. That paper achieved
roughly 86 % accuracy at a fixed threshold, which is in the same
range as my `max_f1` accuracy of 77 %. The reason mine is lower
on accuracy is that I optimise for PR-AUC and cost-sensitive
recovery, not for accuracy. The numbers aren't directly
comparable because the operating policies differ.

**Source:** Chapter II literature review (references to Antonio et
al. 2017 and the Portugal benchmark).
**Fallback:** *"Different problem — they optimise price, I optimise intervention."*

---

### I3. A panellist says "this is just a black box". How do you respond?

⚫ **Trick** — defensive question, common at every defence.

**30-sec core:**

> Two-part answer. First, this is *not* a black box — every prediction
> ships with the SHAP top-5 features, every decision is traceable to
> a calibrated probability, and the model card in
> `model_metadata.json` records the training data, the algorithm,
> the version, and the lineage. Second, even if the underlying
> classifier *were* opaque, the *deployment* is not — the dashboard
> shows the prediction, the SHAP explanation, the operating threshold,
> and the recommended action. The hotel sees *both* the score and
> the reason. A black-box critique applies to models with no
> attribution layer; this deployment has one.

**Deep dive:** The TreeSHAP approach I use was introduced by
Lundberg, Erion & Lee (2018, *Consistent Individualized Feature
Attribution for Tree Ensembles*, arXiv:1802.03888) as a tree-
specific algorithm with polynomial-time exact Shapley values. The
broader SHAP unified framework is from Lundberg & Lee (NeurIPS
2017). Every gradient-boosted tree decision can be decomposed back
to feature contributions with mathematical guarantees (the Shapley
axioms: efficiency, symmetry, dummy, additivity). The thesis
Chapter IV §4.5 explains this in plain English; the technical
details are in Notebook 05.

**Source:** Chapter IV §4.5, Notebook 05, Lundberg & Lee 2017
(unified SHAP); Lundberg, Erion & Lee 2018 (TreeSHAP).
**Fallback:** *"SHAP top-5 per prediction; the dashboard shows the reasons."*

---

### I4. Why €15 specifically for the false-positive intervention cost? Where does that number come from?

🔴 **Hard** — strict-economics panellist.

**30-sec core:**

> €15 is the industry-range estimate for the marginal cost of an
> automated email reminder — staff time, system maintenance,
> opportunity cost of the inbox. It's documented in `src/config.py`
> as `FP_INTERVENTION_COST` and the sensitivity analysis in
> Notebook 10 demonstrates that the policy ranking
> (cost-sensitive < max_f1 < high_precision) is robust across the
> €5 to €60 range — the *operating point* moves, but the
> three-policy ordering doesn't change. So the €15 specifically
> matters for the absolute cost numbers; it doesn't matter for
> the policy recommendations.

**Deep dive:** In the deployed system, the €15 is a per-property
configurable constant — a hotel that knows its actual SMS or email
unit cost would override it. The Power BI dashboard's Page 4
exposes the cost-sensitivity sweep visually, so a manager can see
"if our intervention costs €30 instead of €15, what's the optimal
threshold?" directly. This is a more defensible answer than
hard-coding a single number — the framework supports any FP cost.

**Source:** `src/config.py::FP_INTERVENTION_COST`, Notebook 10
sensitivity analysis, Chapter IV §4.7.2.
**Fallback:** *"Industry-range estimate; sensitivity-tested across €5 – €60."*

---

# J. Business & Industry Framing

### J1. How would you explain this thesis to a hotel GM in 30 seconds?

🟢 **Easy** — common closing question.

**30-sec core:**

> *Hotels lose three million euros to cancellations across a typical
> summer. I built a model that scores every new booking the moment
> it's made, flags the ones likely to cancel, and tells the front
> desk what to do — send a reminder to the medium-risk bookings,
> call the high-risk ones, leave the low-risk ones alone. On the
> 2017 Portugal benchmark, the model recovers ninety-seven point
> five percent of that three million euros. It runs in five hundred
> milliseconds on a laptop, lives inside the existing PMS workflow,
> and ships with a Power BI dashboard the revenue manager can refresh
> on demand.*

**Deep dive:** Notice this skips every ML term — no SHAP, no
calibration, no PR-AUC. The GM doesn't care how it works; they care
that it pays for itself. The three key business hooks: (1) money
recovered (€2.94 M / 97.5 %), (2) action per tier (clear playbook),
(3) zero IT lift (laptop + Power BI). If the GM asks a technical
follow-up, escalate to me; if they ask "what's the ROI?", the
answer is positive within one booking cycle.

**Source:** Chapter V §5.1 + §5.3 R1.
**Fallback:** *"€3M loss → model flags → tiered action → €2.94M recovered."*

---

### J2. What's the ROI per booking the hotel processes? Is the operational cost worth it?

🟡 **Medium** — hospitality-adviser panellist.

**30-sec core:**

> Per booking, the deployment cost is essentially zero — once
> trained, predictions cost under 2 ms on a laptop, and the
> dashboard refresh is free. The recovery is up to €250 per
> high-risk booking, depending on the booking's revenue at risk.
> Operationally, the cost is the time the front desk spends on
> reminders and confirmation calls — roughly 2 minutes per
> Medium-tier reminder and 10 minutes per High-tier call.
> Multiplied by 3,108 High-tier bookings (test sample), that's
> about 520 staff-hours per quarter, against €1.57 M of revenue
> exposure in that tier alone. ROI is at least 100×.

**Deep dive:** Recommendation 4 in Chapter V also notes that the
intervention shouldn't be uniform across the High tier — the
hotel can sort the action list by predicted ADR (Page 2 of the
dashboard) and call the highest-revenue bookings first. That
pareto-optimises staff time. The deployment specifically supports
this prioritisation because the predict endpoint returns both
cancel probability AND predicted ADR.

**Source:** Chapter V §5.3 R3 + R4, dashboard Page 2.
**Fallback:** *"€250/booking recovery vs 10 min staff time. ROI ~100×."*

---

### J3. Will this work in the Philippine market — different culture, different booking patterns?

🟡 **Medium** — likely from a Filipino industry panellist.

**30-sec core:**

> The Philippine sub-study at Punta Villa Resort tested exactly that —
> 193 real PMS bookings from 2022-2025. The methodology survives the
> transfer: the pre-flight duplicate-cluster diagnostic ran cleanly,
> isotonic calibration worked, the SHAP analysis ran. The *headline
> metrics* are directional only at twenty test rows (PR-AUC 0.54 with
> ±15 pp confidence intervals), but the substantive finding — that
> `deposit_type` is the #1 driver on *both* datasets — suggests
> the same operational signal is at work. Replication on 10-15
> Philippine properties is Future Research extension 2.

**Deep dive:** Cultural differences in booking behaviour do exist —
e.g., Philippine resort bookings have a much higher proportion of
domestic bookings vs international, different agent/distribution mix,
and shorter average lead times. But the SHAP top features should
still capture them — the model would learn that a different mix of
channels and customer types matters, but the *type of feature*
(channel reliability, booking source) is universal. The honest claim
is "methodology validated, region-specific calibration required."

**Source:** Chapter V §5.4 PH limitation, §5.5 FR2.
**Fallback:** *"Methodology survives; metrics need region-specific calibration."*

---

### J4. If this is so good, why hasn't every hotel chain deployed it already?

⚫ **Trick** — the "but does it really work?" attack.

**30-sec core:**

> Two reasons. First, **commercial RMS vendors aren't transparent** —
> hotels using IDeaS or Duetto pay for closed-source predictive
> services that might do similar things, but the customer can't
> audit them. My thesis open-sources the methodology so any hotel
> can deploy without licensing. Second, **deployment friction is
> still real** — you need a Python environment, a Power BI
> licence, weekly drift monitoring, and a front-desk team willing
> to follow the intervention playbook. Many hotels don't have the
> data-engineering capacity. This thesis lowers the friction by
> making everything a flat file + a single Python process.

**Deep dive:** Big chains (Marriott, Hilton) probably do something
similar internally. Mid-market chains and independent properties
typically don't — they either rely on commercial RMS or do nothing.
This thesis fills the gap for the independent / mid-market segment
where commercial RMS is too expensive but the data is rich enough
to support a custom solution. The Mapúa BI angle is that the
methodology is *teachable* — a graduate student can replicate the
entire pipeline in a semester.

**Source:** Chapter I motivation, CLAUDE.md project overview.
**Fallback:** *"Vendors are closed-source; deployment friction is real but solvable."*

---

# K. Codebase & Implementation Tour

### K1. Where exactly did you do the machine learning in Python? Walk me through the codebase.

🟢 **Easy** — but easy to fumble. Rehearse the spoken answer.

**30-sec core:**

> All the machine-learning code lives in the `src/` package, organised
> by responsibility — `src/data/` for loading, `src/features/` for
> engineering and the chronological split, `src/models/` for
> algorithm-specific trainers and metrics, `src/utils/` for threshold
> logic, and `src/pipelines/train.py` for master orchestration. The
> stack is pandas, numpy, scikit-learn, LightGBM, XGBoost, and SHAP.
> Training runs via `python scripts/train.py` or `make train`. Serving
> is in `src/serving/inference.py` behind a FastAPI endpoint. There
> are 130 tests in `tests/` at 88 % coverage that verify the pipeline
> end-to-end.

**Deep dive (if asked to show on screen):**

> Open three files in order:
>
> 1. **`src/pipelines/train.py`** — the master entry point. The
>    `run_training_pipeline` function (around line 667) loads the CSV,
>    drops the five leakage columns via `LEAKAGE_COLS`, runs
>    `split_time_aware` for the chronological 80/10/10, does
>    rolling-origin champion-challenger selection across LightGBM,
>    XGBoost, and Gradient Boosting (via `_fit_model_family` and
>    `_select_model_family`), fits the isotonic calibrator on the
>    validation set (`_fit_probability_calibrator`), runs the
>    threshold sweep, and saves every artifact under `artifacts/`
>    with SHA-256 lineage in `model_metadata.json` (built by
>    `_artifact_lineage` and persisted via `_save_json`).
> 2. **`src/models/train.py`** — three trainer functions:
>    `train_lgbm`, `train_xgb`, `train_gb`. Each returns an sklearn
>    Pipeline with a ColumnTransformer (for OneHotEncoder +
>    SimpleImputer) plus the classifier. Same shape, apples-to-apples
>    comparison in the selection step. Default hyperparameters come
>    from `get_default_lgbm_params`, `get_default_xgb_params`,
>    `get_default_gb_params`.
> 3. **`src/serving/inference.py`** — the live `/predict` runtime.
>    Loads artifacts once via `load_artifacts` and caches the result
>    in the `_CACHED_ARTIFACTS` singleton protected by the
>    `_CACHED_ARTIFACTS_LOCK` threading lock — that's the
>    double-checked locking pattern. Each call runs Pydantic
>    validation, feature engineering, `predict_proba`, the isotonic
>    calibrator, threshold resolution, the SHAP top-5 explanation
>    (`explain_prediction`), and the parallel ADR forecast
>    (`predict_adr`). End-to-end under 500 ms.

**Source:** `src/pipelines/train.py`, `src/models/train.py`, `src/serving/inference.py`.
**Fallback:** *"`src/` package, master entry `src/pipelines/train.py`, CLI via `scripts/train.py`."*

---

### K2. Show me where you do the chronological train / val / test split.

🟢 **Easy** — they want to see the leakage prevention.

**30-sec core:**

> The split function is `split_time_aware` in `src/features/build.py`.
> It calls a helper `sort_by_arrival_date` that builds a derived
> `_arrival_date` datetime column from `arrival_date_year`,
> `arrival_date_month`, and `arrival_date_day_of_month` (via
> `add_arrival_date`), then slices the sorted DataFrame into the
> 80/10/10 chunks chronologically. The split ratios are constants in
> `src/config.py` (`TRAIN_RATIO = 0.80`, `VAL_RATIO = 0.10`). I
> deliberately avoid `sklearn.model_selection.train_test_split`
> because it shuffles by default. There's a test in
> `tests/test_split_and_leakage.py::test_split_time_aware_is_chronological`
> that asserts `train_dates.max() <= val_dates.min()` and
> `val_dates.max() <= test_dates.min()` — no chronological overlap.

**Deep dive:**

> The function returns three DataFrames in date order — train, val,
> test — with reset indices so downstream code doesn't accidentally
> depend on row positions. The dropped leakage columns (`LEAKAGE_COLS`
> from `src/config.py`) are removed *before* the split function is
> called, in `src/pipelines/train.py` at the line
> `df = df.drop(columns=LEAKAGE_COLS)`. That ordering matters — if
> leakage columns were removed *after* the split, the validation set
> would briefly have access to leaky data during fit.

**Source:** `src/features/build.py::split_time_aware`, `src/config.py::TRAIN_RATIO`/`VAL_RATIO`/`LEAKAGE_COLS`, `tests/test_split_and_leakage.py`.
**Fallback:** *"`src/features/build.py::split_time_aware`. Chronological, ratios in `src/config.py`."*

---

### K3. Where does the isotonic calibrator come from and how is it fitted?

🟡 **Medium** — calibration-savvy panellist.

**30-sec core:**

> The calibrator is `sklearn.isotonic.IsotonicRegression`, fitted in
> `src/pipelines/train.py` in the function `_fit_probability_calibrator`.
> It's fitted on the *validation* set's raw model probabilities and
> the validation labels — never on the test set. The fitted object is
> pickled to `artifacts/probability_calibrator.pkl` alongside the
> model. At inference time, `src/serving/inference.py` loads the
> calibrator once and applies it to every raw probability via
> `calibrator.predict(raw_proba)`.

**Deep dive:**

> The reason the validation set is used (not the training set) is to
> avoid optimistic calibration — the model's training-set probabilities
> are over-confident due to fit. The validation set is a held-out
> slice the model hasn't seen during training, so the
> probability-to-frequency mapping is honest. The calibrator's output
> is clipped to [0, 1] in inference.py defensively, even though
> isotonic regression already returns values in that range — that's a
> belt-and-braces guarantee for the API.

**Source:** `src/pipelines/train.py::_fit_probability_calibrator`, `src/serving/inference.py`, `artifacts/probability_calibrator.pkl`.
**Fallback:** *"`IsotonicRegression` fitted on val set, in `src/pipelines/train.py`."*

---

### K4. Where are the thresholds (`max_f1`, `high_precision`, `cost_sensitive`) chosen — show me the code.

🟡 **Medium** — operational-mechanics question.

**30-sec core:**

> Threshold selection logic is in `src/utils/thresholds.py`. The
> three selectors are `select_max_f1_threshold`,
> `select_high_precision_threshold`, and `select_min_cost_threshold`
> (with helpers `threshold_sweep` and `cost_threshold_sweep` that
> build the grids via `_make_threshold_grid`). Each sweeps across
> the validation set's predicted probabilities at 0.01 increments
> using `np.linspace`, computes the policy's metric at each candidate
> threshold, and picks the optimum. The chosen thresholds are saved
> to `artifacts/thresholds.json` and the full sweep grid is saved to
> `artifacts/threshold_sweep.csv` for the dashboard.

**Deep dive:**

> The cost-sensitive function uses
> `FP_INTERVENTION_COST = 15.0` from `src/config.py` and computes
> total cost as `FP_count * 15 + FN_revenue_at_risk`. It picks the
> threshold that minimises this total. There's also a
> `resolve_thresholds` helper that the FastAPI endpoint calls — it
> handles the falsy-zero edge case (a cost-sensitive threshold of
> 0.0 is valid, but `0.0 or 0.5` would silently fallback to 0.5; I
> use explicit None-checks instead). Tests for this edge case are in
> `tests/test_thresholds.py`.

**Source:** `src/utils/thresholds.py`, `artifacts/thresholds.json`, `tests/test_thresholds.py`.
**Fallback:** *"`src/utils/thresholds.py`. Sweep on val set at 0.01 steps."*

---

# Final rehearsal protocol

> **One week before defense:**
> Read every question cold, write 30-sec answers. Score yourself —
> green / yellow / red per question. Focus rehearsal on yellows
> and reds.
>
> **Three days before defense:**
> Practise out loud against a stopwatch. Each answer should land at
> 25 – 45 seconds. If you're consistently going over 50 s, your
> answer is too long; cut it.
>
> **One day before defense:**
> Read only the 30-sec cores. Don't re-read the deep dives — they're
> for emergency cases only. Sleep early.
>
> **Defense morning:**
> Re-read the "Final rehearsal protocol" and section J3 (the
> Filipino-market question). Those are the questions you're most
> likely to be asked at the very end, when the panel is gauging
> whether you understand the broader impact.

---

# Cross-reference to defense script Q&A

These are the 10 questions already in `defense_script.md` (don't
double-prepare):

- Q1: Why LightGBM over XGBoost (PR-AUC gap 0.011)?
- Q2: Why `deposit_type` #1 instead of `lead_time` (H3)?
- Q3: Won't €2.94 M overstate real savings?
- Q4: 16.2 pp PR-AUC gap CV vs chronological — what's happening?
- Q5: Post-pandemic behaviour?
- Q6: How does the dashboard know when to retrain?
- Q7: Non-refundable deposit predicts more cancellation — why?
- Q8: PH sub-study only has 20 test rows — what's the point?
- Q9: Per-segment fairness?
- Q10: Won't the hotel over-flag Direct bookings?

This file (40 new questions) + script Q1–Q10 = **50 total questions
prepared.** That's enough rehearsal material to face any reasonable
panel.

---

*End of defense practice Q&A.*
