# Jargon Translation Guide

> Senior Data Scientist → BI undergrad: this is the master reference for
> rewriting any technical text in the project so it reads cleanly to a
> business panel. Use it for notebook markdown cells, slide bullets,
> defence answers, and any future code-level documentation.

## Guiding principle

Translate the **concept**, not the term. Keep the metric name (so a
reader can trace it to the artefact), but always pair it with a
plain-English line that says **what it actually means for the
business**. Never drop a number without context — every number in the
thesis should answer the implicit question *"so what?"*.

Three rules to apply consistently:

1. **First mention of any acronym gets an explanation in plain English**,
   even if it appears twenty times later in the same document.
2. **State the business consequence**, not just the statistical fact.
   "ECE dropped from 0.058 to 0.029" is a statistical fact; "after
   calibration, when the model says 30 % chance of cancellation, the
   truth really is around 30 %" is the business consequence.
3. **If a sentence has more than one acronym in it, rewrite it.** Most
   business readers tune out after the second acronym in a sentence.

---

## Master glossary

The right-hand column is the translation to use on first mention. After
the first mention, the original term is fine.

| Technical term | Plain-English version | Example sentence |
|---|---|---|
| **PR-AUC** (Precision-Recall Area Under the Curve) | A single score from 0 to 1 that measures how well the model balances catching real cancellations against not raising false alarms. 0.5 = no better than chance on a balanced dataset; 1.0 = perfect. | "Our champion model scores 0.760 — meaningfully above the natural cancellation rate of 0.37." |
| **ROC-AUC** | A single score from 0 to 1 that answers: "If we pick one random canceller and one random stayer, how often does the model give the canceller a higher score?" 1.0 = always correctly ordered; 0.5 = coin-flip. | "A ROC-AUC of 0.86 means the model correctly orders cancellers above stayers about 86 % of the time." |
| **F1 score** | A single score that balances precision (don't cry wolf) with recall (catch the cancellations). Useful for comparing models at one specific decision cut-off. | "At the balanced decision cut-off, our model achieves F1 = 0.73." |
| **Precision** | Of the bookings the model flagged as "will cancel", what fraction actually cancelled? Higher means fewer false alarms. | "A precision of 0.65 means roughly 2 of every 3 flagged bookings really cancel." |
| **Recall (Sensitivity)** | Of the bookings that actually cancelled, what fraction did the model catch? Higher means fewer missed cancellations. | "A recall of 0.84 means the model catches 84 % of the cancellations that actually happen." |
| **Brier score** | A combined measure of how close the predicted percentages are to the actual outcomes. Lower is better; 0 is perfect. | "A Brier score of 0.146 — solid for business decision support." |
| **ECE** (Expected Calibration Error) | The average gap between the model's predicted percentages and the actual cancellation rates. Smaller = more honest percentages. | "ECE of 0.029 means a '30 %' prediction matches a real cancellation rate of about 27-33 %." |
| **Isotonic regression / calibration** | A simple mathematical adjustment after training that re-maps the model's raw scores so the percentages match real-world frequencies. Without it, "70 %" might really mean 50 %. | "We apply isotonic calibration so the percentages a manager sees are honest." |
| **Threshold (decision cut-off)** | The percentage above which we flag a booking as "will cancel". Different cut-offs trade off precision and recall. | "At a 40 % cut-off, the model flags any booking with predicted risk ≥ 40 %." |
| **Rolling-origin cross-validation** | A time-respecting test: train on the oldest 60 % of bookings, evaluate on the next slice; then train on 70 %, evaluate on the next slice; then 80 %. Avoids using future bookings to train a model. | "We train the model three times on growing time windows to confirm it works on each future slice." |
| **Bootstrap paired-significance test** | Re-draw thousands of random samples from the test set with replacement; check whether one model's lead over another holds up across those re-draws. If it does, the lead is statistically real, not luck. | "Across 2,000 resamples LightGBM stayed ahead of every other model, so the lead is real." |
| **Bootstrap confidence interval** | The range we'd expect a number to fall in if we re-ran the study many times. Wider intervals mean less certainty. | "Our PR-AUC of 0.54 has a 95 % confidence interval of [0.32, 0.82] — a wide range at this sample size." |
| **SHAP (SHapley Additive exPlanations)** | A score showing how much each booking detail (lead time, deposit type, etc.) pushed the model toward "cancel" or "will arrive" on a single booking. The bigger the score, the bigger the influence. | "On this booking, deposit type contributed +1.80 toward cancel — by far the strongest signal." |
| **mean(\|SHAP\|)** | The average size of a feature's contribution across all test bookings. Used to rank which booking details matter most overall. | "Across the whole test set, deposit type is the #1 ranked driver." |
| **Calibration reliability diagram** | A plot that confirms the model's percentages match real cancellation rates. A perfect line up the diagonal = perfect calibration. | "The reliability plot stays close to the ideal diagonal, confirming the percentages are honest." |
| **Chronological 80/10/10 split** | We sort bookings by arrival date, train on the oldest 80 %, tune the cut-off on the next 10 %, and reserve the most-recent 10 % to score the model once. | "We test the model only on bookings the model hasn't seen yet." |
| **Out-of-time test** | Same as chronological split — the test set is in the future relative to the training set. | (Used as synonym.) |
| **Class imbalance** | When one outcome (cancellation) is less common than the other (arrival). PR-AUC and the cost-sensitive cut-off handle this naturally. | "Cancellations are 37 % of the Portugal data and 15 % of the PH data — we use PR-AUC instead of accuracy because of this imbalance." |
| **Positive class probability** | The model's predicted percentage that a booking will cancel. (Avoid this phrase — use **Cancellation Risk** instead.) | "Cancellation Risk: 78 %" |
| **False positive (FP)** | A booking the model flagged as "will cancel" that actually arrived. Costs a small intervention (e.g., a reminder email). | "Each false positive costs about €15 in unnecessary outreach." |
| **False negative (FN)** | A booking the model said would arrive that actually cancelled. Costs the lost revenue from an unsold room. | "Each false negative costs roughly one night's ADR." |
| **Cost-sensitive threshold** | Picking the decision cut-off that minimises total expected cost, given the relative cost of each error type. | "Because each false negative is more expensive than each false positive, we set the cut-off low (0.04) to catch nearly every cancellation." |
| **PSI (Population Stability Index)** | A measure of how much the live booking mix has shifted compared to the data the model was trained on. > 0.25 = material drift, retrain. | "We monitor PSI weekly to catch behaviour shifts before model quality degrades." |
| **Drift monitoring** | The practice of checking whether the live data still looks like the data the model was trained on. | "If guest behaviour shifts post-pandemic, the drift monitor will flag it before performance drops." |
| **Validation set** | The slice of data used to tune the model (calibration, decision cut-off). Never used to report final numbers. | "We tune the cut-off on the validation set, then test on a separate held-out test set." |
| **Held-out test set** | The final, untouched slice of data. Used once at the end to report honest performance. | "Our headline ROC-AUC of 0.864 comes from the held-out test set the model never saw during training." |
| **Sense → Seize → Transform** | The three-stage cycle from Dynamic Capability Theory: spot what's happening in the data, decide what to do, then change how the business operates to capture the value. | (Used throughout the thesis structure.) |
| **Transferability probe** | A small follow-up study that checks whether the methodology developed on one dataset works on a second, different dataset. | "The Punta Villa study is a transferability probe of the methodology developed on Portugal." |
| **Pre-flight duplicate-cluster diagnostic** | A check we run before training a small-dataset model: do many rows share identical feature combinations? If yes, the chronological split could leak look-alikes across train/test. | "The diagnostic does not fire on the real Punta Villa data — methodology proceeds cleanly." |
| **Feature engineering** | The process of computing derived booking signals from raw fields — total stay length, weekend-heavy flag, revenue at risk, etc. | "We engineer 18 booking-time features from 10 raw columns." |
| **Hyperparameter** | A configuration setting of the model itself (e.g., tree depth, learning rate). Different from the things the model learns from data. | "We use sensible defaults rather than tuning hyperparameters extensively, because the Portugal champion already meets the metric gates." |
| **Pipeline** | The full chain of steps a booking goes through: clean → engineer features → score → calibrate → assign risk tier. | "The pipeline runs the same way for every booking, whether it's a Portugal historical record or a new Punta Villa reservation." |

---

## User-facing label translation table

These are the strings that appear in the Gradio UI, FastAPI responses,
and Power BI columns. The "Display label" column is what the
user/manager should see; the "Internal key" column is what the code
uses (and must NOT be renamed — that would break dashboards and APIs).

| Internal key (do not rename) | Display label (use this in UI / docs) |
|---|---|
| `probability` / `cancel_probability` | **Cancellation Risk** |
| `predicted_max_f1` / `label_max_f1` | **Cancel Flag — Balanced policy** |
| `label_high_precision` | **Cancel Flag — High-confidence policy** |
| `label_cost_sensitive` | **Cancel Flag — Cost-aware policy** |
| `threshold_max_f1` | **Cut-off — Balanced policy** |
| `threshold_high_precision` | **Cut-off — High-confidence policy** |
| `threshold_cost_sensitive` | **Cut-off — Cost-aware policy** |
| `risk_tier` | **Risk Level** (or just **Tier**) |
| `top_features` | **Top Booking-Detail Influences** |
| `predicted_adr` | **Predicted Daily Rate** |
| `adr_residual` | **Daily-Rate Variance** (i.e., how much the entered rate differs from what the model expects) |
| `roc_auc_test` | **Ranking accuracy (Test)** |
| `pr_auc_test` | **Class-balanced score (Test)** |
| `f1` / `f1_at_max_f1_threshold` | **Balanced score (Test)** |
| `ece` / `ece_calibrated` | **Honesty gap** (lower = more honest percentages) |
| `brier_calibrated` | **Probability error (Test)** |
| `n_train` / `n_val` / `n_test` | **Training rows / Tuning rows / Test rows** |
| `positive_rate` | **% flagged as cancel** |
| `recall` | **Cancel-catch rate** |
| `precision` | **Flag-accuracy rate** |
| `feature_importance` | **Booking-detail influence ranking** |
| `mean_abs_shap` | **Average influence score** |
| `shap_summary_plot` | **What drives predictions (chart)** |
| `dataset_diagnostics` | **Data health check** |
| `duplicate_rate` | **% of bookings that look identical** |
| `cost_sensitive_threshold` | **Cost-aware cut-off** |
| `savings_vs_no_model` | **Estimated savings** |
| `arrival_date` | **Arrival date** |
| `lead_time` | **Days from booking to arrival** |
| `deposit_type` | **Deposit policy** |
| `reserved_room_type` | **Room type** |
| `total_of_special_requests` | **Special requests count** |

### Important: keep internal keys unchanged

The right column changes only the **display label** — the underlying
column names in `data/predictions/predictions_live.csv`, the field
names in `/predict` responses, and the SQLite schema must remain
unchanged because Power BI bindings, automated tests, and external
integrations depend on them. The mapping is applied at the moment of
display (Gradio render, Power BI column rename in the dashboard
visual), not at data persistence.

---

## Notebook markdown rewrite patterns

The following patterns apply across most notebooks. When you see the
**left** wording, replace with the **right** wording.

| Replace this | With this |
|---|---|
| "calibrated probability" | "the honest percentage chance of cancellation" |
| "log-loss" (if it appears) | "average prediction error" |
| "binary classification" | "predicting cancel-or-arrive" |
| "supervised learning" | "the model learns by example from past bookings whose outcome we already know" |
| "out-of-time evaluation" | "we test on more-recent bookings that the model never saw during training" |
| "feature importance" | "which booking details drive the prediction" |
| "the positive class" | "cancellations" |
| "the negative class" | "arrivals" |
| "logits" / "raw scores" | "uncalibrated model scores" |
| "TreeSHAP" / "KernelSHAP" | "SHAP feature attributions" |
| "ensemble" | "a model that combines several smaller models for better predictions" |
| "gradient boosting" | "a tree-based model that improves by learning from its own mistakes" |
| "stratified split" | "a split that preserves the cancellation rate in each piece" |
| "leak" / "look-ahead bias" | "letting the model peek at the future during training" |
| "ROC curve" | "the trade-off curve between catching cancellations and avoiding false alarms" |
| "operating point" | "the chosen decision cut-off" |
| "regularisation" | "a penalty that stops the model from over-memorising the training data" |

### Programmatic substitution

A safe substitution script that handles the most common patterns is
provided in `scripts/_translate_notebooks_helper.py` (one-off
utility — delete after running). Run it once per notebook directory,
then manually polish the cells where context requires a more nuanced
rewrite than a simple find-and-replace.

---

## What NOT to translate

A few terms are accepted business vocabulary in hospitality analytics
and should be **kept** rather than translated — translating them
would actually make the thesis harder to defend, not easier.

| Keep these | Why |
|---|---|
| **ADR** (Average Daily Rate) | Standard hospitality industry term; every revenue manager uses it. |
| **RevPAR** (Revenue per Available Room) | Same — industry-standard KPI. |
| **PMS** (Property Management System) | Industry-standard term. |
| **CRM** (Customer Relationship Management) | Industry-standard term. |
| **BI** / **BIA** (Business Intelligence / Analytics) | Theme of the thesis; the audience is BI-literate. |
| **OTA** (Online Travel Agency) | Industry-standard term. |
| **DCT** (Dynamic Capability Theory) | The thesis's theoretical framework. |
| **Sense / Seize / Transform** | DCT terms — these are the thesis's structural anchors. |
| **Power BI** | Product name; no translation needed. |
| **Python**, **FastAPI**, **Gradio**, **SQLite** | Tool names — translate only if your audience won't recognise them. |
| **LightGBM**, **XGBoost**, **Random Forest** | Model family names — name them, but pair with the plain-English "a tree-based model that ..." on first mention. |

---

## Defence-day quick-reference card

If a panellist asks "what does X mean?", these are the answers you
should be able to give in one breath.

| Q | One-breath answer |
|---|---|
| What is PR-AUC? | A single score from 0 to 1 that measures how well the model balances catching cancellations with not crying wolf. 0.76 is meaningfully above our natural cancel rate of 0.37, which means the model is real signal, not noise. |
| What does calibration do? | It re-scales the model's outputs so that "70 %" really means 70 %. Without it, our managers couldn't trust the percentages. |
| Why isotonic calibration specifically? | It's the most flexible method — it doesn't force the calibration curve into a sigmoid shape, which would be wrong for gradient-boosted trees. |
| What's a bootstrap? | Drawing samples from your test set with replacement to see how stable a number is. We do it 2,000 times to confirm LightGBM's lead is real, not luck. |
| Why a cost-sensitive cut-off? | Because each missed cancellation (FN) is more expensive than each unnecessary intervention (FP). Setting the cut-off below 50 % catches almost every cancellation at modest extra cost. |
| What does SHAP do? | It shows you, for each individual booking, which fields pushed the prediction toward cancel or arrive — and how much each one mattered. It's how we make the model explainable. |
| What is drift monitoring? | Checking that the live mix of bookings still looks like the mix we trained on. If guest behaviour shifts, we retrain before the model gets stale. |
| Why didn't you use accuracy? | Because cancellations are 37 % of the data on Portugal and 15 % on PH. A model that always said "no cancel" would still be 63 %-or-85 % accurate but useless to a revenue manager. PR-AUC and F1 are the right metrics under class imbalance. |
