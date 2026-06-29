# Defense Prep — Hard Questions & Crisp Answers

**Thesis:** A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations

Anticipated tough questions from a data-science panel, with answers grounded in the thesis's
actual numbers. Each answer is written to be spoken in 30–60 seconds. Lead with the concession
where one exists (it disarms the examiner), then pivot to your real contribution.

---

## 0. Your 30-second pitch (memorize this)

> "We don't claim a better cancellation *model* — we show that a **calibrated, cost-aware
> decision system** built on an honest, time-aware evaluation beats the trivial policies hotels
> actually use today, and that the *method* transfers to a real Philippine resort. The
> contribution is the system and the rigor, not the algorithm."

**Your strongest cards — play these often:**
1. **Honesty.** You report a non-significant champion (p = 0.177), disclose the duplicate
   sensitivity as a range, and caveat the pilot. Examiners reward this.
2. **Time-aware (chronological) evaluation** — no look-ahead leakage. Most cancellation papers
   get this wrong.
3. **Calibration** — isotonic + ECE 0.062→0.031 + Brier + reliability diagram.
4. **Cost-sensitive thresholding tied to euros**, deployed in FastAPI + Power BI.
5. **A real PMS dataset (Punta Villa)** — genuine novelty vs. recycled Kaggle work.
6. **Reproducibility** — fixed seeds, CI, tests, one-command regeneration.

**Traps to avoid (do NOT say these):**
- ❌ "LightGBM is the most accurate model." (It's statistically tied with logistic regression.)
- ❌ "The model saves 36% of costs." (Say: "*reduces expected cost by 36% vs. the intervene-on-all
  baseline, under our cost assumptions*.")
- ❌ "The method works in the Philippines." (Say: "*shows directional evidence of transfer; it's a
  pilot, not a benchmark*.")

---

## 1. The hard questions

### Q1. "If logistic regression is statistically tied with LightGBM (p = 0.177), why use gradient boosting at all? What did it buy you?"
**This is the most likely and most important question.**
> "You're right that on raw ranking they're tied — LightGBM beats logistic regression by only
> +0.005 PR-AUC, not significant. We're transparent about that. We kept LightGBM for three
> reasons that matter for a *decision tool*, not for a leaderboard: it has the **highest recall
> among the strong models (0.895)**, the **best probability calibration** (which is what the
> cost-sensitive thresholds depend on), and it **trains ~3× faster than XGBoost** at equal
> capacity. It also won our **pre-specified** rolling-origin selection criterion, so the choice
> wasn't made on the test set. If a panel preferred logistic regression for interpretability,
> our framework supports that — the calibration and cost layer are model-agnostic."

### Q2. "Defend the €15 false-positive cost. Your 'optimal' policy intervenes on ~72% of bookings — isn't that just 'warn everyone'?"
**Concede the mechanism, then bound it.**
> "Correct that the cost-optimal threshold is low (0.06) and flags about 72% of bookings — because
> we set the false-negative cost (lost room revenue) far above the €15 false-positive cost, the
> math pushes the threshold down. That's why we **don't rest the contribution on the 36% number
> alone**. Two defenses: first, we ran a **sensitivity analysis (Figure 4.13)** showing how the
> optimal threshold moves as the false-positive cost rises. Second, the model's *discrimination*
> value is clearest at the **max-F1 operating point (precision 0.625, recall 0.895)**, which is
> the policy we'd recommend for routine operations. The cost-optimal policy answers a specific
> question — 'minimize total expected euros' — and even there it beats intervene-on-all by 36%
> while flagging fewer bookings."

*If pressed on where €15 comes from:* "It's a conservative proxy for a reminder/confirmation
contact. We treat it as an assumption and test sensitivity around it; a hotel would calibrate it
to its own intervention cost."

### Q3. "Why Dynamic Capability Theory? It doesn't influence your feature engineering or model choice."
**Don't pretend it's analytical — own it as the framing.**
> "DCT isn't a modeling input — it's the **organizing lens** that connects a prediction to a
> management action: *sense* the data, *seize* it by building the calibrated model, *transform*
> it into operating policy with a feedback loop for retraining. It's what keeps this a *business
> intelligence* study rather than a pure benchmark. The technical work stands on its own; DCT
> explains why each stage exists in managerial terms."

### Q4. "27% of your data are exact duplicates. Isn't your 0.759 PR-AUC inflated?"
**You already disclose this — lead with that.**
> "We flagged that explicitly. About 27% are exact-duplicate block bookings. We verified they
> **don't cross the train/test boundary**, so they don't leak. To be safe we retrained on a fully
> de-duplicated copy: PR-AUC moved from 0.759 to **0.703**. So we report the headline as a
> **range, 0.70–0.76**, in the abstract — depending on whether you count per booking record or per
> unique profile. We report per-record as primary only because booking IDs aren't available to
> tell genuine repeat bookings from artifacts."

### Q5. "Your Philippine test set is 20 bookings. How is that a valid hypothesis test?"
**Reframe it yourself before they do.**
> "It isn't a benchmark and we don't present it as one — it's a **methodological pilot**. With 20
> test bookings the confidence interval on PR-AUC spans ±15 points, so we treat every metric as
> directional. What actually transfers is the *method*: the identical pipeline ran end-to-end on a
> **real, independently collected PMS export** with only currency constants changed, and the
> dominant feature — **deposit_type — was #1 by SHAP in both markets**. The contribution is 'the
> approach ports cleanly,' plus a concrete data-collection target (one full season) for a future
> confirmatory study."

*If pressed on the Spearman 0.71:* "With only seven algorithms ranked, that correlation is
indicative, not statistically conclusive — we present it as qualitative support, not an effect size."

### Q6. "Why is Random Forest in Tables 4.2/4.3 but not in the cross-validation Table 4.1?"
> "Deliberate. The cross-validation table shows the **full complexity ladder**, so it keeps the
> trivial Dummy and Naïve Bayes anchors to demonstrate the value each modeling step adds. The
> chronological test drops those non-competitive baselines and adds **Random Forest as a
> bagged-ensemble contrast** to the boosted models — the comparison that actually matters for
> deployment."

### Q7. "This dataset has been used in hundreds of projects. What's new here?"
> "The dataset is a *benchmark*, not the contribution. Four things differentiate this from a
> typical Kaggle project: a **chronological evaluation** (most use random splits and leak),
> **probability calibration** (rarely done), a **cost-sensitive, euro-denominated decision layer**
> with a deployed dashboard, and a **real second market** (Punta Villa) to test transfer. The
> novelty is turning prediction into a calibrated, auditable *decision system*."

### Q8. "How do you know there's no target leakage? `assigned_room_type` is in the dataset."
> "We explicitly drop the five post-booking fields — **reservation_status,
> reservation_status_date, assigned_room_type, booking_changes, days_in_waiting_list** — before any
> model sees the data, because they're only known during or after the stay. `reservation_status`
> in particular *is* the outcome. Every remaining predictor is fixed at booking time. This is
> stated in the methodology and the field list."

### Q9. "Your data is 2015–2017, pre-pandemic. Is it still relevant?"
> "We name this as a limitation. The value is **methodological** — a reproducible, transferable
> framework — not a deployable 2026 model. That's exactly why we built the drift-monitoring page
> and retraining trigger, and why we piloted on a recent (2022–2025) Philippine dataset. A hotel
> would retrain on its own current data; the pipeline does that with one command."

### Q10. "Why PR-AUC instead of accuracy or ROC-AUC as your primary metric?"
> "Cancellations are the minority class (~37%), and accuracy rewards predicting 'not cancelled.'
> PR-AUC focuses on the positive class we actually care about and ignores the large pool of easy
> non-cancellations, so it's the honest discrimination metric here. We select the champion on
> rolling-origin PR-AUC and report ROC-AUC, F1, precision, and recall alongside it."

### Q11. "Your hypothesis said lead time would be the top predictor, but SHAP says deposit type. Didn't your hypothesis fail?"
**Own it — a corrected hypothesis is a finding.**
> "Yes — H3 is **not supported on ordering**, and we say so. Deposit type, not lead time, is the
> dominant driver, consistently in *both* markets. That's a substantive finding, not a failure:
> booking-payment commitment is the strongest cancellation signal. The named features still
> matter; the predicted *ranking* was wrong, and we corrected it with evidence."

### Q12. "What is calibration and why does it matter more than accuracy here?"
> "A calibrated model means when it says '70% chance of cancellation,' about 70% actually cancel.
> That matters because our decisions are driven by **probability thresholds and risk tiers**, not
> by a yes/no label — so the probabilities have to be trustworthy. We measure it with expected
> calibration error, which **halved from 0.062 to 0.031** after isotonic correction, and show it
> with a reliability diagram (Figure 4.5)."

### Q13. "Your high-precision policy catches only ~10% of cancellations. What's it for?"
> "It's the threshold-0.98 policy — near-perfect precision but recall 0.095. We report it for
> completeness as one end of the operating-point spectrum. It only makes sense when intervention
> capacity is extremely scarce and a false alarm is very costly. For normal operations we
> recommend the max-F1 point; for cost minimization, the cost-sensitive point."

### Q14 (if ADR comes up). "Your ADR regressor uses post-booking features — isn't that leakage?"
> "The ADR forecast is a **parallel, experimental** component of the dashboard, not part of the
> cancellation results. At training it used some post-booking fields; in live serving those are
> passed as placeholders, which we disclose. The methodologically clean fix is retraining on
> booking-time features only — we flag it as such. None of the cancellation findings depend on it."

### Q15. "Your abstract says LightGBM has *superior discriminatory power* and *validated* transferability — but the body says not significant and directional. Which is it?"
**The body is correct; the abstract wording overreaches. Concede and align.**
> "The body is the accurate statement: the advantage over logistic regression is **not statistically
> significant (p = 0.177)**, and the Philippine pilot is **directional (n = 20)**, not a validated
> benchmark. The abstract should read the same way — reporting the 0.70–0.76 PR-AUC range and calling
> the pilot directional — and we've aligned it. Consistency between the abstract and the results is
> exactly the standard we hold ourselves to."

*(Best handled by fixing the abstract before the defense — see `THESIS_ESSENTIALS.md` Issue 1 — so the
question never arises.)*

---

## 2. Quick-reference numbers (have these cold)

| Thing | Value |
|---|---|
| Champion | LightGBM (rolling-origin PR-AUC) |
| Test ROC-AUC / PR-AUC / F1 | 0.863 / 0.759 (0.70–0.76 de-dup range) / 0.736 |
| Test precision / recall (max-F1) | 0.625 / 0.895 |
| Calibration ECE (raw → isotonic) | 0.062 → 0.031 |
| H2: LightGBM vs Logistic Regression | +0.005 PR-AUC, p = 0.177 (not significant) |
| H2: vs Decision Tree / Random Forest | +0.246 / +0.032 (both significant) |
| Cost: no-model / intervene-all / cost-sensitive | €2,322,794 / €111,240 / €71,135.50 |
| Saving vs intervene-all | €40,105 = **36%** |
| Cost-sensitive threshold / recall / % flagged | 0.06 / 0.991 / ~72% of bookings |
| FP cost / FN cost | €15 / proportional to revenue at risk |
| Duplicates | ~27% exact; champion unchanged after de-dup |
| Philippine pilot | 193 bookings, n_test = 20 (9 cancellations), ±15pp CI |
| Top SHAP feature (both markets) | deposit_type |
| Split | chronological 80 / 10 / 10 (reported as 80/20 holdout) |

---

## 3. How to concede gracefully (without collapsing)

When an examiner lands a real hit, the move is: **agree → bound → pivot**.
- "That's a fair point, and we say so in the limitations…" (agree)
- "…its effect is bounded because [number/sensitivity analysis]…" (bound)
- "…and it doesn't change the core contribution, which is [calibrated cost-aware system]." (pivot)

Never defend a weak point to the death. Your paper's credibility comes from the honesty — leaning
into a limitation *strengthens* you in the room. The examiners have already read the caveats; they
want to see that you understand them.
