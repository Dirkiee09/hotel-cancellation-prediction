# Thesis Defense Script — Hotel Booking Cancellation Prediction

> **Mapúa University — 15-minute presentation + Q&A.**
> The script is **paragraph-tagged**: every blockquote is marked
> **(mandatory)** or **(droppable)**. Read the mandatory paragraphs
> always; drop the droppables only if you fall behind. The pacing
> model below shows how the two tracks fit a 15-minute window across
> typical reading speeds. Companion file: `slide_content_guide.md`.

### Pacing model

| Track | Words | @ 130 wpm | @ 145 wpm (typical defense) | @ 155 wpm (nerves) |
|---|---:|---:|---:|---:|
| **Mandatory only** | ~1,990 | 15 min 18 s | 13 min 43 s | 12 min 51 s |
| **Full (mand + drop)** | ~2,390 | 18 min 23 s | 16 min 29 s | 15 min 25 s |

**Recommended delivery:** read the mandatory track at ~145 wpm; insert
droppable paragraphs only on the stages where you're visibly ahead of
the clock at your podium. The clock should sit on the lectern, not on
the slide deck. Slide-transition overhead is ~3 s per click × 22
clicks ≈ 1 minute of unavoidable silence on top of the spoken time.

### Defense-day instructions

| Setting | Value |
|---|---|
| Reading rate target | 145 wpm |
| Headline-number cue | **Bold** word + 1-second hold after speaking it |
| Stage tag legend | **(mandatory)** = always read; **(droppable)** = skip if behind |
| Slide count | 23 main + 5 backup = 28 total |
| Print this script | 14 pt, single-sided, tape to back of laptop |

---

## 0. Pre-defense checklist

*Do this twenty minutes before the panel walks in.* No exceptions.

- [ ] Laptop plugged in; battery > 70 %; charger packed anyway
- [ ] HDMI / USB-C adapter tested against the projector
- [ ] PowerPoint file open in *Slide Show* mode, slide 1 visible
- [ ] Live Gradio UI open in a second tab at `localhost:8000/ui`
      (in case you want the live demo on slide 20)
- [ ] Power BI Desktop open with the 8-page dashboard, on Page 1
      (for the Alt+Tab demo on slide 20)
- [ ] This script printed in 14 pt, taped to the back of the laptop
- [ ] Slide guide printed as a backup in your folder
- [ ] Phone on Do Not Disturb
- [ ] Water bottle on the lectern, not at the table
- [ ] One deep breath. You wrote this. You know it.

---

## Stage 1 — Title & Agenda (slides 1 – 2, ~30 s)

*Stand still. Read only what is in the blockquote — do not improvise
on the title slide. Click forward as soon as you finish line 2.*

> **(mandatory)** Good morning panel. My name is Dirk Vincent Viñas,
> and the title of my thesis is *"A Machine-Learning Framework for
> Booking-Time Cancellation Prediction in the Hotel Industry."*

*Click to slide 2.*

> **(mandatory)** In the next fifteen minutes I'll walk you through
> five things: the problem, the model that solved it, the business
> numbers it produced, the live deployment, and six recommendations
> the hotel can act on Monday morning.

---

## Stage 2 — The €3 Million Problem (slide 3, ~90 s)

*Click to slide 3. Pause one full second before speaking.*

> **(mandatory)** Hotel cancellations are expensive, and worse, they
> are invisible until it is too late. On the Portugal benchmark used
> in this study — **one hundred nineteen thousand, two hundred and
> ten** bookings spanning 2015 to 2017 — **thirty-seven percent** of
> all bookings were cancelled before check-in. That is not a tail
> event. That is more than one in three.

*Pause. Point at the red bar on the slide.*

> **(mandatory)** On the two-month test window I'll report today,
> those cancellations cost the property **three million and fourteen
> thousand euros** in lost room revenue. The hotel only learns a
> booking is dead after the check-in window has passed — by then
> there's no chance to resell the room.

> **(mandatory)** So the question this study set out to answer is
> simple: *can we tell, at the moment a booking is made, which
> bookings are likely to cancel — and use that signal to act
> before the loss happens?* If the answer is yes, the hotel
> recovers revenue. If the answer is no, this is an interesting
> classifier and nothing more. The remainder of this presentation
> is about exactly how *yes* the answer turned out to be.

> **(droppable)** Note that this study does not look at
> *post-booking* signals like room reassignments or waiting-list
> changes. The model has to predict using only what the front desk
> knows at the moment of reservation — because that is the only
> moment when intervention is possible.

---

## Stage 3 — Research Questions and Hypotheses (slide 4, ~45 s)

*Click to slide 4.*

> **(mandatory)** Four research questions, and five pre-registered
> hypotheses. The questions: which model wins, what features drive
> it, does cost-sensitive thresholding pay its keep, and does the
> methodology transfer.

> **(mandatory)** Of the five hypotheses, four were fully
> supported by the data, and one — hypothesis three, on the
> *order* of the top SHAP features — was only partially supported.
> The three predicted features all appear in the top ten, but the
> rank order differs. I'll return to that finding in stage ten.

---

## Stage 4 — Significance and Scope (slides 5 – 6, ~30 s)

*Click to slide 5, hold briefly, click to slide 6 mid-paragraph.*

> **(mandatory)** This work serves four audiences: revenue managers
> get a calibrated risk score, BI practitioners get a reusable
> methodology, academic research gets a reproducible benchmark,
> and the hospitality sector gets the first publicly-available
> end-to-end deployment artefact.

*Click to slide 6.*

> **(mandatory)** Scope: Portugal benchmark, booking-time features
> only, binary classifier plus calibration plus three thresholds,
> with the live FastAPI and Power BI infrastructure. Out of scope:
> external features, A/B-tested intervention policies, and
> headline numbers on the Philippine sub-study — that last one is
> directional only at twenty test rows.

---

## Stage 5 — Conceptual Framework (slide 7, ~30 s)

*Click to slide 7. Trace the arrows from left to right with the cursor.*

> **(mandatory)** The framework is a five-stage decision loop:
> property-management data flows into engineered features, the
> features flow into the LightGBM classifier, the classifier
> produces a calibrated probability, the probability is bucketed
> into a low, medium, or high risk tier, and the tier triggers a
> specific operational action. Every action either prevents a loss
> or absorbs a small known cost. The closing loop is the revenue
> impact, which feeds back into the next training cycle.

---

## Stage 6 — Methodology and Dataset (slide 8, ~60 s)

*Click to slide 8.*

> **(mandatory)** Six algorithms were trained on the same data
> under identical preprocessing: LightGBM, XGBoost, Gradient
> Boosting, Random Forest, Logistic Regression, and a baseline
> Decision Tree.

> **(mandatory)** The split is the credibility move of this study.
> The data was split **chronologically**, not randomly — oldest
> **eighty percent** trains the model, next ten percent calibrates
> and tunes thresholds, most recent ten percent is held out for the
> reported numbers. This is harder than random shuffling but mimics
> production: the model always predicts the *future* from data on
> the past.

> **(droppable)** Thirty-three booking-time features were used,
> with explicit exclusion of post-booking leakage columns like
> `reservation_status` and `assigned_room_type` — columns that
> would inflate test metrics academically but are useless at the
> booking desk where they don't yet exist.

---

## Stage 7 — Model Comparison (slides 9 – 10, ~90 s)

*Click to slide 9.*

> **(mandatory)** Here is the head-to-head on the chronological
> test set. The floor is the Dummy baseline — guessing the
> majority class — at PR-AUC **point three seven eight**. Every
> trained model beats it. LightGBM wins on every threshold-
> dependent metric — ROC-AUC **point eight six four**, PR-AUC
> **point seven six**, and F1 **point seven three five** — but the
> field is close above the baseline.

> **(mandatory)** The gap to second-place Gradient Boosting on
> PR-AUC is **point zero zero six**, small enough to dismiss as
> noise if we didn't test it. So we tested it.

*Click to slide 10.*

> **(mandatory)** Paired bootstrap resampling — two thousand
> resamples — confirms the lead at **p equals point zero zero one**
> against Gradient Boosting, and at **p less than point zero zero
> one** against every other algorithm. The ranking is real.

> **(droppable)** Why LightGBM specifically? Three practical
> reasons. First, hotel data mixes numeric and categorical signals
> and gradient-boosted trees handle both natively. Second,
> LightGBM trains in roughly thirty seconds on a laptop, which
> matters when the property wants to retrain monthly against
> fresh data. Third, inference is under two milliseconds per
> booking, well inside the latency budget of a live booking-desk
> API.

---

## Stage 8 — Champion Deep-Dive (slides 11 – 12, ~90 s)

*Click to slide 11. Point at the PR curve, not the ROC.*

> **(mandatory)** ROC-AUC of **point eight six four** means a
> random cancelled booking is correctly ranked above a random kept
> booking **eighty-six percent** of the time. PR-AUC of **point
> seven six** means precision stays high even as we chase recall —
> the curve does not collapse.

> **(mandatory)** The bootstrap ninety-five percent confidence
> interval on PR-AUC is **point seven four eight to point seven
> seven two** — a width of only point zero two four. The headline
> number is tight.

*Click to slide 12.*

> **(mandatory)** And here is the same model at its production
> threshold, translated into business terms. At threshold
> **point four zero**, the model catches **eighty-four percent**
> of all real cancellations — three thousand, seven hundred and
> ninety-one of four thousand, five hundred and six.

> **(droppable)** The false-positive count is two thousand and
> twenty-four — flagged-but-actually-fine bookings. Each one of
> those costs the hotel about fifteen euros in a reminder email.
> The false-negative count — the missed cancellations — is seven
> hundred and fifteen, and each of those costs the full booking
> revenue. The asymmetry between fifteen euros and full revenue is
> what drives the cost-sensitive policy on slide eighteen.

---

## Stage 9 — Calibration (slide 13, ~30 s)

*Click to slide 13.*

> **(mandatory)** Isotonic calibration halves the test-set
> Expected Calibration Error from **point zero five eight** to
> **point zero two nine**. In plain language: when the model says
> seventy-five percent, it really means about seventy-five percent
> in observed cancellation rate. The probability number is
> directly usable as a policy band — no fudge factor needed.

---

## Stage 10 — Feature Importance (slides 14 – 15, ~60 s)

*Click to slide 14.*

> **(mandatory)** This is the SHAP global importance plot. Each
> row is a feature, each dot is a booking, the colour is the
> feature value, and the horizontal position is how much that
> feature pushed the prediction toward *cancel* or *keep*.

> **(mandatory)** The number one driver — and this is the finding
> that surprised us — is **deposit type**, not **lead time** as
> hypothesised. Country is second, agent is third, and lead time
> only appears at rank seven.

*Click to slide 15. Pause two seconds before the next line.*

> **(mandatory)** And here is the counter-intuitive part.
> **Non-refundable** deposits correlate with **higher**
> cancellation rates, not lower. The hotel's instinct says a
> non-refundable deposit should *deter* cancellation; the data
> says the opposite. The explanation is that non-refundable rates
> are concentrated in channels whose customers cancel frequently
> regardless of the deposit policy. The deposit doesn't change
> behaviour — it changes *who books*. Action item: audit the
> *channels*, not the deposit policy itself.

---

## Stage 11 — Risk Tier × Revenue Exposure (slide 16, ~60 s)

*Click to slide 16.*

> **(mandatory)** Risk is heavily concentrated. **Twenty-six
> percent** of bookings — the High risk tier, with calibrated
> probability above point seven — account for **fifty-two
> percent** of all realised cancellation losses. That is one and
> a half million euros of the three million we started with,
> sitting in one quarter of the bookings.

> **(mandatory)** This is why the deployment uses *tiered*
> intervention, not blanket. Confirmation calls to the High tier,
> reminder emails to the Medium tier, silence on the Low tier —
> the policy pattern matches where the money actually is.

---

## Stage 12 — Threshold Policies (slide 17, ~30 s)

*Click to slide 17.*

> **(mandatory)** Three operating points, three use cases. The
> balanced max-F1 policy at threshold point four zero is the
> default for weekly operations. The high-precision policy at
> point nine eight is for executive audits where every flag must
> survive scrutiny. And the cost-sensitive policy at point zero
> four is the recommended deployment default, because it has the
> lowest total expected cost.

---

## Stage 13 — The €2.94 Million Headline (slide 18, ~60 s)

*Click to slide 18. Pause two seconds before speaking.*

> **(mandatory)** Under the cost-sensitive operating policy, the
> model recovers **two million, nine hundred and thirty-seven
> thousand, seven hundred and fifty-four euros** of revenue at
> risk on the test set — that is **ninety-seven point five
> percent** of the theoretical maximum.

*Pause one full second. Let the number land.*

> **(mandatory)** The mechanic is asymmetric. The model is willing
> to flag three quarters of all bookings, because the cost of a
> wrongly flagged booking is fifteen euros and the cost of a
> *missed* cancellation is the full booking revenue. It rationally
> trades many cheap false positives for the recovery of a few
> expensive false negatives.

> **(droppable)** And even under the more conservative max-F1
> policy used for normal weekly operations, the model still saves
> two point six one million euros. The model is not just
> academically accurate — it pays for itself many times over per
> booking cycle.

---

## Stage 14 — Live Deployment (slide 19, ~45 s)

*Click to slide 19. Trace the diagram with the cursor.*

> **(mandatory)** The model is not a notebook artefact. It is
> wired into a FastAPI server on `localhost:8000`, fronted by a
> Gradio user interface, that scores any booking in under five
> hundred milliseconds. Every successful prediction writes one
> row to an audit log in SQLite, which is exported to a CSV that
> Power BI consumes on refresh.

> **(mandatory)** And there is a closing loop, in amber on the
> diagram. A weekly drift script computes the Population
> Stability Index for every feature. When two or more features
> cross PSI equal to zero point two five, the dashboard's
> monitoring page flags a retrain — the model never silently
> degrades.

---

## Stage 15 — Power BI Dashboard (slide 20, ~45 s)

*Click to slide 20.*

> **(mandatory)** All of the operational signal feeds an eight-page
> Power BI dashboard. Page one is a risk overview with KPI cards.
> Page two is an action list of high-risk bookings the front desk
> should call. Page three shows risk patterns by segment. Page four
> compares the three threshold policies. Page five shows the live
> ADR forecast. Page six shows revenue exposure. Page seven shows
> calibration and fairness. And page eight is the drift monitoring
> page that triggers retraining.

> **(droppable)** Every page refreshes from the same live
> prediction log, so the dashboard always reflects the most recent
> bookings the model has scored. No ETL job — the model is the
> source of truth.

---

## Stage 16 — Six Managerial Recommendations (slide 21, ~90 s)

*Click to slide 21.*

> **(mandatory)** The findings translate into six concrete actions
> a hotel revenue manager can put on their Monday-morning
> checklist.

> **(mandatory)** **One:** adopt the risk-tier policy. Bucket every
> new booking into Low, Medium, or High at calibrated probabilities
> point four zero and point seven zero. The Power BI dashboard
> auto-refreshes the counts.

> **(mandatory)** **Two:** tighten policy by *booking source*, not
> by guest history. The top three SHAP drivers — deposit type,
> country, and agent — are all channel signals. The hotel's
> leverage is auditing which agents and which countries cancel
> most, not changing individual guest treatment.

> **(mandatory)** **Three:** run a seventy-two-hour reminder
> workflow on Medium-tier bookings. At fifteen euros per
> intervention, this is the cheapest layer of the policy stack
> and addresses the largest single slice of revenue at risk in
> absolute terms.

> **(mandatory)** **Four:** reserve confirmation calls and partial
> deposit requests for the High tier. That tier carried a
> seventy-six percent observed cancellation rate on the test set —
> the hit rate justifies the manual effort.

> **(droppable)** **Five:** use the live API and Gradio interface
> as a frontline tool. Any booking entered through the existing
> PMS can be scored in under five hundred milliseconds. And
> **six:** treat the dashboard's drift page as the retraining
> trigger. Without that monitoring, last quarter's model
> silently degrades and the hotel never notices.

---

## Stage 17 — Limitations and Future Research (slide 22, ~60 s)

*Click to slide 22.*

> **(mandatory)** Honest reporting matters as much as the headline
> number. The biggest limitation is the single benchmark dataset —
> Portugal pre-pandemic, two properties. The Philippine sub-study
> at Punta Villa Resort showed the methodology transfers, but at
> twenty test rows the metric confidence intervals are too wide
> for headline use. The two-point-nine-four million figure is also
> an *upper bound* — it assumes guests respond to reminders and
> deposit requests at the rates the cost model assumes; the
> measured response rate awaits live A/B testing.

> **(mandatory)** Future research extends in five directions: add
> external context features like weather and local events,
> replicate on ten to fifteen Philippine resorts, run randomised
> A/B trials of the intervention policies, retrain the ADR
> regressor on booking-time features only, and package the
> pre-flight duplicate-cluster diagnostic and the
> feature-availability mapping as a standalone Python library
> for the broader hospitality analytics community.

---

## Stage 18 — Closing Statement (slide 22 hold, ~30 s)

*Hold on slide 22. Look at the panel, not the screen.*

> **(mandatory)** To close: this study set out to show that
> cancellation risk is predictable at the moment of booking with
> calibrated probabilities honest enough to drive cost-sensitive
> action. The Portugal benchmark gave a clean, defensible answer:
> **yes, it is** — and the revenue recovery is large enough that
> the model pays for itself many times over per booking cycle.
> The operational pipeline is in place. The dashboard is built.
> The recommendations are concrete. Thank you for your time —
> I welcome your questions.

*Click to slide 23. Smile. Hold eye contact for two seconds.*

---

# Q&A Appendix — Ten Anticipated Questions

> Each answer is paced for **30–60 spoken seconds** (75–150 written
> words). Read aloud only what's in the blockquote. The slide
> reference in *italics* tells you which backup slide to un-hide
> while you answer.

---

### Q1 — *"Why LightGBM over XGBoost? The PR-AUC gap is only point zero one one."*

*Show backup slide B1 (bootstrap CI forest).*

> The gap is real but small, so the deciding factor is operational
> rather than statistical. Paired-bootstrap p equals point zero
> zero one means the lead survives two thousand resamples, but
> the practical difference is dominated by speed and footprint.
> LightGBM trains in about thirty seconds on a laptop and infers
> in under two milliseconds per booking; XGBoost is roughly twice
> as slow on both. For a property that wants to retrain monthly
> and serve a live booking-desk API, the LightGBM choice
> minimises operational friction without sacrificing measurable
> accuracy. If the hotel preferred XGBoost for ecosystem reasons,
> the framework would still work — the methodology is
> algorithm-agnostic.

---

### Q2 — *"Why is `deposit_type` the #1 driver and not `lead_time` — your hypothesis predicted the opposite?"*

> Hypothesis three was the most interesting finding precisely
> because it was only partially supported. All three pre-registered
> features — lead time, deposit type, and previous cancellations —
> appear in the top ten SHAP features, which validates the
> *substantive* prediction. But the rank order was wrong: the
> actual order is deposit type at rank one, country at rank two,
> agent at rank three, and lead time only at rank seven. The
> explanation, which we developed in Chapter four section five
> point two, is that channel features dominate over guest-level
> features in this dataset because cancellation is heavily driven
> by *which channel the booking comes through*, not by the guest's
> stay-length decision. This is a finding the hospitality
> literature should pay attention to.

---

### Q3 — *"Won't the €2.94 million figure overstate real savings? Guests don't actually respond to reminders at the rates you assume."*

*Show backup slide B3 (ADR regression) if pricing comes up.*

> Yes — and this is explicitly flagged as a limitation in Chapter
> five. The two-point-nine-four-million figure is an *upper
> bound*. It is the revenue at risk that the model correctly
> identifies; the *recovered* revenue depends on whether
> reminders, calls, and deposit requests actually prevent the
> cancellations they flag. Our cost model assumes a deterministic
> response, which is not realistic. Future-research extension
> three — randomised A/B testing of the intervention policies —
> converts this upper bound into a measured treatment effect.
> Until then, the honest framing is "the model identifies the
> losses; the policy decides how much to actually recover."

---

### Q4 — *"PR-AUC drops from 0.922 in CV to 0.760 on the chronological test — that's a 16-point gap. What's happening?"*

*Show backup slide B1 (bootstrap CI forest).*

> That sixteen-point gap is the empirical signature of *concept
> drift over time*. The CV number runs on a random shuffle of the
> dataset — every fold sees rows from every time period, so the
> algorithms compete on the easiest possible footing. The
> chronological test forces the model to predict 2017 bookings
> using a model trained on 2015 and 2016, which exposes it to
> shifts in guest mix, deposit policy, and booking channels that
> accumulated between training and test. The gap is *not* a
> defect of the model — it is the cost of honest evaluation. The
> 0.760 number is what the hotel actually sees in production; the
> 0.922 is what the academic literature reports under the easier
> protocol. We report both for transparency.

---

### Q5 — *"How will the model behave on post-pandemic data?"*

> The honest answer is *we don't know yet*, and that is itself a
> limitation listed in Chapter five. The training data is 2015
> to 2017, pre-pandemic. Cancellation behaviour likely shifted
> after 2020 — guests cancel more on average, more late, and for
> different reasons. The methodology handles this in two ways.
> First, the chronological evaluation protocol is robust to the
> *kind* of drift we'd expect. Second, the PSI drift monitoring
> page on the Power BI dashboard would detect the shift and
> trigger retraining — that is exactly what the loop is built
> for. A hotel deploying this model in 2025 should validate the
> metrics on their own holdout, not assume the 0.864 ROC-AUC
> transfers unchanged.

---

### Q6 — *"How does the dashboard know when to retrain?"*

*Show backup slide B5 (PSI drift heatmap).*

> The retraining trigger is the Population Stability Index — the
> standard distribution-shift measure used in credit-risk
> monitoring. The dashboard's page eight computes PSI for every
> feature against the training baseline, weekly. Three zones:
> below point one zero is safe, point one zero to point two five
> is watch, and above point two five is retrain. The trigger
> fires when *two or more* features cross point two five
> simultaneously — single-feature triggers are too noisy and would
> retrain too often. The PSI rule is conservative on purpose: a
> false alarm costs only a retrain cycle, but missed drift costs
> the entire recovery number on the headline slide.

---

### Q7 — *"Why does a non-refundable deposit predict cancellation? Shouldn't it deter it?"*

> This is the most counter-intuitive finding in the entire study,
> and it took us a full SHAP-dependence analysis to interpret
> correctly. The deposit type itself doesn't *cause* cancellation
> — it acts as a marker for the channel and customer type behind
> the booking. Non-refundable rates are disproportionately offered
> by aggregator channels with low-trust speculative bookers, and
> by certain corporate-buyer agents whose own cancellation rates
> are elevated. The deposit doesn't change guest behaviour; it
> changes *which guests book*. The operational implication —
> recommendation two in Chapter five — is that the hotel's
> leverage is auditing the *channels* that offer non-refundable
> rates, not the deposit-policy structure itself.

---

### Q8 — *"The Philippine sub-study only has 20 test rows. What's the point?"*

*Show backup slide B2 (PH cluster diagnostic).*

> The point isn't the metric — the point is the methodology
> contribution. At twenty test rows, the bootstrap confidence
> interval on PR-AUC is roughly plus or minus fifteen percentage
> points; we report those metrics as *directional only* and never
> as headline numbers. But the pre-flight duplicate-cluster
> diagnostic — which detects datasets organised around recurring
> booking archetypes that would leak twins across the
> chronological split — *did* run on the Punta Villa export, and
> *did not* fire. That tells us the methodology operates honestly
> on that data: small but not contaminated. The diagnostic itself
> is one of the two methodology contributions we propose
> packaging as a library in future research five.

---

### Q9 — *"How is the model auditable for fairness across customer segments?"*

*Show backup slide B4 (per-segment fairness heatmap).*

> Chapter four section seven point three breaks the test-set
> metrics out by hotel type and market segment. The strongest
> segment is Groups bookings — PR-AUC point nine eight five — and
> the weakest is Direct bookings at point four eight nine. Direct
> bookings cancel rarely, so the prediction problem is intrinsically
> harder there. The dashboard's page seven shows this matrix live
> and lets the hotel see whether any segment is systematically
> mis-served. Recommendation ten in the recommendations section
> addresses this directly: on Direct-tier flags below probability
> point seven zero, the policy should add human review rather than
> auto-acting.

---

### Q10 — *"Won't the hotel over-flag Direct bookings and annoy loyal guests?"*

*Show backup slide B4 (per-segment fairness heatmap).*

> That is the right concern, and it is exactly why the deployment
> policy is tiered. The Direct segment has the lowest PR-AUC and
> the lowest base cancellation rate, so the cost-sensitive
> threshold flags many Direct bookings that don't cancel — those
> are false positives. The operational mitigation is two-fold.
> First, for Direct bookings the recommended action below
> probability point seven zero is *no action* — the email
> reminder is reserved for the High tier only on this segment.
> Second, the dashboard's page two shows the action list filtered
> by segment, so the front desk sees the *list of Direct bookings
> to call* — typically a handful per week — rather than a mass
> email. The model is the signal; the policy decides who to
> actually contact.

---

# Speaker Tips Appendix

## Pacing rules

- **First thirty seconds:** read every word slowly. Your nerves
  will compress the pace; budget for it.
- **After thirty seconds:** settle into 130 wpm. Use the bolded
  numbers as natural emphasis breaks.
- **After every headline number** (in bold): hold one full second
  before continuing. The number is the slide; the pause is the
  microphone.

## Slide-transition rules

- Click forward *before* speaking the first line on the new
  slide — never read the slide title aloud while pointing at it.
- Pause one second after every click so the panel re-orients.
- Never have two charts visible while you're speaking. Build slides
  with click animations if you need to reveal in sequence.

## Pointer rules

- Use the cursor only to indicate **the one number** on each slide
  that matters. Resist the urge to draw circles or trace lines.
- On figures, point at one feature, one bar, or one curve — never
  multiple.

## Body-language rules

- **Stand still** during numbers (slides 9 – 18).
- **Move two steps** during the recommendations (slide 21) — one
  step per pair of recommendations.
- **Stand still and lean slightly forward** during the closing
  statement and during the Q&A.
- Make eye contact with each panellist once during the closing
  statement, in clockwise order.

## If interrupted

- If a panellist interrupts mid-stage, finish the current sentence,
  then turn to face them fully. Answer their question. *Then*
  return to the script — never abandon the closing statement.
- If you lose time, drop in this order: Stage 4 (significance),
  Stage 6's droppable feature-list paragraph, Stage 7's droppable
  paragraph on why LightGBM, Stage 8's droppable paragraph on
  false-positive cost, Stage 15's droppable single-source-of-truth
  line. That is roughly 150 words of cuts — about 70 seconds of
  recovery.

## If you blank

- Look at the blockquote on the printed script taped to the back
  of the laptop.
- Take a sip of water. Buy three seconds.
- Restart the current paragraph from the first word. Do not
  apologise — just restart.

## Closing line

- The last sentence — *"the model pays for itself many times over
  per booking cycle"* — is the line the panel will remember.
  Practise it twenty times before the day. Speak it slowly,
  clearly, and **never** rush past it.
