# Slide-by-Slide Content Guide — Thesis Defense Deck

> Companion to `defense_script.md`. Every slide is rendered as a
> **9-field card** so you can paste the content straight into PowerPoint
> without re-thinking layout decisions on the day. Twenty-three main
> slides cover the 15-minute talk; six backup slides at the end stay
> hidden in the deck and are only un-hidden during Q&A.

---

## Deck-level conventions

| Setting | Value |
|---|---|
| Aspect ratio | 16 : 9 (Widescreen) |
| Slide size | 13.33 × 7.5 in (33.87 × 19.05 cm) |
| Primary brand colour | `#1F4E79` (deep navy — titles, headlines, accents) |
| Danger / loss colour | `#A6192E` (red — money lost, errors, drift) |
| Success / saved colour | `#107C41` (green — money saved, calibration gains) |
| Highlight / call-out | `#F5A623` (amber — drift loop, retraining loop) |
| Neutral body text | `#3B3B3B` (charcoal grey) |
| Surface background | `#F4F4F4` (light grey panels) |
| Title font | Calibri, 32 pt, Bold |
| Body font | Calibri, 22 pt, Regular |
| Footer (source citation) | Calibri, 12 pt, Italic, `#7A7A7A` |
| Slide template | One headline number per slide, maximum |
| File name | `defense_deck.pptx` |
| Total slides | 23 main + 6 backup (hidden) = 29 |

**Footer rule:** Every chart slide has a 12-pt italic footer with the
data source (e.g., *Source: `reports/metrics.json`, n = 11,922 test
rows*). This earns trust with the panel and makes the deck self-citing.

**Polish-pass rule:** No slide gets more than ~30 words of body text.
If you need more, split the slide. The script carries the language;
the slide carries the proof.

---

## Slide 1 — Title

1. **Title:** A Machine-Learning Framework for Booking-Time
   Cancellation Prediction in the Hotel Industry
2. **Layout:** Title-only (centered)
3. **Headline number:** *None* (this is the cover slide)
4. **Body bullets:**
   - Author: *Dirk Werner B. Viñas*
   - Program: Mapúa University — BS Business Intelligence & Analytics
   - Adviser: *[insert adviser name]*
   - Defense date: *[insert date]*
5. **Figure/image:** Mapúa logo top-left (small); a faint cropped
   region of `fig_23_risk_tier_business_overview.png` as a subtle
   background watermark at 8 % opacity
6. **Caption / footer credit:** *Defense presentation, Mapúa
   University*
7. **Color emphasis:** `#1F4E79` on the title; everything else neutral
8. **Script reference:** Stage 1
9. **Speaker notes:** Stand still. Read only the title and your name —
   nothing else. Smile, take one breath, click forward.

---

## Slide 2 — Agenda

1. **Title:** Today's Roadmap
2. **Layout:** Title + 5-bullet list (Two-content with the right pane
   showing a vertical progress strip)
3. **Headline number:** *None* (or "5 sections" as a small accent)
4. **Body bullets:**
   - The problem — €3 M of revenue that walks out the door
   - The model — how LightGBM beat five challengers
   - The business numbers — €2.94 M recovered on the test set
   - The deployment — Live API + Power BI dashboard
   - The recommendations — six concrete actions for the hotel
5. **Figure/image:** Vertical progress strip with five filled circles
   (icons: euro, gear, chart, server, checklist)
6. **Caption / footer credit:** *15-minute presentation*
7. **Color emphasis:** `#1F4E79` circle ring outlines; `#F5A623` fill
   for the active step
8. **Script reference:** Stage 1 (back half)
9. **Speaker notes:** Five fingers — count one finger per bullet as
   you say it. This sets a visual memory anchor for the panel.

---

## Slide 3 — The €3 Million Problem

1. **Title:** Hotel Cancellations Are Expensive — And Invisible Until
   Too Late
2. **Layout:** Two-content (left: numbers; right: stacked bar chart)
3. **Headline number:** **€3,014,266** in cancellation revenue lost
   on a single test window (≈ two months of bookings)
4. **Body bullets:**
   - **37 %** average cancellation rate on the Portugal benchmark
     (119,210 bookings, 2015 – 2017)
   - The hotel only learns a booking is dead *after* the check-in
     window passes — no chance to resell the room
   - Cancellations cluster in a *minority* of bookings — most are
     fine. The question is *which ones*.
5. **Figure/image:** Vertical stacked bar split 63 % kept (`#107C41`)
   vs 37 % cancelled (`#A6192E`); overlay the €3 M figure as a
   call-out arrow pointing at the red segment
6. **Caption / footer credit:** *Source: `reports/metrics.json`,
   Portugal test set n = 11,922*
7. **Color emphasis:** `#A6192E` on the €3 M number — the *loss*
   colour matches the dashboard
8. **Script reference:** Stage 2
9. **Speaker notes:** Pause for 1 full second after saying "three
   million euros." Let the number land before continuing.

---

## Slide 4 — Research Questions & Hypotheses

1. **Title:** Five Questions This Study Answers
2. **Layout:** Two-content (left: 4 RQ bullets; right: 5-row compact
   hypothesis table)
3. **Headline number:** **5** hypotheses pre-registered; **5** closed
4. **Body bullets (left pane):**
   - Which model performs best at booking-time prediction?
   - What features actually drive cancellation risk?
   - Does cost-sensitive thresholding pay its operational keep?
   - Does the methodology transfer beyond Portugal?
5. **Body bullets (right pane — micro-table):**
   | # | Hypothesis | Verdict |
   |---|---|---|
   | H1 | Lead time, deposit, prev. cancels are top predictors | ✅ Supported |
   | H2 | Gradient-boosted trees beat baselines (p < .001) | ✅ Supported |
   | H3 | SHAP order is lead_time > deposit > prev_cancels | ⚠ Partial |
   | H4 | Cost-sensitive thresholds reduce loss | ✅ Supported |
   | H5 | Top SHAP feature transfers PT ↔ PH | ✅ Supported |
6. **Figure/image:** None — the micro-table *is* the visual
7. **Caption / footer credit:** *Source: Table 4.6, hypothesis
   evidence verdict pack*
8. **Color emphasis:** Green check marks `#107C41`; amber warning
   `#F5A623` on H3 (partial)
9. **Script reference:** Stage 3
10. **Speaker notes:** Land the "4 of 5 supported, 1 partial" line
    crisply — it pre-empts the most likely panel critique.

---

## Slide 5 — Significance of the Study

1. **Title:** Why This Matters
2. **Layout:** Title + 4-icon row (Picture-with-caption)
3. **Headline number:** *None* (qualitative slide)
4. **Body bullets:**
   - **For revenue managers** — a calibrated risk score on every new
     booking, with a clear action per tier
   - **For BI practitioners** — reusable methodology bridging
     classification to revenue impact, with live dashboard support
   - **For academic research** — a reproducible benchmark on a
     widely-used dataset with documented chronological splits
   - **For the hospitality sector** — first publicly-available
     end-to-end deployment artefact with audit logging
5. **Figure/image:** Four flat icons in a horizontal row (hotel,
   chart, lab beaker, pin)
6. **Caption / footer credit:** *None*
7. **Color emphasis:** `#1F4E79` for the icons
8. **Script reference:** Stage 4
9. **Speaker notes:** Skip the second bullet entirely if running
   long — it's the most droppable on the slide.

---

## Slide 6 — Scope of the Study

1. **Title:** What's In and What's Out
2. **Layout:** Two-content (left: "in scope"; right: "out of scope")
3. **Headline number:** *None* (qualitative)
4. **Body bullets (left — IN SCOPE):**
   - Portugal benchmark, 119,210 bookings, 2015 – 2017
   - Booking-time features only (no post-outcome leakage)
   - Binary classifier + isotonic calibration + 3 thresholds
   - Live FastAPI + Gradio + Power BI dashboard
5. **Body bullets (right — OUT OF SCOPE):**
   - External features (weather, events, FX) — Future Research 1
   - A/B-tested intervention policies — Future Research 3
   - Post-pandemic data — Limitation
   - Headline numbers on PH (n = 20 test rows) — directional only
6. **Figure/image:** None — pure two-column text
7. **Caption / footer credit:** *Scope detailed in Chapter I §1.6*
8. **Color emphasis:** Green border `#107C41` on left; amber border
   `#F5A623` on right
9. **Script reference:** Stage 4
10. **Speaker notes:** Naming the *out-of-scope* items proactively
    closes the most common Q&A traps before they open.

---

## Slide 7 — Conceptual Framework

1. **Title:** The Decision Loop
2. **Layout:** Full-bleed image (horizontal flow diagram)
3. **Headline number:** *None* (framework slide)
4. **Body bullets:** *None on the slide* — the diagram speaks
5. **Figure/image:** Custom horizontal flow:
   **Data** (PMS export) → **Features** (33 booking-time fields) →
   **Model** (LightGBM + isotonic) → **Action** (Low / Medium / High
   tier) → **Revenue impact** (€2.94 M recovered).
   Re-use the same arrow style as `fig_deployment_framework.png` for
   visual continuity.
6. **Caption / footer credit:** *Framework adapted from CRISP-DM with
   a revenue feedback loop*
7. **Color emphasis:** Each box uses the palette in sequence — navy →
   navy → navy → amber → green
8. **Script reference:** Stage 5
9. **Speaker notes:** Point at each box once with the cursor as you
   say its name. Five points, one second each.

---

## Slide 8 — Methodology & Dataset Split

1. **Title:** How the Data Was Prepared
2. **Layout:** Two-content (left: 4 bullets; right: Table 4.1)
3. **Headline number:** **119,210** cleaned bookings; **chronological**
   80 / 10 / 10 split
4. **Body bullets (left):**
   - Six algorithms trained: LightGBM, XGBoost, GB, RF, LR, DT
   - Chronological — not random — split. Oldest 80 % trains;
     most recent 10 % tests.
   - 33 booking-time features; explicit exclusion of post-outcome
     leakage (e.g., `reservation_status`, `assigned_room_type`)
   - Isotonic probability calibration on the validation slice
5. **Body bullets (right — Table 4.1):**
   | Split | Rows | Date range | Cancel rate |
   |---|---:|---|---:|
   | Train | 95,367 | 2015-07 → 2017-04 | 36.1 % |
   | Val | 11,920 | 2017-04 → 2017-06 | 43.9 % |
   | Test | 11,922 | 2017-06 → 2017-08 | 37.8 % |
6. **Figure/image:** None
7. **Caption / footer credit:** *Source: Chapter IV §4.2, Table 4.1*
8. **Color emphasis:** `#1F4E79` for the headline numbers in body
9. **Script reference:** Stage 6
10. **Speaker notes:** Emphasise the word *chronological* — this is
    the credibility move that separates this study from random-shuffle
    work.

---

## Slide 9 — Model Comparison (Table)

1. **Title:** LightGBM Wins — But the Field Is Close
2. **Layout:** Title + table (Comparison)
3. **Headline number:** **PR-AUC 0.760** (LightGBM, calibrated)
4. **Body bullets (compressed Table 4.2):**
   | Algorithm | F1 | ROC-AUC | PR-AUC |
   |---|---:|---:|---:|
   | **LightGBM (champion)** | **0.735** | **0.864** | **0.760** |
   | Gradient Boosting | 0.734 | 0.861 | 0.754 |
   | XGBoost | 0.729 | 0.855 | 0.749 |
   | Random Forest | 0.704 | 0.851 | 0.739 |
   | Logistic Regression | 0.713 | 0.839 | 0.739 |
   | Decision Tree | 0.596 | 0.675 | 0.508 |
5. **Figure/image:** None — table is the visual
6. **Caption / footer credit:** *Source: Table 4.2, n = 11,922 test
   rows; threshold = `max_f1` per model*
7. **Color emphasis:** Highlight the LightGBM row with a `#1F4E79`
   left-border accent
8. **Script reference:** Stage 7
9. **Speaker notes:** The PR-AUC gap to GB is 0.006 — don't oversell
   the lead. The bootstrap p = .001 line on the next slide does that
   for you.

---

## Slide 10 — Model Comparison (Figure)

1. **Title:** The Performance Ladder, At a Glance
2. **Layout:** Picture-with-caption
3. **Headline number:** **+0.006** PR-AUC over runner-up (p = 0.001
   paired bootstrap)
4. **Body bullets:**
   - LightGBM ahead of every challenger on PR-AUC
   - 2 ms inference per booking — fits inside an API call
   - Trains in ~30 s on a laptop — friendly for monthly retraining
5. **Figure/image:** `reports/figures/thesis/fig_02_grouped_bar_model_selection.png`
6. **Caption / footer credit:** *Source: `reports/figures/thesis/fig_02_…`,
   PR-AUC ranking across 6 algorithms*
7. **Color emphasis:** `#1F4E79` on the LightGBM bar in the figure
8. **Script reference:** Stage 7
9. **Speaker notes:** Point only at the leftmost bar. Resist the urge
   to walk through the others.

---

## Slide 11 — Champion Deep-Dive (ROC + PR Curves)

1. **Title:** How Sharply Does LightGBM Separate Cancellers from
   Non-Cancellers?
2. **Layout:** Picture-with-caption (image spans the slide; small
   bullets below)
3. **Headline number:** **ROC-AUC 0.864 / PR-AUC 0.760**
4. **Body bullets:**
   - ROC-AUC 0.864 — a random cancelled booking outranks a random
     kept booking **86 %** of the time
   - PR-AUC 0.760 — precision stays high while recall climbs
   - Bootstrap 95 % CI on PR-AUC = [0.748, 0.772]
5. **Figure/image:** `reports/figures/thesis/fig_01_roc_pr_curves.png`
6. **Caption / footer credit:** *Source: `fig_01_roc_pr_curves.png`,
   n = 11,922 test rows*
7. **Color emphasis:** `#1F4E79` for the champion curve; `#A6192E`
   for the no-skill diagonal in the ROC plot
8. **Script reference:** Stage 8
9. **Speaker notes:** Don't read both axes — say one sentence per
   curve, then move on.

---

## Slide 12 — Confusion Matrix in Business Terms

1. **Title:** Where Does the Model Get It Right and Wrong?
2. **Layout:** Picture-with-caption (matrix left; business reading right)
3. **Headline number:** **84.1 %** of real cancellations caught
   (recall) at threshold 0.40
4. **Body bullets:**
   - **3,791 TP** — cancellations correctly flagged → revenue
     recovered
   - **715 FN** — cancellations missed → revenue lost (€405 k)
   - **2,024 FP** — false alarms → €15 each (reminder cost only)
   - **5,392 TN** — silent, correct, no action needed
5. **Figure/image:** `reports/figures/thesis/fig_03_normalized_confusion_matrix_max_f1.png`
6. **Caption / footer credit:** *Source: `fig_03_…`, threshold =
   `max_f1` = 0.40*
7. **Color emphasis:** `#107C41` on TP cell; `#A6192E` on FN cell
8. **Script reference:** Stage 8
9. **Speaker notes:** The FP cost is *small* (€15) and the FN cost is
   *large* (full revenue) — that asymmetry is what Stage 13 will
   exploit.

---

## Slide 13 — Calibration: Before vs After

1. **Title:** When the Model Says 75 % — Does It Mean 75 %?
2. **Layout:** Picture-with-caption
3. **Headline number:** **ECE 0.058 → 0.029** after isotonic
   calibration (**halved**)
4. **Body bullets:**
   - Probabilities now correspond to observed cancellation rates
     within ~3 %
   - Operational consequence: deposit policies can be set off the
     probability directly — no safety margin needed
   - Same model; calibration adds ~2 lines of code at training time
5. **Figure/image:** `reports/figures/thesis/fig_05_calibration_reliability_and_histogram.png`
6. **Caption / footer credit:** *Source: `fig_05_…`, ECE measured on
   the test set*
7. **Color emphasis:** `#107C41` on the calibrated curve; faded
   `#A6192E` on the uncalibrated curve
8. **Script reference:** Stages 8 & 9
9. **Speaker notes:** This is a 30-second slide — read the headline
   number, the operational consequence, and click on.

---

## Slide 14 — SHAP Global Feature Importance

1. **Title:** What Drives the Predictions?
2. **Layout:** Picture-with-caption (full SHAP beeswarm)
3. **Headline number:** **`deposit_type`** is the #1 driver — *not*
   `lead_time` as hypothesised
4. **Body bullets:**
   - Top three are *channel* features: `deposit_type`, `country`,
     `agent`
   - `lead_time` matters but only at rank 7
   - Operational features push *toward* keeping the booking
     (`required_car_parking_spaces`, `total_of_special_requests`)
5. **Figure/image:** `reports/thesis/shap_summary_plot.png`
6. **Caption / footer credit:** *Source: rebuilt SHAP beeswarm,
   `scripts/rebuild_shap_summary_plot.py`, n = 2,000 test rows
   sampled*
7. **Color emphasis:** *None overlaid on the figure* — let the
   blue/red gradient speak
8. **Script reference:** Stage 10
9. **Speaker notes:** Point at the top row (`deposit_type`) only.
   The rest of the figure earns itself.

---

## Slide 15 — SHAP Key Insight (Deposit Counter-Intuition)

1. **Title:** The Counter-Intuitive Finding
2. **Layout:** Two-content (left: claim; right: explanation)
3. **Headline number:** **Non-refundable** deposits correlate with
   *higher* cancellation rates — not lower
4. **Body bullets (left — claim):**
   - Hotels assume a non-refundable deposit *deters* cancellation
   - SHAP says the opposite — `Non Refund` is a *risk amplifier*
5. **Body bullets (right — why):**
   - Channels offering non-refundable rates skew toward
     speculative bookings (low-trust agents)
   - The deposit doesn't change behaviour; it changes *who books*
   - Action: audit the *channels* selling non-refundable rates, not
     the deposit policy itself
6. **Figure/image:** None — text-only insight
7. **Caption / footer credit:** *Source: Chapter IV §4.5.2,
   SHAP-based interpretation*
8. **Color emphasis:** Amber `#F5A623` on "the opposite" phrase
9. **Script reference:** Stage 10
10. **Speaker notes:** Pause after "non-refundable deposits correlate
    with *higher* cancellation rates" for 2 full seconds. This is the
    most memorable slide for a hospitality panel.

---

## Slide 16 — Risk Tier × Revenue Exposure

1. **Title:** Where Is the Money Actually Lost?
2. **Layout:** Picture-with-caption (figure left; mini-table right)
3. **Headline number:** **26 %** of bookings (High tier) account for
   **52 %** of cancellation losses
4. **Body bullets (right — mini Table 4.7):**
   | Tier | % Bookings | % Losses |
   |---|---:|---:|
   | Low | 51.0 % | 6.4 % |
   | Medium | 22.9 % | 41.5 % |
   | **High** | **26.1 %** | **52.2 %** |
5. **Figure/image:** `reports/figures/thesis/fig_23_risk_tier_business_overview.png`
6. **Caption / footer credit:** *Source: Table 4.7, calibrated
   probabilities thresholded at 0.40 / 0.70*
7. **Color emphasis:** `#A6192E` on the High-tier row
8. **Script reference:** Stage 11
9. **Speaker notes:** This is the slide that justifies *tiered* —
    not blanket — intervention. Land the 26 % / 52 % numbers
    together; never separately.

---

## Slide 17 — Three Threshold Policies

1. **Title:** Three Operating Points, Three Use Cases
2. **Layout:** Title + table (Comparison, three-row table)
3. **Headline number:** **97.5 %** revenue recovery under
   `cost_sensitive`
4. **Body bullets (Table 4.10 compressed):**
   | Policy | Threshold | % Flagged | Recall | Total Cost (€) | Use case |
   |---|---:|---:|---:|---:|---|
   | `max_f1` | 0.40 | 48.8 % | 0.841 | 405,743 | Weekly ops |
   | `high_precision` | 0.98 | 3.6 % | 0.095 | 2,874,599 | Audit |
   | **`cost_sensitive`** | **0.04** | **75.1 %** | **0.996** | **76,512** | **Default** |
5. **Figure/image:** None — table is the visual
6. **Caption / footer credit:** *Source: Table 4.10,
   `FP_INTERVENTION_COST = €15`*
7. **Color emphasis:** `#107C41` left-border on the
   `cost_sensitive` row
8. **Script reference:** Stage 12
9. **Speaker notes:** Don't read every column. Say *"three policies,
    three use cases"*, then read just the use-case column.

---

## Slide 18 — The €2.94 Million Headline

1. **Title:** What Cost-Sensitive Thresholding Saves
2. **Layout:** Picture-with-caption (cost curve + giant call-out)
3. **Headline number:** **€2,937,754** recovered — **97.5 %** of the
   theoretical maximum
4. **Body bullets:**
   - The model rationally trades many cheap false positives for the
     recovery of a few expensive false negatives
   - Even the conservative `max_f1` policy saves **€2.61 M**
   - The €2.94 M is on a *two-month* test window; annualised the
     leverage compounds
5. **Figure/image:** `reports/figures/thesis/fig_11_cost_sensitive_threshold_sweep.png`
6. **Caption / footer credit:** *Source: Table 4.8 +
   `fig_11_cost_sensitive_threshold_sweep.png`*
7. **Color emphasis:** Large `#107C41` call-out box around the
   €2.94 M number
8. **Script reference:** Stage 13
9. **Speaker notes:** Linger on this slide. This is the *single
    most quotable* number in the entire defense. Read it twice if
    you must.

---

## Slide 19 — Live Deployment Framework

1. **Title:** From Booking Entry to Power BI Refresh
2. **Layout:** Full-bleed image
3. **Headline number:** **< 500 ms** end-to-end per `/predict` call
4. **Body bullets:**
   - FastAPI + Gradio on `localhost:8000`
   - Async SQLite audit log → CSV → Power BI dashboard
   - Drift loop (PSI) triggers retraining when ≥ 2 features cross
     PSI = 0.25
5. **Figure/image:** `reports/figures/thesis/fig_deployment_framework.png`
6. **Caption / footer credit:** *Source: Chapter IV §4.8, custom
   diagram generated by `scripts/create_deployment_diagram.py`*
7. **Color emphasis:** Diagram already colour-coded by flow type
   (request, persistence, drift, artifact)
8. **Script reference:** Stage 14
9. **Speaker notes:** Trace the path with the cursor: front-desk
    box → FastAPI → SQLite → Power BI. Five seconds, no more.

---

## Slide 20 — Power BI Dashboard Tour

1. **Title:** Eight Pages, One Decision Support Tool
2. **Layout:** Picture-with-caption (8-page mosaic OR live demo
   switch via Alt+Tab)
3. **Headline number:** **8** dashboard pages built from one CSV
4. **Body bullets:**
   - Page 1 Risk Overview, Page 2 Action List, Page 3 Patterns
   - Page 4 Policies, Page 5 ADR, Page 6 Revenue, Page 7 Trust
   - Page 8 Monitoring (PSI drift heatmap)
   - **All pages refresh from the live `/predict` audit log**
5. **Figure/image:** 8-page screenshot mosaic (user supplies — take
   a 2 × 4 grid screenshot of all pages at 1080p)
   *Alternative:* Show the live dashboard via Alt+Tab if confident
6. **Caption / footer credit:** *Source: `data/predictions/predictions_live.csv`,
   refreshed by `make export-predictions`*
7. **Color emphasis:** None — let the dashboard's own colours show
8. **Script reference:** Stage 15
9. **Speaker notes:** Optional live demo: Alt+Tab to the open
    `.pbix`, hover the Page 1 KPI cards, return in ≤ 10 seconds.
    Only do this if you've rehearsed it twice.

---

## Slide 21 — Six Managerial Recommendations

1. **Title:** Six Things the Hotel Can Do Monday Morning
2. **Layout:** Icon grid 2 × 3 (Comparison layout with 6 panes)
3. **Headline number:** *None* (qualitative action slide)
4. **Body bullets (one per icon):**
   - **R1 — Adopt risk-tier policy** (Low / Med / High)
   - **R2 — Tighten policy by booking source**, not guest history
   - **R3 — 72-hr reminder email** for Medium-tier bookings
   - **R4 — Confirmation calls + partial deposit** for High tier
   - **R5 — Use the live API as a frontline tool**
   - **R6 — Treat the PSI drift page as a retrain trigger**
5. **Figure/image:** Six flat icons in a 2 × 3 grid (medal, gear,
   envelope, phone, server, gauge)
6. **Caption / footer credit:** *Source: Chapter V §5.3*
7. **Color emphasis:** `#1F4E79` border on every tile; `#F5A623`
   fill on the icon background
8. **Script reference:** Stage 16
9. **Speaker notes:** Say each recommendation in *one* sentence.
    Resist sub-bullets. The slide visual carries the rest.

---

## Slide 22 — Limitations + Future Research

1. **Title:** What This Study Did Not Do — And What Comes Next
2. **Layout:** Two-content (left: limitations; right: future research)
3. **Headline number:** *None* (qualitative)
4. **Body bullets (left — LIMITATIONS):**
   - Single benchmark; Portugal pre-pandemic
   - No external features (weather, events, FX)
   - Cost model is a single-point estimate
   - PH headline metrics directional only (n_test = 20)
   - €2.94 M is an *upper bound*; not A/B-tested
5. **Body bullets (right — FUTURE RESEARCH):**
   - Add external context features (FR1)
   - Replicate on 10 – 15 PH properties (FR2)
   - A/B test the intervention policies (FR3)
   - Booking-time-only ADR regressor (FR4)
   - Package methodology contributions as a library (FR5)
6. **Figure/image:** None
7. **Caption / footer credit:** *Source: Chapter V §5.4 + §5.5*
8. **Color emphasis:** Amber `#F5A623` left border on limitations;
   green `#107C41` left border on future research
9. **Script reference:** Stage 17
10. **Speaker notes:** Reading limitations *before* recommendations
    fails the audience. Always come *out* of limitations into the
    closing statement — never the other way round.

---

## Slide 23 — Closing Statement & Q&A

1. **Title:** Thank You — Ready for Your Questions
2. **Layout:** Title-only (centered) with a small contact strip
3. **Headline number:** **€2.94 M / 97.5 %** repeated one last time
4. **Body bullets:**
   - *"Cancellation risk is predictable at the moment of booking
     with calibrated probabilities honest enough to drive
     cost-sensitive action."*
   - Repo: github.com/[user]/[repo]
   - Email: dwbvinas@mymail.mapua.edu.ph
5. **Figure/image:** Faded Mapúa logo at 20 % opacity, bottom-right
6. **Caption / footer credit:** *None*
7. **Color emphasis:** `#1F4E79` on the closing sentence
8. **Script reference:** Stage 18
9. **Speaker notes:** Smile. Say *"thank you"* clearly, hold the
   eye contact for 2 seconds, then click to the title slide as a
   neutral background while the panel asks questions.

---

# Q&A Backup Slides (HIDDEN in the deck — un-hide as needed)

> Right-click each backup slide in PowerPoint → *Hide Slide*. They
> stay numbered in the deck order but are skipped during normal
> presentation. Un-hide live when a panel question lands on one.

---

## Slide B1 — Bootstrap Confidence Intervals

1. **Title:** How Tight Are the Headline Numbers?
2. **Layout:** Picture-with-caption
3. **Headline number:** **PR-AUC 95 % CI = [0.748, 0.772]**
   (width 0.024)
4. **Body bullets:**
   - 2,000 bootstrap resamples on the test set
   - ROC-AUC 95 % CI = [0.858, 0.871] (width 0.013)
   - F1 95 % CI = [0.725, 0.744] (width 0.019)
5. **Figure/image:** `reports/figures/thesis/fig_06_bootstrap_ci_forest.png`
6. **Caption / footer credit:** *Source: `reports/benchmarks/13_*.csv`*
7. **Color emphasis:** `#1F4E79` on the champion row
8. **Script reference:** Q1, Q4
9. **Speaker notes:** Cue this slide if the panel pushes on whether
    the LightGBM lead is real or noise.

---

## Slide B2 — Philippine Sub-Study

1. **Title:** Did the Methodology Transfer?
2. **Layout:** Two-content (left: bullets; right: cluster diagnostic)
3. **Headline number:** **`deposit_type` is #1 on both** datasets
4. **Body bullets:**
   - Punta Villa Resort, **n = 193** real PMS bookings
   - Test n = 20 → bootstrap 95 % CI width ≈ ±15 pp
   - Pre-flight duplicate-cluster diagnostic ran and *did not fire* —
     methodology operates honestly
   - PH PR-AUC ≈ 0.54 chronological — directional only, not headline
5. **Figure/image:** `reports/figures/thesis/fig_11.1_ph_cluster_structure.png`
6. **Caption / footer credit:** *Source: `reports/ph/ph_transferability.json`*
7. **Color emphasis:** Amber `#F5A623` border around the slide (small
   sample, hence caveat colour)
8. **Script reference:** Q8
9. **Speaker notes:** Lead with the diagnostic *did not fire* — that
   is the methodological contribution; the metrics are secondary.

---

## Slide B3 — ADR Regression

1. **Title:** What Does the Booking Actually Charge?
2. **Layout:** Picture-with-caption (scatter left; mini table right)
3. **Headline number:** **Test RMSE = €44.31** (Gradient Boosting
   champion, selected by *validation* RMSE)
4. **Body bullets:**
   - 8 regressors compared; **Gradient Boosting** wins on the
     validation set (28.76 €) — the methodologically honest selection
   - **XGBoost is fractionally better on the test set** (44.06 €) but
     loses on validation (29.30 €) — we never select by test to avoid
     test-set peeking
   - R² 0.234 — directional pricing signal, not exact prediction
   - Used live by the Power BI Page 5 (ADR Forecasting)
5. **Body bullets (right — top 4 by Test RMSE, from Table 4.8):**
   | Regressor | Val RMSE | Test RMSE | Test R² |
   |---|---:|---:|---:|
   | XGBoost | 29.30 | **44.06** | 0.243 |
   | **Gradient Boosting** *(champion)* | **28.76** | **44.31** | **0.234** |
   | Random Forest | 31.89 | 44.52 | 0.227 |
   | Decision Tree | 31.28 | 45.87 | 0.179 |
6. **Figure/image:** `reports/figures/thesis/fig_45_adr_pred_vs_actual.png`
7. **Caption / footer credit:** *Source: Chapter IV Table 4.8 +
   `reports/regression_results.csv`*
8. **Color emphasis:** `#1F4E79` on the Gradient Boosting row + a
   `#107C41` highlight on its **Val RMSE 28.76** cell (the
   tie-breaker)
9. **Script reference:** Stage 14 backup + Q3
10. **Speaker notes:** If a panellist points at the XGBoost test
    RMSE 44.06 and asks "why not XGBoost as champion?" — answer
    *"selected by validation RMSE, not test RMSE; XGBoost lost on
    val 29.30 vs Gradient Boosting 28.76 — we never select on the
    test set"*. Then concede *"R² is moderate by design; ADR is
    dominated by rate-card noise the model can't see"*.

---

## Slide B4 — Per-Segment Fairness

1. **Title:** Does the Model Work Equally Well Across Segments?
2. **Layout:** Picture-with-caption
3. **Headline number:** **Groups PR-AUC 0.985** vs **Direct PR-AUC
   0.489**
4. **Body bullets:**
   - Strongest segment: Groups (large, patterned bookings)
   - Weakest segment: Direct (small, idiosyncratic, low base rate)
   - Resort Hotel slightly outperforms City Hotel (+0.029 PR-AUC)
   - Action: human review on Direct-tier flags below probability 0.70
5. **Figure/image:** `reports/figures/thesis/fig_17_segment_performance_heatmap.png`
6. **Caption / footer credit:** *Source: Table 4.11,
   `reports/segment_metrics.csv`*
7. **Color emphasis:** Heatmap already encodes its own gradient
8. **Script reference:** Q9, Q10
9. **Speaker notes:** Acknowledge the Direct-segment gap *before* the
    panel asks — it makes the answer feel earned, not defensive.

---

## Slide B5 — PSI Drift Monitoring

1. **Title:** How Will the Hotel Know When to Retrain?
2. **Layout:** Picture-with-caption
3. **Headline number:** **PSI ≥ 0.25 on ≥ 2 features → retrain**
4. **Body bullets:**
   - Zones: safe < 0.10, watch 0.10 – 0.25, retrain ≥ 0.25
   - Page 8 of the Power BI dashboard refreshes from
     `drift_metrics.csv` weekly
   - The PSI rule is conservative — false alarms cost only a
     re-train cycle, missed drift costs the recovery numbers
5. **Figure/image:** `reports/figures/thesis/fig_8.4_psi_feature_drift_heatmap.png`
6. **Caption / footer credit:** *Source: `scripts/compute_live_drift.py`,
   `src/utils/drift.py`*
7. **Color emphasis:** Heatmap already encodes its own gradient
8. **Script reference:** Q6
9. **Speaker notes:** Use this slide if the panel pushes on
   *"what happens after deployment?"* — it answers the question
   in one diagram.

---

## Slide B6 — Where Does the Model Plug Into the Hotel's IT Stack?

1. **Title:** Where Does the Model Sit in the Hotel's Systems?
2. **Layout:** Picture-with-caption (figure dominates; 3 short bullets
   on the right)
3. **Headline number:** **CRS** — the model lives inside the Central
   Reservation System layer, alongside the dashboard
4. **Body bullets:**
   - **PMS is the hub** — exchanges inventory/prices + bookings with
     the channel manager, OTAs, and other distribution channels
   - **CRS hosts the BI stack** — LightGBM classifier, ADR regressor,
     three threshold policies, TreeSHAP, SQLite audit log, Power BI
   - **Dashed feedback loop** — model output revises inventory/price
     signals back through the channel manager
5. **Figure/image:**
   `reports/figures/thesis/fig_conceptual_systems_positioning.png`
6. **Caption / footer credit:** *Source: Chapter IV §4.8.1, Figure 4.9;
   framework adapted from António, Almeida, & Nunes (2017), Figure 6.*
7. **Color emphasis:** `#1F4E79` on the **CRS** word in the headline
   (matches the navy border on the CRS box in the figure)
8. **Script reference:** General operational-positioning question —
   use this if a panellist asks *"where does this model plug into our
   distribution stack?"* or *"how does it interact with the existing
   PMS / channel manager?"* (no scripted Stage; deliver the three
   bullets verbatim and trace the arrows with the cursor in ≤ 30 s)
9. **Speaker notes:** Open with *"the PMS is the centre of gravity;
   the CRS layer is where the model lives"*, then trace one solid
   arrow (PMS → CRS "all bookings") and one dashed arrow (CRS →
   Channel Manager "revised inventory/prices"). Reference António et
   al. by name so the panel recognises the framework template — they
   wrote the source dataset paper, so naming them earns credibility
   in three words.

---

# Design Notes Appendix

## PowerPoint shortcuts you'll want on the day

| Shortcut | Action |
|---|---|
| **F5** | Start presentation from slide 1 |
| **Shift + F5** | Start presentation from current slide |
| **B** | Black-out screen (use when answering a Q without slides) |
| **W** | White-out screen |
| **Esc** | Exit presentation |
| **Alt + Tab** | Swap to live Gradio UI / `.pbix` for slide 20 demo |
| **Ctrl + Shift + G** | Group selected objects (for icon grids) |
| **N** / **→** | Next slide |
| **P** / **←** | Previous slide |
| **G** | Slide thumbnail grid (jump to slide by number) |

## Final polish checklist (do all six before the day)

- [ ] Open every chart slide and confirm the figure embeds at
      ≥ 1500 × 1000 px (no pixelation when projected)
- [ ] Confirm the palette is applied consistently — `#1F4E79` on
      every primary title, no accidental theme colours
- [ ] Confirm every chart slide has a 12 pt italic source footer
- [ ] Confirm no slide exceeds ~30 words of body text
- [ ] Confirm Q&A backup slides B1 – B6 are hidden (right-click
      → *Hide Slide*) but not deleted
- [ ] Print this slide guide + the script as a paper backup —
      tape the script to the back of the laptop, paper-clip the
      slide guide to your folder

## A note on the figures

All 14 referenced figures already exist on disk under
`reports/figures/thesis/` or `reports/thesis/` — they were
generated by the training and reporting pipeline and do not need
regeneration before the defense. The only image the user must
supply is the 8-page Power BI mosaic for slide 20, which is taken
by screenshotting each page of the live `.pbix` and assembling them
in a 2 × 4 grid (any image editor works — PowerPoint's
*Insert → Photo Album → Grid* layout also works).
