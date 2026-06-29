
# Thesis Defense Guide — 2026-06 (post-audit, canonical numbers)

> Supersedes numeric content in the older defense pack where they conflict:
> this guide is built from the retrained model's canonical result files
> (verified 2026-06-11 against reports/metrics.json). For stale-vs-current
> values see ../thesis_drafts/NUMBER_CORRECTIONS.md.

## The thesis in three sentences

"I built and validated an end-to-end ML system that predicts hotel booking
cancellations at booking time, calibrated so its probabilities are
trustworthy and connected to a cost model so its decisions are economically
grounded. On 119k Portuguese bookings it achieves ROC-AUC 0.863 with a 3%
calibration error, and the cost-aware policy beats even the best trivial
strategy by 36% on held-out data. I then demonstrated the METHOD transfers
to a Philippine resort's real PMS data, and every number is regenerable by
one command — verified on independent hardware."

## Tier-1 numbers (memorize)

- Test (max-F1): ROC-AUC 0.863 · PR-AUC 0.759 · F1 0.736 · P 0.625 · R 0.895
- Base rate 37% (test 37.8%, n = 11,922) — PR-AUC baseline is 0.378
- Calibration ECE 0.062 -> 0.031 (isotonic, fitted on val only)
- Thresholds: max_f1 0.41 · high_precision 0.98 · cost_sensitive 0.06
- Test cost ladder: nothing EUR 2,322,794 -> thr-0.5 EUR 669,637 ->
  intervene-all EUR 111,240 -> cost policy EUR 71,136
  (savings: +598,502 vs 0.5; **+40,104 = 36% vs intervene-all <- headline**)
- H2 vs LogisticRegression: +0.0045 PR-AUC, p = 0.177 -> NOT significant
- Selection (rolling val PR-AUC, matched 300/7/0.05): LGBM 0.8693 >
  XGB 0.8684 > GB 0.8669
- Split decision (test, paired n=2000): XGB +0.0036 PR-AUC (p=.01);
  LGBM +0.0087 F1 (p=.002); ROC tie (p=.183); LGBM 2.8x faster
- Duplicates: 31,994 (26.8%); ZERO cross train/test; PR-AUC 0.759
  per-record vs 0.703 per-unique-profile (dedup_sensitivity.json)
- SHAP top-3: deposit_type, country, agent (lead_time #7, prev_canc #10)
- Philippines: 193 rows, 15% cancel; Spearman rank rho = 0.71 (p=.071);
  PH #1 = XGBoost; CIs +/- 15pp -> directional only
- Engineering: 148 tests / 88.9% cov; determinism delta = 0.0; cloud CI
  retrains + regenerates thesis analysis green (37 min)

## Crucial tables/figures

1. E12 cost ladder — the "so what" answered in money, on test, vs the
   strongest trivial baseline
2. E04 significance forest — paired CIs; champion-vs-leader gap 0.0036,
   baselines 6-70x worse
3. E07 calibration before/after — a methodological choice that earned its keep
4. benchmarks/14_paired_significance_vs_champion.csv — statistical backbone
5. E17 rank-slope — transferability in five seconds
6. Notebook 11 Table 11.2 — 7 algos x 2 markets; compare RANKS not values
7. reports/dedup_sensitivity.json — the measured duplicates answer

## Seven danger zones (rehearse the answers)

1. **H2 not significant** — own it: marginal discrimination gap measured
   honestly; GBT value = calibration + cost policy + nonlinearity without
   hand-engineering. Defend a small honest gap, not a large fragile one.
2. **H3 ordering wrong** — partially supported: features right, order wrong
   (deposit #1, lead_time #7). Say it before they do.
3. **Why LightGBM if XGBoost wins test PR-AUC?** — prespecified validation
   protocol; choosing on test = selection bias; metric-dependent winners
   within 0.01; 2.8x faster training.
4. **Duplicates** — zero cross-boundary (verified); report 0.759/0.703 as a
   per-record vs per-unique-profile range; no booking IDs -> both views valid.
5. **Non-Refund 99.4% artifact** — known dataset artifact; ablation bounds it
   (masking deposit barely hurts — correlated features absorb it; country
   masking costs most); same signal tops independent PH data. Attribution
   != necessity.
6. **n=193 PH pilot** — thesis says it first: +/-15pp CIs, directional only;
   method-transfer proven, findings-transfer directional (rho=.71);
   deliverable = data-collection roadmap (>= 1 season, missing fields).
7. **Cost model assumes intervention works** — acknowledged simplification;
   FP-cost sweep EUR 1-100 shows smooth adaptation; intervene-all baseline
   strips most optimism and 36% incremental value survives.

## Never say / always say

- NOT "saves EUR 1.5M" -> "EUR 40k (36%) beyond the best trivial policy"
- NOT "LightGBM is the best algorithm" -> "won our prespecified protocol;
  modern GBTs statistically interchangeable here"
- NOT "threshold is robust" -> "policy adapts smoothly, no instability"
- NOT any number from pre-audit drafts (0.864/0.760, thr 0.04, "H2 all
  p<0.001") — see NUMBER_CORRECTIONS.md

## Strategy

- Open with the live demo (`make demo`) — a working system reframes the room
- Volunteer H2/H3 honesty in your own words before questions
- Trump card under fire: "one command regenerates every number; GitHub's
  servers verified it independently"
- Close hard exchanges on undisputed ground: calibration, cost ladder,
  reproducibility

## Night-before checklist

1. `make check` (all gates green)
2. `make demo` (app boots, one test prediction)
3. Print this guide + tier-1 numbers
4. E04 / E07 / E12 / E17 open as backup images
5. Manuscript corrected against NUMBER_CORRECTIONS.md




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
3. **Headline number:** **PR-AUC 0.760** (LightGBM, calibrated) —
   **+0.38 over the Dummy baseline**
4. **Body bullets (compressed Table 4.2 + Dummy anchor row):**
   | Algorithm | F1 | ROC-AUC | PR-AUC |
   |---|---:|---:|---:|
   | **LightGBM (champion)** | **0.735** | **0.864** | **0.760** |
   | Gradient Boosting | 0.734 | 0.861 | 0.754 |
   | XGBoost | 0.729 | 0.855 | 0.749 |
   | Random Forest | 0.704 | 0.851 | 0.739 |
   | Logistic Regression | 0.713 | 0.839 | 0.739 |
   | Decision Tree | 0.596 | 0.675 | 0.508 |
   | *Dummy (majority-class baseline)* | *0.000* | *0.500* | *0.378* |
5. **Figure/image:** None — table is the visual
6. **Caption / footer credit:** *Source: Table 4.2 chronological test
   set (n = 11,922; threshold = `max_f1` per model); Dummy row is the
   theoretical floor at the 37.8 % positive-class base rate*
7. **Color emphasis:** Highlight the LightGBM row with a `#1F4E79`
   left-border accent; render the Dummy row in italic grey (`#7A7A7A`)
   to mark it as the baseline-floor reference
8. **Script reference:** Stage 7
9. **Speaker notes:** Anchor against the Dummy first ("our floor is
   PR-AUC 0.378") *then* point at LightGBM ("the champion delivers
   0.760 — a 38-point lift over guessing the majority class"). The
   PR-AUC gap to GB is 0.006 — don't oversell the *internal* lead.
   The bootstrap p = .001 line on the next slide does that for you.

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
