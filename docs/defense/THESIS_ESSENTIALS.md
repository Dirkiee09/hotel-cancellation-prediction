# Thesis Essentials & Validation Report

**Validated against:** `Predicting Hotel Booking Cancellations_ A Machine Learning Approach Finishing touch.pdf`
(121 pages, text-layer verified). This file = the at-a-glance facts you must know cold + the issues
found during validation.

---

## 1. Identity

| | |
|---|---|
| **Title** | A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations |
| **Authors** | Avanceña, Luis Miguel C.; Montecino, Nathaniel; Viñas, Dirk Werner |
| **Advisers** | Prof. John Edward Manalac; Dr. Donn Enrique L. Moreno |
| **One line** | A calibrated, cost-aware decision system that predicts cancellations at booking time and turns the scores into euro-denominated actions, framed by Dynamic Capability Theory (Sense → Seize → Transform). |

---

## 2. Numbers you must know cold

| Item | Value |
|---|---|
| Champion | **LightGBM** (selected by rolling-origin PR-AUC) |
| Test ROC-AUC / PR-AUC / F1 | **0.863 / 0.759 (range 0.70–0.76 de-dup) / 0.736** |
| Precision / Recall (max-F1) | 0.625 / 0.895 |
| Calibration ECE (raw → isotonic) | 0.062 → 0.031 |
| LightGBM vs Logistic Regression | +0.005 PR-AUC, **p = 0.177 (NOT significant)** |
| LightGBM vs Decision Tree / Random Forest | +0.246 / +0.032 (both significant) |
| Cost: no-model / intervene-all / model | €2,322,794 / €111,240 / €71,135.50 |
| Saving vs intervene-all | €40,105 = **36%** |
| Cost threshold / recall / % flagged | 0.06 / 0.991 / ~72% of bookings |
| FP / FN cost | €15 / proportional to revenue at risk |
| Duplicates | ~27% exact; champion unchanged after de-dup |
| Philippine pilot | 193 rows, test n = 20, ±15pp CI (directional) |
| Top SHAP feature (both markets) | **deposit_type** |
| Split | chronological 80 / 10 / 10 (reported as 80/20 holdout) |

---

## 3. What the figures/tables prove (1.1 → 4.20, all present)

- **Fig 1.2** — conceptual framework (Sense→Seize→Transform; full model ladder Dummy→NB→LR→DT→RF→GB→XGB→LightGBM)
- **Fig 4.8 / 4.9** — system positioning + technical serving architecture (the deployment story)
- **Fig 4.1–4.7** — significance, SHAP, ROC/PR, confusion matrix, calibration, risk tiers, cost ladder
- **Fig 4.10–4.12** — cross-market rank, temporal stability, calibration over time
- **Fig 4.20** — Power BI dashboard
- **Tables 4.1–4.6** — CV benchmark, chronological test, paired significance, SHAP, cost-savings, hypothesis verdicts

---

## 4. VALIDATION FINDINGS — ✅ ALL APPLIED (2026-06-22)

All three issues below were applied to `Revisions of Predicting Hotel Booking Cancellations_ A Machine
Learning Approach.docx` and **highlighted yellow** for your review. Clear the highlights (Word ▸ select
all ▸ Text Highlight Color ▸ No Color) to produce the final clean copy.

### ✅ Issue 1 (FIXED): the abstract overclaimed and contradicted the body
The reworded abstract said LightGBM *"demonstrated **superior discriminatory power**"* and that the
Philippine pilot *"**validated** the transferability"* — which **contradicted the body** (advantage
over logistic regression *not significant, p = 0.177*; pilot *directional only, n = 20*) and dropped
the **0.70–0.76 PR-AUC range**. The abstract now states the range, the p = 0.177 honesty, and "directional
evidence … a 20-booking proof of concept rather than a benchmark."

**Replacement text that was applied:**
> "The champion LightGBM model achieved a ROC-AUC of 0.863 and a PR-AUC of 0.759 — reported as a
> range of approximately 0.70–0.76 depending on whether exact-duplicate block bookings are retained
> or removed — and was refined through isotonic calibration to halve its calibration error. A paired
> bootstrap test found its advantage over a well-tuned logistic regression to be small and **not
> statistically significant (p = 0.177)**, so LightGBM was retained for its calibration, recall, and
> efficiency rather than a decisive accuracy margin. SHAP analysis identified deposit type, not lead
> time, as the dominant driver. A cost-minimizing threshold lowered expected cost by 36% relative to
> the strongest trivial policy. … a pilot on an independent Philippine resort dataset provided
> **directional evidence** for the transferability of the pipeline (a 20-booking proof of concept,
> not a benchmark)."

### ✅ Issue 2 (FIXED): the 2017 António reference was missing
Figure 4.8's caption cites *"Adapted from António, Almeida, & Nunes (2017)"*, but the References list
only had the **2019** paper. The 2017 source was added (and the 2019 entry's spelling corrected to
"António, N., de Almeida"):
> António, N., de Almeida, A., & Nunes, L. (2017). Predicting hotel booking cancellations to decrease
> uncertainty and increase revenue. *Tourism & Management Studies, 13*(2), 25–39.
> https://doi.org/10.18089/tms.2017.13203

### ✅ Issue 3 (FIXED): dual-roster note was absent
Table 4.1 (CV) has no Random Forest; Tables 4.2/4.3 add it and drop Dummy/NB. A sentence was added to
the methodology: *"The CV benchmark (Table 4.1) retains the trivial Dummy/Naïve Bayes anchors to show
the full complexity ladder; the chronological test (Table 4.2) drops them and adds Random Forest as a
bagged-ensemble contrast."*

> **Note on lineage:** this PDF is a **reworded version** (new title, new abstract) separate from the
> `D:\Documents Dirk\Thesis\` build pipeline. Fixes above must be made in **whatever Word source
> produced this PDF**, not the `.txt` pipeline — they won't propagate automatically.

---

## 5. The three things to never get wrong in the room
1. LightGBM is **not** statistically better than logistic regression — lead on calibration + cost system.
2. The 36% saving is **conditional on the €15/revenue cost assumptions** (sensitivity tested).
3. The Philippine result is a **directional pilot (n = 20)**, not a validated benchmark.
