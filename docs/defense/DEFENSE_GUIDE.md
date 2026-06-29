# Defense Guide — Updated

**Thesis:** A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations
**Companion files:** `THESIS_ESSENTIALS.md` (facts + fixes), `defense_qa_hard_questions.md` (Q&A),
`Defense_Cheat_Sheet.docx` (one-page print).

---

## 1. The spine (say this story, in this order)

> **Problem → Prediction → Probability → Action → Proof of transfer.**
> Hotels lose money to cancellations they can't see coming. We predict each booking's cancellation
> risk **at booking time**, **calibrate** the probability so it's trustworthy, turn it into a
> **cost-aware action** (risk tier → intervention), deploy it in **Power BI**, and show the **method
> transfers** to a real Philippine resort. The contribution is the *system and the rigor*, not a
> new algorithm.

Everything you present should hang on that one sentence.

---

## 2. Suggested 8–10 minute flow

| # | Slide | Land this point | Visual |
|---|---|---|---|
| 1 | Title + team | — | — |
| 2 | Problem & business cost | Cancellations = revenue instability; late ones hurt most | — |
| 3 | Framework (DCT) | Sense → Seize → Transform organizes the whole study | **Fig 1.2** |
| 4 | Data & honest setup | 119k Portugal bookings; **chronological** split (no leakage); booking-time features only | leakage exclusion list |
| 5 | Models tried | Complexity ladder Dummy → … → LightGBM | Table 4.1 |
| 6 | Headline result | ROC-AUC 0.863, PR-AUC 0.759; **calibration halved (0.062→0.031)** | **Fig 4.3 + 4.5** |
| 7 | Honesty slide | LightGBM ties logistic regression (p=0.177) → kept for recall/calibration/speed | Table 4.3 |
| 8 | Why it matters (cost) | 36% cheaper than intervene-on-all, **under stated cost assumptions** | **Fig 4.7** cost ladder |
| 9 | Drivers | deposit_type is #1 (not lead time) — corrects H3 | **Fig 4.2 / SHAP** |
| 10 | Deployment | FastAPI + Power BI decision system | **Fig 4.8 / 4.9 / 4.20** |
| 11 | Transfer pilot | Method ports to real PH resort; deposit_type #1 in both — **directional (n=20)** | **Fig 4.10** |
| 12 | Contribution & future work | Calibrated, cost-aware, reproducible template; external-data fusion = future | — |

---

## 3. Five messages to land (and the proof for each)

1. **"We didn't leak the future."** — chronological split, booking-time features, explicit exclusion
   of `reservation_status` etc.
2. **"Our probabilities are trustworthy."** — isotonic calibration, ECE 0.062 → 0.031, reliability diagram.
3. **"We're honest about the model."** — p = 0.177 vs logistic regression is reported, not hidden.
4. **"It pays off in euros."** — 36% vs the strongest trivial policy, with a sensitivity analysis.
5. **"The method travels."** — real second market (Punta Villa), same top driver, framed as a pilot.

---

## 4. Lead with your strengths (examiners undervalue these)
- **Intellectual honesty** — your rarest, strongest card. Volunteer the limitations.
- **Reproducibility** — fixed seeds, CI, tests, one-command regeneration.
- **A real PMS dataset** (Punta Villa) — genuine novelty vs. recycled Kaggle work.
- **Calibration + cost framing** — most cancellation papers stop at accuracy.

## 5. Preempt your weaknesses (raise them before the panel does)
- "LightGBM isn't statistically better than LR" → that's why the contribution is the *system*.
- "Cost-optimal flags ~72% of bookings" → the value is avoiding the costliest misses; max-F1 is the
  selective operating point.
- "€15 is an assumption" → tested via sensitivity analysis (Fig 4.13).
- "DCT doesn't drive the modeling" → it's the organizing managerial lens, by design.
- "Pilot is n=20" → proof of concept, not a benchmark.

---

## 6. Opening & closing lines

**Open:**
> "Cancellations are the single biggest source of revenue uncertainty for a hotel — and the ones that
> hurt most arrive too late to resell. Our study turns each booking into a calibrated, cost-aware
> decision the front desk can act on the moment the reservation is made."

**Close:**
> "We're not claiming the smartest algorithm — a tuned logistic regression matches it statistically,
> and we say so. What we contribute is a calibrated, interpretable, cost-aware decision *system*,
> evaluated honestly on time-ordered data and shown to transfer to a real second market. That's the
> part a hotel can actually deploy."

---

## 7. Before you walk in — checklist
- [x] **abstract overclaim** fixed + highlighted in the docx (Issue 1) — clear highlights for the final copy.
- [x] **2017 António reference** added; 2019 spelling corrected (Issue 2).
- [x] **dual-roster note** added to methodology (Issue 3).
- [ ] Know the **number table** cold (cheat sheet).
- [ ] Rehearse the **3 killer answers** (LR-tie, €15/72%, DCT).
- [ ] Confirm **Figure 4.20** (dashboard) renders in the final PDF.
- [ ] Have the **agree → bound → pivot** reflex ready for any hit.
