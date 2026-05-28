# Table 4.7 — Risk Tier Distribution × Revenue Exposure

**Source.** `reports/test_predictions_for_powerbi.csv` (n = 11,922
chronological test bookings).

**Method.** Risk tier boundaries are taken from `src/config.py`:
**Low** (P < 0.40), **Medium** (0.40 ≤ P < 0.70), **High** (P ≥ 0.70).
For each tier, two financial columns are reported:

- **"Total Revenue in Tier"** = the sum of `revenue_at_risk` across **all
  bookings** in the tier (the booking value at stake if every flagged
  booking cancelled).
- **"Realised Revenue Lost"** = the sum of `revenue_at_risk` for bookings
  that **actually cancelled** in the tier — the ground-truth financial
  exposure observed on the held-out test set.

| Risk Tier | Calibrated Probability Band | Bookings (count) | % of Total | Avg Revenue / Booking (€) | Total Revenue in Tier (€) | Actual Cancellations | Realised Revenue Lost (€) | Recommended Action |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Low** | P < 0.40 | 6,107 | 51.22 % | 539.47 | 3,294,519.39 | 715 | 375,383.06 | No action — accept the booking on standard terms. |
| **Medium** | 0.40 ≤ P < 0.70 | 2,707 | 22.71 % | 706.14 | 1,911,520.91 | 1,435 | 1,066,904.98 | Reminder email 72 h before arrival; soft confirmation request. |
| **High** | P ≥ 0.70 | 3,108 | 26.07 % | 650.56 | 2,021,930.98 | 2,356 | 1,571,977.80 | Confirmation call + partial deposit request before arrival week. |
| **Total** | — | **11,922** | **100.00 %** | **606.27** | **7,227,971.28** | **4,506** | **3,014,265.84** | — |

**Two findings a panelist will probe.**

1. **Concentration of risk.** 26.07 % of bookings (the High tier) carry
   **52.15 %** of all realised cancellation losses
   (€1,571,977.80 / €3,014,265.84). Operationally: focusing intervention
   effort on the High tier alone recovers more than half of the total
   revenue exposure.

2. **Empirical calibration honesty (per-tier cancellation rates).**
   Low tier 11.71 %, Medium tier 53.01 %, High tier **75.80 %**. The
   model's "75 %" probability really does correspond to ~76 % observed
   cancellations — the isotonic calibration step (Table 4.10) is doing
   real work, not just looking good.

**Why this table is the headline BI deliverable.** It converts a
probability score (which a hotel manager cannot act on) into three
operational decisions (which she can). The "Recommended Action" column
is the bridge from machine learning to revenue management.
