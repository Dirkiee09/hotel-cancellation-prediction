# Table 4.5 — Financial Cost-Savings by Threshold Policy

**Source.** `reports/cost_threshold_summary.json`, `reports/metrics.json`,
`artifacts/thresholds.json`, `reports/cost_threshold_sweep.csv` (101-step
threshold grid).

**Cost model.**
- **FP intervention cost** = €15 per false-positive flag (SMS + staff
  review) — see `FP_INTERVENTION_COST` in `src/config.py`.
- **FN recovery cost** = `revenue_at_risk` (ADR × `total_stay`) per missed
  cancellation, with one recovery night assumed (`FN_RECOVERY_NIGHTS = 1`).
- **No-model baseline cost** = total `revenue_at_risk` of all actual
  cancellations in the validation set = €1,606,669.92.
- **Threshold-0.50 baseline cost** = the cost of flagging at the
  conventional 50% threshold = €387,350.44.

**Three operating policies compared (LightGBM champion, Portugal validation set).**

| Policy | Threshold | TP / FP / FN / TN (val) | Precision | Recall | F1 | Total Expected Cost (€) | Savings vs No Model (€) | Savings vs Thr 0.50 (€) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **`cost_sensitive`** (champion) | 0.04 | TP=6,652 / FP=3,716 / FN=27 / TN=1,529 | 0.501 | 0.996 | 0.666 | **59,160.04** | **+1,547,509.88** | **+328,190.40** |
| `max_f1` (balanced) | 0.40 | (see Table 4.2 chronological row) | 0.652 | 0.841 | 0.735 | not the cost-optimised policy | — | — |
| `high_precision` (precision-first) | 0.98 | low TP, near-zero FP | 1.000 | 0.095 | 0.173 | not the cost-optimised policy | — | — |
| Threshold = 0.50 (naïve baseline) | 0.50 | — | — | — | — | 387,350.44 | +1,219,319.48 | — |
| No model (catch nothing) | — | TP=0, FN=all positives | — | — | — | 1,606,669.92 | — | — |

**Headline result.** The `cost_sensitive` policy saves **€1,547,509.88
versus no model** and **€328,190.40 versus the naïve 0.50 threshold** on
the Portugal validation set — that's **95.4 % of the theoretical
maximum** (€1,606,669.92, the cost of catching nothing) recovered by
the operationalised pipeline. This is the headline BI claim of the
thesis.

**Why the threshold is so low (0.04).** The cost asymmetry is severe:
each FP costs €15 but each FN costs ~€430 average revenue-at-risk. The
optimiser rationally trades many cheap false positives for the recovery
of a few expensive false negatives. The recall climbs to 99.6 % — the
model essentially catches all real cancellations at this threshold.

**Operational policy table (decision support for revenue managers).**

| Risk Tier | Calibrated Probability | Action Recommended | Expected Volume (Portugal val) |
|---|---|---|---:|
| Low | < 0.40 | No action | majority of bookings |
| Medium | 0.40 – 0.70 | Reminder email 72h before arrival | moderate count |
| High | ≥ 0.70 | Confirmation call + partial deposit request | smallest count |

(Risk tier boundaries from `RISK_TIER_MEDIUM_THRESHOLD = 0.40` and
`RISK_TIER_HIGH_THRESHOLD = 0.70` in `src/config.py`.)

**Philippines sub-study.** No cost-sensitive policy is reported for PH —
the validation set (n_val ≈ 19) is too small to fit a reliable cost
curve, so PH uses only `max_f1` and `high_precision`. Caveat documented
in `CLAUDE.md` and Section 4.5.
