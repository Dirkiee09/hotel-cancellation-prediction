# Table 4.8 — Threshold Policy Operational Comparison

**Source.** `reports/test_predictions_for_powerbi.csv` (LightGBM champion
at calibrated probabilities), `reports/confusion_matrix_*.csv`,
`reports/cost_threshold_summary.json`. Computed on the Portugal
chronological test set (n = 11,922; 4,506 actual cancellations).

**Cost model.** FP intervention = €15 per false flag
(`FP_INTERVENTION_COST` in `src/config.py`); FN cost = full
`revenue_at_risk` of the missed cancellation (ADR × `total_stay`).
"No-model baseline" = catch nothing → all 4,506 cancellations become
FNs and their full revenue at risk is lost.

| Policy | Threshold | Bookings Flagged | % Flagged | TP | FP | FN | TN | Recall | Precision | FP Cost (€) | FN Revenue Lost (€) | **Total Cost (€)** | Savings vs No Model (€) | Recommended Use Case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `max_f1` (balanced) | 0.40 | 5,815 | 48.78 % | 3,791 | 2,024 | 715 | 5,392 | 0.841 | 0.652 | 30,360.00 | 375,383.06 | **405,743.06** | 2,608,522.78 | Default weekly operations — balances FP cost against recall. |
| `high_precision` | 0.98 | 426 | 3.57 % | 426 | 0 | 4,080 | 7,416 | 0.095 | 1.000 | 0.00 | 2,874,599.08 | **2,874,599.08** | 139,666.76 | Quarterly executive audit — only the most certain cancellations. |
| **`cost_sensitive`** (champion deployment) | 0.04 | 8,957 | 75.13 % | 4,486 | 4,471 | 20 | 2,945 | 0.996 | 0.501 | 67,065.00 | 9,446.98 | **76,511.98** | **2,937,753.86** | **Recommended deployment default — minimises total expected cost.** |
| No model (catch nothing) | — | 0 | 0.00 % | 0 | 0 | 4,506 | 7,416 | 0.000 | — | 0.00 | 3,014,265.84 | **3,014,265.84** | — | Baseline reference only. |

**Headline finding.** The cost-sensitive policy recovers
**€2,937,753.86 of revenue at risk** on the test set — **97.5 % of the
theoretical maximum** (€3,014,265.84). It does so by trading 4,471 false
positives at €15 each (€67,065) to catch all but 20 of the 4,506 actual
cancellations. Compared to the conventional balanced threshold
(`max_f1`), the cost-sensitive policy saves an additional
**€329,231.08** per cycle on the test set alone.

**Why the `high_precision` policy looks catastrophic here.** At
threshold 0.98 the model only flags 426 bookings — it catches 9.5 % of
cancellations and lets the other 4,080 cost the hotel their full
revenue at risk. This is *by design*: `high_precision` is for cases
where every flag must be defensible (e.g., charging a deposit), not
for maximising revenue recovery. Including this row in the table
demonstrates that the system offers a *spectrum* of operating points,
not a single forced trade-off.

**Sensitivity caveat.** The €15 FP intervention cost is the assumed
industry-range value (`FP_INTERVENTION_COST` in `src/config.py`).
Section 4.4 and Notebook 10 demonstrate that the cost-sensitive
threshold (0.04) is robust across the €5–€60 FP-cost sweep — the policy
ranking does not change for any reasonable cost assumption.

**Link back to H4.** This table is the empirical evidence behind H4
(Table 4.6): cost-sensitive thresholding (€76,512 total cost) reduces
expected revenue loss by €2,937,754 vs no model — a 97.5 % recovery rate
that supports H4 with a margin large enough to survive any reasonable
cost-model perturbation.
