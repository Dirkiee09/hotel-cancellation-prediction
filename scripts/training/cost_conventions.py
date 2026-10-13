"""How much the headline cost cut depends on how a contact is priced.

The paper's cost model (equation 8) charges EUR 15 only for a wasted contact: a
contact made to a guest who would have cancelled costs nothing and always saves
the booking. Neither of those is measured, and none of the 150 scenarios in
``revision_diagnostics.py`` varies them. This re-prices the same two policies,
contacting everyone and the cost-sensitive threshold already chosen on
validation, under stricter conventions. No retraining, no re-tuning: the
threshold stays where the pipeline put it.

Conventions:
  paper            EUR 15 per wasted contact; a contacted canceller is saved free
  charge_all       EUR 15 for every contact, including real cancellers
  charge_all_e50   ...and a contact saves only half of would-be cancellations
  charge_all_e25   ...and only a quarter

Input:  reports/dedup/test_predictions_for_powerbi.csv, reports/dedup/thresholds.json
Output: reports/thesis/cost_conventions.csv (this script is the sole writer)

Usage:
    python scripts/training/cost_conventions.py
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.config import FN_RECOVERY_NIGHTS, FP_INTERVENTION_COST, REPORTS_DIR, TARGET_COL
from src.utils import configure_logging
from src.utils.business import compute_fn_cost_vector

logger = logging.getLogger(__name__)

CONVENTIONS = (
    ("paper", "EUR 15 per wasted contact; a contacted canceller is saved for free", False, 1.0),
    ("charge_all", "EUR 15 for every contact, including real cancellers", True, 1.0),
    (
        "charge_all_e50",
        "EUR 15 for every contact; a contact saves half of would-be cancellations",
        True,
        0.5,
    ),
    (
        "charge_all_e25",
        "EUR 15 for every contact; a contact saves a quarter of would-be cancellations",
        True,
        0.25,
    ),
)


def _cost(y, prob, loss, threshold, *, charge_true_positives, effectiveness):
    flagged = prob >= threshold
    fp = flagged & (y == 0)
    tp = flagged & (y == 1)
    fn = ~flagged & (y == 1)
    total = (
        FP_INTERVENTION_COST * fp.sum() + loss[fn].sum() + (1.0 - effectiveness) * loss[tp].sum()
    )
    if charge_true_positives:
        total += FP_INTERVENTION_COST * tp.sum()
    return float(total)


def main() -> None:
    configure_logging()
    preds = pd.read_csv(REPORTS_DIR / "dedup" / "test_predictions_for_powerbi.csv")
    thresholds = json.loads((REPORTS_DIR / "dedup" / "thresholds.json").read_text(encoding="utf-8"))
    tau = float(thresholds["cost_sensitive"]["threshold"])
    y = preds[TARGET_COL].to_numpy().astype(int)
    prob = preds["cancel_probability"].to_numpy()
    loss = compute_fn_cost_vector(preds, fn_recovery_nights=FN_RECOVERY_NIGHTS)

    rows = []
    for key, label, charge_tp, effectiveness in CONVENTIONS:
        blanket = _cost(
            y, prob, loss, 0.0, charge_true_positives=charge_tp, effectiveness=effectiveness
        )
        policy = _cost(
            y, prob, loss, tau, charge_true_positives=charge_tp, effectiveness=effectiveness
        )
        rows.append(
            {
                "convention": key,
                "description": label,
                "threshold": tau,
                "contact_everyone_cost_eur": round(blanket, 2),
                "cost_sensitive_cost_eur": round(policy, 2),
                "saving_eur": round(blanket - policy, 2),
                "saving_pct": round(100.0 * (blanket - policy) / blanket, 2),
            }
        )
        logger.info("cost_convention %s saving_pct=%.2f", key, rows[-1]["saving_pct"])

    out = REPORTS_DIR / "thesis" / "cost_conventions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(
        "cost_conventions_written path=%s flagged=%d of %d", out, int((prob >= tau).sum()), len(y)
    )
    print(pd.DataFrame(rows)[["convention", "saving_eur", "saving_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
