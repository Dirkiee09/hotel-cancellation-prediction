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
