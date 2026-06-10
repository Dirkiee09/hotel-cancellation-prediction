# 30-Minute Professor Consultation — Runbook & Script (2026-06)

## Prep checklist (10 min before)
1. `make demo` — app running at localhost:8000/ui
2. Open `notebooks/11_transferability_ph.ipynb` (rendered)
3. GitHub Actions tab open (green run visible)
4. Excel: `reports/benchmarks/16_rankings.csv` + `14_paired_significance_vs_champion.csv`
5. Terminal ready: `python demo/sample_requests.py`
6. Print the numbers card (bottom)

Golden rule: demo is the opener, not the closer. Fallback = notebook screenshots.

## Timeboxes
- 0:00–2:00  Framing: working system + validated results + two guidance asks
- 2:00–9:00  Live demo: high-risk booking (Non-Refund, lead 200, prev cancel) →
  calibrated probability, TreeSHAP drivers, ADR forecast; low-risk booking
  (repeat guest, Direct); storage (CSV/SQLite/Power BI); sample_requests.py
- 9:00–16:00 Results + audit story: test metrics; matched-capacity selection;
  de-circularized H4; zero row-leakage verified; deterministic (Δ=0.0);
  cost result framed vs intervene-all (+36%), not just vs no-model
- 16:00–21:00 Honest findings FIRST: H2 p=0.177 vs LR (marginal advantage —
  sell calibration+cost+speed package); champion split decision (prespecified
  validation protocol; switching on test metrics = selection bias)
- 21:00–26:00 PH transferability (NB 11): method transfers (same code, 2
  constants) vs findings directional (ρ=0.71, GBTs top both, ±15pp CIs at
  n=193); output = data-collection roadmap for the resort
- 26:00–30:00 Asks: (1) honest H2 framing in main text vs limitations,
  (2) ADR chapter: exploratory vs re-produce first, (3) PH pilot: chapter vs
  appendix scope

## Hard-question bank
- 0.86 good? Published range 0.85–0.93; >0.90 typically uses post-booking
  leakage features; mine is booking-time only, verified leak-free.
- Why not XGBoost? Prespecified validation rule; test-based switching =
  selection bias; statistically interchangeable; LightGBM 2.8× faster.
- 193 rows useless? For metric confirmation yes (and the notebook says so);
  for method-generalization proof + a collection roadmap, exactly enough.
- Duplicates? 32k exact dups; verified zero cross train/test boundary
  (arrival-date features pin them inside partitions); widens true CIs →
  limitations section.
- Non-Refund 99% artifact? Known dataset artifact (tour-operator blocks);
  ablation quantifies dependence; in limitations.
- Reproducible? `make train`, deterministic, identical across double runs and
  on CI.

## Numbers card
```
TEST SET (chronological, untouched until final eval; n=11,922)
  ROC-AUC 0.8634 | PR-AUC 0.7590 | F1 0.7356 | ECE 0.062→0.031
  Thresholds: max_f1=0.41 · high_precision=0.98 (P=1.00, R=0.094) · cost=0.06
SELECTION (rolling-origin val PR-AUC, matched capacity 300/7/0.05)
  LightGBM 0.8693 > XGBoost 0.8684 > GradBoost 0.8669
H2 vs LogisticRegression: dPR-AUC +0.0045, p=0.177 (not significant)
H4 cost (test): saves EUR 599k vs thr-0.5 · +EUR 40k (36%) vs intervene-all
PHILIPPINES: 193 rows, 15% cancel · Spearman rank rho=0.71 across 7 algos ·
  GBTs top both markets · CIs ±15pp → directional only
ENGINEERING: 147 tests, 88.9% cov · mypy 55 files · bandit/pip-audit clean ·
  CI retrains from scratch · repro check Δ=0.0 across double training
```
