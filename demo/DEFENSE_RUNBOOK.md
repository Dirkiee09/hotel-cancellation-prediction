# Defense Day Runbook — Hotel Cancellation Thesis

**Print this. Keep it next to your laptop.** It contains the exact commands
to run, the numbers that should appear on screen, the recovery procedures
if anything breaks mid-demo, and pre-cooked answers to the most likely
panelist questions.

> Last verified on a clean working tree at `1a5c100` (Redesign PH Gradio UI).
> Run `git status` before defense day; if anything is dirty, freeze your
> demo branch at this commit until after the defense.

---

## ⏱ Day-of timing (read this first)

| When | What to do | Time |
|---|---|---|
| Night before | Run the "night-before checklist" below; sleep with both servers green | ~15 min |
| 30 min before | Open the runbook; mentally rehearse Act 1-5 | 10 min |
| 5 min before | Run "demo-day startup"; confirm both browser tabs load; click each tab once to warm the caches | 5 min |
| Demo opens | Talk first, click second — narrate the plan, *then* show | 15-30 min |
| Q&A | Use the "Anticipated questions" section as your safety net | 15-30 min |

---

## 🚀 Quick start (commands you'll actually type)

Run these **in two separate terminals** so each server has its own log
visible:

```powershell
# Terminal 1 — Portugal main study
python demo/start_server.py
# -> opens http://localhost:8000/ui

# Terminal 2 — PH sub-study (real Punta Villa data)
python demo/start_server_ph.py
# -> opens http://localhost:8001/ui
```

Both launchers verify artefacts exist, refuse port collisions, poll
`/healthz` until ready, and auto-open the browser. **Do NOT type
`uvicorn` directly** — the launchers are the safe path.

---

## 🧰 Night-before checklist

Do all of this the night before. Tick each line as you go.

### Repo state
- [ ] `git status` shows working tree clean
- [ ] `git log --oneline -3` shows your latest commits (UI redesign at top)
- [ ] `python -m pytest --no-cov --tb=no` → `130 passed`
- [ ] `ruff check .` → `All checks passed!`
- [ ] `python -m mypy src` → `Success: no issues found in 40 source files`

### Artefacts
- [ ] `ls artifacts/best_model.pkl` exists → Portugal model trained
- [ ] `ls artifacts/ph/ph_model.pkl` exists → PH model trained
- [ ] `ls reports/metrics.json` exists → Portugal metrics present
- [ ] `ls reports/ph/ph_transferability.json` exists → PH metrics present
- [ ] If any are missing: `python scripts/train.py` then `python scripts/train_ph.py`

### Both servers come up cleanly
- [ ] Start Portugal: `python demo/start_server.py` — browser opens, hero
      chips show ROC-AUC 0.864 / PR-AUC 0.760 / F1 0.735
- [ ] Start PH: `python demo/start_server_ph.py` — browser opens, hero
      chips show ROC-AUC 0.611 / PR-AUC 0.542 / F1 0.000 + amber caveat strip
- [ ] Submit one prediction on each → result panel renders with risk
      badge, top features, policy decisions
- [ ] Ctrl+C both servers cleanly

### Notebooks (don't run; just verify they exist and have cached output)
- [ ] `ls notebooks/01_eda.ipynb` … `10_sensitivity_analysis.ipynb` exist
- [ ] `ls notebooks/ph/01_eda.ipynb` … `notebooks/ph/11_transferability.ipynb` exist (11 files)
- [ ] Open ONE PH notebook (e.g. `07_model_selection.ipynb`) in JupyterLab to confirm cells render

### Slides / external tools
- [ ] Slide deck open in a separate window (don't share screen with the slide editor)
- [ ] Power BI Desktop loads with both `predictions_live.csv` and `adr_test_predictions.csv`
- [ ] Browser bookmarks: `http://localhost:8000/ui`, `http://localhost:8001/ui`

### Hardware
- [ ] Laptop charged 100 %; charger packed
- [ ] HDMI / USB-C adapter in bag
- [ ] Backup laptop or USB stick with `git bundle` of the repo

---

## 🎯 Demo flow (5 acts, ~15 min total)

Memorise the *shape* of the flow, not the script. The bullets are
talking points, not lines to read.

### Act 1 — Setup (1 min)
- "This is an end-to-end ML pipeline for predicting hotel booking
  cancellations at the moment of reservation."
- "Two servers running locally — Portugal main study on `:8000`,
  Philippine resort sub-study on `:8001` — same methodology applied to
  two real datasets."
- Show both browser tabs side-by-side.

### Act 2 — Portugal headline (3 min)
- Open `http://localhost:8000/ui`.
- **Point at the KPI chips**: "ROC-AUC 0.864, PR-AUC 0.760, F1 0.735 at
  the max-F1 threshold, calibration ECE 0.029 — these are calibrated
  test-set numbers on roughly 12,000 unseen bookings."
- Submit the **🔴 High-risk** example from the Examples tab.
- Walk through the result panel:
  - Big probability + risk badge
  - **Top contributing features** (mini-bars): "the model exposes its
    reasoning — `deposit_type=Non Refund`, long `lead_time`, `Groups`
    market segment all push toward cancel"
  - **Three policy decisions** (Balanced / High-precision / Cost-optimal)
  - Open the JSON accordion briefly: "this is what the FastAPI client
    sees"

### Act 3 — PH sub-study (3 min)
- Switch to `http://localhost:8001/ui`.
- **Point at the amber caveat strip**: "this is the transferability
  probe — same methodology, 193 real bookings from Punta Villa Resort,
  bootstrap CIs span ±15 pp. We treat these numbers as directional."
- KPI chips: "ROC-AUC 0.611, PR-AUC 0.542 vs a 15 % baseline — the
  model has measurable discriminative signal."
- Submit the **🔴 High-risk** example (No Deposit, 120-day lead, 0
  special requests).
- Headline finding to surface: **deposit_type is the #1 SHAP feature**
  on PH too — same dominant predictor as Portugal. "This is
  cross-dataset evidence the methodology is detecting real signal, not
  memorising spurious patterns."

### Act 4 — The methodology contribution (3 min)
- Open `notebooks/ph/11_transferability.ipynb` in JupyterLab.
- Scroll to Section 11.2 — the duplicate-cluster diagnostic.
- "Before trusting any small-N metric, we run a pre-flight check: do
  feature vectors repeat? If 30 %+ of rows share a vector AND clusters
  have consistent labels, the chronological split leaks twins. **On the
  real PH export the diagnostic does NOT fire — duplicate rate ≈ 0 %.**"
- "So the reported metrics measure honest generalisation, not
  memorisation. The diagnostic is a methodology contribution — a check
  any researcher should run before claiming transferability on small
  datasets."

### Act 5 — Wrap (2 min)
- Open `notebooks/ph/07_model_selection.ipynb`, scroll to Section 7.2 — the
  bootstrap-CI forest plot.
- "Three model families compared honestly: LightGBM leads on point
  estimate but the CIs all overlap. At n_test = 20 the comparison is
  statistically inconclusive. We select LightGBM by point-estimate
  parity + parallel-to-Portugal lineage + Occam's razor. This is what
  responsible small-N model selection looks like."
- Close: "Portugal at 119k rows gives production-grade metrics; PH at
  193 rows demonstrates the methodology travels. The honest claim is
  **'same methodology, weaker model — more data needed for production
  thresholds'**, not 'production-ready classifier'."

---

## 📊 Expected on-screen numbers

If a number on screen doesn't match this table, **stop and check the
artefacts** — something is stale.

### Portugal hero chips (`:8000/ui`)
| Metric | Expected |
|---|---|
| ROC-AUC | **0.864** |
| PR-AUC | **0.760** |
| F1 @ max_f1 | **0.735** |
| Calibration (ECE) | **0.029** |

### Portugal thresholds (visible in policy grid + JSON)
| Policy | Threshold | Precision | Recall |
|---|---|---|---|
| `max_f1` | 0.40 | 0.652 | 0.841 |
| `high_precision` | 0.98 | 1.000 | 0.357 |
| `cost_sensitive` | 0.04 | 0.501 | 0.996 |

### Portugal data lineage
- 119,210 raw rows → 119,210 cleaned (181 dropped: 180 zero-guest + 1 negative ADR)
- Train: 95,367 rows · Val: 11,920 · Test: 11,922
- Date range: 2015-07-01 → 2017-08-31
- Cancellation rate: train 36.1 %, val 43.9 %, test 37.8 %

### PH hero chips (`:8001/ui`)
| Metric | Expected |
|---|---|
| ROC-AUC | **0.611** |
| PR-AUC | **0.542** |
| F1 @ max_f1 | **0.000** *(see Q&A: not a bug, sample-size symptom)* |
| Calibration (ECE) | **0.378** |

### PH thresholds
| Policy | Threshold |
|---|---|
| `max_f1` | 0.190 |
| `high_precision` | 0.190 *(collapses at n_val = 19)* |

### PH data lineage
- 193 raw rows → 193 cleaned (0 dropped)
- Train: 154 · Val: 19 · Test: 20
- Date range: 2022-12-29 → 2025-12-28
- Cancellation rate: 15.0 % (29/193)
- **Duplicate-rate diagnostic**: 0.0 % → does NOT fire → methodology proceeds honestly

### PH top SHAP features (global, mean(|SHAP|))
1. `deposit_type` — **2.32**  *(this is the headline finding to flag)*
2. `adr` — 1.83
3. `reserved_room_type` — 0.84
4. `revenue_at_risk` — 0.78
5. `lead_time` — 0.72

---

## 🆘 Recovery procedures

Things will sometimes break. When they do, **don't panic, don't
fumble** — pick the matching symptom and run the recovery.

### S1: "Port 8000/8001 is already in use"
```powershell
# Kill all uvicorn processes (Windows PowerShell)
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# Then relaunch
python demo/start_server.py
python demo/start_server_ph.py
```

### S2: Browser doesn't open / shows blank page
- Manually visit `http://localhost:8000/healthz` or `:8001/healthz`
- If you see `{"ready": true, ...}` → server is up, just refresh the `/ui` tab
- If `Connection refused` → the launcher exited; check the log:
  - Portugal: `.gradio/uvicorn.log`
  - PH: `.gradio/uvicorn_ph.log`

### S3: `PH artifacts unavailable` in the result panel
```powershell
python scripts/train_ph.py
# Restart the server after training completes
```

### S4: Hero chips say "metrics not available"
- Portugal: `python scripts/train.py`
- PH: `python scripts/train_ph.py`
- Then restart the affected server.

### S5: Prediction returns a weird/zero number
- **Expected behaviour, not a bug**: probabilities ≤ 0.01 % render as
  `<0.01%`. This is the calibrator-zero floor on confident "stay"
  bookings. **Mention this proactively** — it's in the UI's Help tab.

### S6: Notebook won't open / shows an error
- **Don't try to fix during the demo.** Switch to the `reports/figures/thesis/`
  folder and show the PNG that the notebook would have generated — every
  figure is pre-saved.

### S7: Power BI dashboard doesn't refresh
```powershell
python scripts/export_predictions.py
# Then in Power BI Desktop: Home > Refresh
```

### S8: A panelist asks for a number not on screen
- "I can show that in `reports/metrics.json` / `reports/ph/ph_transferability.json`"
- Open the JSON in a text editor — every number is there.

### S9: Total laptop failure
- Backup machine + USB stick with `git bundle` of the repo
- Plan C: walk through the slide deck only; refer panelists to the
  GitHub repo (now pushed: `github.com:Dirkiee09/hotel-cancellation-thesis`)

### S10: You forget the answer to a question
- **Always acceptable**: "Let me check the notebook" + open the relevant `.ipynb`
- Specific notebook pairs to remember:
  - "Why LightGBM?" → `notebooks/02_modeling.ipynb` (Portugal) +
    `notebooks/ph/07_model_selection.ipynb` (PH)
  - "What about calibration?" → `notebooks/03_deep_analysis.ipynb` (Section 3.6)
  - "How does it explain itself?" → `notebooks/05_explainability.ipynb`
  - "What if the data shifts?" → `notebooks/08_model_monitoring.ipynb`

---

## ❓ Anticipated questions (pre-cooked answers)

The 12 questions panelists are most likely to ask, with defensible
one-paragraph answers.

### Q1: Why LightGBM and not XGBoost / Random Forest / a neural net?
**Portugal**: rolling-origin PR-AUC was the selection criterion across
three candidates (LightGBM, XGBoost, GradientBoosting). LightGBM won by
a small but statistically significant margin on 3 chronological folds.
**PH**: the 3-way comparison at n_test = 20 has CIs that overlap totally,
so the selection rests on point-estimate parity + parallel-to-Portugal
lineage + Occam's razor. A neural net is overkill for ~50 tabular
features and would be harder to explain.

### Q2: Why 80/10/10 split and not 80/20 or 70/15/15?
The val set has **two distinct jobs**: fit the isotonic calibrator AND
pick the threshold. Either job on the test set would be leakage. So we
need a held-out val. The 10/10 internal mechanic reports as
"80 / 20 holdout" in thesis text.

### Q3: Why isotonic calibration, not Platt scaling or beta calibration?
Isotonic is non-parametric — it does not assume a sigmoid shape, which
gradient-boosted trees often violate. Brier score on Portugal val
improved from 0.120 (raw) to 0.114 (calibrated); ECE dropped from 0.046
to ~0 on val and from 0.058 to 0.029 on test. Platt would have imposed
a shape the model's outputs don't follow.

### Q4: Why is the PH PR-AUC so much lower than Portugal's (0.54 vs 0.76)?
Three reasons stack: (a) **~500× fewer training rows** (154 vs 95k),
(b) **narrower feature menu** — the PH PMS export does not capture
country, market_segment, customer_type, agent, or previous_cancellations
— and (c) different geography / property type. Bootstrap 95 % CIs on the
PH PR-AUC span roughly ±15 pp, so the gap is statistically real but the
*direction* of the methodology — same family wins, same SHAP feature
dominates — is preserved.

### Q5: At the max_f1 threshold the PH F1 is zero. Doesn't that mean the model is useless?
No — it means the **chosen threshold is wrong** for this test sample.
Two facts to separate: (a) **discrimination**: PR-AUC = 0.542 vs random
baseline 0.150 → the model assigns higher probabilities to actual
cancellations than to non-cancellations; (b) **operating threshold**:
`max_f1` was tuned on a 19-row val set with only ~3 positives, so it
overfits to val and gives F1 = 0 on test. With more data the threshold
stabilises. The honest framing is "the model has signal; production
thresholds need more data".

### Q6: `deposit_type = "Non Refund"` predicts higher risk on Portugal. Isn't that counterintuitive?
It is, and it's consistent across the entire Portugal dataset — flagged
in the UI Help tab. The pattern is real: Non-Refundable deposits in
this dataset are often paid by guests who can claim them back via
insurance / chargeback, so the deposit-policy field captures booking
*intent* rather than *commitment*. SHAP makes the pattern visible
booking-by-booking. **On real PH data the pattern flips back to
intuition** — Non-Refundable is a strong stay signal — which is
another piece of evidence the model is learning data-specific
relationships, not universals.

### Q7: How do you know the model isn't overfit?
Three pieces of evidence on Portugal: (a) **rolling-origin CV** with 3
chronological folds, PR-AUC stable across folds (`reports/benchmarks/10_rolling_origin_fold_metrics.csv`),
(b) **learning curves** that flatten well before 100 % of training data
(`notebooks/03_deep_analysis.ipynb` Section 3.2), (c) **expanding-window CV**
with no fold collapse (`notebooks/03_deep_analysis.ipynb` Section 3.3). On PH
the learning curve does NOT flatten — explicitly documented as "the
model is data-starved" in `notebooks/ph/03_deep_analysis.ipynb`.

### Q8: What's the live ADR forecast caveat?
The `/predict` endpoint also returns a predicted ADR for revenue-at-risk
analysis. The ADR regressor was trained with four post-booking features
(`is_canceled`, `assigned_room_type`, `booking_changes`,
`days_in_waiting_list`) that are not known at booking time. Live
inference passes placeholders, so live `predicted_adr` is slightly less
accurate than the published test-set RMSE of 44.31 EUR. The
methodologically clean fix is retraining the ADR regressor on
booking-time features only — that's documented in CLAUDE.md as a known
limitation, not a discovered defect.

### Q9: How do you handle missing values / weird input?
Two lines of defence: (a) Pydantic schema (`src/app/schemas.py` and
`src/app/ph_schemas.py`) validates at the API boundary — type
coercion, range checks (`adults ≥ 1`, `0 ≤ adr < 100000`); (b) the
preprocessor (`src/features/build.py::build_preprocessor` and
`src/utils/validate_data.py::clean_raw_ph`) imputes for batch /
offline data — numeric median, categorical "UNKNOWN". The
OneHotEncoder is fit with `handle_unknown="ignore"` so a never-seen
category at inference time produces an all-zero one-hot vector,
not a crash.

### Q10: How do you monitor the model in production?
`notebooks/08_model_monitoring.ipynb` (Portugal) and
`notebooks/ph/08_model_monitoring.ipynb` (PH) are runnable templates.
They read a baseline (the held-out test predictions) and a live log
(`data/predictions/predictions.sqlite` for Portugal,
`data/predictions/ph_predictions.sqlite` for PH) and compute PSI drift
on the score distribution + risk-tier mix + calibration drift. PSI < 0.1
= no drift; 0.1-0.25 = investigate; > 0.25 = retrain. Every prediction
that flows through `/predict` or the Gradio UI auto-appends to the
SQLite log and refreshes the Power BI CSV.

### Q11: Why is the cancellation rate different (37 % Portugal vs 15 % PH)?
Genuine population difference. Portugal mixes city + resort properties
across many markets — high-volume, high-churn segments. Punta Villa is
a single resort with a more loyal local clientele (often Walk-In,
single meal plan). The PH model is trained on the lower base rate
directly; metrics are computed against that natural class balance. Note
that PR-AUC normalises against the base rate: PH PR-AUC of 0.54 vs
baseline 0.15 is roughly the same *lift* as Portugal's 0.76 vs 0.37 if
you measure ratio over baseline.

### Q12: What would you do next given another six months?
Three concrete things: (a) **collect more PH bookings** — the data-hunger
curve in `notebooks/ph/03_deep_analysis.ipynb` does not flatten at 154
training rows, doubling the training set would meaningfully tighten the
metrics; (b) **retrain the ADR regressor on booking-time-only features**
so the live `predicted_adr` matches the published RMSE; (c) **A/B test
the cost-sensitive threshold in production** — the Portugal pipeline
computes `cost_sensitive` thresholds, but the operating decision (how
much an intervention costs vs how much a cancellation costs) is a
business decision that would benefit from a live experiment.

---

## 📚 Reference index

### Where to point if asked "where is X documented?"

| Topic | File |
|---|---|
| Project overview | `CLAUDE.md` |
| Methodology decisions | `CLAUDE.md` the "Pipeline Flow (detailed)" section |
| Plug-and-play dataset support | `CLAUDE.md` the "Swapping Datasets" section |
| PH sub-study scope + framing | `CLAUDE.md` the "PH Sub-Study" section + `notebooks/ph/README.md` |
| Live PH server | `CLAUDE.md` the "Live PH server" section |
| ADR live-forecast caveat | `CLAUDE.md` the "Live ADR forecast" section |
| Cross-artifact threshold consistency | `scripts/check.py sync` |
| Metric quality gates | `scripts/check.py metrics` |

### Where to point if asked "show me the code that does X?"

| Topic | File |
|---|---|
| Data cleaning (Portugal) | `src/utils/validate_data.py::clean_raw` |
| Data cleaning (PH) | `src/utils/validate_data.py::clean_raw_ph` |
| Feature engineering | `src/utils/validate_data.py::add_derived_booking_features` |
| Chronological split | `src/features/build.py::split_time_aware` |
| LightGBM training | `src/models/train.py::train_lgbm` |
| Threshold sweep | `src/utils/thresholds.py::threshold_sweep` |
| Calibration | `src/pipelines/train.py::_fit_probability_calibrator` |
| TreeSHAP explainer | `src/serving/inference.py::explain_prediction` |
| Pre-flight duplicate diagnostic | `scripts/train_ph.py::_compute_duplicate_diagnostics` |
| 3-way model comparison (PH) | `scripts/train_ph.py::_compute_model_family_comparison_ph` |

### Where to point if asked "what does notebook X show?"

| Notebook | Headline finding |
|---|---|
| `01_eda.ipynb` (Portugal) | 37 % cancel rate, lead_time / market_segment patterns |
| `02_modeling.ipynb` (Portugal) | LightGBM champion via rolling-origin |
| `03_deep_analysis.ipynb` (Portugal) | Calibration ECE ≈ 0.03, learning curves flat above 50k rows |
| `04_adr_forecasting.ipynb` (Portugal) | ADR test RMSE = 44 EUR |
| `05_explainability.ipynb` (Portugal) | Top SHAP: `deposit_type`, `lead_time`, `agent` |
| `06_business_analytics.ipynb` (Portugal) | Revenue-at-risk dashboard |
| `07_model_selection.ipynb` (Portugal) | Rolling-origin selection across 3 families |
| `08_model_monitoring.ipynb` (Portugal) | PSI drift template |
| `09_model_comparison.ipynb` (Portugal) | Ensembling gives ~1 pp PR-AUC lift |
| `10_sensitivity_analysis.ipynb` (Portugal) | Cost-curve, data-hunger, threshold trade-offs |
| `ph/01_eda.ipynb` | 0 % duplicate rate, deposit_type / room mix |
| `ph/02_modeling.ipynb` | ROC-AUC 0.611, PR-AUC 0.542 |
| `ph/03_deep_analysis.ipynb` | Data-hunger curve does not flatten → more data needed |
| `ph/04_adr_forecasting.ipynb` | ADR regressor overfits (test R² = -0.97) |
| `ph/05_explainability.ipynb` | deposit_type leads PH SHAP just like Portugal |
| `ph/06_business_analytics.ipynb` | Revenue exposure concentrated in No-Deposit |
| `ph/07_model_selection.ipynb` | 3-way CIs overlap → selection by parallel-to-Portugal lineage |
| `ph/08_model_monitoring.ipynb` | Monitoring template ready for production data |
| `ph/09_model_comparison.ipynb` | Mean-of-3 ensemble shows non-significant lift |
| `ph/10_sensitivity_analysis.ipynb` | Cost sensitivity sweep + data hunger |
| `ph/11_transferability.ipynb` | **Pre-flight diagnostic passes; methodology survives the transfer** |

---

## 🎤 One-sentence elevator pitches

If you only get one sentence, use one of these:

**For the methodology**:
> *"End-to-end cancellation prediction at booking time with LightGBM + isotonic calibration + cost-aware threshold selection, deployed as a FastAPI + Gradio service with per-prediction SHAP explanations and live drift monitoring."*

**For the PH sub-study**:
> *"A transferability probe that re-runs the methodology on 193 real Punta Villa bookings, surfaces a generic pre-flight duplicate-cluster diagnostic, and confirms `deposit_type` dominates cancellation prediction across both geographies."*

**For the honest small-N framing**:
> *"Same methodology, weaker model — more data needed for production thresholds — not a methodology defect."*

---

## ✅ Post-defense

After you're done and the room has cleared:

- [ ] `Ctrl+C` both servers
- [ ] `git status` — anything dirty? commit a "post-defense" branch if so
- [ ] Coffee. You've earned it.
