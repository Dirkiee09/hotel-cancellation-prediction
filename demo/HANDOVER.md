# Demo Handover — ICBIM 2026, Bangkok

For the presenters. Everything needed to get the live demo running on your own
laptop, verify it works, and know what to say if something goes wrong.

Paper: *A Strategic Business Intelligence Approach to Predicting Hotel Booking
Cancellations* (BM4117).

---

## 1. Setup (once, ~10 minutes)

You need **Python 3.11 or 3.12**. Check with `python --version`; newer versions
have no matching packages and the install fails. Windows commands shown; on
macOS/Linux use `source .venv/bin/activate` instead of the `activate` line.

Get the code either way:

- **Download ZIP** from https://github.com/Dirkiee09/hotel-cancellation-prediction
  (green **Code** button), extract it somewhere simple like `C:\hotel-demo`; or
- `git clone https://github.com/Dirkiee09/hotel-cancellation-prediction.git`

Then, in that folder:

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -e . -r requirements.txt
```

**The trained model is included** in `artifacts/`, so there is nothing to train.
If you want to rebuild it anyway, `python scripts/training/train.py` takes about
ten minutes and reproduces the committed model exactly on the same Python and
library versions.

> `python demo/quick_train.py` trains on 10k rows in under a minute. **Do not
> present with that model** — the numbers will not match the paper.

---

## 2. Run it

**Double-click `START DEMO.bat`** in the project folder. That is the whole thing.
It finds the virtual environment itself, so there is no way to start it on the
wrong Python — the most common setup failure.

If you prefer a terminal:

```bash
python demo/start_server.py
```

Either way a browser opens at <http://localhost:8000/ui>. If it doesn't, open
that URL yourself.

**To stop: click the server window and press `Ctrl+C`.** Do not just close the
window — the server keeps running, port 8000 stays occupied, and the next launch
fails with `Port 8000 is already in use`. If that happens, run this in
PowerShell:

```bash
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn' } | Stop-Process -Force
```

---

## 3. The 90-second demo script

Three buttons under the form load prepared bookings. **Use them — do not type
during the talk.**

| Step | Action | What to say |
|------|--------|-------------|
| 1 | Click **① Non-refundable, 120d lead** → **Predict** | "Same hotel, same rate. Non-refundable deposit, booked four months out. The model puts this above 99 percent and flags it High. In this dataset non-refundable bookings almost all cancel, which looks backwards; it partly reflects how group bookings were recorded, and the paper says so." |
| 2 | Point at the risk meter | "The green/amber/red bands are the 0.40 and 0.70 tier boundaries from the paper. The marker is where this booking sits." |
| 3 | Point at **Revenue at risk** | "€900 of exposure on this one booking. That's ADR times nights — Equation 1." |
| 4 | Click **③ Direct repeat guest** → **Predict** | "Direct channel, repeat guest, special requests. Under 1 percent — the other end of the same calibrator, so you see the matching note. No contact needed; we save the staff effort." |
| 5 | Open **Model Capabilities & Benchmarks** tab | "The full model ladder with bootstrap confidence intervals, and the SHAP attribution." |

---

## 4. Questions you should expect

**"It says >99% — is the model claiming certainty?"**
No. Isotonic calibration is a step function, and its end bins collapse to exactly
0 or 1 when every validation booking in that bin shared the same outcome. We
report it as a bound rather than as certainty, and the card on screen says so.

Quote the figures for the stream you are showing — the demo serves the per-record
model, so those are the ones that match the screen:

| | Served here (per-record) | Paper's primary (de-duplicated) |
|---|---|---|
| Bookings at an exact 0 or 1 | 16.0% | 13.2% |
| Expected calibration error | 0.031 | 0.018 |
| Observed cancel rate, Low / Med / High | 0.087 / 0.493 / 0.746 | 0.140 / 0.573 / 0.789 |

Either way the ordering is monotone across the three tiers, which is the point:
the tiers mean what they claim. If you are asked which number the paper reports,
it is the right-hand column.

**"Your slide says 0.848 but the screen says 0.863."**
Two evaluation streams, both in the paper. 0.848 / 0.703 is the de-duplicated
primary estimate. 0.863 / 0.759 is the per-record robustness view, which is the
model this demo serves. The banner shows the primary figures and its caption
names the served model's.

**"Is LightGBM really the best model?"**
Not decisively, and we say so. At matched capacity XGBoost edges it on PR-AUC by
0.0036 (p = 0.010) while LightGBM wins F1 by 0.0087 (p = 0.002); ROC-AUC is a
tie (p = 0.183). LightGBM was selected in advance by a rolling-origin criterion,
has the higher recall our cost structure rewards, and trains in 2.7 s versus
7.5 s. We do not claim it dominates.

**"Why present LightGBM when XGBoost scored higher on your own table?"**
This is the sharpest question you will get. The answer is a strength, not an
excuse: the champion was fixed *before* the test set was touched, by the
rolling-origin criterion — LightGBM 0.869 mean PR-AUC against XGBoost 0.868 and
Gradient Boosting 0.867. Switching to XGBoost because it won on the held-out
test is selecting on the test set, which is precisely the error the whole design
exists to prevent. Across-fold standard deviation is about 0.032 for every
candidate, so the ranking is stable in order but not in magnitude, and the
paired bootstrap uses XGBoost — the higher scorer — as its reference rather than
our own champion.

The paper does not say which stream those rolling-origin figures come from: they
are per-record. On the primary stream the order is the same: 0.714, 0.711, 0.708.

**"Does boosting actually beat logistic regression?"**
Yes, modestly. On the per-record test set, with both models calibrated the same
way, LightGBM's PR-AUC is 0.759 against 0.740: a gap of 0.019, p < 0.001 under
the paired bootstrap. Uncalibrated, the gap is 0.021, p < 0.001. On the
de-duplicated stream it is 0.026 (0.703 against 0.677), p < 0.001. The paper's own figure is XGBoost
against logistic regression, 0.023, p < 0.001. Most of the booking-time signal is
linear, the boosted gain is real but small, and the business result comes from
calibration plus cost-sensitive thresholding rather than from raw discrimination.

If someone has seen an older copy of the repository: until October 2026 the
pipeline compared the calibrated LightGBM with an *uncalibrated* logistic
regression and recorded H2 as "not supported" (+0.005, p = 0.177). Calibration
lowers PR-AUC for any model, so that comparison was lopsided. It is fixed, and
`reports/metrics.json` now records both the calibrated and uncalibrated pairs.

**"Where does 31.2% come from?"**
Cost-sensitive threshold versus contacting every guest. €15 per unnecessary
contact against booking-specific lost revenue (ADR × nights, minus one resold
night). €84,045 → €57,815 on the de-duplicated test set. The served per-record
model's figure is 36.1%.

Two assumptions sit underneath it, and a careful panellist may find them: a
contact costs €15 only when the guest was not going to cancel, and contacting a
guest who was going to cancel always saves the booking. If you are challenged,
give the euro figure, which barely moves:

| How a contact is priced | Saving vs contacting everyone |
|---|---|
| The paper's model | €26,230 (31.2%) |
| €15 for every contact, including real cancellers | €26,485 (20.2%) |
| …and a contact saves only half of would-be cancellations | €28,880 (2.7%) |

The saving comes from not contacting the 2,085 safest bookings while missing 17
cancellations, and that holds under every pricing. The percentage depends on what
it is divided by. Whether contacts actually prevent cancellations has not been
measured; the paper's conclusion calls for a field test for exactly this reason.

**"Does the probability of exactly 100% appear in your API?"**
Yes — the stored value is untouched; only the display is bounded. We did not want
to alter the model's output to make a demo look better.

---

## 5. Points the printed paper states loosely

The paper is final, so these are handled out loud. Each answer below is checked
against the committed files.

**"Is `required_car_parking_spaces` known when the booking is made?"**
We cannot tell from the data. Every one of the 7,416 bookings that requested
parking arrived, a 0.0% cancellation rate, which suggests the field may be filled
in at check-in. We retrained the primary stream without it: ROC-AUC 0.848 → 0.828,
PR-AUC 0.703 → 0.674, and the cost cut 31.2% → 25.4%. The method's conclusion
holds; the headline accuracy partly depends on that field. Say this plainly if
asked; do not defend the field as booking-time.

**"Your deck says deposit terms are a lever. Should hotels drop non-refundable rates?"**
No. Non-refundable bookings cancel 99.4% of the time in this data, which partly
reflects how block bookings were recorded rather than guest behaviour; the paper
says so.
Retrained without deposit type, the model loses under 0.001 of PR-AUC, so it does
not depend on it.

**"How far ahead is deposit type?"**
Narrowly. Summing SHAP over each feature's one-hot columns: deposit type 1.116,
country 1.085, agent 0.926. Those come from the per-record model. Do not say "by a
wide margin".

**"Did the Philippine pilot work?"**
The procedure ran end to end on 193 local bookings, using a schema adapter for
that property's export and an 18-feature set (no country, agent or market
segment). The paper's "without modification" means the modelling stages. With 20
test bookings it could not predict: it flagged none of the 9 cancellations, and
simpler models scored higher. That is why the paper estimates 1,500 to 3,000
bookings are needed before a local study means anything.

**"The paper says Python 3.11."**
The pipeline is tested on 3.11 in CI on every push. The committed results were
produced on Windows with Python 3.12.10, and a retrain there reproduces the model
byte for byte. CI retrains on Linux with 3.11 and posts how far each figure moves.

**"How well does it handle last-minute bookings?"**
Weakly, and the paper does not report it. Bookings made 0–3 days ahead are 6.7% of
the per-record test set and cancel 9.7% of the time. PR-AUC there is 0.28; the
cost-sensitive threshold catches 81% of them at 18% precision.

**Smaller precision points**, if anyone reads that closely:

| The paper says | The files show |
|---|---|
| 119,209 after cleaning, then 87,229 after de-duplicating | The pipeline de-duplicates the raw file first (87,396) and then cleans (87,229). Cleaning first gives 87,222. |
| XGBoost and LightGBM matched at depth 7, subsample 0.8 | LightGBM's row subsampling was inactive and its trees stop at 31 leaves, where XGBoost at depth 7 can grow 128. LightGBM was the smaller model. |
| A 60-day test window, ~130 high-risk bookings a week | 2 July to 31 August is 61 days: 128 a week, 18 a day. |
| Random Forest underperforms logistic regression | On PR-AUC and F1. On ROC-AUC Random Forest is higher (0.848 vs 0.841). |
| Withholding deposit type: 0.703 → 0.702 | Correct; the ablated run re-selected XGBoost. |

---

## 6. If it breaks on the day

| Symptom | Fix |
|---------|-----|
| `TRAINED MODEL MISSING` / `No trained model found` | The download was only partly extracted. Extract the whole zip again and run from the extracted folder |
| `ERROR: Port 8000 is already in use` | `python demo/start_server.py --port 9000` |
| Browser doesn't open | Go to <http://localhost:8000/ui> manually |
| Page loads but Predict does nothing | Check the terminal for errors; restart the server |

**The demo does not need internet.** Everything is served from localhost. Conference
wifi cannot break it.

**Record a backup video.** Before you fly, run through section 3 with screen
capture and keep the file on the presenting laptop. If anything goes wrong on
stage, play the video and keep talking. This is standard practice and costs you
ten minutes.

---

## 7. Before you present

- [ ] Run the full demo on the **actual presenting laptop**, not just your own
- [ ] Clear the activity log so old rows don't show: delete `data/predictions/predictions.csv`
- [ ] Close other apps using port 8000
- [ ] Set the laptop to **never sleep** during the session
- [ ] Record the backup video
- [ ] Check the projector: the UI is dark-themed, confirm it's readable from the back of the room
