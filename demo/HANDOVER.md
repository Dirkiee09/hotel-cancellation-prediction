# Demo Handover — ICBIM 2026, Bangkok

For the presenters. Everything needed to get the live demo running on your own
laptop, verify it works, and know what to say if something goes wrong.

Paper: *A Strategic Business Intelligence Approach to Predicting Hotel Booking
Cancellations* (BM4117).

---

## 1. Setup (once, ~15 minutes)

You need **Python 3.11+** and **git**. Windows commands shown; on macOS/Linux use
`source .venv/bin/activate` instead of the `activate` line.

```bash
git clone https://github.com/Dirkiee09/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction

python -m venv .venv
.venv\Scripts\activate

pip install -e . -r requirements.txt
```

### Then get a trained model

The repository does **not** contain the trained model — `artifacts/` is
git-ignored because the files are large. You have two options:

**Option A — train it yourself (recommended).** Reproducible: every seed is fixed
at 42, so you get the same numbers as the paper. Allow up to ~10 minutes; it
fits several model families across three rolling-origin folds before calibrating.

```bash
python scripts/training/train.py
```

**Option B — copy the `artifacts/` and `reports/` folders** from whoever already
has them, into the project root. Faster, no training needed.

> If you only want to check the UI quickly and don't care about accuracy:
> `python demo/quick_train.py` trains on 10k rows in under a minute. **Do not
> present with this model** — the numbers will not match the paper.

---

## 2. Run it

```bash
python demo/start_server.py
```

A browser opens at <http://localhost:8000/ui>. If it doesn't, open that URL
yourself.

To stop: `Ctrl+C` in the terminal.

---

## 3. The 90-second demo script

Three buttons under the form load prepared bookings. **Use them — do not type
during the talk.**

| Step | Action | What to say |
|------|--------|-------------|
| 1 | Click **① Non-refundable, 120d lead** → **Predict** | "Same hotel, same rate. Non-refundable deposit, booked four months out. The model puts this above 99 percent, flags it High, and the drivers show deposit type dominating — which is our paper's central finding." |
| 2 | Point at the risk meter | "The green/amber/red bands are the 0.40 and 0.70 tier boundaries from the paper. The marker is where this booking sits." |
| 3 | Point at **Revenue at risk** | "€900 of exposure on this one booking. That's ADR times nights — Equation 1." |
| 4 | Click **③ Direct repeat guest** → **Predict** | "Direct channel, repeat guest, parking and special requests. Under 1 percent — the other end of the same calibrator, so you see the matching note. No contact needed; we save the staff effort." |
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
model this demo serves. The banner states which one is on screen.

**"Is LightGBM really the best model?"**
Not decisively, and we say so. At matched capacity XGBoost edges it on PR-AUC by
0.0036 (p = 0.010) while LightGBM wins F1 by 0.0087 (p = 0.002); ROC-AUC is a
tie (p = 0.183). LightGBM was selected in advance by a rolling-origin criterion,
has the higher recall our cost structure rewards, and trains in 2.7 s versus
7.5 s. We do not claim it dominates.

**"Where does 31.2% come from?"**
Cost-sensitive threshold versus contacting every guest. €15 per unnecessary
contact against booking-specific lost revenue (ADR × nights, minus one resold
night). €84,045 → €57,815 on the de-duplicated test set.

The banner reads this from whichever stream it is showing and labels it, so on a
fresh clone without `reports/dedup/` it will say **36.1% · per-record**
(€111,240 → €71,136) instead. Both are real; the paper reports 31.2%. Check which
one is on screen before you quote a number.

**"Does the probability of exactly 100% appear in your API?"**
Yes — the stored value is untouched; only the display is bounded. We did not want
to alter the model's output to make a demo look better.

---

## 5. If it breaks on the day

| Symptom | Fix |
|---------|-----|
| `ERROR: No trained model found` | `python scripts/training/train.py` (or copy `artifacts/`) |
| `ERROR: Port 8000 is already in use` | `python demo/start_server.py --port 9000` |
| Browser doesn't open | Go to <http://localhost:8000/ui> manually |
| Page loads but Predict does nothing | Check the terminal for errors; restart the server |
| Banner shows per-record numbers | Expected on a fresh clone — see Q2 above, or run `python scripts/training/dedup_sensitivity.py` |

**The demo does not need internet.** Everything is served from localhost. Conference
wifi cannot break it.

**Record a backup video.** Before you fly, run through section 3 with screen
capture and keep the file on the presenting laptop. If anything goes wrong on
stage, play the video and keep talking. This is standard practice and costs you
ten minutes.

---

## 6. Before you present

- [ ] Run the full demo on the **actual presenting laptop**, not just your own
- [ ] Clear the activity log so old rows don't show: delete `data/predictions/predictions.csv`
- [ ] Close other apps using port 8000
- [ ] Set the laptop to **never sleep** during the session
- [ ] Record the backup video
- [ ] Check the projector: the UI is dark-themed, confirm it's readable from the back of the room
