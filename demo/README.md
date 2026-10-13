# `demo/` — running and presenting the predictor

Three documents live here. Pick the one that matches what you are doing.

| Document | For | What it covers |
|---|---|---|
| [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md) | **Presenters** | The talk: timing checkpoints, a 10-minute cut list, the two-presenter split, panel questions, and the phrases that would overclaim |
| [`HANDOVER.md`](HANDOVER.md) | **Presenters** | The live demo: setup, the 90-second script, the questions you will be asked, and a failure runbook |
| This file | **Anyone running it** | Commands, the walkthrough, and what each script does |

If you are presenting, read **section 4 of `HANDOVER.md`** before anything else.
It covers the three questions most likely to catch you out.

## Quick start

**Windows:** double-click **`START DEMO.bat`** in the project root. It finds the
virtual environment itself, so it cannot launch on the wrong Python — the most
common setup failure. `demo\run_demo.bat` forwards to the same place.

**Any platform:**

```bash
# 1. Install (skip if already done)
pip install -e . -r requirements.txt

# 2. Train the model (skip if artifacts/ already has files)
python scripts/training/train.py

# 3. Start server + open browser
python demo/start_server.py
```

The Gradio UI opens at http://localhost:8000/ui

> **No trained model?** `artifacts/` is committed, so a clone or a "Download
> ZIP" from GitHub already has one — step 2 is only needed if you want to
> rebuild it yourself, or you deleted `artifacts/`. If the app says the model
> is missing after a ZIP download, the archive was only partly extracted.

## Demo Scripts

| Script | Purpose |
|--------|---------|
| `run_demo.bat` | Windows one-click launcher — resolves the venv, then runs `start_server.py` |
| `start_server.py` | Launch the API + Gradio UI, auto-opens browser |
| `sample_requests.py` | Send 4 contrasting scenarios, print comparison table |
| `quick_train.py` | Fast 10k-row smoke train (~30s) when you just need artifacts |

## Demo Walkthrough

### 1. Gradio UI (http://localhost:8000/ui)

Load a booking with one of the three buttons under the form, then click Predict.
The result shows:
- **Cancellation probability** (0-100%)
- **Risk tier** (Low / Medium / High)
- **Three policy decisions** (max F1, high precision, cost-sensitive)
- **Top contributing features** (what's driving the prediction)

| Button | Key fields | Expected |
|--------|-----------|----------|
| ① Non-refundable, 120d lead | City Hotel, Online TA, Non Refund deposit | above 99%, High |
| ② Online agent, 60d lead | City Hotel, Online TA, No Deposit, one special request | 60%, Medium |
| ③ Direct repeat guest | Resort Hotel, Direct, repeat guest, special requests, no parking request | about 4%, Low |

The arrival date defaults to 30 days from today, so the month feature changes with
the day you present. The figures above hold for arrivals in November and December.
In this dataset a non-refundable deposit marks the highest-risk bookings, not the
lowest; see `HANDOVER.md` §5.

### 2. API Comparison Table

While the server is running, open a second terminal:

```bash
python demo/sample_requests.py
```

Prints a side-by-side comparison of 4 scenarios with probabilities, risk tiers,
policy labels, and top SHAP drivers.

### 3. API Docs (http://localhost:8000/docs)

Swagger UI with:
- `POST /predict` - the prediction endpoint (try it live)
- `GET /model-info` - model metadata, thresholds, feature count
- `GET /healthz` - readiness check

### 4. Quality Gates

```bash
python scripts/utils/check.py all      # artifact integrity + metrics + sync + fairness
python -m pytest -q                    # 153 tests, coverage gate >=80%
```

### 5. Notebooks

Open any notebook in `notebooks/` — all have cached outputs, no need to run cells:
- `01_eda.ipynb` - data exploration
- `05_explainability.ipynb` - SHAP feature importance
- `06_business_analytics.ipynb` - revenue dashboard
- `10_sensitivity_analysis.ipynb` - cost/threshold sensitivity

## Talking Points

- **No data leakage**: only booking-time features, no post-booking info
- **Three thresholds**: different cutoffs for different business needs
- **Calibrated probabilities**: isotonic calibration makes probabilities reliable
- **Champion selection**: LightGBM was chosen in advance by rolling-origin validation, before the test set was opened; XGBoost is level with it on the test set
- **Built like a deployable service, presented as a prototype**: FastAPI, health checks, thread-safe caching, Docker, 153 tests. The paper calls it a prototype, so do not call it production-ready
