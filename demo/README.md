# Demo Guide

Everything you need to demo the hotel booking cancellation predictor.

## Quick Start (3 commands)

```bash
# 1. Install (skip if already done)
pip install -e . -r requirements.txt

# 2. Train the model (skip if artifacts/ already has files)
python scripts/train.py

# 3. Start server + open browser
python demo/start_server.py
```

The Gradio UI opens automatically at http://localhost:8000/ui

## Demo Scripts

| Script | Purpose |
|--------|---------|
| `start_server.py` | Launch the API + Gradio UI, auto-opens browser |
| `sample_requests.py` | Send 4 contrasting scenarios, print comparison table |
| `quick_train.py` | Fast 10k-row smoke train (~30s) when you just need artifacts |

## Demo Walkthrough

### 1. Gradio UI (http://localhost:8000/ui)

Fill in a booking and click Predict. The result shows:
- **Cancellation probability** (0-100%)
- **Risk tier** (Low / Medium / High)
- **Three policy decisions** (max F1, high precision, cost-sensitive)
- **Top contributing features** (what's driving the prediction)

Try these contrasting bookings:

| Scenario | Key fields | Expected |
|----------|-----------|----------|
| Low risk | Lead time: 2, Deposit: Non Refund, Repeated guest: 1 | ~5%, Low |
| Medium risk | Lead time: 90, Online TA, No deposit, Family (2+2) | ~40-60%, Medium |
| High risk | Lead time: 200, Groups, Previous cancellations: 1 | ~70%+, High |

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
python scripts/check.py all      # artifact integrity + metrics + sync + fairness
python -m pytest tests/ -q       # 91 tests, 89% coverage
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
- **Champion selection**: LightGBM wins via rolling-origin cross-validation, not a lucky split
- **Production-ready**: FastAPI, health checks, thread-safe caching, Docker, 91 tests
