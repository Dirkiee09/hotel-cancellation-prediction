# Notebooks

All notebooks **load pre-computed artifacts** — they do not retrain models.
Run `make full-pipeline` before opening any notebook for the first time.

---

## Run Order & Purpose

| # | Notebook | Purpose | Required artifacts |
|---|----------|---------|-------------------|
| 01 | `01_eda.ipynb` | Exploratory data analysis — distributions, correlations, cancellation patterns | `data/hotel_bookings.csv` |
| 02 | `02_modeling.ipynb` | Rolling-origin model selection, champion/challenger comparison, PR-AUC curves | `artifacts/best_model.pkl`, `reports/metrics.json` |
| 03 | `03_deep_analysis.ipynb` | Calibration, SHAP global/local, learning curves, ablation, baselines | `artifacts/`, `reports/thesis/` |
| 04 | `04_adr_forecasting.ipynb` | ADR time-series decomposition + regression (revenue forecasting) | `artifacts/adr_regressor*.pkl` |
| 05 | `05_explainability.ipynb` | SHAP beeswarm, segment insights, cost analysis, segment-specific thresholds | `artifacts/`, `reports/segment_metrics.csv` |
| 06 | `06_business_analytics.ipynb` | Revenue management dashboard — overbooking buffer, ROI, segment P&L | `reports/test_predictions_for_powerbi.csv` |
| 07 | `07_model_selection.ipynb` | Model family comparison, dumbbell charts, hyperparameter fairness | `reports/benchmarks/`, `reports/thesis/` |
| 08 | `08_model_monitoring.ipynb` | Drift detection baseline — score distribution, feature shift vs production | `reports/test_predictions_for_powerbi.csv` |
| 09 | `09_model_comparison.ipynb` | Champion vs challengers — complexity ladder, significance tests, cost tradeoffs, why boosting wins | `reports/benchmarks/`, `reports/thesis/` |
| 10 | `10_sensitivity_analysis.ipynb` | Cost sensitivity, dataset size impact, threshold policy comparison | `artifacts/`, `reports/` |

---

## Quick Start

```bash
# 1. Generate all artifacts and reports
make full-pipeline
# Windows without make: python scripts/train.py && python scripts/train.py --verify-only

# 2. Open a notebook
jupyter lab notebooks/01_eda.ipynb
```

---

## Conventions

- Every section: **markdown header → code cell → markdown insight**
- Plots: always call `setup_plotting()` first (Space Grotesk, publication style)
- Save figures: `save_thesis_figure(fig, fig_no, stem, FIG_DIR)` → `reports/figures/thesis/`
- Tables: `df.style.format(...).set_caption(...)` — never `print(df)`
- No model retraining inside notebooks — load from `artifacts/` only

---

## Helper Functions

All reusable plot/analysis helpers live in `src/eval/notebook_utils.py`.
Import them with:

```python
from src.eval.notebook_utils import (
    load_main_context,        # loads metrics, thresholds, model metadata
    load_analysis_context,    # loads SHAP, CI, ablation results
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_segment_heatmap,
    setup_plotting,
    save_thesis_figure,
)
```
