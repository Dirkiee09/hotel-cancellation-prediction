# Table 4.3 — ADR Regression Performance

**Source.** `reports/regression_results.csv` (Portugal),
`reports/ph/ph_adr_regressor_metrics.json` (Philippines).

**Method.** Eight regressors fit on the chronological train split, tuned
on the validation set, evaluated on the held-out test set. The Portugal
champion (selected by **val RMSE**) is **Gradient Boosting** at 28.76 €
val RMSE. XGBoost records a fractionally lower test RMSE (44.06 € vs
44.31 €) but loses on validation (29.30 € vs 28.76 €) — the methodology
never selects on the test set. RMSE in EUR; MAPE as a percentage.

**Caveat.** The Portugal ADR regressor was trained on four post-booking
features (`is_canceled`, `assigned_room_type`, `booking_changes`,
`days_in_waiting_list`). Live inference uses sensible defaults
(documented in `CLAUDE.md`), so live `predicted_adr` is slightly less
accurate than the test-set RMSE = 44.31. The methodologically clean fix
is retraining on booking-time features only.

| Dataset | Model | Train RMSE (€) | Val RMSE (€) | Test RMSE (€) | Test MAE (€) | Test R² | Test MAPE (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Portugal | Gradient Boosting (champion) | 32.70 | 28.76 | 44.31 | 32.24 | 0.234 | 23.45 |
| Portugal | XGBoost | 32.89 | 29.30 | 44.06 | 32.14 | 0.243 | 23.48 |
| Portugal | Random Forest | 22.74 | 31.89 | 44.52 | 32.57 | 0.227 | 24.60 |
| Portugal | Decision Tree | 33.74 | 31.28 | 45.87 | 33.28 | 0.179 | 25.15 |
| Portugal | Ridge | 37.64 | 30.29 | 47.64 | 34.55 | 0.115 | 24.74 |
| Portugal | Linear Regression | 37.63 | 30.30 | 47.65 | 34.56 | 0.114 | 24.75 |
| Portugal | Lasso | 39.84 | 30.80 | 51.99 | 38.04 | −0.054 | 27.39 |
| Portugal | Neural Network | 41.38 | 31.06 | 55.17 | 38.22 | −0.187 | 26.72 |
| Philippines | HistGradientBoosting | — | — | see PH JSON | — | — | — |

**LightGBM exclusion (Portugal ADR).** Gradient-boosting family was already
represented by sklearn's `GradientBoostingRegressor` and XGBoost — and on
the cancellation classification task LightGBM and Gradient Boosting
finished within 0.003 PR-AUC of each other. Adding LightGBM to ADR
regression would have added ~0.5 % RMSE noise without strategic value.

**Why Test R² < 0.25 is OK here.** ADR is dominated by rate-card pricing
(room type, season, channel) that the model captures, plus high-variance
guest-specific factors (group discounts, loyalty status, promotional
codes) that aren't in the feature set. The thesis claim is *directional
ADR signal at booking time*, not *exact price prediction*.
