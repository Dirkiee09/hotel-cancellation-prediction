# Table 4.4 — Top 5 SHAP Feature Importances (Both Datasets, Side-by-Side)

**Source.** Portugal: thesis Section 4.3.4 Table 4.7, derived from
`reports/thesis/shap_feature_importance.csv` (encoded-column SHAP
aggregated back to raw feature names — one-hot dummies summed). PH:
`reports/ph/shap_feature_importance.csv` and `reports/ph/shap_analysis.json`.

**Method.** `shap.TreeExplainer` on the calibrated LightGBM champion of
each pipeline. Encoded-feature SHAP values are summed back to the
originating raw feature so the table reports the 33 (Portugal) or 18 (PH)
business-meaningful inputs, not the post-encoding dummies.

**Top 5 per dataset.**

| Rank | Portugal: Top Raw Feature | Aggregated mean(\|SHAP\|) | Philippines: Top Raw Feature | Aggregated mean(\|SHAP\|) |
|---|---|---:|---|---:|
| 1 | **`deposit_type`** | 1.150 | **`deposit_type`** | 2.323 |
| 2 | `country` | 1.095 | `adr` | 1.829 |
| 3 | `agent` | 0.911 | `reserved_room_type` | 0.844 |
| 4 | `required_car_parking_spaces` | 0.746 | `revenue_at_risk` | 0.783 |
| 5 | `total_of_special_requests` | 0.576 | `lead_time` | 0.718 |

**Cross-dataset finding (H5 supported).** `deposit_type` ranks #1 in both
datasets — strong evidence that the methodology detects a consistent
cancellation driver across geographies, properties, and dataset sizes.

**Hypothesis-vs-data tension (H3 partially supported).** The thesis
pre-registered "lead_time has highest SHAP, then deposit_type, then
previous_cancellations". The Portugal data shows the actual order is
**deposit_type > country > agent > parking > special_requests > market_segment > lead_time (rank 7) > … > previous_cancellations (rank 10)**.
All three pre-registered features appear in the top-10, but the rank
order differs from the hypothesis — flagged in Section 4.3.4 as
"partially supported".

**Plain-English interpretation.** Two of the top three Portugal drivers
(`country`, `agent`) are **booking-source identity**, not customer
behaviour. The model is learning *which travel agents and which countries
of origin reliably honour their bookings* — a finding consistent with
hotel-industry intuition: cancellation rates are dominated by channel
reliability, not by guest fickleness. This is a defensible thesis story:
the model isn't a black box; it codifies what experienced revenue
managers already know.
