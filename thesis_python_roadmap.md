# Thesis-Defensible Python Roadmap
## Predicting Hotel Booking Cancellations: A Machine Learning Approach

---

## 0 — Critical Leakage Traps (Read First)

Your single biggest thesis-defense risk is **data leakage**. The dataset contains columns that encode the *outcome itself*. If any model sees these, your AUC will be artificially inflated and your panel will catch it.

| Column | Why It Leaks | Action |
|---|---|---|
| `reservation_status` | Literally says "Canceled" / "Check-Out" / "No-Show" — this *is* the label. | **DROP immediately** |
| `reservation_status_date` | Date the cancellation or check-out was recorded — post-outcome information. | **DROP immediately** |

Drop these two columns as the very first operation after loading the CSV, before any EDA. Document this decision explicitly in your methodology chapter.

Additionally, there is a subtle issue: `booking_changes` records amendments up to check-in. Some of those changes may have occurred *after* a cancellation decision was already made (or after the guest's behavior changed). Your thesis already scopes to "information available at or near the time of reservation," so either treat `booking_changes` as a known limitation or drop it. If you keep it, state the assumption explicitly.

---

## 1 — Data Loading & Cleaning (Sense Phase)

```python
import pandas as pd
import numpy as np

df = pd.read_csv("hotel_bookings.csv")

# ── STEP 1: Kill leakage columns ──────────────────────────
df.drop(columns=["reservation_status", "reservation_status_date"], inplace=True)

# ── STEP 2: Fix NULL-string encoding ──────────────────────
# agent and company are stored as "NULL" strings, not real NaN
df["agent"] = df["agent"].replace("NULL", np.nan).astype(float)
df["company"] = df["company"].replace("NULL", np.nan).astype(float)

# ── STEP 3: Handle missing values ─────────────────────────
# children: 4 nulls → fill with 0 (most common, makes domain sense)
df["children"] = df["children"].fillna(0).astype(int)

# country: 488 nulls → fill with "Unknown"
df["country"] = df["country"].fillna("Unknown")

# agent: 16,340 nulls → flag with 0 (meaning "no agent / direct")
df["agent"] = df["agent"].fillna(0).astype(int)

# company: 112,593 nulls (94%) → too sparse to use as raw ID
# Create a binary: had_company = 1 if company is not null
df["had_company"] = df["company"].notna().astype(int)
df.drop(columns=["company"], inplace=True)

# ── STEP 4: Sanity checks ─────────────────────────────────
# Remove ADR outliers (negative ADR is a data error; extreme ADR is noise)
df = df[df["adr"] >= 0]
df = df[df["adr"] < 1000]  # or use a percentile cutoff, e.g., 99.5th

# Remove zero-guest bookings (0 adults + 0 children + 0 babies)
df = df[~((df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0))]
```

**Justify every decision in your thesis.** Each cleaning step should be a sentence in your methodology. Panelists love seeing that you didn't just run code — you *thought* about what each column means.

---

## 2 — Feature Engineering (Sense Phase)

These features align directly with the variables described in your Chapter III methodology table and conceptual framework.

```python
# ── Derived features from your thesis scope ───────────────
df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df["adr_per_person"] = df["adr"] / df["total_guests"].replace(0, 1)
df["is_weekend_heavy"] = (
    df["stays_in_weekend_nights"] > df["stays_in_week_nights"]
).astype(int)

# Revenue at risk proxy (used later for cost-sensitive threshold)
df["revenue_at_risk"] = df["adr"] * df["total_stay"]

# ── Construct an arrival date for time-based splitting ────
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
df["arrival_month_num"] = df["arrival_date_month"].map(month_map)
df["arrival_date"] = pd.to_datetime(
    df["arrival_date_year"].astype(str) + "-"
    + df["arrival_month_num"].astype(str).str.zfill(2) + "-"
    + df["arrival_date_day_of_month"].astype(str).str.zfill(2)
)

# ── Cyclical encoding of month (better than one-hot for trees) ─
df["month_sin"] = np.sin(2 * np.pi * df["arrival_month_num"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["arrival_month_num"] / 12)

# ── Late-window flag (your thesis focus: ≤ 3 days lead time) ──
df["is_late_window"] = (df["lead_time"] <= 3).astype(int)
```

---

## 3 — Time-Aware Train / Validation / Test Split (CRITICAL)

Your thesis explicitly promises time-aware splits. This is **non-negotiable** for methodological soundness. A random 80/20 split allows future data to leak into training, which inflates metrics and is indefensible.

```python
# Dataset spans: July 2015 → August 2017
# Strategy: temporal cutoffs based on arrival_date

train = df[df["arrival_date"] < "2017-01-01"]          # ~Jul 2015 – Dec 2016
val   = df[(df["arrival_date"] >= "2017-01-01")
         & (df["arrival_date"] < "2017-05-01")]         # Jan – Apr 2017
test  = df[df["arrival_date"] >= "2017-05-01"]          # May – Aug 2017

print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
print(f"Train cancel rate: {train['is_canceled'].mean():.3f}")
print(f"Val cancel rate:   {val['is_canceled'].mean():.3f}")
print(f"Test cancel rate:  {test['is_canceled'].mean():.3f}")
```

**Why this matters for your defense:** If a panelist asks "how do you prevent leakage?" you answer: "we used a strict temporal split — the model never sees any booking with an arrival date after the training cutoff. This simulates real deployment where you only have historical data."

---

## 4 — Encoding Categoricals

```python
from sklearn.preprocessing import LabelEncoder

# Identify columns to encode
cat_cols = [
    "hotel", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type",
    "assigned_room_type", "deposit_type", "customer_type",
]

# For tree-based models: label encoding is sufficient and preferred
# (LightGBM/XGBoost handle label-encoded categoricals natively)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(train[col].astype(str))  # fit only on train
    for split in [train, val, test]:
        split[col + "_enc"] = split[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
        )
    label_encoders[col] = le

# For logistic regression: use one-hot on low-cardinality cols only
# (country has ~170 levels — either group into top-N + "Other" or drop for LR)
```

---

## 5 — Define Feature Sets

```python
# Features your thesis lists as independent variables
FEATURES = [
    "lead_time", "adr", "total_stay", "total_guests", "adr_per_person",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests",
    "arrival_date_week_number", "month_sin", "month_cos",
    "had_company", "is_weekend_heavy",
    # Encoded categoricals
    "hotel_enc", "meal_enc", "market_segment_enc",
    "distribution_channel_enc", "reserved_room_type_enc",
    "assigned_room_type_enc", "deposit_type_enc",
    "customer_type_enc",
]
# Optionally include country_enc for tree models (high cardinality is fine)

TARGET = "is_canceled"

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]
```

---

## 6 — Model Training & Evaluation (Seize Phase)

### 6a. Logistic Regression Baseline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, brier_score_loss, classification_report,
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train.fillna(0))
X_val_sc = scaler.transform(X_val.fillna(0))
X_test_sc = scaler.transform(X_test.fillna(0))

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # handles imbalance
    penalty="l2",
    C=1.0,
)
lr.fit(X_train_sc, y_train)
```

### 6b. Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
```

### 6c. XGBoost

```python
import xgboost as xgb

# Calculate scale_pos_weight for imbalance
spw = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    use_label_encoder=False,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)
```

### 6d. LightGBM (your thesis Hypothesis 2 centers on gradient-boosted trees)

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    is_unbalance=True,
    random_state=42,
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)
```

### 6e. Evaluation on the Held-Out Test Set

Report these metrics on the **test** set only (never tune on test). This maps to the metrics listed in your Objective 2.

```python
def evaluate(name, model, X, y, needs_scaling=False):
    if needs_scaling:
        X = scaler.transform(X.fillna(0))
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUC:       {roc_auc_score(y, y_prob):.4f}")
    print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"  Precision: {precision_score(y, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y, y_pred):.4f}")
    print(f"  F1:        {f1_score(y, y_pred):.4f}")
    print(f"  Brier:     {brier_score_loss(y, y_prob):.4f}")
    return y_prob

prob_lr  = evaluate("Logistic Regression", lr,        X_test, y_test, needs_scaling=True)
prob_rf  = evaluate("Random Forest",       rf,        X_test, y_test)
prob_xgb = evaluate("XGBoost",             xgb_model, X_test, y_test)
prob_lgb = evaluate("LightGBM",            lgb_model, X_test, y_test)
```

### 6f. ROC & Precision-Recall Curves (for the thesis figures)

```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, probs in [("LR", prob_lr), ("RF", prob_rf),
                     ("XGB", prob_xgb), ("LGBM", prob_lgb)]:
    RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, probs, name=name, ax=axes[1])
axes[0].set_title("ROC Curve (Out-of-Time Test Set)")
axes[1].set_title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("roc_pr_curves.png", dpi=150)
```

---

## 7 — Probability Calibration (Required by Your Lit Review)

Your literature review (citing Lynn, 2025) correctly notes that uncalibrated probabilities are a known gap. Calibrating is what makes your scores *actionable* (e.g., "this booking has a 72% cancellation probability" actually means 72%).

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Calibrate the best tree model using the validation set
# Use isotonic regression (non-parametric, works well with trees)
cal_model = CalibratedClassifierCV(
    lgb_model,  # or whichever wins
    method="isotonic",
    cv="prefit",  # already fitted; calibrate on val set
)
cal_model.fit(X_val, y_val)

# Evaluate calibration on test set
prob_cal = cal_model.predict_proba(X_test)[:, 1]

# Reliability diagram
fraction_pos, mean_pred = calibration_curve(y_test, prob_cal, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(mean_pred, fraction_pos, "s-", label="Calibrated LightGBM")
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability Diagram")
plt.legend()
plt.savefig("calibration_plot.png", dpi=150)
```

---

## 8 — SHAP Analysis (Objective 3, Hypothesis 3)

This directly tests your Hypothesis 3 (lead time > deposit type > previous cancellations).

```python
import shap

# Use TreeExplainer for speed with tree-based models
explainer = shap.TreeExplainer(lgb_model)  # or xgb_model
shap_values = explainer.shap_values(X_test)

# If binary classification returns a list, take class 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# ── Global feature importance (bar plot) ──────────────────
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  max_display=15, show=False)
plt.tight_layout()
plt.savefig("shap_global_importance.png", dpi=150)

# ── Beeswarm (shows direction of effect) ─────────────────
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150)

# ── Single-booking explanation (for the dashboard concept) ─
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value
            if not isinstance(explainer.expected_value, list)
            else explainer.expected_value[1],
        data=X_test.iloc[0],
        feature_names=FEATURES,
    )
)
```

**For your defense:** Compare the SHAP ranking against your Hypothesis 3. If lead time is #1 and deposit type is #2, your hypothesis is supported. If the ordering differs, that is *still a valid finding* — discuss why the data may differ from prior literature.

---

## 9 — Cost-Sensitive Decision Threshold (Objective 4, Hypothesis 4)

This is where your study moves from "yet another ML paper" to something with **business value**. Your thesis conceptual framework (Transform phase) explicitly calls for this.

```python
# ── Define asymmetric costs ───────────────────────────────
# False Negative (FN): missed cancellation → lose revenue_at_risk
#   Assume: hotel loses 1 night of ADR as penalty recovery,
#   so net loss = ADR × (total_stay - 1), minimum 0
# False Positive (FP): unnecessary intervention on a good booking
#   Assume: friction cost = e.g., $10-$30 per intervention (reminder/deposit request)

FP_COST = 15   # cost of intervening on a booking that wouldn't have canceled
# FN cost is booking-specific: use revenue_at_risk from test set

test_revenue = test["revenue_at_risk"].values

# ── Sweep thresholds to find cost-minimizing cutoff ───────
thresholds = np.arange(0.05, 0.95, 0.01)
total_costs = []

for t in thresholds:
    preds = (prob_cal >= t).astype(int)
    fp = ((preds == 1) & (y_test.values == 0))  # intervened, didn't cancel
    fn = ((preds == 0) & (y_test.values == 1))  # missed cancellation
    cost = fp.sum() * FP_COST + (fn * test_revenue).sum()
    total_costs.append(cost)

optimal_t = thresholds[np.argmin(total_costs)]
print(f"Cost-minimizing threshold: {optimal_t:.2f}")

plt.figure(figsize=(8, 4))
plt.plot(thresholds, total_costs)
plt.axvline(optimal_t, color="r", linestyle="--", label=f"Optimal = {optimal_t:.2f}")
plt.xlabel("Decision Threshold")
plt.ylabel("Total Expected Cost ($)")
plt.title("Cost-Sensitive Threshold Selection")
plt.legend()
plt.savefig("cost_threshold.png", dpi=150)
```

### Risk-Based Deposit Tiers (for the Power BI playbook)

```python
# Assign risk bands based on calibrated probabilities
def risk_tier(prob):
    if prob >= 0.70:
        return "High Risk → Require non-refundable deposit"
    elif prob >= 0.40:
        return "Medium Risk → Request partial deposit or send reminder"
    else:
        return "Low Risk → No intervention"

test_results = test.copy()
test_results["cancel_prob"] = prob_cal
test_results["risk_tier"] = test_results["cancel_prob"].apply(risk_tier)
test_results["predicted_cancel"] = (prob_cal >= optimal_t).astype(int)

# Summary for Power BI export
print(test_results["risk_tier"].value_counts())
test_results.to_csv("test_predictions_for_powerbi.csv", index=False)
```

---

## 10 — Late-Window Analysis (Your Thesis Focus)

Your scope explicitly mentions "special attention to short-notice cancellations (within 3 days before arrival)." This section provides the evidence.

```python
late = test[test["lead_time"] <= 3]
not_late = test[test["lead_time"] > 3]

print(f"Late-window bookings: {len(late):,}  "
      f"(cancel rate: {late['is_canceled'].mean():.3f})")
print(f"Normal bookings:     {len(not_late):,}  "
      f"(cancel rate: {not_late['is_canceled'].mean():.3f})")

# Evaluate model performance specifically on the late window
prob_late = cal_model.predict_proba(late[FEATURES])[:, 1]
print(f"\nLate-window AUC:  {roc_auc_score(late['is_canceled'], prob_late):.4f}")
print(f"Overall test AUC: {roc_auc_score(y_test, prob_cal):.4f}")
```

**Expected finding:** AUC will likely be *lower* for the late window. This is consistent with your literature review (C-Sánchez & Sánchez-Medina, 2024). Document this honestly — it strengthens rather than weakens your thesis because it shows you understand the problem's difficulty.

---

## 11 — Reproducibility Checklist

| Item | How to Implement |
|---|---|
| Random seeds | Set `random_state=42` everywhere; `np.random.seed(42)` at top of notebook |
| Environment | Export `pip freeze > requirements.txt` |
| Data version | Record SHA-256 hash of the CSV: `sha256sum hotel_bookings.csv` |
| Notebook | Run "Restart & Run All" before submission — every cell must execute in order |
| No test-set tuning | Hyperparameters tuned on validation only; test set touched exactly once |

---

## 12 — Mapping Results to Your Hypotheses

| Hypothesis | How to Evaluate | Where in Code |
|---|---|---|
| **H1:** Lead time, deposit type, previous cancellations are significant predictors | Check if these appear in the top SHAP features with non-trivial magnitude | Section 8 |
| **H2:** Gradient-boosted trees beat LR, RF, XGBoost baseline | Compare AUC / F1 on the out-of-time test set across all 4 models | Section 6e |
| **H3:** Lead time has greatest SHAP importance, then deposit, then prev. cancellations | Read the SHAP bar plot rank ordering | Section 8 |
| **H4:** Cost-minimizing threshold + deposit tiers reduce expected revenue loss | Compare total cost at optimal threshold vs. cost at default 0.5 threshold, and vs. "no model" baseline (assume all honored) | Section 9 |

---

## 13 — Suggested Notebook Structure

```
01_data_loading_and_cleaning.ipynb
02_eda_and_feature_engineering.ipynb
03_time_split_and_encoding.ipynb
04_model_training_and_evaluation.ipynb
05_calibration_and_shap.ipynb
06_cost_threshold_and_tiers.ipynb
07_late_window_analysis.ipynb
08_export_for_powerbi.ipynb
```

Alternatively, a single well-organized notebook with clear markdown section headers is fine — just make sure it runs top-to-bottom without manual cell reordering.

---

## 14 — Common Defense Questions & How to Answer Them

**Q: "Why not use deep learning / LSTM?"**  
A: Your thesis scope explicitly delimits to classical ML and gradient-boosted trees. Your lit review cites Xiao et al. (2024) on temporal deep models — acknowledge this as a future direction, but justify that tree-based methods with SHAP provide the interpretability needed for your Transform phase (Power BI dashboard, deposit tiers).

**Q: "Your AUC is X — is that good enough?"**  
A: Reference your own literature review benchmarks (AUC 0.85–0.98 in prior work). Also emphasize that AUC alone is insufficient — you also report Brier score and calibration curves, addressing the gap identified by Lynn (2025).

**Q: "How do you know the model won't degrade in production?"**  
A: You used time-aware splits (simulating deployment). You recommend a monitoring plan (Section 10 of your methodology) tracking forecast error and cancel rate drift. Acknowledge that concept drift is a known limitation (your lit review, Section on Hewapathirana, 2023).

**Q: "Why is `reservation_status` not in the model?"**  
A: Because it is the target variable encoded differently. Including it would constitute a textbook case of target leakage, producing a model with near-perfect accuracy and zero real-world utility.

---

## Key Packages to Install

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn
```

---

*This roadmap is designed to be copy-pasted into Jupyter cells and adapted. Every design choice maps back to a specific section of your thesis manuscript (Chapters I–III). Good luck with the defense.*
