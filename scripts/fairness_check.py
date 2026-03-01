"""Quick hyperparameter fairness sensitivity check.

Answers: is LightGBM > XGBoost due to algorithm quality or just capacity budget?

Tests XGBoost at (a) default params and (b) LightGBM's exact capacity budget,
then compares both against the saved LightGBM champion artifact.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import ARTIFACTS_DIR
from src.data.load import load_raw_data
from src.features.build import build_preprocessor, split_time_aware
from src.models.train import train_xgb
from src.utils.validate_data import clean_raw

warnings.filterwarnings("ignore")

# ── Data ────────────────────────────────────────────────────────────────────
print("Loading and splitting data...")
raw = load_raw_data()
cleaned, _ = clean_raw(raw)
train_df, val_df, test_df = split_time_aware(cleaned)

TGT = "is_canceled"
X_tr, y_tr = train_df.drop(columns=[TGT]), train_df[TGT].to_numpy()
X_va, y_va = val_df.drop(columns=[TGT]),   val_df[TGT].to_numpy()
X_te, y_te = test_df.drop(columns=[TGT]),  test_df[TGT].to_numpy()

prep = build_preprocessor()
X_tr_t = prep.fit_transform(X_tr)
X_va_t = prep.transform(X_va)
X_te_t = prep.transform(X_te)
print(f"  Train {X_tr_t.shape} | Val {X_va_t.shape} | Test {X_te_t.shape}\n")


def calibrated_eval(model: Any) -> tuple[float, float]:
    """Isotonic-calibrate on val, evaluate on test."""
    cal = IsotonicRegression(out_of_bounds="clip").fit(
        model.predict_proba(X_va_t)[:, 1], y_va
    )
    probs = np.clip(cal.predict(model.predict_proba(X_te_t)[:, 1]), 0.0, 1.0)
    return float(average_precision_score(y_te, probs)), float(roc_auc_score(y_te, probs))


# ── 1. XGBoost default ───────────────────────────────────────────────────────
print("[1/3] XGBoost — default     (n_est=100, depth=5,  lr=0.10)")
t0 = time.perf_counter()
m1 = train_xgb(X_tr_t, y_tr, X_va_t, y_va,
               params={"n_estimators": 100, "max_depth": 5, "learning_rate": 0.10})
pr1, roc1 = calibrated_eval(m1)
print(f"  PR-AUC={pr1:.4f}  ROC-AUC={roc1:.4f}  ({time.perf_counter()-t0:.1f}s)\n")

# ── 2. XGBoost matched capacity ──────────────────────────────────────────────
print("[2/3] XGBoost — matched     (n_est=300, depth=7,  lr=0.05)")
t0 = time.perf_counter()
m2 = train_xgb(X_tr_t, y_tr, X_va_t, y_va,
               params={"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05})
pr2, roc2 = calibrated_eval(m2)
print(f"  PR-AUC={pr2:.4f}  ROC-AUC={roc2:.4f}  ({time.perf_counter()-t0:.1f}s)\n")

# ── 3. LightGBM champion (from artifact) ─────────────────────────────────────
print("[3/3] LightGBM — champion   (n_est=300, depth=7,  lr=0.05)  [artifact]")
pipeline   = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
calibrator = joblib.load(ARTIFACTS_DIR / "probability_calibrator.pkl")
inner      = pipeline.named_steps["model"]
probs_lgbm = np.clip(calibrator.predict(inner.predict_proba(X_te_t)[:, 1]), 0.0, 1.0)
pr_lgbm    = float(average_precision_score(y_te, probs_lgbm))
roc_lgbm   = float(roc_auc_score(y_te, probs_lgbm))
print(f"  PR-AUC={pr_lgbm:.4f}  ROC-AUC={roc_lgbm:.4f}\n")

# ── Results ──────────────────────────────────────────────────────────────────
print("=" * 68)
print(f"  {'Model':<42} {'PR-AUC':>7}   {'ROC-AUC':>7}")
print(f"  {'LightGBM champion':<42} {pr_lgbm:>7.4f}   {roc_lgbm:>7.4f}")
print(f"  {'XGBoost matched  (n300/d7/lr0.05)':<42} {pr2:>7.4f}   {roc2:>7.4f}")
print(f"  {'XGBoost default  (n100/d5/lr0.10)':<42} {pr1:>7.4f}   {roc1:>7.4f}")
print("=" * 68)

delta_pr      = pr_lgbm - pr2
delta_roc     = roc_lgbm - roc2
capacity_gain = pr2 - pr1
total_gap     = pr_lgbm - pr1

print(f"\n  Delta (LightGBM vs XGBoost-matched):   PR-AUC {delta_pr:+.4f}   ROC-AUC {delta_roc:+.4f}")
print(f"  Capacity effect on XGBoost:            PR-AUC {capacity_gain:+.4f}   (matched minus default)")

if total_gap != 0:
    pct_capacity = 100.0 * capacity_gain / total_gap
    pct_algo     = 100.0 - pct_capacity
    print(f"\n  Gap decomposition  (LightGBM default vs XGBoost default = {total_gap:+.4f} PR-AUC):")
    print(f"    - Capacity accounts for: {pct_capacity:.0f}%")
    print(f"    - Algorithm quality    : {pct_algo:.0f}%")

print()
if delta_pr > 0:
    print("  VERDICT: LightGBM STILL LEADS at equal hyperparameter budget.")
    print("  The champion ranking is NOT an artifact of the capacity gap.")
    print("  Leaf-wise growth + histogram binning explain the residual advantage.")
else:
    print("  VERDICT: XGBoost matches or surpasses LightGBM at equal capacity.")
    print("  The original margin was driven by budget, not algorithm quality.")
    print("  Review champion selection — ranking may not be robust.")
print()
