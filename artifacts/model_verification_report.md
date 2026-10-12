# Model Verification Report

- Generated: 2026-04-24 07:07:33
- Seed policy: `random=42`, `numpy=42`, model seeds set to `42`, single-thread model fitting where supported

## 1) Exact Notebook Definitions and Usages
```text
                                                      Topic                            Notebook                                                  Notes
        Holdout split construction (chronological 80/10/10)                         02_modeling   Loads pre-computed artifacts via load_main_context()
                Holdout train/test usage (feature matrices)       02_modeling, 03_deep_analysis           Artifacts loaded; no retraining in notebooks
                      Expanding-window CV (TimeSeriesSplit)                    03_deep_analysis           Section on expanding-window cross-validation
        Preprocessing pipeline (OneHotEncoder + imputation)       02_modeling, 03_deep_analysis          Pipeline loaded from artifacts/best_model.pkl
                Gradient Boosting & XGBoost hyperparameters     02_modeling, 07_model_selection Rolling-origin champion selection; fairness comparison
Thresholding logic (max_f1, high_precision, cost_sensitive) 03_deep_analysis, 05_explainability                  Loaded from artifacts/thresholds.json
                        SHAP explainability (TreeExplainer)                   05_explainability    Loads pre-computed SHAP values from reports/thesis/
                Business analytics (cost model, risk tiers)               06_business_analytics                 Revenue-at-risk, segment cost analysis
                 ADR forecasting (regression + time series)                  04_adr_forecasting       Loads adr_regressor.pkl; optional Keras NN model
                    Model monitoring (drift detection, PSI)                 08_model_monitoring         Baseline from test_predictions_for_powerbi.csv
```

## 2) Exact Split and Pipeline Parameters Used
- Data cleaned with `clean_raw`, validated with `validate_raw`, then sorted by `arrival_date`.
- Holdout split: train+val ends at index `107287` of `119209` rows (TRAIN_RATIO=0.8, VAL_RATIO=0.1).
- Holdout cutoff date: `2017-06-21`.
- Holdout train rows: `107287`, holdout test rows: `11922`.
- TimeSeriesSplit config for CV: `TimeSeriesSplit(n_splits=5)` with default `gap=0`, default fold test size.
- Preprocessing pipeline (identical for both compared models):
  - Categorical: `SimpleImputer(constant='UNKNOWN')` -> cast to string -> `OneHotEncoder(handle_unknown='ignore', min_frequency=0.01)`
  - Numeric: `SimpleImputer(strategy='median')`

## 3) Notebook Model Hyperparameters (as actually used)
```text
            Model                                                                                                        Notebook params
Gradient Boosting                                                              {"max_depth": 5, "n_estimators": 100, "random_state": 42}
          XGBoost {"eval_metric": "logloss", "learning_rate": 0.1, "max_depth": 5, "n_estimators": 100, "n_jobs": 1, "random_state": 42}
```

### Diagnostic Search Spaces (for under-tuning check)
```text
            Model                                                                                                                                                                                                                                                                                                                                                                      Grid tested
Gradient Boosting                                                                                                                                                                               [{'max_depth': 3, 'n_estimators': 100}, {'max_depth': 5, 'n_estimators': 300}, {'max_depth': 5, 'n_estimators': 200}, {'max_depth': 5, 'n_estimators': 100}, {'max_depth': 3, 'n_estimators': 50}]
          XGBoost [{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}, {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}, {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}, {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 300}, {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}, {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200}]
```

## 4) Comparison A: Single Holdout (current split)
```text
            Model  ROC-AUC  PR-AUC  F1@0.50  Precision@0.50  Recall@0.50  Brier     TN     FP     FN     TP
Gradient Boosting   0.8721  0.7917   0.7226          0.7106       0.7351 0.1411 6050.0 1353.0 1197.0 3322.0
          XGBoost   0.8684  0.7841   0.7010          0.7271       0.6767 0.1436 6255.0 1148.0 1461.0 3058.0
```

## 5) Comparison B: Rolling-Origin / Backtest (5 cutoffs)
### Per-cutoff detail
```text
            Model  cutoff_frac cutoff_date test_start_date test_end_date  n_train  n_test  ROC-AUC  PR-AUC  F1@0.50
Gradient Boosting         0.50  2016-09-06      2016-09-06    2016-11-07  59604.0 11920.0   0.9342  0.9136   0.7014
          XGBoost         0.50  2016-09-06      2016-09-06    2016-11-07  59604.0 11920.0   0.9288  0.9076   0.6563
Gradient Boosting         0.58  2016-10-26      2016-10-26    2017-01-21  69141.0 11920.0   0.9009  0.8560   0.7081
          XGBoost         0.58  2016-10-26      2016-10-26    2017-01-21  69141.0 11920.0   0.9022  0.8575   0.6975
Gradient Boosting         0.66  2017-01-01      2017-01-01    2017-03-27  78677.0 11920.0   0.9028  0.8423   0.6932
          XGBoost         0.66  2017-01-01      2017-01-01    2017-03-27  78677.0 11920.0   0.9065  0.8456   0.6969
Gradient Boosting         0.74  2017-03-11      2017-03-11    2017-05-16  88214.0 11920.0   0.9172  0.8889   0.7743
          XGBoost         0.74  2017-03-11      2017-03-11    2017-05-16  88214.0 11920.0   0.9164  0.8884   0.7650
Gradient Boosting         0.82  2017-05-04      2017-05-04    2017-07-04  97751.0 11920.0   0.9095  0.8874   0.7719
          XGBoost         0.82  2017-05-04      2017-05-04    2017-07-04  97751.0 11920.0   0.9065  0.8844   0.7675
```

### Distribution summary (mean/std across cutoffs)
```text
            Model  ROC-AUC mean  ROC-AUC std  PR-AUC mean  PR-AUC std  F1@0.50 mean  F1@0.50 std
Gradient Boosting        0.9129       0.0135       0.8776      0.0284        0.7297       0.0399
          XGBoost        0.9121       0.0107       0.8767      0.0249        0.7166       0.0483
```

## 6) Comparison C: TimeSeriesSplit CV (n_splits=5)
### Fold-level detail
```text
            Model  fold train_end_date val_start_date val_end_date  ROC-AUC  PR-AUC  Precision@0.50  Recall@0.50  F1@0.50
Gradient Boosting   1.0     2015-11-18     2015-11-18   2016-04-18   0.8294  0.7913          0.8492       0.5713   0.6831
          XGBoost   1.0     2015-11-18     2015-11-18   2016-04-18   0.8390  0.7943          0.8538       0.5726   0.6855
Gradient Boosting   2.0     2016-04-18     2016-04-18   2016-07-31   0.9102  0.8752          0.8592       0.6260   0.7243
          XGBoost   2.0     2016-04-18     2016-04-18   2016-07-31   0.9150  0.8824          0.8821       0.6214   0.7291
Gradient Boosting   3.0     2016-07-31     2016-07-31   2016-11-07   0.9150  0.8801          0.8808       0.5871   0.7046
          XGBoost   3.0     2016-07-31     2016-07-31   2016-11-07   0.9083  0.8728          0.8935       0.5177   0.6555
Gradient Boosting   4.0     2016-11-07     2016-11-07   2017-03-18   0.8986  0.8366          0.8114       0.5966   0.6876
          XGBoost   4.0     2016-11-07     2016-11-07   2017-03-18   0.8992  0.8377          0.8106       0.5886   0.6820
Gradient Boosting   5.0     2017-03-18     2017-03-18   2017-06-21   0.9121  0.8861          0.8482       0.6968   0.7651
          XGBoost   5.0     2017-03-18     2017-03-18   2017-06-21   0.9117  0.8873          0.8483       0.6881   0.7598
```

### Mean/std summary
```text
            Model  ROC-AUC mean  ROC-AUC std  PR-AUC mean  PR-AUC std  Precision mean  Precision std  Recall mean  Recall std  F1 mean  F1 std
          XGBoost        0.8946       0.0316       0.8549       0.039          0.8577         0.0324       0.5977      0.0629   0.7024  0.0416
Gradient Boosting        0.8930       0.0361       0.8539       0.040          0.8498         0.0251       0.6156      0.0496   0.7129  0.0334
```

## 7) Diagnosis
### 7.1 Under-tuning check
- Gradient Boosting: Potential under-tuning: best grid ROC-AUC (0.8963) exceeds notebook baseline (0.8930) by 0.0033.
- XGBoost: No material under-tuning signal in this grid: best ROC-AUC (0.8973) vs baseline (0.8946), delta 0.0027.

Gradient Boosting grid results:
```text
            Model                                params  Mean ROC-AUC  Std ROC-AUC  Mean PR-AUC  Mean F1
Gradient Boosting {"max_depth": 3, "n_estimators": 100}        0.8963       0.0213       0.8535   0.7082
Gradient Boosting {"max_depth": 5, "n_estimators": 300}        0.8938       0.0257       0.8500   0.7145
Gradient Boosting {"max_depth": 5, "n_estimators": 200}        0.8935       0.0285       0.8541   0.7152
Gradient Boosting {"max_depth": 5, "n_estimators": 100}        0.8930       0.0323       0.8539   0.7129
Gradient Boosting  {"max_depth": 3, "n_estimators": 50}        0.8839       0.0284       0.8418   0.6861
```

XGBoost grid results:
```text
  Model                                                       params  Mean ROC-AUC  Std ROC-AUC  Mean PR-AUC  Mean F1
XGBoost  {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 200}        0.8973       0.0220       0.8562   0.7114
XGBoost  {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 300}        0.8973       0.0190       0.8562   0.7120
XGBoost  {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100}        0.8946       0.0283       0.8549   0.7024
XGBoost {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 300}        0.8913       0.0309       0.8516   0.6981
XGBoost  {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100}        0.8902       0.0291       0.8494   0.6944
XGBoost {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 200}        0.8873       0.0312       0.8474   0.6914
```

### 7.2 Calibration + threshold-on-validation experiment
- Calibration experiment used `CalibratedClassifierCV(method='isotonic', cv=TimeSeriesSplit(n_splits=3))` on subtrain only.
```text
            Model  Brier (uncalibrated)  Brier (calibrated-CV)  Threshold@maxF1 on validation (uncalibrated)  Threshold@maxF1 on validation (calibrated-CV)  Test F1 using validation-selected threshold (uncalibrated)  Test F1 using validation-selected threshold (calibrated-CV)
Gradient Boosting                0.1543                 0.1789                                          0.24                                           0.34                                                      0.7401                                                       0.6532
          XGBoost                0.1563                 0.1756                                          0.28                                           0.34                                                      0.7285                                                       0.6460
```

### 7.3 Feature drift check (metric by late-time bucket on holdout)
```text
 bucket bucket_start bucket_end             Model  n_rows  cancel_rate  ROC-AUC  PR-AUC  F1@0.50
    1.0   2017-06-21 2017-07-04 Gradient Boosting  2385.0       0.4025   0.8919  0.8402   0.7671
    1.0   2017-06-21 2017-07-04           XGBoost  2385.0       0.4025   0.8887  0.8344   0.7631
    2.0   2017-07-04 2017-07-18 Gradient Boosting  2385.0       0.3732   0.8640  0.7705   0.7169
    2.0   2017-07-04 2017-07-18           XGBoost  2385.0       0.3732   0.8604  0.7608   0.6774
    3.0   2017-07-18 2017-08-01 Gradient Boosting  2384.0       0.3909   0.8785  0.8124   0.7348
    3.0   2017-07-18 2017-08-01           XGBoost  2384.0       0.3909   0.8758  0.8058   0.7158
    4.0   2017-08-01 2017-08-16 Gradient Boosting  2384.0       0.3704   0.8581  0.7566   0.6912
    4.0   2017-08-01 2017-08-16           XGBoost  2384.0       0.3704   0.8518  0.7473   0.6793
    5.0   2017-08-16 2017-08-31 Gradient Boosting  2384.0       0.3582   0.8638  0.7603   0.6970
    5.0   2017-08-16 2017-08-31           XGBoost  2384.0       0.3582   0.8613  0.7535   0.6590
```

### 7.4 Learning curves vs `n_estimators` / `max_depth`
```text
            Model      varying  value  Mean ROC-AUC
Gradient Boosting n_estimators   50.0        0.9004
Gradient Boosting n_estimators  100.0        0.9049
Gradient Boosting n_estimators  200.0        0.9042
Gradient Boosting n_estimators  300.0        0.9017
          XGBoost n_estimators   50.0        0.8931
          XGBoost n_estimators  100.0        0.9004
          XGBoost n_estimators  200.0        0.9000
          XGBoost n_estimators  300.0        0.8987
Gradient Boosting    max_depth    2.0        0.8904
Gradient Boosting    max_depth    3.0        0.9009
Gradient Boosting    max_depth    5.0        0.9049
Gradient Boosting    max_depth    7.0        0.8994
          XGBoost    max_depth    2.0        0.8857
          XGBoost    max_depth    3.0        0.8972
          XGBoost    max_depth    5.0        0.9004
          XGBoost    max_depth    7.0        0.8977
```

## 8) Deterministic Selection Policy Status
- Policy implemented: `rolling_origin_mean_roc_auc_then_pr_auc_v1` (rank by rolling-origin mean ROC-AUC, tie-break by rolling-origin mean PR-AUC).
- Deterministic policy selects `Gradient Boosting` and holdout confirmation agrees (holdout top is also `Gradient Boosting`).
- Policy code path references:
  - `main.ipynb` uses `select_best_model` in cell 17.
  - `analysis.ipynb` uses `select_best_model` in cell 22.
- Holdout is retained as a confirmation signal only (not the primary selector).

## 9) Figures
- Holdout ROC curve: `D:/PythonProject1/artifacts/figures/verification_holdout_roc_curve.png`
- Holdout PR curve: `D:/PythonProject1/artifacts/figures/verification_holdout_pr_curve.png`
- Rolling metric vs cutoff: `D:/PythonProject1/artifacts/figures/verification_metric_vs_cutoff.png`
- Calibration curves: `D:/PythonProject1/artifacts/figures/verification_calibration_curve.png`
- Learning curves: `D:/PythonProject1/artifacts/figures/verification_learning_curves.png`
- Metric by time bucket: `D:/PythonProject1/artifacts/figures/verification_metric_by_time_bucket.png`

## 10) Conclusion
- Policy winner: `Gradient Boosting`
- Holdout top (confirmation): `Gradient Boosting`
- Gradient Boosting is more robust on future windows (holdout + rolling-origin), while TimeSeriesSplit CV is close. This suggests GB generalizes slightly better to the late timeline.