# Olist Model Evaluation Protocol

## Leakage-safe preprocessing

Split chronologically before fitting anything. Fit imputers, scalers, encoders, frequency maps, target encoders, feature selectors, geographic statistics, historical aggregates, and calibration only on training data. Validation/test unknowns map to an explicit unknown representation without refitting. Target encoding is prohibited unless out-of-fold within chronologically valid training folds. Historical outcomes must precede each approval time.

Do not apply SMOTE or synthetic oversampling before splitting. Start with class weights, validation-only threshold selection, and cost-sensitive evaluation. Any resampling stays inside training folds and must be compared with a non-resampled baseline.

## Metrics and models

Primary metric: PR-AUC. Secondary metrics: ROC-AUC, recall, precision, F1, confusion matrix, Brier score, calibration curve, risk-score distribution, observed prevalence, and later inference latency. Accuracy is descriptive only and never selects a model.

Mandatory future comparison: `DummyClassifier`, Logistic Regression, CatBoost, and LightGBM under the same cohort, split, features, and metrics. Random Forest/XGBoost are optional and do not expand the mandatory scope.

## Thresholds

Select thresholds using validation only and freeze before test evaluation:

1. threshold maximizing a declared validation objective (default candidate: F1, explicitly reported);
2. threshold meeting a stakeholder-approved minimum recall;
3. threshold minimizing operational scenario cost.

No recall requirement or business cost is currently verified. Scenario analysis may use clearly labeled assumptions only:

```text
Expected Intervention Cost =
    FP × cost_of_unnecessary_intervention
  + FN × cost_of_missed_late_delivery
```

Phase 1D classifies this gap as an operational decision-policy blocker, not a blocker to threshold-independent offline ranking and calibration evaluation. No threshold may be described as production-optimal until a cost matrix or approved recall/precision constraint is verified. The final test set never selects a threshold.

## Reproducibility record

Every Phase 2 run must store: experiment ID; fixed seed; exact split boundaries; raw and processed hashes; Python/package versions; configuration; eligibility/anomaly/feature-contract versions; metric definitions; hyperparameters; selected threshold and rule; counts/prevalence; UTC execution time; hardware; calibration method; and untouched-test predictions. Predictions must include only audit identifiers, label, score, frozen prediction, split, and experiment ID under ignored experiment storage.
