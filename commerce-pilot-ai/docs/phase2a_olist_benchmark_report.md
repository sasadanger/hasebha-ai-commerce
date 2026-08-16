# Phase 2A Olist Strict-Core Benchmark Report

Status: **COMPLETE**. Validation selected **CatBoost** before Test access. Test results did not change that locked decision.

| Model | Val AP | Val ROC-AUC | Val Brier | Test AP | Test ROC-AUC | Test Brier | Test AP lift | Locked threshold | Refit time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dummy | 0.124935 | 0.500000 | 0.112791 | 0.063288 | 0.500000 | 0.059919 | 1.000 | 0.066068 | 0.011s |
| logistic_regression | 0.129828 | 0.524325 | 0.111629 | 0.089085 | 0.568641 | 0.074276 | 1.408 | 0.142386 | 0.044s |
| catboost | 0.144503 | 0.569622 | 0.109136 | 0.079478 | 0.563441 | 0.086984 | 1.256 | 0.129332 | 0.237s |
| lightgbm | 0.134970 | 0.530222 | 0.112835 | 0.068864 | 0.536512 | 0.059862 | 1.088 | 0.065042 | 0.038s |

Validation prevalence was 12.493476%, versus 6.328807% on Test. CatBoost AP fell from 0.144503 to 0.079478; Logistic Regression achieved the highest descriptive Test AP (0.089085), but Test was not used to revise selection. This indicates temporal instability and weak strict-core ranking signal. Confidence intervals in `final_test_metrics.json` represent historical Test resampling uncertainty only.

Feature importance is associative, not causal. Every predictor is a purchase/approval calendar or approval-duration signal and may encode period-specific Olist operations. No conditional feature, target component, identifier, resampling, calibration, or class weighting was used.

Legal/licensing readiness: **NO-GO**. Production readiness: **NO-GO**. Egyptian-market external validity: **NO-GO**. This is local retrospective research only.
