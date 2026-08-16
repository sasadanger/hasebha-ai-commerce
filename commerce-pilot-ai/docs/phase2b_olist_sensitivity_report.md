# Phase 2B Olist Development-Only Sensitivity Report

`PHASE2B_STATUS = PARTIAL` because only the explicitly authorized retrospective sensitivity—with unverified as-of semantics—was scientifically eligible. No primary as-of-compatible Contract B feature was approved.

| Model | Strict mean AP | Expanded mean AP | Delta AP | Paired 95% CI | Strict ROC | Expanded ROC | Strict Brier | Expanded Brier | Expanded AP lift | Folds improved | Evidence | Duration |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| dummy | 0.088589 | 0.088589 | 0.000000 | [0.000000, 0.000000] | 0.500000 | 0.500000 | 0.081743 | 0.081743 | 1.000 | 0/3 | DEGRADED_PERFORMANCE | 0.022s |
| logistic_regression | 0.099894 | 0.110798 | 0.014504 | [0.011916, 0.017154] | 0.526603 | 0.561433 | 0.081422 | 0.081251 | 1.269 | 3/3 | SUPPORTED_INCREMENTAL_SIGNAL | 0.244s |
| catboost | 0.094946 | 0.104244 | 0.022466 | [0.019130, 0.025827] | 0.517291 | 0.544863 | 0.095123 | 0.081247 | 1.169 | 3/3 | SUPPORTED_INCREMENTAL_SIGNAL | 0.818s |
| lightgbm | 0.094040 | 0.099618 | 0.005667 | [0.002925, 0.008359] | 0.519653 | 0.531950 | 0.081748 | 0.081756 | 1.120 | 3/3 | SUPPORTED_INCREMENTAL_SIGNAL | 0.467s |

The expanded aggregates improved development ranking for Logistic Regression, CatBoost, and LightGBM under the retrospective snapshot assumption. This is not deployable approval-time evidence because final-export immutability was not verified. Fold prevalence and effects varied, so temporal stability remains limited. Dummy receives identical constant information; any machine-scale delta is scientifically zero.

Importance values are associative, not causal. Legal/licensing, production readiness, and Egyptian-market external validity remain NO-GO. Phase 2A CatBoost remains historically locked. No Test target, prediction content, or metric was read or calculated.
