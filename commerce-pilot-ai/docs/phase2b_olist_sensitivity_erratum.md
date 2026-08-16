# Phase 2B Corrective Reporting Erratum

This revision preserves the original artifacts and recalculates reporting from the unchanged development predictions. Correct labels are Dummy `NO_INCREMENTAL_SIGNAL`; Logistic Regression, CatBoost, and LightGBM `INCONCLUSIVE_INCREMENTAL_SIGNAL`. No expanded feature has proven approval-time semantics.

The previous `0.014504` was the mean of bootstrap replicates for a pooled out-of-fold delta, displayed beside macro fold means; it was a valid nearby pooled estimand but mislabeled and its point definition was improper. Corrected outputs separate macro and pooled estimands. The previous `1.269` is the macro expanded AP lift ratio: the mean of fold AP/prevalence ratios.

No models were trained, no predictions or model artifacts were rewritten, and the consumed Test set was not reopened.
