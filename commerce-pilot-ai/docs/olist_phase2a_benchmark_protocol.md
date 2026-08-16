# Olist Phase 2A Strict-Core Benchmark Protocol

Phase 2A is a local, offline retrospective benchmark using only Contract A. Its label is whether an eligible delivered order's recorded delivery time exceeded its stored estimated-delivery time. This does not establish that the estimate was immutable or customer-visible at approval.

The four and only model families are a prior-probability `DummyClassifier`, scaled unweighted Logistic Regression, CPU CatBoost, and deterministic CPU LightGBM. Bounded grids, seed 42, and one thread are fixed in `configs/olist_phase2a_benchmark.yaml` before model fitting.

Train fits all candidate preprocessing and models; Validation selects configurations, maximum-F1 research thresholds, and the champion. Average Precision from scikit-learn is the canonical primary metric (not trapezoidal PR integration). Exact ties use lower Brier score and then the earlier, simpler configured candidate. The selection manifest is hashed and reproducibility checked before a separate final-test command may access Test. The ledger prohibits a second final evaluation.

Test results cannot change the Validation-selected champion. Thresholds and capacity tables are research descriptions, not operational policies. No calibration, resampling, target encoding, class weighting, conditional features, deployment work, or cross-dataset integration is permitted.

Legal/licensing readiness, production readiness, and Egyptian-market external validity remain **NO-GO** regardless of scores.
