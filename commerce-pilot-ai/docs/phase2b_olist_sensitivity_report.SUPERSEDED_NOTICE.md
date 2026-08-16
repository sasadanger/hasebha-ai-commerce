# SUPERSEDED NOTICE

The sibling file `phase2b_olist_sensitivity_report.md` in this directory is a **historical, pre-correction artifact**. Its results table Evidence column reads `SUPPORTED_INCREMENTAL_SIGNAL` for `logistic_regression`, `catboost`, and `lightgbm`, and its Delta AP column for `logistic_regression` (`0.014504`) mixes a pooled bootstrap-replicate mean with macro AP means displayed beside it — the exact defect documented in `phase2b_olist_sensitivity_erratum.md`. That table is **not** the current authoritative conclusion.

`phase2b_olist_sensitivity_report.md` itself was **not modified** by this notice — it is preserved unmodified for audit-trail traceability.

The current authoritative report is: `reports/generated/olist/phase2b/correction_v3/phase2b_sensitivity_report_v3.md`.

Current authoritative evidence labels:

- `dummy`: `NO_INCREMENTAL_SIGNAL`
- `logistic_regression`: `INCONCLUSIVE_INCREMENTAL_SIGNAL`
- `catboost`: `INCONCLUSIVE_INCREMENTAL_SIGNAL`
- `lightgbm`: `INCONCLUSIVE_INCREMENTAL_SIGNAL`

Machine-readable precedence: `reports/generated/olist/phase2b/correction_v3/evidence_precedence.json`.
