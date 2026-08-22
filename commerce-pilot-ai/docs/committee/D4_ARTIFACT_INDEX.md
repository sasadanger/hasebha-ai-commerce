# D4 — Master Artifact Index

Verified: 2026-08-22. One line + path per artifact category. This is an index, not a copy —
every claim in D1/D2/D5/D6 traces back to one of these.

## Forensic study (this cycle — the newest, most decisive evidence)
- `scripts/forensics/production_model_forensics.py` — the forensic experiment code (real raw-data loading, real training, spot-checked and verified this session).
- `reports/generated/olist_v3_multistage/forensics/FORENSIC_EXPERIMENT_RESULTS.json` — raw per-period, per-experiment results.
- `reports/generated/olist_v3_multistage/forensics/FORENSIC_EXPERIMENT_SUMMARY.json` — condensed summary.
- `commerce-pilot-ai/PRODUCTION_MODEL_FORENSIC_REPORT.md` — full narrative forensic report.
- `commerce-pilot-ai/PRODUCTION_MODEL_FORENSIC_SCORECARD.json` — machine-readable scorecard.
- `commerce-pilot-ai/PRODUCTION_MODEL_FEATURE_INVENTORY.md` — full feature-by-feature classification (A-F).
- `commerce-pilot-ai/PRODUCTION_MODEL_EXPERIMENT_MATRIX.json` — every experiment, structured.
- `commerce-pilot-ai/PRODUCTION_MODEL_COMMITTEE_BRIEF.md` — pre-existing committee-facing summary of the forensics.

## Committee-defense package (prior cycle)
- `reports/generated/committee_defense/MASTER_MODEL_EVIDENCE_MATRIX.json` — 12-track evidence matrix.
- `reports/generated/committee_defense/STRONGEST_RESULTS_BY_CATEGORY.json` — 8-category strongest-result audit.
- `reports/generated/committee_defense/LOW_METRIC_SCIENTIFIC_INTERPRETATION.md` — why the numbers are low, explained.
- `reports/generated/committee_defense/NEGATIVE_RESULTS_REGISTER.json` — 15 formally registered negative results.
- `reports/generated/committee_defense/PRODUCTION_REALITY_MATRIX.json` — binary, evidence-based production-execution audit.
- `reports/generated/committee_defense/COMMITTEE_DEFENSE_BRIEF.md` — 20 Q&A.
- `reports/generated/committee_defense/60_SECOND_PROJECT_EXPLANATION.md` — plain-language summary.
- `reports/generated/committee_defense/COMMITTEE_DEFENSE_SCORECARD.json`.
- `reports/generated/committee_defense/FINAL_COMMITTEE_READINESS_REPORT.md` — 22-section prior readiness report.

## Complete ML results reconstruction (prior cycle)
- `reports/COMMERCEPILOT_COMPLETE_ML_RESULTS_REPORT.md` — full 20-section results reconstruction.
- `reports/COMMERCEPILOT_COMPLETE_ML_RESULTS_SCORECARD.json`.
- `reports/COMMERCEPILOT_SUPERVISOR_HANDOFF_CURRENT_STATE.md` / `.json` — first-principles state reconstruction, including the 5-orders/0-fulfillment-outcomes live DB finding.

## Planning
- `docs/FINAL_PROJECT_EXECUTION_PLAN.md` — 22-section forward roadmap, Feature Availability Matrix, Execution Authorization Matrix, GREEN/YELLOW/RED framework.
- `docs/MASTER_EXECUTION_STATUS.md` — living status file, including the C:→D: migration record and venv-usability correction.

## Per-track evidence
### Arabic
- `reports/generated/arabic_sota/ARABIC_FINAL_DECISION.json` — champion selection, 5 rejected candidates, SARF cost-rejection.
- `configs/nlp_champion_registry.yaml` — champion/finalist freeze registry, 3-seed confirmation (seeds 101/202/303).
- `artifacts/experiments/arabic_foundation/baseline/predictions/` — raw prediction parquets.

### Amazon
- `reports/generated/amazon/metrics.json` — classical champion selection (TF-IDF+LinearSVC, plateau rule).
- `reports/generated/amazon/transformer_final_eval.json` — DistilRoBERTa evaluation (reported, not the designated champion).
- `reports/generated/amazon/predictions/test_balanced_predictions.parquet` — raw predictions (this session's bootstrap CI source).
- `artifacts/experiments/amazon/models/amazon_tfidf_wordchar_linearsvc_size100000.joblib` — the fitted, hash-verified, but currently unwired classical model artifact.

### Instacart
- `reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/` — freeze + protected-test-access record.
- `reports/generated/instacart/protected_test_final_results.json` — full protected-test metrics.

### Olist V1 (production)
- `reports/generated/olist/phase2a/final_test_metrics.json` — frozen test-set metrics for all evaluated model types.
- `artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions/test_catboost.parquet` — raw predictions (this session's bootstrap CI source).
- `src/ai_service/config.py` — hash-pinned production artifact reference.

### Olist V2
- `reports/generated/olist_v2/FINAL_SCORECARD.json`, `OLIST_V2_TEMPORAL_STABILITY_FINAL_REPORT.md`, `CURRENT_STATE.json` — regime-shift collapse, trucker-strike refutation, recency-weighted adaptation.

### Olist V3
- `reports/generated/olist_v3_multistage/SELLER_SLA_TEMPORAL_EVAL.json` — seller-SLA research model, 5-period temporal eval.
- `reports/generated/olist_v3_multistage/TASK_B_C_RESULTS.json` — customer T0 stacking and T1 dynamic results.
- `reports/generated/olist_v3_multistage/SELLER_SLA_LEAKAGE_TESTS.json` — leakage audit (0/4 failures).
- `reports/generated/olist_v3_multistage/SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json` — feature-parity audit against real HASEBHA architecture.
- `reports/generated/olist_v3_multistage/OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json` — the 0.7702→0.5188 sentinel-substitution collapse.
- `reports/generated/olist_v3_multistage/PRODUCTION_PARITY_MODEL_COMPARISON.json` — the 13-feature production-parity retrain.
- `reports/generated/olist_v3_multistage/SELLER_SLA_CALIBRATION_REPORT.json`, `PRODUCTION_PARITY_CALIBRATION_REPORT.json` — isotonic calibration.
- `reports/generated/olist_v3_multistage/OLIST_SELLER_SLA_MODEL_CARD.md`, `PRODUCTION_PARITY_MODEL_CARD.md` — model cards.
- `reports/generated/olist_v3_multistage/HASEBHA_SHIPPING_SLA_PRODUCT_REQUIREMENT.md` — the specified, not-yet-implemented business SLA mechanism.

### Olist Marketing Funnel Enrichment (negative result, NR-16)
- `reports/generated/olist_funnel/OLIST_FUNNEL_ENRICHMENT_FINAL_REPORT.md` — full report.
- `reports/generated/olist_funnel/OLIST_FUNNEL_SCORECARD.json` — machine-readable scorecard.
- `reports/generated/olist_funnel/FUNNEL_DATA_QUALITY_AUDIT.json` — acquisition, provenance, coverage audit.
- `reports/generated/olist_funnel/FUNNEL_FEATURE_LEAKAGE_CHECK.json` — 0/4384 leakage violations.
- `reports/generated/olist_funnel/FUNNEL_EXPERIMENT_RESULTS.json` — full per-period baseline vs. treatment results.
- `scripts/funnel/build_funnel_features.py`, `scripts/funnel/funnel_experiment.py` — new code, does not touch frozen pipelines.
- `docs/data_provenance.md` — funnel dataset registered alongside every other dataset.

### DataCo/EAGLE
- `reports/generated/dataco/DATACO_ACQUISITION_PROVENANCE.json` — hash-verified dataset acquisition.
- `reports/generated/dataco/DATACO_LSTM_REPRODUCTION.json` — 4-seed LSTM reproduction (0.6454 vs published 0.9679).
- `reports/generated/dataco/DATACO_TARGET_FORENSIC_CORRECTION.json` — mathematical diagnosis of the target-prevalence mismatch.
- `reports/generated/dataco/LSTM_REPRODUCTION_GATE_DECISION.json` — decision to not attempt EAGLE.

### Jumia (excluded, historical only)
- `reports/checkpoints/jumia_phase9_freeze_and_protected_test_2026-08-15/` and related — historical evidence that Jumia was worked on before the current permanent-exclusion policy; not part of the current defense narrative.

## Engineering / infrastructure
- `src/ai_service/services/seller_sla_risk.py`, `production_parity_seller_sla.py`, `prediction_feedback_store.py` — hash-verified model services, raw-feature persistence.
- `src/ai_service/routers/fulfillment.py`, `schemas.py`, `main.py`, `config.py`, `health.py` — API surface.
- `medusa-app/commercepilot-medusa/apps/backend/src/subscribers/order-placed.ts` — the only Medusa integration point; live V1 wiring + shadow wiring + `same_zone` resolution (fixed this project cycle).
- `reports/generated/final_release/RAW_FEATURE_PERSISTENCE_FINAL_REPORT.md`, `SELLER_SLA_INTEGRATION_SCORECARD.json`, `HASEBHA_SINGLE_VENDOR_FULFILLMENT_SCORECARD.json` — engineering-cycle final reports.

## Live evidence (not a file, a direct query)
- PostgreSQL database (`commercepilot_medusa_postgres` container): 5 real orders (2026-08-15/16), 0 fulfillment records, 0 shadow-metadata rows — verified by direct SQL query in a prior session, re-confirmed structurally consistent in every session since.
