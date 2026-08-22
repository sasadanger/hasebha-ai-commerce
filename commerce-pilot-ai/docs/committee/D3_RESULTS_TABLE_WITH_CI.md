# D3 — Unified Results Table with Confidence Intervals

Verified: 2026-08-22. All CIs computed READ-ONLY this session from saved per-example
predictions (bootstrap, 2000 resamples, seed=42) — no retraining. Full machine-readable
detail: `D3_RESULTS_TABLE_WITH_CI.json`.

| Track | Metric | Point Estimate | Uncertainty | n | Type |
|---|---|---|---|---|---|
| Olist V1 (production) | ROC-AUC | 0.5634 | 95% CI [0.5483, 0.5774] | 24,744 | Bootstrap (exact-match verified) |
| Olist V3 Seller-SLA (research) | mean temporal AUC | 0.7702 | fold std 0.0483, worst period 0.6762 | 5 periods | Temporal dispersion (not bootstrap) |
| HASEBHA production-parity | mean temporal AUC | 0.5551 | worst period 0.5289; independently reproduced at 0.5540 | 5 periods | Temporal dispersion + independent reproduction |
| Amazon Appliances (classical) | Macro-F1 | 0.9454 | 95% CI [0.9432, 0.9476] | 40,000 | Bootstrap (exact-match verified) |
| Arabic MPOLD (MARBERTv2) | Macro-F1 | 0.8130 (1-seed) / 0.7906 (3-seed mean) | 3-seed dispersion (existing); bootstrap CI NOT_COMPUTED this session | — | Multi-seed (existing) |
| Instacart (hybrid recommender) | Precision@5 | 0.3674 | NOT_COMPUTED this session (ranking metric, needs per-user artifact reconstruction) | 26,314 users | Point estimate only |

**Honest disclosure**: 2 of 6 rows have genuine bootstrap CIs newly computed this session
(both point estimates reproduced bit-for-bit against previously reported values, confirming
the source files are authoritative). 2 rows use the existing, already-computed temporal
fold dispersion (a different, also legitimate uncertainty measure — not relabeled as a
bootstrap CI). 2 rows (Arabic bootstrap, Instacart bootstrap) are explicitly marked
NOT_COMPUTED rather than approximated, per this project's standing "no fabricated numbers"
rule — a partial deliverable stated honestly beats a complete one with invented figures.
