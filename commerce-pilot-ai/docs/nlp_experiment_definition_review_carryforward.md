# Carryforward of Independent Review Findings into Experiment Definition

Source: `reports/checkpoints/phase2c_nlp_provenance_remediation_independent_review_2026-08-09/INDEPENDENT_REVIEW_REPORT.json` (`REVIEW_DECISION = APPROVE_WITH_NONBLOCKING_FINDINGS`).

This document does not edit, rewrite, or reinterpret any historical provenance artifact. It records how this experiment-definition gate addresses the two material non-blocking findings from that review.

## Finding 1 — Provenance wording ("this session" overstatement)

**Original finding:** `evidence_sources` fields for Amazon Appliances, ASTD, LABR, and MPOLD in the remediation session's erratum/ledger/portfolio-decision artifacts read "direct download+hash this session" / "direct local schema/hash re-verification this session," but file mtimes (and the remediation session's own, more careful `acquisition_manifest_v2.json`) show these four datasets were not actually touched during that session — they were carried forward unchanged from an earlier session. The underlying facts (hashes, licenses, row counts) were independently confirmed correct; only the timing claim was wrong.

**How this gate addresses it:** `configs/nlp_experiment_dataset_roles.yaml` and `configs/nlp_experiment_manifest.yaml` describe these four datasets' roles based on their **verified current state** (hash, license, row count as independently re-confirmed by the review), not on when that state was last re-verified. No experiment definition in this gate depends on the "this session" framing being accurate — it depends only on the state being accurate, which the independent review separately confirmed. No historical artifact (the remediation checkpoint, its erratum, or its ledger) is modified by this gate.

## Finding 2 — Duplicate/conflicting-label methodology not bit-exact reproducible

**Original finding:** `quarantine_quality_audit_v2.json`'s duplicate and conflicting-label-duplicate counts for Egyptian Tweets 40K (379/13) and ArSAS (99/31) were not bit-exact reproducible by the independent reviewer using a straightforward normalization (reviewer recompute: 323–382/11–12 for Egyptian Tweets 40K; 99 exact/26 for ArSAS). Row counts, missing-text counts, and label distributions all matched exactly — only the dedup grouping definition was under-specified.

**How this gate addresses it:** This gate does not attempt to guess or retroactively justify the original 379/13/31 figures. Instead it creates `configs/nlp_duplicate_control_contract.yaml`, a new, forward-looking, fully pinned deterministic specification (`NORMALIZED_EXACT_KEY`, fixed pipeline order, fixed tie-breaking rule), and explicitly declares `EGYPTIAN_TWEETS_DUPLICATE_REAUDIT_REQUIRED = YES` and `ARSAS_DUPLICATE_REAUDIT_REQUIRED = YES`. No experiment using either dataset (B1, D1, D2, and the relevant direction of B3) may proceed to a future training gate until a fresh, contract-compliant duplicate count is computed and recorded. This is a stricter posture than simply trusting either the original count or the reviewer's recompute — both are now considered provisional pending the pinned re-audit.

## No historical rewriting performed

This gate did not modify: `reports/generated/nlp/provenance_remediation_erratum.json`, `reports/generated/nlp/quarantine_quality_audit_v2.json`, `reports/checkpoints/phase2c_nlp_provenance_remediation_2026-08-09/*`, or `reports/checkpoints/phase2c_nlp_provenance_remediation_independent_review_2026-08-09/*`. All clarification is additive, in this new document and in the new contracts it references.
