# NLP Training Authorization Requirements

`PHASE2C_AUTHORIZATION_SCOPE = NLP_EXPERIMENT_DEFINITION_ONLY`. This document defines the gate a **future**, separately-authorized `PHASE2C_NLP_TRAINING_AUTHORIZATION_GATE` must pass before any NLP model training, fine-tuning, or embedding generation begins. Nothing in this document itself authorizes training.

## Minimum requirements before any training gate may pass

| # | Requirement | Status at end of this gate | Evidence |
|---|---|---|---|
| 1 | Experiment definition approved | DONE for A, B2, C, E; BLOCKED for B1/D1/D2 pending reaudit; PENDING for F/G/H | `configs/nlp_experiment_manifest.yaml` |
| 2 | Dataset hashes locked | DONE for all 6 active datasets | `reports/generated/nlp/acquisition_manifest_v2.json`, independent-review re-hash confirmation |
| 3 | Duplicate methodology locked | DONE (contract exists); reaudits for Egyptian Tweets 40K and ArSAS still **outstanding** | `configs/nlp_duplicate_control_contract.yaml` |
| 4 | Canonical preprocessing locked | DONE | `configs/nlp_text_normalization_contract.yaml` |
| 5 | Split policy locked | DONE (methodology only, no split materialized) | `configs/nlp_split_policy.yaml` |
| 6 | Label ontology locked | DONE | `configs/nlp_label_ontology.yaml` |
| 7 | Metrics locked | DONE | `configs/nlp_metric_contract.yaml` |
| 8 | Baseline plan locked | DONE (design only) | `docs/nlp_experiment_methodology_policies.md` |
| 9 | Licenses reviewed | PARTIAL — every active dataset's license/rights status is recorded, but several remain `NOT_INDEPENDENTLY_CLEARED` / `NOT_STATED` (ArSAS) / `UNRESOLVED_*_RIGHTS` (LABR, ASTD); full legal review is a prerequisite for anything beyond research use, not yet performed | `configs/nlp_experiment_dataset_roles.yaml`, `reports/generated/nlp/dataset_registry_v2.json` |
| 10 | Commercial vs. research use separated | DONE — every dataset's `commercial_use_status` is distinct from its research-use eligibility | `reports/generated/nlp/dataset_registry_v2.json` |
| 11 | Protected NLP evaluation policy defined | DONE (design only; no set sealed yet) | `configs/nlp_split_policy.yaml#nlp_test_set_governance` |

## Additional, experiment-specific blockers a training gate must check

- `EGYPTIAN_TWEETS_DUPLICATE_REAUDIT_REQUIRED = YES` blocks EXPERIMENT_B1 and the ASTD→Egyptian-Tweets direction of EXPERIMENT_B3.
- `ARSAS_DUPLICATE_REAUDIT_REQUIRED = YES` blocks EXPERIMENT_D1 and EXPERIMENT_D2.
- EESA (Experiment F) and ADAB (Experiment G) remain unacquired; no training gate may authorize them until data is obtained through a confirmed official channel and independently re-audited using the same discipline applied to Egyptian Tweets 40K and ArSAS in this project's history.
- Experiment H remains structurally blocked pending first-party Egyptian commerce data that does not yet exist.

## What a training gate must NOT do even once all rows above read DONE

- Train on the sealed Phase 2A structured Test set, or increase its `access_count` beyond 1.
- Treat any research-only license status (GPLv2 share-alike, unresolved platform-text rights, no-license-stated) as cleared for commercial/production use without a separate legal review step.
- Claim Egyptian-market or Egyptian-commerce readiness from any experiment other than Experiment H.
- Merge datasets outside the `merge_policy` in `configs/nlp_task_dataset_matrix.yaml`.

## Current overall readiness

`NLP_TRAINING_GATE_READINESS = READY_FOR_AUTHORIZATION_REVIEW`

This means the definitional prerequisites (dataset roles, experiment definitions, duplicate contract, normalization contract, split contract, metrics, label ontology all locked; zero closure-critical provenance issues per the independent review) are complete enough to hand off to a `PHASE2C_NLP_TRAINING_AUTHORIZATION_GATE`. It does **not** mean training may start: that future gate must still resolve the two outstanding reaudits (blocking B1, D1, D2, and the corresponding direction of B3) and the full legal license review (row 9) on a per-experiment basis before authorizing any specific run. Experiments A, B2, C, and E carry no currently-known blocker beyond that general legal-review requirement; B1, D1, D2, and the ASTD→Egyptian-Tweets direction of B3 additionally require their reaudits first.
