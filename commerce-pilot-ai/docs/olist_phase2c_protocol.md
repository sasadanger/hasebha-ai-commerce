# Olist Phase 2C Protocol (planning only)

Status: `PLANNING_ONLY`. This document defines protocol that will govern Phase 2C execution if authorized. No model was trained, no prediction was generated, and no protected data was accessed to produce this document.

## Structured feature eligibility taxonomy

Every candidate feature for Phase 2C must be assigned exactly one of the following operational-availability states before it may be used as model input. This extends, and does not replace, the existing classifications in `docs/olist_asof_feature_contract.md` and `configs/olist_feature_contract_v1.yaml`.

| State | Meaning |
|---|---|
| `PROVEN_PRIMARY_ASOF` | Source-system evidence proves the value was fixed, correct, and retrievable at the decision timestamp, with no possibility of later mutation or backfill. |
| `UNVERIFIED_ASOF` | Present in the historical export but approval-time availability is not proven. Current status of all 15 Expanded features. |
| `RETROSPECTIVE_ONLY` | Eligible for offline/retrospective sensitivity study but not eligible as a production feature (`CONDITIONALLY_ALLOWED_RETROSPECTIVE_SENSITIVITY`, per `reports/generated/olist/phase2b/feature_availability_audit.json`). |
| `FORBIDDEN_LEAKAGE` | Structurally post-decision or post-outcome (e.g. delivery timestamp, review table, final order status). Never eligible regardless of future verification. |
| `IDENTIFIER_ONLY` | Join/audit key; never a predictive input (e.g. `order_id`, `customer_id`, `product_id`, `seller_id`). |
| `POST_OUTCOME` | Recorded only after the label-generating event occurs; subset of `FORBIDDEN_LEAKAGE` called out separately because it is often mistaken for a feature (e.g. review score, review text, cancellation reason logged post-decision). |
| `UNKNOWN` | Not yet classified. No feature may be used in `UNKNOWN` state. |

For any feature proposed for Phase 2C model input, the audit must record:

- source table/system
- event timestamp (when the fact became true in the real world)
- ingestion timestamp, if the source system logs it separately from the event
- decision timestamp it is being evaluated against (`order_approved_at` per `configs/olist_phase2c_target_contract.yaml`)
- latest safe cutoff (the last moment the value could be read without leakage)
- availability delay (event timestamp to safe-read timestamp)
- null behavior (rate and cause of missingness)
- revision/backfill behavior (can the source system change this value after the fact?)
- whether the value could plausibly change after the decision time
- leakage risk classification
- production availability evidence (not just historical-export presence)

This is the same audit discipline already partially applied in `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json` and `docs/olist_asof_feature_contract.md`; Phase 2C's job is to complete it to the point of a `PROVEN_PRIMARY_ASOF` or definitive non-approval determination for every one of the 15 Expanded features and the outstanding `CONDITIONALLY_ALLOWED`/`REQUIRES_VERIFICATION` entries (`shipping_limit_date`, payment-record completeness, catalog snapshot completeness).

## Leakage control (structured and text)

Prohibited in Phase 2C, without exception:

- post-outcome fields (`order_delivered_carrier_date`, `order_delivered_customer_date`, final `order_status`, cancellation events occurring after the decision timestamp)
- refund/return outcome data recorded before the decision but describing a later event
- delivery-completion data of any kind before the decision timestamp
- review text written after the outcome, used to predict that same or an earlier outcome (structurally true for the entire Olist review table per `docs/olist_asof_feature_contract.md`)
- customer-support or other text occurring after the decision timestamp
- aggregate statistics (counts, sums, means) computed using rows/events with timestamps after the decision timestamp for the order being scored
- target encoding, mean encoding, or any statistic fitted outside the training fold
- text embeddings/vectorizers/topic models fitted using validation or test text
- global normalization/scaling fitted across the full dataset instead of per training fold
- retrospective joins performed without an explicit timestamp constraint tying the joined value to a moment at or before the decision timestamp

## Temporal validation contract

Phase 2C inherits, and does not loosen, the fold construction already frozen in `docs/olist_temporal_split_spec.md` and `configs/olist_phase2b_sensitivity.yaml`:

- Fold construction: fixed expanding-window development folds (`src/modeling/olist/temporal_validation.py::fold_indices`), forward temporal split on `order_approved_at`.
- Train period strictly precedes validation period for every fold; overlap raises `ValueError` (existing enforced behavior).
- Gap: none currently enforced beyond strict precedence; whether a buffer gap is needed is a Phase 2C open question tied to feature revision/backfill risk discovered during the AS-OF audit.
- Customer/seller/product overlap policy: not currently constrained across fold boundaries; Phase 2C must assess and document seller/customer recurrence across boundaries, per `docs/olist_future_delivery_risk_spec.md`'s "Split and transfer plan."
- Time-based leakage checks: reuse `src/modeling/olist/temporal_validation.py::validate_paired_predictions` and extend equivalent checks to any new feature/text source.
- Feature cutoff enforcement: every feature's "latest safe cutoff" (see taxonomy above) must be enforced programmatically, not just documented.
- Text cutoff enforcement: any future text feature must be filtered to `text_creation_timestamp <= decision_timestamp` before use.
- Reproducibility requirement: same discipline as `src/modeling/olist/phase2b_reporting.py`/`reproduce()` — deterministic seeds, environment capture, prediction-hash verification.
- **No future Test data is selected, previewed, or accessed by this planning document.**

## Protected Test policy for Phase 2C and beyond

The consumed Phase 2A Test set (`reports/generated/olist/phase2a/test_access_ledger.json`, status `CONSUMED`, `access_count = 1`) **remains sealed**. Phase 2C must not reuse it for tuning, comparison, threshold selection, feature engineering, or NLP decisions of any kind.

Phase 2C requires, in order:

1. **Development-only evidence** for the AS-OF audit and feature-contract hardening (no Test access needed at all — this is a documentation/source-system audit, not a model evaluation).
2. If AS-OF proof succeeds for some Expanded features, **a new validation protocol** using development folds only (existing development predictions or a newly authorized, separately gated retraining run — not performed by this document) to re-establish whether the AS-OF-proven subset shows incremental value under the same paired-bootstrap/AS-OF-gated methodology as Phase 2B.
3. Only after (1) and (2) both succeed, and only through a **separately authorized governance process** (not created, opened, or accessed by this document), would **a new, independently sealed Test set** be considered for a final confirmatory evaluation. That governance process must define: who authorizes access, how `access_count` is incremented and audited, how the new Test set's provenance is proven independent of any development/tuning decision, and how the resulting ledger update is itself independently reviewed — mirroring the existing Phase 2A ledger discipline.

## Governance artifacts required for any Phase 2C execution sub-step

- experiment ID
- dataset hash
- code hash
- config hash
- dependency snapshot
- seed control record
- split manifest
- feature manifest
- prediction manifest
- metric definitions
- model card
- evaluation report
- leakage audit
- AS-OF audit (the primary Phase 2C deliverable)
- NLP data card (if/when the NLP track executes)
- Egyptian-market evidence statement (see `docs/phase2c_egyptian_market_evidence_requirements.md`)
- reproducibility report
- protected-test ledger (read-only reference; not mutated except through the separate governance process above)
- evidence precedence declaration, if any correction to Phase 2C evidence is later needed (same pattern as `reports/generated/olist/phase2b/correction_v3/evidence_precedence.json`)

## Production architecture boundary

```text
Data ingestion            -> DESIGN_ONLY
Feature computation        -> IMPLEMENT_IN_PHASE2C (AS-OF audit + contract hardening only; not a production pipeline)
NLP processing              -> DEFERRED
Feature store / assembly   -> DEFERRED
Predictive model            -> DEFERRED (no retraining authorized by this gate)
Calibration                 -> DEFERRED
Decision rules               -> DEFERRED
API                          -> OUT_OF_SCOPE (charter Phase 5)
Monitoring                   -> DESIGN_ONLY
Feedback loop                 -> OUT_OF_SCOPE (charter Phase 5/6)
```

Phase 2C may design the shape of feature computation and monitoring for future phases but does not build, deploy, or operate any of them.

## LLM / agent boundary

Decision: an LLM/agent layer is **DEFERRED**, not in-scope and not out-of-scope permanently. CommercePilot AI's charter anticipates a future admin dashboard consuming Decision Action Cards; an LLM could plausibly assist with several roles, each requiring separate future evaluation:

- explanation (describing why a score/evidence pattern was flagged) — lowest risk, most plausible near-term future role
- summarization (condensing evidence/limitations for a reviewer) — plausible future role
- recommendation drafting (proposing the "next action" text of a Decision Action Card) — requires human review per charter non-goals
- customer-feedback classification (a possible NLP-track role, not a delivery-risk role) — deferred with the NLP track
- structured extraction (e.g. pulling fields from free text) — deferred with the NLP track
- analyst interface (natural-language query over evidence) — future, undesigned
- autonomous action (an LLM initiating or executing a business action without human review) — **explicitly excluded from any current or near-term phase**; the charter's non-goals state "Fully autonomous business decisions or actions without accountable human review" are out of scope for the whole project, not just Phase 2C. Any future autonomous-action proposal requires a separate, dedicated safety/governance gate not created by this document.

No LLM/agent component is built, called, or evaluated by this planning gate.
