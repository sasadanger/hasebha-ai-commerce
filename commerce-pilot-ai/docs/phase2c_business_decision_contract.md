# Phase 2C Business Decision Contract (planning only)

Status: `PLANNING_ONLY`. No Decision Action Card is produced. No business action is triggered. This document defines the contract that a future capability service would need to satisfy, per `docs/project_charter.md` Phase 4 ("Decision Action Card contract") and `docs/olist_future_delivery_risk_spec.md`.

## System-type decision (not yet made)

The fulfillment/delivery capability could be built as a **ranking**, a **classification**, a **risk score**, or a **decision recommendation**. These are not interchangeable and Phase 2C must not conflate them (see `configs/olist_phase2c_evaluation.yaml`).

- `docs/olist_future_delivery_risk_spec.md` describes a "risk level" feeding a "suggested operational action" — this points toward **risk scoring feeding a decision recommendation**, not plain binary classification.
- Confirming this requires operations-stakeholder input on how reviewers actually consume a flagged order (a ranked worklist? a binary alert? a probability displayed alongside evidence?). **Not resolved by this gate.**

## Prediction target vs. business action vs. downstream recommendation (kept distinct)

| Concept | Content | Status |
|---|---|---|
| Prediction target | Calibrated probability that an approved order is delivered later than its recorded estimate | Model exists (Phase 2A CatBoost, Strict core); Expanded-contract status `INCONCLUSIVE_INCREMENTAL_SIGNAL` |
| Business action | What an operations reviewer does with a flagged order (contact carrier, expedite, manual review, no action) | `NOT_YET_DEFINED`; requires an operations/fulfillment stakeholder |
| Downstream recommendation | The Decision Action Card itself: risk level, evidence available at prediction time, suggested action, prediction time, confidence, limitations | `NOT_YET_DEFINED`; contract owned by the shared Decision Action Card API (charter Phase 4), which has not been designed yet |

## Cost inputs (explicitly unknown, not invented)

Per the project charter's success criteria ("Numerical targets will be established only after data validation and stakeholder baseline review; none are assumed in this charter"):

- cost of a false positive (unnecessary operational attention on an order that would have arrived on time): `UNKNOWN_REQUIRES_BUSINESS_INPUT`
- cost of a false negative (a late order that received no proactive attention): `UNKNOWN_REQUIRES_BUSINESS_INPUT`
- operational review capacity (how many flagged orders a team can act on per period): `UNKNOWN_REQUIRES_BUSINESS_INPUT`
- prevented-loss estimate: `UNKNOWN_REQUIRES_BUSINESS_INPUT`

No threshold, no cost-sensitive cutoff, and no ROI claim is set anywhere in Phase 2C planning. These require a named business owner's input before any of Steps 16–17 in the parent gate prompt can be resolved beyond "requires business input."

## Decision Action Card field skeleton (design reference only)

Per `docs/olist_future_delivery_risk_spec.md` and `docs/project_charter.md`, a future card is expected to contain, at minimum: a clearly stated observation, the business decision it informs, supporting evidence, relevant limitations, and a proposed next action — specifically for this capability: risk level, evidence available at the prediction point, suggested operational action, prediction time, confidence, and limitations. **The exact field contract is explicitly deferred to charter Phase 4** and is not designed in Phase 2C beyond this reference skeleton.

## What Phase 2C does not do

Phase 2C does not build a Decision Action Card, does not define a threshold, does not estimate a business cost, and does not claim any operational or commercial value. It only keeps the three concepts (prediction target / business action / downstream recommendation) explicit and un-conflated so that future phases have a clean contract to build against.
