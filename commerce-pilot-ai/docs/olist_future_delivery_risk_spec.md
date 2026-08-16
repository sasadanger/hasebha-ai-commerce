# Future Olist Delivery-Risk Specification

## Intended decision

Help an operations reviewer decide which approved order may need proactive fulfillment attention before customer delivery.

## Prediction points

- **Recommended first option:** immediately after confirmed payment approval.
- **Alternative:** confirmed shipment handoff, for a narrower carrier-stage intervention.

## Candidate target

For records with credible customer-delivery and contemporaneous estimated-delivery timestamps, a candidate late-delivery target is whether actual customer delivery occurred after the estimated date. The dataset contains both fields and 7,827 such comparisons were true. Eligibility, cancellations, timestamp anomalies, and the meaning/versioning of the estimate must be resolved before target approval.

## Conceptual field policy

At payment approval, candidate groups are purchase-time context, confirmed approval/payment facts, customer location at permitted granularity, and item/seller attributes demonstrably known by then. At handoff, confirmed handoff time and prior elapsed time may additionally be eligible.

Forbidden fields include actual delivery, reviews, final status, post-prediction cancellation events, and any outcome or measurement created later. Estimated dates are allowed only when their historical-as-of value can be established.

## Split and transfer plan

Use a forward temporal split on the prediction timestamp with non-overlapping train/validation/test periods. Assess seller/customer recurrence across boundaries and avoid random row splitting. Olist's 2016–2018 Brazilian marketplace cannot validate Egyptian carriers, geography, holidays, Arabic operations, currency, regulation, or current store processes. Production work requires timestamped Egyptian order events and documented as-of availability.

## Future Decision Action Card

A future card may contain risk level, evidence available at the prediction point, suggested operational action, prediction time, confidence, and limitations. No risk score or card is produced in Phase 1B.

