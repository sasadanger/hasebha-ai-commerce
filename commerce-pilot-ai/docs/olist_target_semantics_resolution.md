# Olist Target Semantics Resolution

Decision: `VALID_RETROSPECTIVE_PROMISE_BREACH_TARGET_WITH_UNVERIFIED_ASOF_SEMANTICS`.

The scientifically permitted claim is:

> Retrospective offline classification of whether a delivered order exceeded the estimated delivery date stored in the Olist dataset.

The orders Parquet contains typed `order_delivered_customer_date` and `order_estimated_delivery_date`. After the documented eligibility filter, the target is `actual > estimate`; equality is negative. The cohort has zero missing target components. Tests cover a positive example, equality, and missing-component exclusion.

Future outcome information is valid for constructing a supervised label when it is never an input. Actual delivery and stored estimate are target-only. Reviews, status, handoff, target derivatives, and future aggregates are forbidden inputs.

This decision does not verify that the stored estimate was customer-facing, unchanged, or available at approval. Therefore “approval-time promise breach,” promised lead time, or operational promise-management claims remain unsupported. The confirmed approval timestamp remains the cutoff for strict inputs; it names when prediction is hypothetically evaluated, not proof that every final-export field existed then.

