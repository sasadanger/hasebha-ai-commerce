# Phase 2C Egyptian-Market Evidence Requirements (planning only)

Status: `PLANNING_ONLY`. No Egyptian data is collected, fabricated, or substituted by this document. No Egyptian performance is inferred from Brazilian Olist results.

## Boundary statement (required)

**Proven on Olist:** structured late-delivery ranking signal exists in development-fold data for Logistic Regression, CatBoost, and LightGBM under the retrospective Expanded feature contract (`INCONCLUSIVE_INCREMENTAL_SIGNAL` — statistically present, operationally unproven). Nothing about Egypt is proven or disproven by this.

**Unproven for Egypt:** everything. Per `docs/olist_future_delivery_risk_spec.md`: "Olist's 2016–2018 Brazilian marketplace cannot validate Egyptian carriers, geography, holidays, Arabic operations, currency, regulation, or current store processes. Production work requires timestamped Egyptian order events and documented as-of availability." This remains fully true after Phase 2B/2C. Any future claim of Egyptian readiness requires direct Egyptian evidence, not an inference from Olist or Amazon performance.

## Dataset role classification

| Source | Role | Notes |
|---|---|---|
| Olist | `DEVELOPMENT_BENCHMARK` | Brazilian marketplace, 2016–2018. Used to develop and sensitivity-test methodology only. Must never be silently reclassified as Egyptian external validation. |
| Amazon Reviews 2023 (Appliances) | `DEVELOPMENT_BENCHMARK` | English-language, US marketplace. Voice-of-customer methodology development only; not Egyptian, not Arabic. |
| Instacart | `DEVELOPMENT_BENCHMARK` | US grocery marketplace. Personalization methodology development only; unrelated to this Phase 2C track. |
| Egyptian order/customer data | `EXTERNAL_VALIDATION` (once it exists) | Does not currently exist in this repository. Required before any Egyptian-market claim. |
| A future independently sealed Egyptian Test-equivalent set | `PRODUCTION_MONITORING` precursor / `EXTERNAL_VALIDATION` | Governance process not yet defined; see protected-Test policy in `docs/olist_phase2c_protocol.md`. |

Olist remains `DEVELOPMENT_BENCHMARK` throughout Phase 2C. It is never promoted to `INTERNAL_VALIDATION`, `EXTERNAL_VALIDATION`, or `PRODUCTION_MONITORING` for an Egyptian claim.

## Requirement classification

| Item | Classification | Rationale |
|---|---|---|
| Egyptian order lifecycle timestamps (purchase, approval, handoff, delivery, estimate) with the same as-of rigor as Olist | `REQUIRED` | Cannot repeat the AS-OF verification exercise, let alone train, without them. |
| Cash-on-delivery behavior | `REQUIRED` | Materially different failure/risk mode than the Olist card/boleto-dominant market; likely a first-order predictor in Egypt. |
| Failed delivery / delivery attempt records | `REQUIRED` | Egypt-specific delivery-attempt patterns are not represented in Olist at all. |
| Cancellation | `REQUIRED` | Needed to replicate the same exclusion-waterfall discipline as `docs/olist_target_eligibility_policy.md`. |
| Return/refund | `USEFUL` | Relevant to a broader customer-experience signal but not required for the narrow late-delivery target. |
| Governorate (geography) | `REQUIRED` | Direct analogue of Olist customer/seller state; needed for any segment or leakage analysis. |
| Courier/carrier identity | `REQUIRED` | Egypt's carrier landscape differs entirely from Brazilian carriers; cannot be inferred from Olist. |
| Payment method | `REQUIRED` | Cash-on-delivery prevalence materially changes the risk model. |
| Delivery SLA / promised window | `REQUIRED` | Direct analogue of the disputed `order_estimated_delivery_date` field; its AS-OF status must be proven for Egypt independently — Olist's resolution does not transfer. |
| Product category | `USEFUL` | Useful segment; not required for the minimal target replication. |
| Seller behavior / seller diversity | `USEFUL` | Relevant to marketplace vs. D2C distinction below. |
| Arabic customer feedback (reviews) | `USEFUL` for a future voice-of-customer track; `NOT_CURRENTLY_JUSTIFIED` for the delivery-risk target, mirroring the Olist review-table leakage prohibition. |
| Egyptian Arabic complaints (support channel text) | `USEFUL` for voice-of-customer; same leakage caveat as above if post-outcome. |
| WhatsApp/support interactions | `OPTIONAL`, contingent on legal/consent clearance; `NOT_CURRENTLY_JUSTIFIED` without a documented lawful basis and data-governance review. |
| Seasonality (general) | `USEFUL` | Standard temporal-validation consideration. |
| Ramadan/Eid effects | `REQUIRED` if the deployment market has meaningful order-volume/delivery-capacity swings around these periods | Absent from Olist entirely; a Brazil-trained model cannot be assumed to generalize to Egyptian religious/calendar seasonality. |
| Inflation/price changes | `USEFUL` | Relevant to payment-related features; not required for the minimal replication. |
| Promotions | `USEFUL` | Same as above. |
| Marketplace vs. D2C differences | `REQUIRED` | Changes seller-behavior semantics materially; must be documented before any seller-level feature is reused. |
| Privacy/legal constraints | `REQUIRED` | Per `docs/risk_register.md` privacy row; must be resolved before any customer-level Egyptian data is used, exactly as Olist's own customer/seller/location fields already require access minimization. |
| Consent/data governance | `REQUIRED` | Same basis; no Egyptian data may be collected or used without a documented lawful basis. |

## Minimum viable evidence (structure, not fabricated numbers)

- **Minimum schema:** order identity, purchase timestamp, approval-equivalent timestamp, delivery-promise timestamp (with proven as-of provenance), actual delivery timestamp or definitive non-delivery status, governorate, courier/carrier identity, payment method, cancellation status.
- **Required timestamps:** every field above must carry both an event timestamp and evidence of when it was knowable to the system (mirroring the taxonomy in `docs/olist_phase2c_protocol.md`).
- **Label definition:** must be independently defined for the Egyptian delivery-promise semantics; the Olist formula (`order_delivered_customer_date > order_estimated_delivery_date`) must not be assumed to transfer without verifying the Egyptian promise field's own as-of provenance.
- **Minimum observation period:** not stated as a fixed number here. **Sample-size/statistical-power analysis is required before execution** to justify a minimum observation period and minimum positive-class event count for the target prevalence actually observed in Egyptian data; no number is invented in this document.
- **Minimum event count:** same — requires a power analysis once true prevalence is known; not assumed to match Olist's ~8.1% late-delivery prevalence.
- **Class balance considerations:** must be assessed from real Egyptian prevalence, not assumed from Olist.
- **Required text coverage if NLP is evaluated:** to be defined jointly with `configs/olist_phase2c_nlp_contract.yaml` once Arabic text data and a language/dialect policy exist.
- **Anonymization requirements:** must meet or exceed the access-minimization treatment already applied to Olist customer/seller/location fields (`docs/olist_data_understanding.md`).
- **Leakage exclusions:** identical structural principle as `docs/olist_asof_feature_contract.md` — post-outcome fields, post-decision events, and any review/complaint text created after the decision timestamp are forbidden as features regardless of language or market.
- **Temporal validation requirement:** forward-chronological split on the Egyptian decision timestamp, no random row splitting, consistent with `docs/olist_temporal_split_spec.md`'s existing discipline.
- **Geographic coverage expectations:** must span enough governorates to support subgroup evaluation; exact minimum requires stakeholder/business input.
- **Channel coverage expectations:** must reflect the actual mix of marketplace vs. D2C, cash-on-delivery vs. prepaid, that the target deployment will see; not assumed.
- **Merchant/business diversity expectations:** enough seller/merchant diversity to avoid a model that only reflects a handful of large sellers.
- **Concept-drift considerations:** Egyptian macro conditions (inflation, promotions, courier capacity) change faster than the 2016–2018 Olist window; a monitoring/re-validation cadence must be defined before any production claim.
- **External-validation criteria:** a held-out Egyptian evaluation set, independent of any development/tuning decision, governed by the same ledger discipline as the Phase 2A Test set, before any Egyptian-market readiness claim is made.

## Explicit non-claim

This document does not claim, and Phase 2C must not claim, that Egyptian-market readiness, production readiness, or deployment readiness exists or is imminent. It defines what evidence would be required if and when Egyptian data becomes available.
