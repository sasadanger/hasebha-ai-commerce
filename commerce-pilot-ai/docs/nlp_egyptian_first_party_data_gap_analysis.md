# NLP Egyptian First-Party Data Gap Analysis

Status: planning document. **No first-party Egyptian customer-language data currently exists in this project.** This document identifies what the 20 investigated public/commercial candidates cannot provide, and specifies the desired schema for future data streams. No data described below is claimed to exist.

## What public datasets cannot provide (as demonstrated by this session's investigation)

- **No dataset investigated is a verified, high-provenance, Egyptian e-commerce customer-service corpus.** The closest domain matches (ADAB: e-commerce + customer service, multi-dialect including Egyptian, but Egypt is only one of four dialect groups; JERD: genuinely Egyptian e-commerce/Jumia, but LEVEL_5 provenance) both fall short of a first-party-quality resource.
- **No dataset investigated contains WhatsApp support data, live chat, or call-center transcripts from an Egyptian e-commerce operation.** The closest resources found are paid commercial vendor products (FutureBeeAI, Macgence) offering Egyptian Arabic retail/e-commerce call-center speech — these are real but require a commercial licensing relationship, not something this project currently has.
- **No dataset investigated contains return/refund/cancellation/COD-refusal reason codes tied to real Egyptian orders.** These concepts do not exist as annotated categories anywhere in the portfolio (`docs/nlp_label_ontology_mapping.md` — `COMPLAINT`, `INTENT` remain `NO_MAPPING` gaps project-wide).
- **No dataset investigated contains verified Franco-Arabic / Arabizi (Latin-script Egyptian Arabic) text at scale** with commerce relevance.
- **No dataset investigated links customer text to an order, product, or seller identifier under a documented consent basis** — the closest (JERD) is unverified in provenance and consent basis is unknown.

## Desired future first-party Egyptian data streams

For each stream below: **does not currently exist**; schema is specified for future collection design only.

### WhatsApp / live chat support

| Field | Description |
|---|---|
| `text` | Message content |
| `event_timestamp` | When the message was sent |
| `ingestion_timestamp` | When the system recorded it |
| `channel` | WhatsApp / live chat / other |
| `customer_pseudonymous_id` | Non-reversible customer identifier |
| `order_id` | If lawful and consented |
| `product_id` | If lawful and consented |
| `seller_id` | If lawful and consented |
| `reason_code` | Structured reason taxonomy (to be designed) |
| `human_outcome_label` | Resolution outcome as recorded by a human agent |
| `pii_redaction_state` | Whether/how PII was redacted |
| `consent_legal_basis` | Documented lawful basis for processing |

### Call-center transcripts

Same schema as above, plus: call duration, agent ID (pseudonymous), call disposition code, language/dialect tag if manually verified.

### Return / refund / cancellation reason text

Same core schema, plus: `return_reason_category` (structured), `refund_amount` (if lawful to retain), `cancellation_initiator` (customer/seller/system).

### Courier / delivery exception feedback

Same core schema, plus: `courier_id` (pseudonymous), `delivery_attempt_number`, `exception_type` (structured taxonomy to be designed).

### Product-quality / merchant-escalation complaints

Same core schema, plus: `product_id`, `seller_id`, `escalation_tier`.

## Explicit non-claim

This document does not claim any of the above data streams currently exist, are being collected, or are approved for collection. It defines what would be required if and when Egyptian first-party operational data becomes available, consistent with `docs/phase2c_egyptian_market_evidence_requirements.md` from the prior Phase 2C gate.
