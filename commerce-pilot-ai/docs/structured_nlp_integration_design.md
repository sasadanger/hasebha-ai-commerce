# Structured + NLP Integration Design (design only — nothing implemented)

Status: planning document. No integration is implemented. This document exists to prevent a specific, previously-identified leakage risk from recurring.

## The core separation

**NLP customer intelligence** (sentiment, complaint categorization, product-quality signal, seller-quality signal, refund/cancellation analysis, voice-of-customer themes) and **the late-delivery decision-time model** (`configs/olist_phase2c_target_contract.yaml`, decision timestamp `order_approved_at`) are two structurally different systems with different timing rules, and must never be silently merged.

## The rule

NLP-derived information may only enter a *prediction* model (a model whose output is used before an outcome occurs, such as the late-delivery risk score) if the underlying **text itself was created at or before that model's decision timestamp**. This is the exact same AS-OF discipline already established in `docs/olist_asof_feature_contract.md` and `docs/olist_phase2c_protocol.md`, extended to text.

Concretely, for the late-delivery target:

- **Forbidden:** using a customer's post-delivery review text, post-outcome complaint, or post-outcome support conversation as a *feature* to predict that same order's (or an earlier decision about that order's) delivery outcome. This is already established as `FORBIDDEN_LEAKAGE`/`POST_OUTCOME` for the Olist review table in `docs/olist_asof_feature_contract.md`, and the same rule applies to any future Egyptian text source.
- **Forbidden:** using aggregate customer-sentiment statistics computed from data that includes the order being scored's own post-decision events.
- **Allowed (separately):** post-outcome text may validly support *other*, explicitly separate tasks that are not early-decision predictions: customer satisfaction measurement, complaint categorization, product-quality signal extraction, seller-quality signal extraction, refund/cancellation-reason analysis, and general voice-of-customer intelligence. These are retrospective analysis tasks, not pre-outcome predictions, and carry no AS-OF requirement in the same sense.

## Why this matters now

No NLP-derived feature has ever been proposed for the late-delivery model in this project. This document exists precisely so that a future session does not casually add "customer sentiment" as an input feature without re-deriving this distinction from scratch, and so the mistake is documented before it can happen rather than after.

## Future conceptual architecture (design only, not built)

```text
Customer Text Sources (future, first-party + approved public benchmarks)
        |
PII / Privacy Layer  (redaction, minimization, consent-basis check)
        |
Language / Dialect Detection  (Egyptian Arabic / MSA / English / code-switch / Franco-Arabic)
        |
Task-specific NLP
        |
   Sentiment | Complaint | Intent | Aspect | Topic | Emotion | Safety
        |
NLP Signal Store  (timestamped, so downstream consumers can enforce their own AS-OF rules)
        |
Business Intelligence / Decision Layer
        |
Monitoring
```

Every box above is `DESIGN_ONLY`. None is implemented, deployed, or fed by real data in this session. The critical design property already fixed by this document: the **NLP Signal Store must carry the original text's creation timestamp forward**, so that any future consumer (including a prediction model) can independently enforce the AS-OF rule above rather than trusting the NLP layer to have already done so.

## Relationship to the Decision Action Card contract

Per `docs/phase2c_business_decision_contract.md` and `docs/project_charter.md`, any future NLP-derived signal reaching a Decision Action Card must retain its originating capability, evidence reference, and limitations — an NLP complaint-theme signal is not permitted to silently masquerade as a delivery-risk signal, or vice versa, in a shared card.
