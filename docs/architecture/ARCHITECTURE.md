# HASEBHA Architecture

## Overview

HASEBHA (public brand) / CommercePilot (internal codename) connects a
real Medusa v2 commerce platform to a real machine-learning fulfillment-
risk signal and a deterministic decision layer, with the result surfaced
directly in Medusa Admin.

```text
Customer
   |
Next.js Storefront (apps/storefront)
   |
Medusa Backend (apps/backend)  ---- PostgreSQL, Redis
   |  order.placed event
HASEBHA / CommercePilot AI Service (FastAPI, src/ai_service)
   |
ML Risk Signal (CatBoost, Olist-trained)
   |
Decision Engine (deterministic rules)
   |
order.metadata.commercepilot_ai
   |
Medusa Admin
   |
HASEBHA Intelligence Widget
```

## Components

### Next.js Storefront (`apps/storefront`)

Official `medusajs/nextjs-starter-medusa` base, customized for HASEBHA
branding and an Egyptian (EGP) demo region alongside the original EUR
region. Talks to Medusa exclusively through the Store API (publishable
API key), never directly to the database or the AI service.

### Medusa Backend (`apps/backend`)

Medusa v2 on Node/TypeScript, backed by PostgreSQL (data) and Redis
(event bus / caching). Owns all commerce state: products, regions,
pricing, carts, orders, fulfillment. The only HASEBHA-specific code here
is:

- `src/subscribers/order-placed.ts` — listens for `order.placed`, calls
  the AI service, persists the result via the Order module's
  `updateOrders` (never a direct database write).
- `src/admin/widgets/commercepilot-ai-order-widget.tsx` — renders the
  persisted result on the order detail page ("HASEBHA Intelligence" in
  the UI).

### HASEBHA / CommercePilot AI Service (`commerce-pilot-ai/src/ai_service`)

A FastAPI application exposing:

- `POST /v1/fulfillment/risk` — CatBoost inference, returns a continuous
  `risk_score` and a threshold-derived `risk_class`.
- `POST /v1/decision` — the Decision Engine: deterministic rules mapping
  risk (and other order features) to `priority` / `action` /
  `reason_codes`.
- `POST /v1/nlp/analyze` — Arabic offensive-language / sentiment
  classification (MPOLD/ASTD finalists; a separate research track,
  JERD/Jumia, is evaluated but not promoted to production — see the root
  README's Known Limitations).
- `POST /v1/recommendations` — an Instacart-derived hybrid recommender
  demo, exposed but not wired into the live storefront order flow.
- `GET /health` / `GET /ready` — service health and readiness.

The AI service never writes to the Medusa database. It is a pure
inference boundary: Medusa calls it synchronously, gets a JSON response,
and persists that response itself through its own module API.

### Decision Engine

Not a machine-learned component — a small, versioned, deterministic
rules layer (`decision-engine-rules-v1`) inside the AI service. Given a
`risk_score`/`risk_class` and order features, it returns a `priority`
(e.g. `P4_ROUTINE`, `P1_HIGH`), an `action` (e.g. `NO_ACTION`), and
`reason_codes` explaining the decision. Determinism and versioning make
every decision reproducible and auditable from its recorded inputs.

### Persistence contract

`order.metadata.commercepilot_ai` is the single integration contract
between the AI service and the rest of the platform:

```json
{
  "processing_status": "COMPLETED",
  "idempotency_key": "order.placed:order_...",
  "processed_at": "...",
  "fulfillment_risk": {
    "risk_score": 0.227,
    "risk_class": "high",
    "model_version": "...",
    "model_experiment_id": "olist-phase2a-strict-core-v1",
    "model_artifact_sha256": "..."
  },
  "decision": {
    "action": "NO_ACTION",
    "priority": "P4_ROUTINE",
    "reason_codes": ["NO_TRIGGERING_SIGNAL"],
    "ruleset_version": "decision-engine-rules-v1"
  },
  "feature_source": { "...": "..." }
}
```

This key name is intentionally left unchanged by the HASEBHA brand rename
— it is a compatibility-sensitive metadata contract read by the Admin
widget and referenced by historical evidence checkpoints.

### Infrastructure (local demo)

- **PostgreSQL** — Medusa's primary datastore (Docker Compose, local
  port 5433 to avoid colliding with unrelated local services).
- **Redis** — Medusa's event bus / cache (Docker Compose, local port
  6381).
- **FastAPI** — the AI service, run separately (not orchestrated by
  Docker Compose in the local demo).

None of this reflects a production deployment topology — see the root
README's "Known limitations" for what would need to change for that
(managed Postgres/Redis, a real payment provider, calibration work on
the risk model, horizontal scaling of the AI service, etc.).

## Reliability characteristics

- **Fails soft**: if the AI service is unavailable, times out, or
  returns a malformed response, the subscriber records that outcome
  (`processing_status: AI_UNAVAILABLE` / `AI_TIMEOUT` / etc.) without
  blocking order placement.
- **Idempotent**: duplicate `order.placed` delivery for an
  already-processed order is detected via the idempotency key and
  skipped rather than reprocessed.
- **No direct database writes from the AI service** — every persisted
  result goes through Medusa's own module API, preserving Medusa's
  normal data-integrity guarantees.
