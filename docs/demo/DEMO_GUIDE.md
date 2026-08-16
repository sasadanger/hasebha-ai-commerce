# HASEBHA (حاسبها) — Demo Guide

**Public brand:** HASEBHA | حاسبها — "AI-Powered Commerce Intelligence"
**Internal technical codename:** CommercePilot (unchanged in code, APIs,
metadata keys, model identifiers, experiment IDs, and artifact paths —
see the root README's "Brand vs. codename" section).

This is the current, live presentation documentation. Prior dated
development-session checkpoints (including the earlier BE3LY brand
iteration) are preserved locally as historical record and are not
rewritten to make this brand appear retroactive.

---

## 1. Startup commands

```bash
# 1. Postgres + Redis
cd medusa-app
docker compose up -d

# 2. AI service (port 8123)
cd ../commerce-pilot-ai
uvicorn src.ai_service.main:app --port 8123

# 3. Medusa backend + Admin (port 9000)
cd ../medusa-app/commercepilot-medusa
npm run backend:dev

# 4. Storefront (port 8000)
cd apps/storefront
npx next dev -p 8000
```

If region/currency/collection data changes on the backend while the
storefront dev server is already running, its on-disk Next.js Data Cache
(`apps/storefront/.next/cache/fetch-cache`) can serve stale results.
Clear that directory and restart `next dev` after backend data changes.

## 2. URLs

| What | URL |
|---|---|
| **Storefront (Egypt / EGP / HASEBHA)** | **http://localhost:8000/eg** |
| Storefront (Europe / EUR demo region) | http://localhost:8000/gb |
| Medusa Admin | http://localhost:9000/app |
| AI service health | http://localhost:8123/health |

Demo Admin login email: `admin@commercepilot.local`. The password is not
stored in this repository or any report — it is a local-only credential
known to the operator running the demo.

## 3. How to create a demo order

1. Open http://localhost:8000/eg.
2. Pick a product, choose a variant, add to cart.
3. Go to cart → checkout, fill in an Egyptian address (country: Egypt).
4. Choose Standard Delivery or Express Delivery.
5. Select the manual payment method (the only configured provider —
   suitable for a local demo, not a real payment gateway) and place the
   order.
6. Open Medusa Admin → Orders → the new order. The **HASEBHA
   Intelligence** panel appears in the order detail sidebar within a few
   seconds (the `order.placed` subscriber processes asynchronously).

## 4. What the risk score and Decision Engine mean

- `risk_score` is the raw output of the frozen Olist-trained CatBoost
  fulfillment-risk model (`olist-phase2a-strict-core-v1`). It is **not**
  a calibrated probability — its ROC-AUC (0.563) and average precision
  (0.079) on the frozen Olist evaluation are modest. `risk_class` (e.g.
  "high") is a threshold applied to that score for display purposes; the
  threshold has a **documented discrepancy** that has not been silently
  corrected — treat `risk_class` as an approximate, rule-based label, not
  a scientific classification.
- The **Decision Engine** is a separate, deterministic rules layer
  (`decision-engine-rules-v1`) mapping the risk signal and other order
  features to a `priority` / `action` / `reason_codes` triple. It is not
  a machine-learned component.

## 5. Known limitations

- **Jumia 5-star Egyptian-domain classifier is research-only.**
  `JUMIA_EGYPT_DOMAIN_VALIDATION = PARTIAL`,
  `EGYPTIAN_ECOMMERCE_5STAR_MODEL_READY_FOR_V1 = NO`. It is not wired
  into the live order flow and is not promoted to production.
- The Olist fulfillment-risk model has a documented risk-class threshold
  discrepancy (not silently fixed); `risk_score` is the primary,
  more-trustworthy output.
- This is a local development demo, not a production deployment: single
  local Postgres/Redis instance, in-memory event bus, manual-only
  payment provider (no real payment gateway), no CDN/production
  build/load testing/production auth hardening.
- A harmless artifact from an early seeding attempt remains in the local
  demo database (a second, unused `Default Store`/`Default Sales
  Channel` pair) — documented, not functionally relevant.
- No claim of "production proven" is made anywhere in this project. The
  standing claim is: **OFFLINE-VALIDATED + EGYPTIAN-DOMAIN-VALIDATED
  (where supported by evidence) + LOCAL-INTEGRATION-VALIDATED.**

## 6. Evidence

Two orders demonstrate the fully proven end-to-end flow
(Storefront → Cart → Checkout → Medusa Order → AI inference → Decision
Engine → `order.metadata` → Admin widget, each with
`processing_status: COMPLETED`): an EUR order on the original `/gb`
region, and an EGP order on the `/eg` region proving the Egyptian
configuration independently. Exact order IDs and full verification
detail are preserved in this project's local (unpublished) evidence
checkpoints.
