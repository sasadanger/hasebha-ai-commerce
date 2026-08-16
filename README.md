# HASEBHA | حاسبها

**الداتا تحسبها… وإنت تقرر.**

AI-Powered Commerce Intelligence for E-commerce — a real, running Medusa
commerce platform wired end-to-end to a machine-learning fulfillment-risk
signal and a deterministic decision engine, demonstrated on a live
Egyptian (EGP) storefront.

> Public brand: **HASEBHA | حاسبها**. Internal technical codename:
> **CommercePilot** — retained throughout code, APIs, database metadata
> keys (`commercepilot_ai`), model/experiment identifiers, and checkpoint
> paths, since renaming those would risk breaking the integration this
> project demonstrates and the audit trail behind it. See
> [Brand vs. codename](#brand-vs-codename).

---

## 1. What is HASEBHA?

HASEBHA is an ecommerce order pipeline that treats every placed order as
an event worth scoring. When a customer checks out on the storefront, the
resulting Medusa order triggers a real machine-learning inference call
(a CatBoost fulfillment-risk model trained on the Olist Brazilian
e-commerce dataset), which feeds a small deterministic **Decision
Engine** that turns the risk signal into a priority/action/reason-code
recommendation. The result is persisted on the order itself and surfaced
directly in Medusa Admin next to the order it describes — no separate
dashboard to context-switch into.

## 2. Problem being solved

Ecommerce operations teams generally see order data and fulfillment
problems as two separate systems: the storefront/OMS on one side, and
whatever spreadsheet or BI tool surfaces "risky" orders on the other.
HASEBHA closes that gap for one concrete signal — fulfillment risk — by
making the scoring and the decision happen automatically, synchronously
with the order lifecycle, and visible exactly where an operator is
already looking (the order detail page).

## 3. Key capabilities

- Real-time fulfillment-risk scoring on every order (`order.placed` →
  FastAPI inference call, not a batch job).
- A deterministic, auditable Decision Engine (priority / action / reason
  codes) — not another opaque model.
- Full persistence of the AI result on `order.metadata`, independently
  queryable and idempotent (safe against duplicate event delivery).
- A Medusa Admin widget ("HASEBHA Intelligence") that renders the signal,
  the decision, and the model/audit trail directly on the order page.
- A live, real-data Egyptian storefront demo (EGP pricing, Egypt region,
  Egypt-specific shipping options) proving the whole chain works beyond a
  single hardcoded currency/region.
- Separate, independently evaluated NLP (Arabic offensive/sentiment
  classification) and recommendation (Instacart-style) components,
  exposed by the same FastAPI service, each with its own honestly
  reported evaluation status (see [§18](#18-evaluation-summary)).

## 4. System architecture

```text
Customer
   |
Next.js Storefront (apps/storefront)
   |  cart -> checkout -> place order
Medusa Backend (apps/backend)  ---- PostgreSQL, Redis
   |  order.placed event
HASEBHA / CommercePilot AI Service (src/ai_service, FastAPI)
   |  fulfillment-risk inference (CatBoost, Olist-trained)
   |
Decision Engine (deterministic rules: priority / action / reason_codes)
   |
order.metadata.commercepilot_ai   (persisted via Medusa's Order module)
   |
Medusa Admin
   |
HASEBHA Intelligence widget (order detail page)
```

The AI service is called synchronously from a Medusa subscriber
(`apps/backend/src/subscribers/order-placed.ts`); the metadata write goes
through Medusa's Order module API (`updateOrders`), never a direct
database write. The Admin widget reads the exact same `order.metadata`
field the order-detail page already loads — no extra API route, no
separate data fetch.

## 5. End-to-end flow

1. Customer adds a product to cart and checks out on the storefront
   (`/eg` for the Egyptian/EGP demo, `/gb` for the original EUR demo
   region).
2. Medusa creates the order and emits `order.placed`.
3. The subscriber calls the AI service: `POST /v1/fulfillment/risk`, then
   `POST /v1/decision`.
4. The result — risk score, risk class, decision, reason codes, model
   identifiers, timestamps, an idempotency key — is written to
   `order.metadata.commercepilot_ai`.
5. Opening the order in Medusa Admin shows the **HASEBHA Intelligence**
   widget with that same data, within a few seconds of the subscriber
   completing.

## 6. Technology stack

| Layer | Technology |
|---|---|
| Storefront | Next.js 15 (App Router), official `medusajs/nextjs-starter-medusa` base |
| Commerce backend | Medusa v2 (Node/TypeScript), PostgreSQL, Redis |
| AI service | Python, FastAPI |
| Fulfillment-risk model | CatBoost (trained on Olist Brazilian e-commerce data) |
| NLP components | MARBERT / AraBERT (Arabic offensive-language and sentiment classification) |
| Recommender | Instacart-style hybrid ranking with popularity backfill |
| Infra (local demo) | Docker Compose (Postgres, Redis) |

## 7. AI/ML components

- **Fulfillment-risk model** (`olist-phase2a-strict-core-v1`, CatBoost):
  outputs a continuous `risk_score`. This is a model score, **not a
  calibrated probability** — its ROC-AUC (0.563) and average precision
  (0.079) on the frozen Olist evaluation are modest, and this is stated
  plainly rather than glossed over. A `risk_class` label ("high"/"low")
  is derived from a threshold for display purposes; **that threshold has
  a documented discrepancy that has not been silently resolved** — treat
  `risk_class` as an approximate, rule-based label and `risk_score` as
  the primary signal.
- **NLP (Arabic offensive/sentiment classification)**: MPOLD/ASTD
  classical+transformer finalists, plus a from-scratch Jumia
  Egyptian-domain validation track (JERD). The Jumia 5-star Egyptian
  classifier is **research-only and not promoted to production** —
  `JUMIA_EGYPT_DOMAIN_VALIDATION = PARTIAL`,
  `EGYPTIAN_ECOMMERCE_5STAR_MODEL_READY_FOR_V1 = NO`. See
  [§19 Known limitations](#19-known-limitations).
- **Recommender**: an Instacart-derived hybrid ranking demo (precision@10
  0.288, recall@10 0.340, NDCG@10 0.412) exposed via the same FastAPI
  service but not wired into the live order flow.

## 8. Decision Engine

A small, deterministic, versioned rules layer
(`decision-engine-rules-v1`) — not a machine-learned component. It maps
the fulfillment-risk signal (and other order features) to a
`priority` (e.g. `P4_ROUTINE`, `P1_HIGH`), an `action` (e.g.
`NO_ACTION`), and a list of `reason_codes` explaining why. Being
rule-based and versioned means every decision is explainable and
reproducible from its inputs — a deliberate choice over a second opaque
model on top of the first.

## 9. Medusa integration

- `apps/backend/src/subscribers/order-placed.ts` — the only integration
  point; calls the AI service and persists the result via the Order
  module.
- No direct database writes from the AI service — all persistence goes
  through Medusa's own module APIs.
- Idempotent: duplicate `order.placed` delivery for an already-processed
  order is detected and skipped (verified in prior evidence checkpoints).
- Fails soft: AI-unavailable, AI-timeout, and malformed-response paths
  are all handled without blocking order placement — the order always
  completes; the AI annotation is best-effort.

## 10. Admin Intelligence widget

`apps/backend/src/admin/widgets/commercepilot-ai-order-widget.tsx`
("HASEBHA Intelligence" in the UI) renders on every order detail page and
separates its content into three sections:

1. **AI Risk Signal** — `risk_score` (raw model output) and the
   `risk_class` business-rule label, with an explicit on-screen note that
   `risk_score` is not a calibrated probability and the threshold has a
   documented, unresolved discrepancy.
2. **Decision Engine** — priority, action, reason codes, ruleset version.
3. **Model / Audit** — model/experiment identifier, artifact hash prefix,
   processing status, timestamp, idempotency key.

Its only data source is `order.metadata.commercepilot_ai`, the same field
the order-detail page already loads.

## 11. Egyptian storefront demo

The storefront at `/eg` runs on a real Medusa "Egypt" region (currency
`egp`, country `eg`) with real EGP product prices and two real shipping
options (Standard Delivery 60 EGP, Express Delivery 120 EGP), all backed
by actual Medusa data — no hardcoded storefront-only prices. The original
EUR/`gb` region from earlier development is left fully intact alongside
it. See `/eg` route support was added purely as backend data (a new
region); no router code changes were needed, since the official template
already resolves valid country routes dynamically from `/store/regions`.

## 12. Project structure

```text
ecommerce_medusa/
├── README.md                     <- you are here
├── docs/
│   ├── architecture/              architecture notes
│   └── demo/                      demo guide
├── commerce-pilot-ai/             AI/ML service + research tracks
│   ├── src/ai_service/            FastAPI app (risk, decision, NLP, recommender)
│   ├── src/                       training/eval pipelines, decision engine
│   ├── tests/                     pytest suite
│   ├── configs/                   YAML configuration (no secrets)
│   ├── notebooks/                 exploratory/analysis notebooks
│   └── requirements.txt
└── medusa-app/
    ├── docker-compose.yml         Postgres + Redis for local dev
    └── commercepilot-medusa/
        └── apps/
            ├── backend/            Medusa v2 backend + Admin extension
            └── storefront/         Next.js storefront
```

## 13. Local setup

Prerequisites: Docker, Node.js 20+, Python 3.11+, npm.

```bash
git clone <this-repo-url>
cd ecommerce_medusa

# Python AI service
cd commerce-pilot-ai
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # if you add one; the service is otherwise configured via configs/*.yaml

# Medusa backend
cd ../medusa-app/commercepilot-medusa/apps/backend
cp .env.example .env   # fill in local secrets (see comments in the file)
cd ..
npm install

# Storefront
cd apps/storefront
cp .env.example .env.local   # fill in your local publishable API key after seeding
```

## 14. Environment variables

See `medusa-app/commercepilot-medusa/apps/backend/.env.example` and
`medusa-app/commercepilot-medusa/apps/storefront/.env.example` for the
full annotated list. Nothing in either example file is a real credential
— generate your own local secrets as noted in the file comments. The
`commerce-pilot-ai` FastAPI service is configured via `configs/*.yaml`
rather than environment variables (dataset-path configs with local
overrides are themselves gitignored — see `commerce-pilot-ai/.gitignore`).

## 15. Startup commands

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

Then open:

- Storefront (Egypt/EGP demo): **http://localhost:8000/eg**
- Medusa Admin: **http://localhost:9000/app**
- AI service health: **http://localhost:8123/health**

## 16. Demo walkthrough

See [`docs/demo/DEMO_GUIDE.md`](docs/demo/DEMO_GUIDE.md) for the full
walkthrough (creating an order, where the AI result appears, what each
field means).

## 17. Tests

- Storefront / backend / Admin: `tsc --noEmit` (0 errors as of the last
  verified pass).
- Python: `pytest` from `commerce-pilot-ai/` — 301 tests passed as of the
  last full run recorded in this project's evidence checkpoints; this AI
  service pass relied on that prior green run rather than re-running the
  full ML suite again for a presentation-only change (see
  `commerce-pilot-ai/reports/` locally for the full history — the raw
  per-session checkpoint archive is not published in this repository, see
  [§20](#20-repository-notes)).

## 18. Evaluation summary

| Component | Status | Headline metric |
|---|---|---|
| Fulfillment-risk (Olist CatBoost) | OFFLINE-VALIDATED | ROC-AUC 0.563, AP 0.079 (modest; threshold discrepancy documented, not hidden) |
| MPOLD/ASTD Arabic offensive-language classifiers | OFFLINE-VALIDATED | frozen finalists, see local model cards |
| Jumia Egyptian-domain 5-star classifier | PARTIAL, research-only | protected-test macro-F1 0.374; not promoted for production |
| Instacart-style recommender | OFFLINE-VALIDATED (demo) | precision@10 0.288, recall@10 0.340, NDCG@10 0.412 |
| Medusa ↔ AI ↔ Decision Engine integration | LOCAL-INTEGRATION-VALIDATED | proven end-to-end on real orders, real EGP checkout |

**Standing claim**: OFFLINE-VALIDATED + EGYPTIAN-DOMAIN-VALIDATED (where
supported by evidence, i.e. not for the Jumia 5-star classifier) +
LOCAL-INTEGRATION-VALIDATED. This project does **not** claim to be
production-proven.

## 19. Known limitations

- `risk_score` is a model score, not a calibrated probability — do not
  treat it as one.
- The Olist risk-class threshold has a documented discrepancy that
  remains under review; it has not been silently corrected.
- The Jumia Egyptian 5-star classifier is research-only
  (`EGYPTIAN_ECOMMERCE_5STAR_MODEL_READY_FOR_V1 = NO`) and is not part of
  the live order flow.
- This is a local development demo: single local Postgres/Redis
  instance, in-memory event bus, manual-only payment provider (no real
  payment gateway), no production build/CDN/load testing/auth hardening.
- A harmless artifact from an early seeding attempt remains in the local
  demo database (a second, unused `Default Store`/`Default Sales
  Channel` pair) — documented, not functionally relevant, not fixed
  since it carries no demo-visible effect.

## 20. Future work

- Calibrate the fulfillment-risk score (e.g. isotonic/Platt scaling) and
  resolve the documented threshold discrepancy before any
  production-facing claim about `risk_class`.
- Revisit the Jumia Egyptian-domain classifier with a larger, more
  balanced dataset before considering production promotion.
- Wire the recommender into a live storefront surface (currently
  API-only).
- Add a real payment provider integration for anything beyond local demo
  use.

Future development should branch from `v1.1-dev` or `v2-dev` — the
`v1.0.0-hasebha` tag marks this release and should not be modified
directly.

## Repository notes

The public repository intentionally excludes: `node_modules`, Python
virtual environments, build output (`.next`, `.medusa`, `dist`), ML
caches and large model artifacts, raw/processed dataset contents, and the
internal dated development-session checkpoint archive under
`commerce-pilot-ai/reports/checkpoints/` (kept in the local/private copy
of this project, not published, to keep the public repository readable
rather than a session-by-session log). See `.gitignore` for the complete,
annotated list.

## Brand vs. codename

**HASEBHA | حاسبها** is the public-facing brand: storefront header,
homepage, footer, checkout, and the Admin AI widget title ("HASEBHA
Intelligence"). **CommercePilot** is the internal technical codename kept
in code for compatibility and auditability: the `commercepilot_ai`
order-metadata key, service/package names, model and experiment
identifiers, and historical checkpoint/report paths. Historical
scientific reports are not rewritten to make the HASEBHA brand appear
retroactive — they reflect the state and terminology at the time they
were written.

## License

See `medusa-app/commercepilot-medusa/LICENSE` (MIT, inherited from the
official Medusa Next.js starter template this storefront is built on).
The HASEBHA/CommercePilot-specific code in this repository (AI service,
Decision Engine, Admin widget, subscriber integration) is provided as-is
for demonstration and portfolio purposes.
