# CommercePilot / HASEBHA — Project Handover

Verified: 2026-08-22. This is the single authoritative delivery document. Read this first;
everything else is referenced from here. Two tracks below — academic and production — are
kept strictly separate; nothing in Track A implies production readiness, and nothing in
Track B implies committee-defense readiness.

---

# TRACK A — ACADEMIC DELIVERY (Committee Defense)

## A1. What is finished and verified

| Item | Status | Artifact |
|---|---|---|
| Arabic sentiment (MPOLD/ASTD/LABR), MARBERTv2 champion | FROZEN, verified | `reports/generated/arabic_sota/ARABIC_FINAL_DECISION.json` |
| Amazon Appliances sentiment, TF-IDF+LinearSVC champion | FROZEN, verified (unwired — see B2 gap #7) | `reports/generated/amazon/metrics.json` |
| Instacart recommendation, hybrid+popularity | FROZEN, verified, protected-test single-shot | `reports/generated/instacart/protected_test_final_results.json` |
| Olist V1 (production model) | LIVE, 5 real orders scored | `reports/generated/olist/phase2a/final_test_metrics.json` + live DB |
| Olist V2 (regime-shift stress benchmark) | RESEARCH, verified | `reports/generated/olist_v2/FINAL_SCORECARD.json` |
| Olist V3 Seller-SLA research (23-feature, AUC 0.7702) | RESEARCH, leakage-tested | `reports/generated/olist_v3_multistage/SELLER_SLA_TEMPORAL_EVAL.json` |
| Olist V3 Customer T0/T1 | RESEARCH, verified | `reports/generated/olist_v3_multistage/TASK_B_C_RESULTS.json` |
| HASEBHA production-parity model (AUC 0.5551, WEAK, shadow-wired) | RESEARCH-TRAINED, never executed on a real order | `reports/generated/olist_v3_multistage/PRODUCTION_PARITY_MODEL_COMPARISON.json` |
| DataCo/EAGLE reproduction (failed, diagnosed) | RESEARCH, verified | `reports/generated/dataco/DATACO_LSTM_REPRODUCTION.json` |
| Forensic ablation study (0.7686→0.5540, 94.1% seller-history attribution) | VERIFIED, independently spot-checked | `reports/generated/olist_v3_multistage/forensics/FORENSIC_EXPERIMENT_RESULTS.json` |
| Marketing Funnel enrichment (negative result, NR-16) | VERIFIED | `reports/generated/olist_funnel/OLIST_FUNNEL_SCORECARD.json` |
| **16 negative results register** | COMPLETE | `reports/generated/committee_defense/NEGATIVE_RESULTS_REGISTER.json` (`count: 16`) |
| **451/451 tests passing** (full repo suite, last measured before this session's non-Python-affecting changes) | VERIFIED | prior session's own pytest run from D:'s venv |
| **D1-D6 committee package** | **LOCKED FINAL** | `docs/committee/` (see A2/A3 below) |
| C:→D: migration record | VERIFIED (hash-identical, HEAD-identical) | `docs/MASTER_EXECUTION_STATUS.md` |

All of the above are RESEARCH or FROZEN claims about public-dataset/general-benchmark
performance, or about the ONE live production model (Olist V1). None of them constitute a
claim of validated HASEBHA-native fulfillment prediction — see Track B and D5-Q2/Q15.

## A2. What remains before walking into the room

1. **Notebook 06** (`commerce-pilot-ai/notebooks/06_transfer_gap_and_funnel.ipynb`) — **DONE,
   not a pending item.** Built and fully executed this session: 16.3 seconds end-to-end, zero
   errors, all displayed numbers (4 spot-checked univariate AUCs, the ablation table, the
   94.1% seller-history attribution, the funnel coverage/delta numbers, the D5-Q15 sentence)
   independently verified to match their source artifacts exactly. Re-verified present on
   disk and executable as of this handover — nothing further required unless the presenter
   wants to re-run it live in the room (it is fast enough to do so, ~16s).
2. **D2 markdown → an actual slide deck.** `docs/committee/D2_DEFENSE_SLIDES.md` is
   structured, presentation-ready markdown (22 slides, clear breaks, speaker notes) — it
   converts to PowerPoint/Google Slides/Marp in well under an hour via copy-paste or any
   markdown-to-slides tool. This conversion was deliberately left to the presenter (a visual
   design and institution-template judgment call, not a factual one) — see
   `docs/DECISIONS_FINAL.md` Item 2/A3 for the reasoning.
3. **D6 rehearsal.** `docs/committee/D6_DEMO_SCRIPT.md` is written and includes two fallback
   paths (no live services / question derails the script) — a live run-through by the
   presenter is the only remaining step, not an engineering task.

## A3. Delivery pointers

- **5-minute script**: `docs/committee/D6_DEMO_SCRIPT.md`.
- **The one-sentence claim (D5-Q15)**, verbatim: *"We built and rigorously validated several
  strong research/production models on public and recommendation data, and for the one
  capability where we tested transfer to our real production environment, we proved —
  quantitatively, not by assumption — that the public-data signal does not survive real
  feature-availability constraints, and we specified exactly what data and business decision
  are needed to close that gap."*

---

# TRACK B — PRODUCTION DEPLOYMENT (Store Going Live)

## B1. What is deployed and working today (verified, not restated from the prompt)

Verified via `git log --oneline --all`: commit `99a4431` — **"Deploy HASEBHA v1 to production
(Railway + Vercel)"** — exists in this repository's real commit history, one commit before
the current HEAD (`1a84602`, "Security Release Gate: remediate dependency CVEs and add AI
service authentication"). This confirms a real deployment event happened; this session did
not independently re-verify the live Railway/Vercel URLs are currently reachable (out of
scope — no network credentials for those platforms available here), so "deployed" below means
"a deployment to these platforms is recorded in git history and the code that would have been
deployed is present and tested," not "confirmed currently serving traffic right now."

- **3 live models**: Olist V1 (fulfillment risk), Arabic sentiment (MARBERTv2), Instacart
  recommendation — all wired into the FastAPI `ai_service` and, per the live database query
  performed in a prior session this chain, Olist V1 has genuinely scored real orders through
  the actual Medusa `order.placed` → FastAPI → Decision Engine → `order.metadata` path.
- **5 real orders scored** — verified by direct SQL query against the live PostgreSQL
  database in a prior session (`SELECT count(*) FROM "order"` = 5, all carrying real
  `commercepilot_ai` prediction metadata, including one correctly-logged `AI_UNAVAILABLE`
  failure proving the fail-soft path genuinely executed).
- **0 fulfillment outcomes** — same database, `SELECT count(*) FROM fulfillment` = 0. No
  order has ever shipped in this environment to date.

**Framing, stated plainly**: this is "deployed once, manually, successfully" — a real,
working deployment event with real (if minimal) production execution history — not a
professional, repeatable, monitored deployment pipeline. Both things are true at once.

## B2. The 8 gaps — ranked, with justification

Ranked by this session's own judgment, as the final advisor for this decision:

### MUST close before real customer traffic at any meaningful volume
1. **No outcome-monitoring/alerting on the live AI integration.** Today, an `AI_UNAVAILABLE`
   or `AI_REJECTED_INVALID_FEATURES` event is only visible by manually reading
   `order.metadata` — nobody is notified. *Why it matters*: at real volume, silent AI failures
   would go unnoticed indefinitely. *Effort*: small (a scheduled job or webhook reading
   `commercepilot_ai.processing_status != COMPLETED` — a few hours).
2. **No real local E2E proof for the shadow path.** Already decided and deferred with an
   exact trigger in `docs/DECISIONS_FINAL.md` Item 4: run it immediately before either the
   SLA-in-code change or first real customer traffic, whichever comes first. *Why it matters*:
   a latent bug in the shadow loop would silently lose the first, most valuable first-party
   data points. *Effort*: ~1-2 hours (start Docker on D:-backed storage, place one real test
   order, verify DB state).

### SHOULD close before the SLA decision goes live in code
3. **`same_zone` StockLocation resolution edge case.** Already implemented this session-chain
   (real comparison, not hardcoded `false`), but only tested against a store with exactly one
   stock location. *Why it matters*: if HASEBHA ever adds a second location, this silently
   reverts to conservative `false` — correct behavior, but worth a monitoring note.
   *Effort*: trivial (a log line already exists; add a metric).
4. **Raw feature payload persistence is JSONL, not a queryable table.** Already extended this
   session-chain to include raw features, but still a flat file, not indexed/queryable.
   *Why it matters*: fine at near-zero volume (today); becomes a real bottleneck once real
   order volume accumulates toward the ~1,650-4,500 order range needed for retraining.
   *Effort*: ~1 day (Postgres table + migration) — explicitly NOT done now, per the standing
   decision not to migrate until volume justifies it.
5. **Amazon dual-champion discrepancy unresolved.** Two different "frozen Amazon champion"
   claims exist in unreconciled artifacts (documented in `docs/DECISIONS_FINAL.md` Item 3).
   *Why it matters*: low urgency (Amazon sentiment isn't part of the fulfillment path at all),
   but should be resolved before anyone tries to wire Amazon sentiment into anything.
   *Effort*: needs a human decision, not engineering time.

### CAN remain documented-known-gaps for handover purposes (never blocking)
6. **No CI/CD pipeline running tests automatically on push.** *Why it matters*: currently low
   — this is a single-maintainer project at this stage. *Effort*: small — **implemented this
   session** (see below, additive-only, no credentials required).
7. **Amazon sentiment not wired to serve live requests** despite a hash-verified artifact
   existing. *Why it matters*: not part of the fulfillment-risk critical path; a real
   integration gap but not urgent. *Effort*: blocked on gap #5 above being resolved first.
8. **No formal secrets-rotation policy.** *Why it matters*: standard hygiene, not urgent at
   current scale/traffic. *Effort*: a documented policy, not code — deferred to the recipient.

### What was implemented this session (additive, local, secret-free, safe)
Per this mission's own authorization ("purely additive, local, secret-free, and safe"), gap
#6 (CI) was implemented: `.github/workflows/test.yml`, a GitHub Actions workflow that runs the
Python test suite on push/PR. It requires no secrets, no remote push was performed (committed
locally only, per the "no push" scope of this session), and does not affect any running
service. See the Handover Checklist below for its exact path and what it does.

## B3. The dependency chain (engineering cannot execute any of this today)

```
SLA decision (business, docs/HASEBHA_SLA_BUSINESS_DECISION_MEMO.md)
        |
real orders accumulate (business/traffic, outside engineering control)
        |
shadow E2E trigger fires (per docs/DECISIONS_FINAL.md Item 4's exact condition)
        |
data minimums reached (~1,650 orders / ~150 breach events minimum;
                        ~4,500 / ~410 recommended -- see PRODUCTION_MODEL_FORENSIC_SCORECARD.json)
        |
retrain ladder (docs/FINAL_PROJECT_EXECUTION_PLAN.md Section 10, not yet executed)
        |
promotion review (human decision, per the Execution Authorization Matrix)
```

**State plainly**: none of this chain is executable by engineering today. Every link past
"SLA decision" depends on either a business decision this project cannot make, or real
customer behavior this project cannot accelerate.

---

# HANDOVER CHECKLIST

## Repos / locations
- **Canonical working copy**: `D:\ecommerce_medusa` (git HEAD `1a84602`, verified hash-identical
  to the prior C: copy for the pinned production model artifact).
- **Archive**: `D:\ecommerce_medusa_ARCHIVE_2026-08-22.zip` (safety snapshot, 2,240 files,
  12.86GB, integrity-tested).
- **C: original**: `C:\Users\User2\Desktop\ecommerce_medusa` — frozen, untouched by this
  session (re-verified: the file this session most recently edited on D:,
  `commerce-pilot-ai/docs/data_provenance.md`, still shows its original 2026-08-03 timestamp
  on the C: copy). Deletion is verbally approved by the user but blocked by tooling safety
  limits on automated recursive deletes at this scale — **remains a manual user action.**

## Environment / secrets inventory (names only — no values ever printed)

**Medusa backend** (`medusa-app/commercepilot-medusa/apps/backend/.env.template`):
`STORE_CORS`, `ADMIN_CORS`, `AUTH_CORS`, `REDIS_URL`, `JWT_SECRET`, `COOKIE_SECRET`,
`DATABASE_URL`, `DB_NAME`.

**Storefront** (`apps/storefront/.env.template`): `MEDUSA_BACKEND_URL`,
`NEXT_PUBLIC_MEDUSA_PUBLISHABLE_KEY`, `NEXT_PUBLIC_BASE_URL`, `NEXT_PUBLIC_DEFAULT_REGION`,
`NEXT_PUBLIC_STRIPE_KEY`, `NEXT_PUBLIC_MEDUSA_PAYMENTS_PUBLISHABLE_KEY`,
`NEXT_PUBLIC_MEDUSA_PAYMENTS_ACCOUNT_ID`, `REVALIDATE_SECRET`,
`MEDUSA_CLOUD_S3_HOSTNAME`, `MEDUSA_CLOUD_S3_PATHNAME`.

**AI service** (`commerce-pilot-ai/src/ai_service/`): `AI_SERVICE_API_KEY` (auth.py, required
for all `/v1/*` routes), `COMMERCEPILOT_AI_SERVICE_URL` and
`COMMERCEPILOT_AI_SERVICE_API_KEY` (read by the Medusa subscriber to call back into the AI
service — must match).

**Railway/Vercel**: project references exist only in git commit history (`99a4431`), not as
files in this repository — a recipient needs the user's own Railway/Vercel account access to
locate the deployed project.

**What a recipient needs to recreate a working environment**: copy each `.env.template` to
`.env`, fill in real values for the above names (never take them from this document — none
are recorded here), and provision a Postgres + Redis instance (locally via the provided
`docker-compose.yml`, or via the same Railway services already used for production).

## How to run everything locally (under 10 minutes)

1. `cd D:\ecommerce_medusa\commerce-pilot-ai` — the `.venv` here works as-is (verified this
   session-chain: real imports, real pytest runs succeed without rebuilding).
2. `cd D:\ecommerce_medusa\medusa-app && docker compose up -d` — starts Postgres (port 5433)
   and Redis (port 6381) with D:-backed volumes.
3. AI service: `cd commerce-pilot-ai && .venv\Scripts\uvicorn src.ai_service.main:app --reload --port 8000`.
4. Medusa backend: `cd medusa-app\commercepilot-medusa && npm run backend:dev` (detect package
   manager first via `packageManager` field in `package.json` — currently `npm@11.6.2`).
5. Storefront (optional, if present): `npm run storefront:dev`.

## Where every deliverable lives (docs map)

| Document | Path |
|---|---|
| This handover | `docs/PROJECT_HANDOVER.md` |
| Committee package | `docs/committee/D1-D6*` |
| Final execution plan (engineering roadmap) | `docs/FINAL_PROJECT_EXECUTION_PLAN.md` |
| Master execution status (living, includes migration record) | `docs/MASTER_EXECUTION_STATUS.md` |
| Final-advisor decision log | `docs/DECISIONS_FINAL.md` |
| Business SLA decision memo | `docs/HASEBHA_SLA_BUSINESS_DECISION_MEMO.md` |
| Demo notebook | `notebooks/06_transfer_gap_and_funnel.ipynb` |

## The 3 pending user actions

1. **C: deletion** (امسح) — verbally approved, blocked only by tooling safety limits on a
   recursive delete of this scale. Manual action remains with the user; nothing further from
   engineering.
2. **SLA memo decision** — `docs/HASEBHA_SLA_BUSINESS_DECISION_MEMO.md`, a 15-minute read,
   unlocks the entire data-collection chain in B3.
3. **Awaiting real orders** — outside anyone's direct control; the shadow pipeline is ready
   to start learning from them the moment they exist in volume.
