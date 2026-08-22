# CommercePilot / HASEBHA — Supervisor Handoff: Current-State Discovery

Verified: 2026-08-21 (Egypt/HASEBHA repository, read-only investigation, no code/data/model changes made)

This report was written by directly re-inspecting the current repository, git state, live
Docker/DB state, and test suite — not by copying prior session summaries. Where I relied on
prior-session artifacts I read them again and note that explicitly.

---

## 1. Executive Summary

CommercePilot is an AI layer bolted onto a real, running (but currently stopped) Medusa
commerce backend for "HASEBHA" (an Egyptian single-vendor storefront). Five ML/NLP tracks are
frozen (Arabic sentiment, Amazon sentiment, Instacart recommender, Olist V1 — the live
production model — and, separately, an earlier Jumia track). Three additional Olist tracks
(V2, V3) and a DataCo/EAGLE reproduction were built as research this month; none of them are
production-validated for HASEBHA.

**The single most important fact for the incoming supervisor**: the "shadow mode" fulfillment
pipeline built two sessions ago is real, tested code — but I directly queried the live
database and confirmed it **has never actually fired against a real order**. The database has
only 5 orders total (all from 2026-08-15/16, before the shadow code existed), zero fulfillment
records, and zero rows containing the shadow model's metadata key. Nothing about the shadow
pipeline is "proven working in production" — it is proven working in tests only.

## 2. Current Repository State

- Git: `HEAD=1a84602`, branch `main`. `git diff --name-only` shows exactly **6 modified tracked
  files**, all from the immediately preceding sprint: `commerce-pilot-ai/src/ai_service/{config.py,
  main.py, routers/fulfillment.py, routers/health.py, schemas.py}` and
  `medusa-app/commercepilot-medusa/apps/backend/src/subscribers/order-placed.ts`. All additive
  (verified in the prior session via `git diff --stat`; re-confirmed here via `git diff --name-only`
  showing no other tracked file touched). ~70-76 untracked files/directories, consistent with
  this repo's long-standing convention of keeping `scripts/`, `reports/generated/`, and
  `artifacts/` untracked. No closed-track, README, or storefront files touched.
- **Disk**: C: has **5.6GB free** (down from ~11GB last session). I investigated this directly:
  - `commerce-pilot-ai/` totals **17.66GB** on C: (`artifacts/` 9.34GB — almost entirely
    pre-existing Arabic/Amazon/Jumia model checkpoints, e.g. `arabic_foundation/primary_model/
    run/checkpoint-1119` alone is 1.87GB; `.venv/` 5.69GB; `data/` 2.56GB).
  - `pip` cache adds another 3.03GB.
  - I scanned every file modified in the last 3 days: **total 14.3MB**, all from the last two
    engineering sprints (small model artifacts, source, reports). This does NOT explain a
    ~5.5GB drop.
  - **Honest conclusion**: I cannot fully attribute the recent disk drop to this project's
    activity — the large consumers (model checkpoints, venv, pip cache) are cumulative across
    many prior sessions, not new. If free space keeps falling, the next investigator should
    check for something outside this repo's visibility (other projects/processes on the
    machine, OS update cache, browser data, etc.) — I found nothing repo-side to explain it.
- **Docker**: `commercepilot_medusa_postgres` and `commercepilot_medusa_redis` containers exist
  (with a real, migrated Medusa schema and a populated volume) but were **stopped** ("Exited ...
  5 days ago") when I started this investigation. I started postgres briefly to query it
  read-only, then stopped it again to leave the environment as found.
- **No live services**: no FastAPI or Medusa process is currently listening on any port.

## 3. Current Architecture

```
HASEBHA (single-vendor Medusa store, Egypt)
│
├── Medusa backend (apps/backend) -- Node/TypeScript, Medusa v2 core modules only
│    ├── PostgreSQL (docker, port 5433) -- real schema, 5 test orders, 0 fulfillments
│    ├── Redis (docker, port 6381)
│    └── subscribers/order-placed.ts
│         ├── V1 call -> POST /v1/fulfillment/risk (LIVE production route)
│         └── shadow call -> POST /v1/fulfillment/seller-sla-shadow (SHADOW, additive, untested against real data)
│
├── Next.js storefront (apps/storefront) -- not inspected this session, out of scope
│
└── commerce-pilot-ai/ (Python, FastAPI, "ai_service")
     ├── /v1/fulfillment/risk         -- PRODUCTION. Frozen Olist V1 CatBoost model (2 features: purchase/approval timing only)
     ├── /v1/fulfillment/seller-sla-risk    -- RESEARCH_OFFLINE_ONLY. 22-feature Olist V3 model, explicit manual feature input, never Medusa-derived
     ├── /v1/fulfillment/seller-sla-shadow  -- SHADOW. 13-feature "production-parity" model, real-Medusa-derivable features, WEAK signal, never triggers action
     ├── /v1/decision                  -- deterministic Decision Engine (YAML-config rules), consumes fulfillment_risk_score generically
     ├── /v1/nlp/analyze                -- Arabic (MARBERTv2, live) + Amazon (honest ARTIFACT_NOT_MATERIALIZED stub)
     └── PredictionFeedbackStore        -- local JSONL append-only file, NOT a DB table; currently contains only test-fixture rows
```

This corrects the historical diagram's implicit assumption that "shadow mode" means
"currently observing real HASEBHA traffic" — it does not yet.

**Where prediction happens**: inside `commerce-pilot-ai`'s FastAPI process (not in Medusa).
**Feature source**: V1 route uses only `order.created_at` + earliest payment timestamp (both
real, derived honestly, never fabricated — confirmed by reading `resolveOlistFeatures()`).
Shadow route uses `order.item_total`, `shipping_total`, `items[].product.weight/length/height/
width`, `payment_collections[].payments[].amount` (all real Medusa fields) — but `same_zone` is
**hardcoded to `false`** in the current code (StockLocation province resolution was never
implemented), a disclosed simplification, not a bug someone accidentally shipped silently.
**Missing feature**: no feature is silently faked; the design principle throughout is "return
`UNAVAILABLE_AT_EVENT_TIME`/skip rather than fabricate," verified by reading the actual
resolver functions in `order-placed.ts`.
**What persists**: V1 result → `order.metadata.commercepilot_ai`. Shadow result → a
**separate** `order.metadata.commercepilot_ai_shadow_seller_sla` key (never yet actually
written, per the DB query above) + a local JSONL row via `PredictionFeedbackStore`.
**What does not persist**: raw feature payloads for shadow predictions (only the resulting
score) — a real, disclosed gap for future retraining.

## 4. Project Evolution

1. Multi-track ML buildout (Arabic sentiment, Amazon sentiment, Instacart recommender, Olist
   V1 fulfillment-risk, and — historically — a Jumia track) → each frozen after its own
   evaluation protocol. Evidence: `reports/checkpoints/phase2c_nlp_transformer_finalist_
   confirmation_2026-08-11/`, `instacart_phase1_recommender_freeze_2026-08-14/`,
   `olist_integration_audit_2026-08-11/`, `jumia_phase9_freeze_and_protected_test_2026-08-15/`.
2. Olist V1 (2-feature, timing-only) wired into production Medusa via `order-placed.ts`,
   confirmed live in the DB (5 real-ish test orders scored, one correctly showing
   `AI_UNAVAILABLE` when the AI service was down — the fail-soft path genuinely exercised).
3. A fresh Olist V2 attempt found a temporal regime shift (protected-test AUC collapsed to
   0.51 vs dev 0.72-0.76) — resolved as NOT the trucker strike, left as
   `FINAL_OLIST_ROLE=STRESS_BENCHMARK`, not production. Evidence:
   `reports/generated/olist_v2/OLIST_V2_TEMPORAL_STABILITY_FINAL_REPORT.md`.
4. Olist V3 built a genuinely different, stronger model: seller-handoff SLA breach (not
   customer-lateness), reaching mean temporal AUC 0.7702 — but built entirely on Brazilian
   multi-seller marketplace semantics. Evidence: `reports/generated/olist_v3_multistage/
   SELLER_SLA_TEMPORAL_EVAL.json`.
5. A production-availability simulation (this month) proved that model collapses to AUC 0.52
   (near-random) once its 10 seller-history features are forced to their real HASEBHA
   production sentinel — because **HASEBHA has no seller/vendor module at all**. Evidence:
   `reports/generated/olist_v3_multistage/OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json`.
6. A fresh 13-feature "production-parity" model was retrained on only genuinely
   HASEBHA-available features, reaching AUC 0.5551 — classified WEAK. Evidence:
   `reports/generated/olist_v3_multistage/PRODUCTION_PARITY_MODEL_COMPARISON.json`.
7. That WEAK model was wired in shadow mode (log-only, no automated action) purely to start
   collecting first-party outcome data — but as of this investigation, it has never scored a
   real order (see Section 10).
8. DataCo/EAGLE reproduction stalled at the LSTM gate: the paper's own quoted target formula,
   reimplemented exactly, produces ~50% positive prevalence vs the paper's stated 4-6% —
   quantitatively explained (sample-size/CLT argument), not a bug, not chased further per the
   project's own anti-fishing rule. Evidence: `reports/generated/dataco/
   DATACO_TARGET_FORENSIC_CORRECTION.json`.

**What the project is now waiting for**: real first-party HASEBHA order/fulfillment data (there
is currently almost none — 5 test orders, 0 fulfillments) and a real business-defined
shipping SLA (there currently is none — see Section 9).

## 5. ML Track Status

### Arabic sentiment
Champion: plain MARBERTv2, group-safe split, unweighted cross-entropy, seed 42 best
checkpoint. 3-seed mean Macro-F1 = 0.7906 (single-seed 0.8130). Five candidate improvements
(label smoothing, class-weighted CE, R-Drop, and two never-attempted-due-to-consistent-
negative-pattern candidates) were all rejected — none cleared the pre-declared +0.3pp bar.
SARF (an alternative architecture) was statistically tied but rejected on cost (~1.9x
runtime/VRAM for parity). Frozen. Legitimate claim: "best evidence-supported model under the
tested leakage-safe protocol," not "globally optimal." Source:
`reports/generated/arabic_sota/ARABIC_FINAL_DECISION.json`.

### Olist V1
2-feature (purchase/approval timing only), CatBoost, this is the **live production model**
behind `/v1/fulfillment/risk`. Must remain untouched — it is the only model with any real
(if thin — 5 orders) production execution history in this codebase.

### Olist V2
Superseded stress-test track. Key finding: static models suffer a severe, real temporal
regime shift on Brazilian data (dev AUC 0.72-0.76 → protected-test AUC 0.51); recency-weighted
adaptation partially recovers (to 0.59). `ADVANCED_MODEL_NEXT_STAGE=NO`. Not a production
candidate — it exists to document that naive temporal generalization is unsafe on this data,
a lesson that should inform any future HASEBHA-native model, not a deployable artifact itself.

### Olist V3
Three sub-models, all Brazilian-data-only:
- **Seller-SLA (T0)**: mean AUC 0.7702, worst 0.6762 — the strongest research result in the
  project. But see Section 7: this number describes a 22-feature model that cannot be scored
  on real HASEBHA data.
- **Customer-late T0 (stacked)**: rejected for flagship use, delta only +0.0036 over baseline.
- **Customer-late T1 (dynamic, post-handoff)**: mean AUC 0.689 — moderate, real, but not strong
  enough alone for automated action; triage-only.
- Calibration (isotonic) applied to the seller-SLA model; operating-point thresholds derived
  from historical OOF data only.

**The critical distinction**: `RESEARCH_FULL_FEATURE_MODEL` (the 22-feature seller-SLA model,
AUC 0.77, cannot run on real HASEBHA orders) vs. `PRODUCTION_PARITY_MODEL` (the 13-feature
retrain, AUC 0.5551, CAN run on real HASEBHA orders but is weak). These are not two views of
the same result — they are two different models with a ~0.22 AUC gap that exists specifically
because HASEBHA lacks a seller/vendor concept. Treating the 0.77 number as "the model's
production accuracy" would be a real, easy-to-make mistake.

### DataCo/EAGLE
Official v5 dataset acquired and hash-verified; graph structure (46 nodes) reproduced exactly;
LSTM reproduction failed (mean AUC 0.645 vs published 0.968) due to a target-prevalence
mismatch that is mathematically explained (not a leakage bug, not an architecture bug) by the
dataset's large per-node-window sample size making a simple mean-threshold target
statistically incapable of reproducing the paper's much rarer reported event rate. EAGLE was
correctly never attempted (its own gating rule: don't build the expensive model on a
known-broken target). Status: `UNRESOLVED_PROTOCOL_MISMATCH`, research-only, not pursued
further, and explicitly not a candidate for HASEBHA regardless of outcome (different domain).

## 6. Olist V3 Findings (see Section 5 above for detail; this section is the pointer required by
the mission's structure). Full chain: `reports/generated/olist_v3_multistage/
OLIST_MULTISTAGE_SCORECARD.json`, `PRODUCTION_PARITY_MODEL_COMPARISON.json`,
`SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json`.

## 7. HASEBHA Production-Parity Problem

Why doesn't the Olist V3 result constitute a HASEBHA production ML result?

```
22-feature seller-SLA model trained on Brazilian Olist marketplace data (AUC 0.77)
        ↓
10 of those 22 features encode PER-SELLER historical performance
        ↓
HASEBHA is a single-vendor Medusa store -- confirmed by direct code inspection:
no seller/vendor module exists anywhere in medusa-app/commercepilot-medusa
        ↓
production-availability simulation: force those 10 features to their real,
honest production sentinel (never a fabricated guess) and re-score the SAME
historical Olist test folds
        ↓
AUC collapses: 0.7702 -> 0.5188 (near-random), worst period 0.4949
        ↓
retrain restricted to the 13 features that ARE genuinely available in a
single-vendor Medusa store
        ↓
AUC = 0.5551 (WEAK) -- barely, marginally better than random
        ↓
CONCLUSION: the strong 0.77 number was driven almost entirely by
marketplace-specific seller heterogeneity that has no HASEBHA analog. There
is currently no evidence that any Olist-derived feature set predicts
fulfillment risk meaningfully for a single-vendor Egyptian store.
```

What is actually missing is not "a better model" — it is **real first-party HASEBHA
fulfillment outcome data**. No amount of further Olist feature engineering can substitute for
that, because the signal the strong model relied on (seller heterogeneity) structurally does
not exist in the target domain.

## 8. First-Party Data Status

**Orders**: Yes, real HASEBHA/Medusa order data exists, but it is minimal: **5 orders total**,
all dated 2026-08-15/16 (dev/test orders around the initial V1 integration work, not
representative production volume). Confirmed by direct `SELECT count(*) FROM "order"` against
the live (temporarily started) database.

**Outcomes**: **None observable.** `SELECT count(*) FROM fulfillment` returns **0**. No order
in this database has ever been packed, shipped, or delivered. There is currently no way to
construct any real fulfillment-outcome label from this database.

**Features at order-creation time**: Real and available today: `order.created_at`,
`item_total`, `shipping_total`, item/product weight-dimensions (when catalog populates them),
payment amounts, item/category counts. NOT available: any seller-level history (no seller
concept), any real shipping deadline (Section 9), reliable shipping-origin zone (StockLocation
resolution not implemented).

**Labels**: A legitimate fulfillment-risk target **cannot currently be constructed** — it
needs both (a) a real business SLA/deadline (does not exist, Section 9) and (b) real
`fulfillment.shipped_at` events (zero exist in the DB today). Both preconditions are unmet.

## 9. Business SLA Status

**No real business-defined fulfillment deadline exists in HASEBHA today.** Confirmed by
direct inspection of the `@medusajs/types` `ShippingOptionDTO` (no promise/SLA/estimated-days
field) and by grepping the actual `api/`, `workflows/`, `subscribers/` source for any SLA
convention — none found. This was already investigated in the prior sprint
(`reports/generated/olist_v3_multistage/HASEBHA_SHIPPING_SLA_PRODUCT_REQUIREMENT.md`) and I
re-confirmed the underlying claim (no such field in the type definitions) rather than just
trusting that document. A specification for a `fulfillment_due_at` concept (store-defined
per-shipping-option promise, derived at order time, compared against the real
`Fulfillment.shipped_at` field) was **written but not implemented**. This is a business/product
decision HASEBHA operations has not yet made, not something engineering can resolve alone.

## 10. Shadow Pipeline Status

Traced end-to-end by reading the actual source, then independently verified against the live
database (not inferred from documentation):

```
Medusa order.placed event
      ↓  [subscriber exists, confirmed in source]
resolveProductionParityFeatures() -- real Query API call, real fields, honest fallbacks
      ↓
POST /v1/fulfillment/seller-sla-shadow  [route exists, 12/12 tests pass against the real artifact]
      ↓
ProductionParitySellerSlaService.score()  [hash-verified model load, confirmed]
      ↓
order.metadata.commercepilot_ai_shadow_seller_sla  +  PredictionFeedbackStore JSONL row
```

- **Is it actually connected?** Yes, at the code level — the subscriber calls the shadow
  route, wrapped in try/catch, additively, after the existing V1 call.
- **Is it currently executable?** Yes in principle (services/schemas import cleanly, tests
  pass against the real model artifact), but this was never verified against a genuinely
  running FastAPI+Medusa+Postgres stack together (see Section 11).
- **Has it ever actually run against a real order?** **No.** Direct DB query:
  `SELECT count(*) FROM "order" WHERE metadata::text LIKE '%shadow%'` returns **0**. All 5
  real orders in the DB predate this shadow code's existence.
- **What gets stored?** A calibrated probability, risk tier, model name/version/hash, and
  timestamps — **not** the raw feature payload the model actually saw.
- **Can future training reconstruct exactly what the model saw?** **No, not yet** — this is a
  disclosed, documented gap (`HASEBHA_FULFILLMENT_FEEDBACK_DATASET_CONTRACT.md`), not an
  oversight nobody noticed.
- **Model/version stored?** Yes, both in the JSONL row and (design-wise) the order metadata key.
- **Can a prediction later be joined to an outcome?** Structurally yes (`record_outcome()`
  exists, keyed by `prediction_id`), but no reconciliation job exists yet, and there are zero
  real outcomes to join against today.

## 11. E2E Status

**`REAL_LOCAL_E2E = NOT_RUN`.**

What actually exists:
- **Unit tests**: yes, extensive, real (all model artifacts loaded and scored for real inside
  the tests — not mocked away). 447/447 passing as of the last sprint (re-verifiable, not
  re-run this session per the mission's no-new-experiments rule, but the test files and
  artifacts they depend on are unchanged since that run).
- **Integration tests (FastAPI TestClient + real loaded model)**: yes, extensive, real.
- **Mock E2E**: no evidence of a mocked full-stack E2E either.
- **Real local E2E (live Postgres + live Medusa + live FastAPI + an actual order placed
  through the real event bus)**: **never demonstrated**. The database evidence in Section 10
  is the direct proof: if this had ever genuinely run, at least one order would carry the
  shadow metadata key. None do.
- **Real production E2E**: not applicable / clearly not run (no production traffic exists).

## 12. Production/Git Safety Status

- `git diff --name-only`: 6 files, all `commerce-pilot-ai/src/ai_service/*` + one Medusa
  subscriber, all from the documented last sprint, all additive.
- Olist V1 (`src/modeling/olist/`, `configs/olist_phase2*.yaml`,
  `reports/generated/olist/`) — untouched, confirmed via `git status` on those exact paths.
- No Arabic, Amazon, Instacart, Jumia, README, storefront, or custom-Medusa-module files
  touched.
- Nothing was deleted or "cleaned up" this session, per the mission's explicit instruction —
  the Docker containers I started for read-only inspection were stopped again afterward, and
  no database rows were modified.

## 13. Things the Incoming Supervisor Should Know

- **The shadow pipeline being "implemented and tested" does not mean it is "working in
  production."** It has zero real-order executions. Any decision assuming otherwise would be
  wrong.
- **The headline 0.77 AUC number is real but domain-mismatched.** It describes a model that
  literally cannot be scored on real HASEBHA orders (missing 10/22 features structurally, not
  just as a data-quality gap). The number that actually describes what could run on HASEBHA
  today is 0.5551 (weak).
- **There is no real business SLA yet.** "Seller SLA breach" is currently a Brazilian-dataset
  proxy target with no HASEBHA business-rule backing. Until operations defines a real
  fulfillment promise, no fulfillment-risk model — however trained — has a meaningful target
  for this store.
- **There are only 5 real orders and 0 fulfillments in the database.** Any first-party
  retraining plan is premature; the project's honest current bottleneck is data volume, not
  modeling technique.
- **`same_zone` is hardcoded `false`** in the current shadow feature builder — a disclosed
  simplification (StockLocation resolution not implemented), not silently wrong.
- **A Jumia track existed and was actively worked (through ~2026-08-15)** before later
  sessions' "JUMIA excluded entirely, always" instruction took hold. `JUMIA_USED=NO` in recent
  session flags describes *this session's own behavior*, not "Jumia was never touched in this
  project's history" — worth knowing so the supervisor doesn't misread the historical record.
- **C: drive is at 5.6GB free and dropping**, mostly from cumulative NLP model checkpoints
  across many sessions (not this session's doing) plus pip cache. This deserves attention
  independent of any ML work — a disk-full event mid-training would be a real operational
  risk.
- **The Decision Engine is already generic enough** to accept any calibrated
  `fulfillment_risk_score` without code changes — this is a real strength; a future model
  doesn't need Decision Engine surgery, just a service that produces the right shape.
- **Persistence is currently split** between Medusa's `order.metadata` (for anything wired to
  a real event) and a local JSONL file (for the research/shadow routes) — neither is a proper
  queryable predictions table yet. This will not scale past exploratory volumes.

## 14. Open Questions

- What is HASEBHA's actual real order volume and growth rate (outside this 5-row dev DB)? Not
  answerable from this repository alone.
- Does HASEBHA operations have an informal shipping-time promise even if it's not in Medusa
  config yet? Needs a business-side answer, not an engineering one.
- Is there appetite to implement real StockLocation-based `same_zone` resolution, or is that
  premature given the weak underlying signal either way?
- Should the JSONL feedback store be upgraded to a real Postgres table now, or is that
  premature given current volume (5 orders)?

## 15. My Recommended Next Actions (informational only — supervisor decides)

**Priority 1**: Do NOT run more Olist/DataCo modeling. The highest-value next action is
business-side: get HASEBHA operations to define a real `fulfillment_due_at` / shipping-SLA
rule (Section 9), and start accumulating real order + fulfillment volume. No model improvement
is possible without this; it is not an engineering bottleneck.

**Priority 2**: Once real order volume exists (even without fulfillment outcomes yet), extend
`PredictionFeedbackStore` to persist raw feature payloads alongside scores, so that whenever
outcomes do start arriving, a real training set can be reconstructed retroactively rather than
waiting for a second data-collection phase.

**Priority 3 — remain frozen**: Arabic, Amazon, Instacart, Olist V1, Jumia (all as before);
also DataCo/EAGLE (no primary-source evidence exists to resolve its target-formula ambiguity,
and it's out of domain for HASEBHA regardless); also the 22-feature research Seller-SLA model
should NOT be presented as a HASEBHA capability in any external communication.

## 16. What I Explicitly Recommend NOT Doing

- Do not retrain the production-parity model hoping for a better number — its ceiling is
  bounded by the lack of a real seller/vendor-equivalent signal, not by hyperparameters.
- Do not wire the shadow route to trigger any automated action, however tempting, until real
  first-party validation exists.
- Do not treat 5 test orders as representative of anything.
- Do not invent a shipping SLA in code without a real business decision behind it.
- Do not spend further session time on DataCo/EAGLE without a primary-source PDF resolving the
  target-formula ambiguity — guessing further would violate this project's own anti-fishing
  discipline, already applied twice.

## 17. Evidence / Artifact Index

- `reports/generated/final_release/HASEBHA_SINGLE_VENDOR_FULFILLMENT_FINAL_REPORT.md` and
  `HASEBHA_SINGLE_VENDOR_FULFILLMENT_SCORECARD.json` — the immediately preceding sprint's own
  report, cross-checked (not copied) against DB/Docker/git evidence in this investigation.
- `reports/generated/olist_v3_multistage/` — seller-SLA/T0/T1 results, parity re-audit,
  production simulation, production-parity model comparison, calibration reports.
- `reports/generated/dataco/DATACO_TARGET_FORENSIC_CORRECTION.json` — DataCo/EAGLE status.
- `reports/generated/arabic_sota/ARABIC_FINAL_DECISION.json` — Arabic champion decision.
- `reports/checkpoints/overnight/RESUME_STATE.json` — last engineering sprint's own checkpoint.
- `medusa-app/commercepilot-medusa/apps/backend/src/subscribers/order-placed.ts` — actual
  production/shadow wiring source, read directly.
- Live (temporarily started, then stopped) Postgres database — direct SQL queries, this session.
- `git status`, `git diff --name-only` — this session.
