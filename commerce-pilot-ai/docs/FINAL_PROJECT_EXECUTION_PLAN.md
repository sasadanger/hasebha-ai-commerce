# CommercePilot / HASEBHA — Final Project Execution Plan

Verified: 2026-08-22. Planning-only document. No code was modified, no model was retrained,
no experiment was run to produce this plan. This synthesizes findings independently verified
across the four immediately preceding evidence-audit sessions in this chain (supervisor
handoff, complete ML results reconstruction, next-action delegation, committee-defense
readiness audit), plus a fresh repository state check performed for this session.

---

## 1. First Principle: Current Repository State (re-verified this session)

- Git HEAD `1a84602`, branch `main`.
- **6 modified tracked files**, unchanged list across every session in this chain:
  `commerce-pilot-ai/src/ai_service/{config.py, main.py, routers/fulfillment.py,
  routers/health.py, schemas.py}` and
  `medusa-app/commercepilot-medusa/apps/backend/src/subscribers/order-placed.ts`. All
  additive, all previously tested (451/451 full suite passing as of the last code change).
- 81 untracked files/directories (report/checkpoint growth from this session chain; no new
  source files added this session).
- No closed-track paths touched (verified via targeted `git status` on
  `src/modeling/olist/`, `configs/olist_phase2*.yaml`, `reports/generated/olist/`, Jumia
  paths, storefront, custom Medusa modules — all clean).
- **C: drive free space: 3.27GB, down from 4.2GB last session, 11.4GB several sessions ago.**
  This has been investigated twice already (two prior sessions each performed a file-mtime
  scan and confirmed under 1MB of this repository's own activity in the preceding 24-72
  hours) — it is **not** caused by this repository. It is now genuinely critical for normal
  system/Docker operation. **This is flagged directly to the user as needing attention outside
  this project's scope — not re-investigated a third time per this mission's own
  instruction.** See Section 6, Blocker B-00.

---

## 2. Frozen Tracks — Protected, Not Reopened

**Olist V1** (production, ROC-AUC 0.5634 on its own frozen test set, `final_test_metrics.json`;
5 real HASEBHA orders scored, verified via direct live-DB query): no scientific defect
discovered this session that would justify reopening it. **Remains frozen.**

**Arabic, Amazon, Instacart, Olist V2, Olist V3 research, DataCo/EAGLE, Jumia**: no new
evidence discovered this session that would justify reopening any of them. **All remain
frozen**, per every prior session's own explicit conclusion.

No track is proposed for change in this plan. If a future session believes a frozen track
should change, that session must stop and justify it per Section 2 of the mission (what's
wrong, why it matters, evidence, why necessary, exact change) — not act unilaterally.

---

## 3. Current Scientific Context — Verified, Not Assumed

| Historical claim | Verification status this session |
|---|---|
| Olist V1 ROC-AUC ≈0.5634 | VERIFIED in a prior session by direct read of `reports/generated/olist/phase2a/final_test_metrics.json` (exact value 0.5634406421203355). Not re-opened this session (planning-only). |
| Olist V3 Seller-SLA, 23 actual features, mean temporal AUC ≈0.7702 | VERIFIED in a prior session by direct Python import of `FEATURE_ORDER` (len==23) and by direct read of `SELLER_SLA_TEMPORAL_EVAL.json` (0.7702329747962686). Note: many earlier artifacts mislabel this "22 features" — corrected and documented, does not change the metric. |
| HASEBHA production-parity mean AUC ≈0.5551, WEAK | VERIFIED in a prior session via `PRODUCTION_PARITY_MODEL_COMPARISON.json` (0.5551351215216622). |
| ~5 real/dev orders, 0 fulfillment outcomes | VERIFIED via a direct live-database query in a prior session (`SELECT count(*) FROM "order"` = 5, `SELECT count(*) FROM fulfillment` = 0). |
| No validated HASEBHA shipping SLA | VERIFIED via direct inspection of `@medusajs/types` `ShippingOptionDTO` (no promise/SLA field) and source grep across `api/`, `workflows/`, `subscribers/` in a prior session. |
| Seller-history unavailable in HASEBHA | VERIFIED — no seller/vendor module exists anywhere in `medusa-app/commercepilot-medusa` (confirmed by directory listing in a prior session). |

All historical numbers cited in this plan are carried forward as VERIFIED-by-prior-direct-read,
not re-derived this session (this mission is planning-only). No number in this plan is
invented or estimated.

---

## 4. Scientific Objective

The question this project must answer is not "can AUC exceed 0.7" but: **does a scientifically
defensible predictive fulfillment-risk signal currently exist for HASEBHA, and if not, what
combination of (A) unused existing signal, (B) cheap feature derivation, (C) small engineering
additions, (D) business decisions, (E) data collection, or (F/G) fundamental target
invalidity/unlearnability explains its absence?**

Current answer, evidence-based: **primarily D and E**, with a secondary, already-answered F
component (see Section 8 — the *current* proxy target, reused from Olist, is not yet a
legitimate HASEBHA target because no business SLA exists to define "late").

---

## 5. Complete Project State Map

| Track | Status | Evidence | Prod/Research/Shadow | Can Change? | Remaining Work | Final Destination |
|---|---|---|---|---|---|---|
| Olist V1 | PRODUCTION | `final_test_metrics.json`, live DB (5 orders) | PRODUCTION | NO (frozen, no defect found) | None authorized | Stays as-is until superseded by a validated HASEBHA-native model |
| Olist V2 | RESEARCH, STRESS_BENCHMARK | `FINAL_SCORECARD.json` | RESEARCH | NO | None | Historical evidence only |
| Olist V3 (seller-SLA/T0/T1) | RESEARCH | `SELLER_SLA_TEMPORAL_EVAL.json`, `TASK_B_C_RESULTS.json` | RESEARCH | NO | None (Olist-side work exhausted, see Section 10) | Research reference; source of the production-parity feature-restriction methodology |
| HASEBHA production-parity model | RESEARCH-TRAINED (on Olist data), WEAK | `PRODUCTION_PARITY_MODEL_COMPARISON.json` | SHADOW (never executed) | Retraining authorized ONLY once real first-party data exists (Section 9) | Await real data; do not retrain on Olist again | Superseded by a HASEBHA-native model once data allows, or retired |
| HASEBHA shadow route | IMPLEMENTED, TESTED, NEVER EXECUTED against a real order | `PRODUCTION_REALITY_MATRIX.json` (0/5 real orders carry its metadata key) | SHADOW | Additive extensions only | Needs real order volume to become meaningful | Becomes the real-world validation mechanism once volume exists |
| Feature persistence (raw payload) | IMPLEMENTED this session-chain (immediately preceding session) | `RAW_FEATURE_PERSISTENCE_SCORECARD.json`, 451/451 tests passing | Engineering | Additive extensions only | Move JSONL→Postgres once volume justifies it | Becomes the future training-set source |
| Prediction feedback store | JSONL, append-only | `prediction_feedback_store.py` | Engineering | Additive extensions only | Outcome-reconciliation job (not yet built, no real outcomes to reconcile against) | Postgres table once volume justifies |
| Medusa integration | V1 wired live; shadow wired, never fired | live DB, `order-placed.ts` | PRODUCTION (V1) + SHADOW | Additive extensions only | None urgent | Stays as-is |
| Fulfillment data | 5 orders, 0 outcomes | live DB query | N/A | N/A — accumulates via real business operation | Wait for real order volume | Becomes the first-party dataset |
| Target definition | Olist-proxy (SELLER_HANDOFF_SLA_BREACH), NOT a real HASEBHA business rule | `HASEBHA_SHIPPING_SLA_PRODUCT_REQUIREMENT.md` | N/A | Requires business decision | HASEBHA ops must define a real shipping SLA | Real target definition (Section 8) |
| Evaluation framework | Rolling-origin temporal, leakage-tested, calibration pipeline all proven on Olist | multiple JSON artifacts, all leakage tests 0/4 or 0/30 failures | Engineering + Research | Reusable as-is | None — reuse for HASEBHA-native modeling later | Reused unchanged for future first-party modeling |
| Tests | 451/451 passing (full repo, last measured) | prior session's own pytest run | Engineering | Add-only | Add HASEBHA-native-data tests once that data exists | Maintained green |
| Deployment | Existing Railway/Vercel config referenced in git log, not re-audited this session | git log (`Deploy HASEBHA v1 to production`) | Engineering | Not touched | Full deployment-readiness audit (Section 15) | Not blocking current phase |
| Documentation | Extensive (this plan + 4 prior audit rounds) | this session chain | Engineering | Add-only | Keep synced with future phases | Maintained |

---

## 6. True Blockers (structured)

### B-00 — C: drive free space critically low (3.27GB)
- **Why it blocks the project**: risk of failed writes, failed Docker operations, failed test
  runs, or corrupted checkpoints if it reaches zero during any future session.
- **Evidence**: PowerShell `Get-PSDrive` checks across 4 sessions (11.4→5.6→4.2→3.27GB),
  file-mtime scans twice confirming <1MB attributable to this repo.
- **Severity**: HIGH (operational risk to the whole machine, not just this project).
- **Dependency**: none.
- **Can engineering solve it**: not within this repository's scope — the consumers identified
  in a prior session (pip cache 3GB, `.venv` 5.7GB, pre-existing NLP model checkpoints 9.3GB)
  are either shared machine state or historical artifacts this project should not delete
  without explicit authorization.
- **Can modeling solve it**: no.
- **Requires business decision**: no — requires a system-administration decision (what to
  clean up on the machine) outside this project's authority.
- **Requires data collection**: no.
- **Estimated effort**: 15-30 minutes for a human with disk access to review and clear
  unrelated caches/files.
- **Next action**: **flagged directly to the user — needs attention outside this project's
  scope, not actioned by this plan.**

### B-01 — No real fulfillment outcome data
- **Why it blocks the project**: zero rows in the live `fulfillment` table means no target
  (however defined) can currently be computed for any real order.
- **Evidence**: direct `SELECT count(*) FROM fulfillment` = 0 (prior session).
- **Severity**: CRITICAL — this is the primary blocker to any HASEBHA-native model.
- **Dependency**: real order volume + real fulfillment events occurring in production.
- **Can engineering solve it**: no — engineering can only ensure the data is captured
  correctly once it exists (already done: `Fulfillment.shipped_at` is a real, confirmed field).
- **Can modeling solve it**: no.
- **Requires business decision**: no (this specific blocker is pure data-volume, not
  definitional).
- **Requires data collection**: YES — this IS the data-collection blocker.
- **Estimated effort**: unknown, depends entirely on real HASEBHA order velocity (outside this
  project's control).
- **Next action**: monitor order/fulfillment volume; do nothing else until it grows (Section 9).

### B-02 — No real business-defined shipping SLA
- **Why it blocks the project**: without a real deadline, "late" has no meaning for HASEBHA;
  the current target is a Brazilian Olist proxy, not a HASEBHA business rule.
- **Evidence**: no promise/SLA field in `ShippingOptionDTO`; no such convention found via
  source grep (`HASEBHA_SHIPPING_SLA_PRODUCT_REQUIREMENT.md`).
- **Severity**: CRITICAL — blocks legitimate target construction independent of data volume.
- **Dependency**: HASEBHA operations decision.
- **Can engineering solve it**: no.
- **Can modeling solve it**: no.
- **Requires business decision**: YES — exactly the kind of decision this plan must not
  invent (per mission Section 2/6 and the project's own long-standing business/engineering
  boundary rule).
- **Requires data collection**: no (definitional, not volume).
- **Estimated effort**: business-side, not estimable by engineering.
- **Next action**: HASEBHA operations must define `fulfillment_due_at` semantics (who sets it,
  when it's known, how it's stored) — the engineering specification already exists and is
  ready to implement the moment this decision is made.

### B-03 — Seller-history signal has no HASEBHA analog
- **Why it blocks the project**: the strongest research signal (0.7702 AUC) depends on
  per-seller heterogeneity that does not exist in a single-vendor store.
- **Evidence**: `SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json`,
  `OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json` (collapse to 0.5188, worst 0.4949).
- **Severity**: HIGH, but already fully characterized — not an open question, a closed finding.
- **Dependency**: none (already resolved scientifically; the "fix" is architectural, not a
  task — HASEBHA would need to become a marketplace, which is explicitly NOT recommended
  merely to salvage this signal).
- **Can engineering solve it**: no.
- **Can modeling solve it**: no — already tested (store-wide operational features made things
  worse, not better, `PRODUCTION_PARITY_MODEL_COMPARISON.json`).
- **Requires business decision**: only if HASEBHA ever becomes a multi-vendor marketplace,
  which is a business-model decision far outside this plan's scope.
- **Requires data collection**: no.
- **Estimated effort**: N/A — closed finding, not an open task.
- **Next action**: none — this blocker is documented, not actionable, and should not be
  revisited without a change in HASEBHA's business model.

### B-04 — No E2E proof for the shadow path
- **Why it blocks the project**: cannot claim the shadow pipeline works end-to-end against
  real infrastructure, only that its parts pass isolated tests.
- **Evidence**: `PRODUCTION_REALITY_MATRIX.json`.
- **Severity**: MEDIUM — lower priority than B-01/B-02 because there is currently no real
  outcome to validate against even if E2E were run.
- **Dependency**: standing up the full local stack (Postgres+Medusa+FastAPI) and placing a
  real test order.
- **Can engineering solve it**: YES, cheaply, whenever justified.
- **Can modeling solve it**: no.
- **Requires business decision**: no.
- **Requires data collection**: no.
- **Estimated effort**: 1-2 hours for a dedicated session.
- **Next action**: low priority; do when B-01/B-02 progress makes it meaningful, not before.

### B-05 — Raw feature payload persistence for training-set reconstruction
- **Status**: **RESOLVED** in the immediately preceding session — `PredictionFeedbackStore`
  now persists `raw_features` and `feature_schema_version` alongside every prediction,
  backward-compatible, 451/451 tests passing. No longer a blocker; listed here only for
  completeness of the historical blocker list.

---

## 7. Feature Availability Matrix

| Feature | Source | Current Status | Prediction-Time Available? | Derivation | Leakage Risk | Scientific Value (from Olist evidence) | Engineering Cost | Category |
|---|---|---|---|---|---|---|---|---|
| purchase_weekday/hour/month | `order.created_at` | Available | Yes | trivial | none | low-moderate (temporal context) | none | A |
| n_items, n_distinct_products, n_categories | `order.items` | Available | Yes | trivial | none | moderate | none | A |
| total_price, total_freight, payment_value | `order.item_total`, `shipping_total`, `payment_collections` | Available | Yes | trivial | none | moderate | none | A |
| weight_g, volume_cm3 | `product.weight/length/height/width` | Available IF catalog populates these fields | Yes, when populated | trivial | none | low (weak in the 13-feature production-parity result) | none (data-quality check only) | A/C (C if catalog gaps need backfilling) |
| same_zone (ship-from vs ship-to) | `order.shipping_address.province` vs fulfilling StockLocation address | NOT YET RESOLVED — hardcoded `false` in the current shadow feature builder | Yes, in principle | requires a StockLocation query not yet implemented | low if implemented correctly | unproven (never isolated in the 13-feature result) | small (1 additional Query API call + comparison logic) | C |
| days_to_shipping_deadline | Olist `shipping_limit_date` analog | NOT AVAILABLE | N/A | requires B-02 | N/A until B-02 resolved | high in Olist (0.7702 model), unknown for HASEBHA | none until B-02 resolved | D |
| seller_past_breach_rate / seller-history block (8 features) | Olist seller_id | NOT AVAILABLE — no seller/vendor concept | N/A | N/A | N/A | HIGH in Olist (dominant signal), NONE in HASEBHA (B-03) | requires HASEBHA to become multi-vendor (out of scope) | F |
| n_installments | Olist `order_payments` | NOT AVAILABLE — no core Medusa field found | N/A | possible via payment-gateway-specific metadata, not investigated | unknown | unproven | small-to-moderate (gateway-specific integration) | C (unexplored) |
| store_recent_breach_rate / store-wide ops features | HASEBHA order+fulfillment history, aggregated store-wide | NOT AVAILABLE (0 fulfillment records) — and even when tested on Olist as an analog (Model P+), this made results WORSE, not better | requires B-01 | requires B-01 | requires care (rolling, causal only) | tested and REJECTED on Olist (-0.0093 delta) | requires B-01 first, then small | E (blocked by data), and already evidence-discouraged even once available |
| real shipping-deadline-based SLA breach target itself | HASEBHA business SLA | NOT AVAILABLE | N/A | requires B-02 | N/A | N/A — this is the target, not a feature | none until B-02 resolved | D |

**Category legend reminder**: A=Available now, B=Derivable now, C=Small engineering addition,
D=Requires business decision, E=Requires historical data, F=Olist-only/research-only.

---

## 8. Target Decision

**Investigated candidate targets**: seller SLA breach (F — no seller concept), fulfillment
delay (requires B-02), handling delay/time-to-shipment (requires a real handling-time promise,
same B-02 dependency), time-to-delivery (requires both a promise AND real delivery-tracking
data, neither of which exists), operational/workload risk (would need a defined "risk" outcome
— none exists), workload risk (same).

**Every viable target candidate reduces to the same blocker: B-02 (no real business SLA) and/or
B-01 (no real outcome data).** There is no candidate target that sidesteps both.

**Explicit statement, as required**: **the current target (`SELLER_HANDOFF_SLA_BREACH`, an
Olist Brazilian marketplace proxy) is NOT a valid HASEBHA production target.** It fails
criterion 1 (business-defined) outright. It is retained only as a research-methodology
artifact, never as a claimed HASEBHA capability.

**Business decision required, precisely specified** (per the project's existing product
requirement doc, referenced not repeated): HASEBHA operations must define, per shipping
option, a real handling/delivery-time promise (`fulfillment_due_at` derivation), stating who
sets it, when it becomes known (must be at or before order placement to support a T0 model),
and how breach is determined (`fulfillment.shipped_at > fulfillment_due_at` is the
already-specified, ready-to-implement mechanism once the promise itself exists).

---

## 9. Data Collection Plan

**Required tables/fields** (already exist in Medusa core, confirmed by prior type inspection,
no new schema needed): `order` (id, created_at, item_total, shipping_total, shipping_address),
`order_item`/`product` (weight/length/height/width, category), `payment_collection`/`payment`
(amount, captured_at), `fulfillment` (shipped_at, packed_at, delivered_at, location_id),
`shipping_option` (extended with a `metadata.promise_business_days` field once B-02 resolves).

**Prediction timestamp**: order.created_at (T0) for a bootstrap model; `fulfillment` creation
for a T1 model (mirroring the Olist V3 T0/T1 distinction already validated on Olist).

**Outcome timestamp**: `fulfillment.shipped_at` (once B-02 defines the deadline to compare
against).

**Feature snapshot**: already implemented — `ProductionParitySellerSlaShadowRequest` payload,
now persisted in full via the raw-feature-payload extension (Section 5, B-05 resolved).

**Prediction storage**: `PredictionFeedbackStore` (JSONL today; Section 13 covers when to
migrate to Postgres).

**Outcome storage**: `record_outcome()` method exists, keyed by `prediction_id`; no automatic
reconciliation job exists yet (correctly deferred — nothing to reconcile against today).

**Retention**: not yet defined — should be defined alongside the eventual Postgres migration,
not before (premature to specify a retention policy for a table that doesn't exist yet).

**Data quality checks required before any modeling** (from the existing
`HASEBHA_FULFILLMENT_FEEDBACK_DATASET_CONTRACT.md`, reused not reinvented): label completeness
(the SLA must be real, not backfilled against the Olist proxy), no leakage (features captured
at T0, never re-derived post-hoc), prevalence sanity check (breach rate should be plausible for
a real store, not near-0% or near-100%, which would suggest an SLA misconfiguration).

**Sample size estimates** — the project's own established methodology (Olist V3, DataCo) used
5-7 rolling temporal periods with thousands of orders per period to get stable AUC estimates
with reasonable positive-class counts (Olist single-seller cohort: 96,380 rows, ~8.8% breach
prevalence, i.e. roughly 8,500 positive events total, split across 5 periods averaging ~1,700
positives per period). HASEBHA will not need Olist's scale, but for a defensible rolling-origin
temporal evaluation with a comparable prevalence regime:

| Quantity | MINIMUM | RECOMMENDED |
|---|---|---|
| Total orders (with real fulfillment linkage) | ~2,000 | ~10,000+ |
| Positive events (SLA breaches, at an assumed 5-15% prevalence — UNVERIFIED for HASEBHA, will only be known once B-02 resolves and real breaches can be measured) | ~100-200 | ~500-1,000+ |
| Minimum observation period | 3 rolling temporal periods (to detect any regime instability, as Olist V2 demonstrated is a real risk) | 6+ rolling periods spanning at least 2-3 months of real operation, ideally covering any known seasonal/demand-peak variation |

**These numbers are estimates grounded in this project's own prior temporal-evaluation
practice, not a HASEBHA-specific measurement** (none exists yet) — they should be treated as a
planning heuristic, re-examined once real prevalence is observed, not as a hard requirement
that HASEBHA data must match Olist's scale.

---

## 10. Production Model Recovery Plan (future experiment ladder, NOT executed now)

| Stage | Hypothesis | Features | Data Requirement | Leakage Risk | Expected Info Gain | Success Criterion | Stop Criterion |
|---|---|---|---|---|---|---|---|
| 0. Baseline | Global prevalence / naive rule beats nothing meaningfully | none (prior-only) | B-01+B-02 resolved | none | establishes floor | N/A (always run) | N/A |
| 1. +Order complexity | Item/price/weight features add lift over baseline | the 13 "Category A" features from Section 7 | same | low | LOW (already shown weak on Olist proxy, 0.5551) | AUC materially above baseline with CI excluding 0 | if CI overlaps baseline, stop here, do not escalate |
| 2. +Temporal features | weekday/hour/month/seasonality add lift | + temporal | same | low | LOW-MODERATE | incremental AUC gain, CI excluding 0 | same |
| 3. +Geography | same_zone (once B-03/Category-C item implemented) adds lift | + same_zone | same + StockLocation resolution implemented | low | UNKNOWN (never isolated on Olist) | incremental gain | same |
| 4. +Customer history | repeat-customer patterns, if HASEBHA has enough repeat customers | + customer-level aggregates | requires enough per-customer order history | moderate (must be causal/rolling) | UNKNOWN | incremental gain | if customer repeat-rate too low, skip this stage entirely |
| 5. +Product complexity | category/catalog-level patterns add lift | + product-level aggregates | same | moderate | UNKNOWN | incremental gain | same |
| 6. +Store history | store-wide rolling breach rate / backlog | + store-level rolling features | requires B-01 volume | moderate (must be causal, shift(1)+rolling only) | **PRE-TESTED ON OLIST AND FOUND NEGATIVE (-0.0093)** — expectation should be LOW, not assumed positive | incremental gain, but this stage carries the HIGHEST prior probability of being useless based on existing evidence | if negative or flat, do not force it in |
| 7. All legitimate features combined | full feature set beats every individual stage | all of the above that passed their own stage | full B-01+B-02 resolution | cumulative | should be the best of the ladder, if any stage helped | best AUC with acceptable worst-period robustness | if the combined model is not meaningfully better than the best single stage, investigate feature redundancy before concluding |
| 8. Model family comparison | LightGBM/CatBoost/HistGB comparable, Logistic Regression as an interpretable floor | same, fixed | same | same | LOW additional (family choice matters less than feature availability, per this project's own repeated finding) | best family per Section 11's protocol | do not run more than the 5 authorized families |
| 9. Calibration | isotonic (per this project's established default) improves Brier/ECE without damaging AUC | N/A | same | none | reuse of proven methodology | AUC drop <= 0.005 (established project threshold) | if calibration damages AUC beyond threshold, keep raw |
| 10. Final temporal validation | the selected model is temporally robust, not just strong on one split | final feature set | full rolling-origin protocol, stress-block diagnostic only, never for selection | none (protocol enforces this) | confirms/denies production-candidacy | worst-period AUC materially above baseline, stable across periods | if worst-period collapses like Olist V2 did, do NOT promote to production candidacy without further adaptation work |

This ladder is a plan for a FUTURE session, contingent on B-01 and B-02 being resolved. **No
stage of this ladder is authorized to execute in the current mission.**

---

## 11. Model Family Strategy (for the future ladder above, not now)

Authorized comparison set, consistent with this project's existing "1 primary + 1 justified
challenger" discipline (never a full zoo): **Logistic Regression** (interpretable floor),
**LightGBM** (this project's established default, used in every Olist V3/production-parity
experiment to date), **CatBoost** (authorized as challenger only if native categorical
handling gives a clear, stated reason — e.g. a genuinely high-cardinality category feature
HASEBHA's catalog exposes that Olist's did not). XGBoost and HistGradientBoosting are
listed by the mission as candidates but are **not recommended for addition** unless LightGBM
and CatBoost both prove clearly insufficient — this project has consistently avoided
uncontrolled model-zoo expansion, and there is no evidence yet that model family, rather than
feature availability, is the limiting factor.

**Protocol** (reusing the project's proven methodology, not inventing a new one): fixed
rolling-origin temporal evaluation (as used throughout Olist V2/V3), a small, evidence-based
hyperparameter configuration (no broad HPO — consistent with every prior session's explicit
"no HPO" rule), a stress block reserved strictly for post-selection diagnostics (never
selection), bootstrap confidence intervals on key comparisons (as already used in the Olist V2
operational-ranking work), and a predeclared comparison rule: the challenger must beat the
primary model's mean AND worst-period metric with a CI excluding zero, or it is rejected —
mirroring the exact acceptance logic already used for SARF vs. MARBERTv2 in the Arabic track.

---

## 12. Research → Production Bridge

```
Olist research (0.7702 AUC, seller-history-dependent)
        |  ENTRY CRITERION: none further needed -- already complete
        v
Production-availability simulation (0.5188, collapse demonstrated)
        |  ENTRY CRITERION: none further needed -- already complete
        v
HASEBHA production-parity retrain (0.5551, WEAK, Olist-trained)
        |  ENTRY CRITERION: none further needed -- already complete
        v
Shadow-mode wiring (implemented, tested, never executed against a real order)
        |  ENTRY CRITERION: none further needed -- already complete
        v
First-party data accumulation  <-- CURRENT STAGE, BLOCKED ON B-01 + B-02
        |  ENTRY CRITERION: B-01 (real fulfillment volume) AND B-02 (real SLA) both resolved
        v
HASEBHA-native training (Section 10 ladder)
        |  ENTRY CRITERION: data-quality checklist (Section 9) passed, minimum sample size met
        v
Temporal validation (rolling-origin, stress-block diagnostic only)
        |  ENTRY CRITERION: HASEBHA-native model beats baseline with CI excluding zero
        v
Shadow validation (real predictions vs real eventual outcomes, still no automated action)
        |  ENTRY CRITERION: temporal validation passed
        v
Production candidate (formal acceptance review against this project's predeclared criteria)
        |  ENTRY CRITERION: shadow validation shows real, stable, calibrated performance
        v
Controlled deployment (staged, reversible, monitored)
        |  ENTRY CRITERION: explicit human approval (Section 21/Authorization Matrix)
```

The project is currently at **"first-party data accumulation," blocked**, having already
legitimately completed every stage above it.

---

## 13. Engineering Completion Plan

| Item | Current State | Remaining Work | Dependencies | Risk | Effort | Done Definition |
|---|---|---|---|---|---|---|
| Backend (FastAPI ai_service) | 3 model-serving routes, hash-verified loading, fail-soft | none blocking; StockLocation resolution for same_zone (Section 7) is the only open item | none | low | small (1-2 hrs) | same_zone genuinely resolved, not hardcoded false |
| Medusa backend | V1 wired live, shadow wired never fired | none blocking | B-01/B-02 for the shadow path to become meaningful | low | none required now | N/A until real volume exists |
| Database (Postgres) | Real schema, 5 orders, 0 fulfillments | none — schema already supports everything needed | B-01 (volume) | low | none | N/A |
| Prediction persistence | JSONL, raw features now included | migrate to Postgres once volume justifies (not yet) | B-01 | low | 1 day when justified | a real prediction table with joinable outcomes |
| Feedback persistence | `record_outcome()` exists, no reconciliation job | build reconciliation job once real outcomes exist | B-01 | low | 0.5-1 day when justified | outcomes automatically linkable to predictions |
| Shadow inference | implemented, tested, never executed live | none required now | B-01 for meaningfulness | low | none | first real shadow prediction recorded |
| Production inference (V1) | live, working | none | none | none | none | already done |
| Monitoring | not audited this session | full audit needed before any production ML claim | none | medium (unknown state) | 0.5-1 day | explicit monitoring/alerting plan exists |
| Logging | structured logging confirmed for AI failures (verified live) | none blocking | none | low | none | already sufficient |
| Tests | 451/451 passing | add HASEBHA-native-data tests once that data exists | B-01/B-02 | low | ongoing | maintained green |
| E2E | never run for shadow path (B-04) | run once meaningful (Section 6, B-04) | B-01/B-02 for meaningfulness | low | 1-2 hrs | a real order provably round-trips through the shadow path |
| Docker | compose file exists, containers correctly stopped when idle | none | none | low | none | already sufficient |
| Deployment | Railway/Vercel referenced in git log, not re-audited this session | full deployment-readiness audit (Section 15) | none | unknown (not measured) | 1-2 hrs | explicit gate-by-gate pass/fail |
| Security | AI service auth already implemented (per git log "add AI service authentication") | not re-audited this session | none | unknown | TBD | explicit audit |
| Documentation | extensive (this plan + 4 audit rounds) | keep synced with future phases | none | low | ongoing | maintained |

---

## 14. Testing Strategy

**Already verified this session-chain** (do not repeat): unit tests for all 3 model services
(hash verification, cold-start handling, tampered-artifact rejection), integration tests
(FastAPI TestClient against real loaded artifacts for all 3 routes), backward-compatibility
tests (old-format JSONL rows alongside new-format rows), auth tests, decision-engine tests,
feature-contract tests (forbidden-field checks), leakage tests (0/4 and 0/30 failures across
Olist V3 and V2 respectively), calibration tests (implicit in the calibration reports' own
methodology), TypeScript compile/lint checks on the Medusa subscriber. Full count: 451/451
passing as of the last code change.

**Required from now on, once B-01/B-02 progress**: real E2E test (Section 6, B-04); HASEBHA
data quality tests (prevalence sanity, label completeness, per Section 9's checklist); a
temporal-split test on real HASEBHA data once enough exists; a Medusa duplicate-event
idempotency test specifically exercised against a live event (the idempotency KEY logic is
implemented and code-reviewed but has not been proven against an actual duplicate delivery);
production-regression tests once any HASEBHA-native model is proposed for deployment.

---

## 15. Deployment Readiness (explicit gates)

| Gate | Status | Reason |
|---|---|---|
| Gate 1 — Code | PASS | 6 modified tracked files, all additive, backward-compatible, TypeScript/lint clean |
| Gate 2 — Tests | PASS | 451/451 passing as of the last code change |
| Gate 3 — Data | **BLOCKED** | 5 orders, 0 fulfillment outcomes — insufficient for any production ML claim |
| Gate 4 — ML | **BLOCKED** | no HASEBHA-native model exists; production-parity model is WEAK and Olist-trained, not HASEBHA-trained |
| Gate 5 — Target | **BLOCKED** | no real business SLA defined (B-02) |
| Gate 6 — Shadow | PARTIAL | implemented and tested, never executed against a real order |
| Gate 7 — E2E | **NOT RUN** | B-04, not yet justified to run given Gates 3-5 are blocked |
| Gate 8 — Monitoring | NOT APPLICABLE YET | not audited this session; not blocking since no ML production claim is being made |
| Gate 9 — Security | PASS (partial) | AI service auth exists per git history; not re-audited in full this session |
| Gate 10 — Production approval | **NOT REQUESTED** | correctly not requested — Gates 3-5 block any legitimate request |

**The system is deployable for its EXISTING scope** (V1 fulfillment risk, Arabic NLP, Instacart
recommendation — all already live). **It is NOT deployment-ready for any NEW HASEBHA-native
fulfillment-risk capability**, because Gates 3, 4, and 5 are blocked by data and business
decisions, not by engineering.

---

## 16. Project End States

**GREEN**: A HASEBHA-native model, trained on real first-party order+fulfillment data against
a real business-defined SLA, survives rolling-origin temporal validation with worst-period
performance materially above baseline and a stable calibration profile. Ship: the trained
model, wired first in continued shadow mode for a probation period, then promoted to
influence a LOW/MEDIUM/HIGH triage tier (never automated refunds/cancellations, per this
project's permanent rule), with full model card and reproducibility manifest.

**YELLOW (current state)**: Infrastructure is production-ready (hash-verified loading,
fail-soft integration, calibration pipeline, feedback persistence with raw-feature
reconstruction) but ML requires first-party data accumulation and a business SLA decision
before any HASEBHA-native model can legitimately exist. Ship: nothing new beyond what's
already live (V1, Arabic, Instacart); continue shadow-mode data collection; do not claim any
new fulfillment-risk capability.

**RED**: If, after sufficient first-party data and a real SLA both exist, a HASEBHA-native
model still cannot beat baseline with a CI excluding zero across multiple stages of the
Section 10 ladder — conclude the current predictive problem is not learnable from the
available feature set, document it as a formal negative result (consistent with this
project's entire prior practice), and do not deploy any model. Ship: the negative-result
documentation itself, plus the existing V1/Arabic/Instacart capabilities, unchanged.

---

## 17. Priority System

**P0 (blocking)**: B-00 (disk space — flagged to user, not actioned here), B-01 (data
volume — outside engineering control), B-02 (business SLA decision — outside engineering
control).
**P1 (essential, engineering-actionable)**: same_zone StockLocation resolution (Section 7,
Category C); monitoring audit (Section 13); deployment-readiness re-audit (Section 15).
**P2 (important, deferred until B-01/B-02 progress)**: E2E validation (B-04); outcome
reconciliation job; Postgres migration for persistence.
**P3 (optional)**: n_installments derivation research (Category C, unexplored); full security
re-audit.

**NEXT 24 HOURS**: none — no P0 item is engineering-actionable within 24 hours; flag B-00 to
the user now (done, this plan).
**NEXT 3 DAYS**: implement same_zone resolution (P1, small, self-contained); begin monitoring
audit.
**NEXT 7 DAYS**: complete deployment-readiness re-audit; await any business SLA decision.
**NEXT 2 WEEKS**: if B-02 resolves, implement the `fulfillment_due_at` mechanism (already
specified); begin monitoring real order/fulfillment volume against the Section 9 sample-size
targets.
**NEXT 30 DAYS**: reassess data volume against Section 9's minimums; if unmet, continue
waiting — do not force premature modeling.
**NEXT 60-90 DAYS**: if minimums are met and B-02 resolved, execute the Section 10 experiment
ladder in a dedicated, explicitly authorized future session.

These timeframes are **planning heuristics tied to task dependencies, not commitments** — P0
items depend on external business/data timelines this plan cannot control or predict.

---

## 18. Stop Conditions (for any future modeling session)

Stop and do not proceed further if: no first-party outcomes exist yet (current state); the
target is not yet business-defined (current state); a required feature genuinely does not
exist and cannot be cheaply derived; two or more experiments in the Section 10 ladder show
negative or flat results in a row (mirroring the Arabic track's own "3 negative results →
stop" precedent); a challenger's confidence interval overlaps the baseline; measured sample
size falls short of Section 9's minimums; worst-period temporal performance collapses the way
Olist V2's did, without a clear adaptation strategy already validated. **Low-information
experimentation is worse than stopping** — this has been this project's operating principle
across every session in this chain and is reaffirmed here.

---

## 19. What NOT To Do

Do not modify, retrain, or delete Olist V1. Do not tune on the protected/exposed stress block.
Do not invent a shipping SLA in code. Do not create synthetic HASEBHA business data. Do not
allow any feature to use information not available at the prediction timestamp (future
leakage). Do not present Olist-only features or results as HASEBHA capabilities. Do not change
the target definition merely to raise AUC. Do not use a random (non-temporal) split as the
primary evaluation for any fulfillment-risk model. Do not run uncontrolled/broad
hyperparameter search. Do not expand beyond the authorized model family list (Section 11)
without a stated, evidence-based reason. Do not rewrite or delete any historical report,
including the negative-results register. Do not modify any frozen track without following the
explicit stop-and-justify procedure (Section 2). Do not treat the 5 existing HASEBHA orders as
a meaningful dataset for any purpose. Do not claim shadow-mode validation on real traffic
until it has actually happened (verified via database evidence, not code/log inference).

---

## 20. Final Execution Roadmap

| Phase | Objective | Inputs | Tasks | Dependencies | Deliverables | Success Criteria | Stop Conditions | Risks | Est. Effort |
|---|---|---|---|---|---|---|---|---|---|
| 0. Repository verification | Establish ground truth | repo, git, live DB | inspect state (COMPLETE, this plan) | none | this document | state matches artifacts | N/A | stale assumptions | DONE (this session) |
| 1. Scientific forensic audit | Trace every headline number to evidence | prior reports | evidence matrix, negative-results register (COMPLETE, prior session) | Phase 0 | committee-defense package | all metrics artifact-traced | untraceable metric found | none identified | DONE (prior session) |
| 2. Target/business definition | Get a real HASEBHA SLA definition | HASEBHA ops decision | ops defines `fulfillment_due_at` semantics | none (business-owned) | a real business rule | SLA is unambiguous, observable, reproducible | ops cannot commit to an unambiguous rule | delay indefinite, outside project control | UNKNOWN (business-owned) |
| 3. Data collection readiness | Confirm schema/capture correctness | Phase 2 | verify `fulfillment.shipped_at` capture works against the real SLA once defined | Phase 2 | a verified capture path | a test fulfillment event correctly triggers breach computation | capture is broken or ambiguous | schema drift | 0.5-1 day once Phase 2 resolves |
| 4. Shadow instrumentation | Already largely complete | existing shadow route | implement same_zone resolution (P1); minor hardening | none | updated shadow feature builder | same_zone genuinely resolved | N/A | low | 1-2 hrs |
| 5. First-party data accumulation | Reach Section 9 minimums | real HASEBHA operation | monitor, do not force | Phase 2/3 | periodic volume check-ins | minimums met (Section 9) | volume stagnates | business-dependent, outside project control | ongoing, unknown duration |
| 6. HASEBHA-native feature engineering | Build the real training frame | Phase 5 data | apply Section 9's data-quality checklist | Phase 5 | a clean HASEBHA-native dataset | quality checklist passes | quality checks fail | data-quality issues discovered late | 1-3 days once triggered |
| 7. Controlled model experiments | Execute Section 10 ladder | Phase 6 | run stages 0-8 with predeclared stop rules | Phase 6 | experiment log, one accepted model or a documented negative result | any stage beats baseline with CI excluding zero | 2+ consecutive negative stages | overfitting to a small first dataset | 3-5 days once triggered |
| 8. Temporal validation | Confirm robustness, not just point estimate | Phase 7 | rolling-origin evaluation, stress-block diagnostic only | Phase 7 | temporal validation report | worst-period AUC materially above baseline | worst-period collapse without a validated fix | regime shift (as seen in Olist V2) | 1-2 days |
| 9. Shadow validation | Confirm against real eventual outcomes | Phase 8 | run the validated model in shadow, wait for real outcomes to accrue, compare | Phase 8, more time for outcomes to occur | shadow-vs-actual comparison report | predictions correlate meaningfully with real outcomes | no correlation, or unstable correlation | requires waiting for real future outcomes | weeks to months (data-dependent) |
| 10. Production approval | Formal go/no-go | Phase 9 | present evidence package, per this project's own acceptance criteria | Phase 9 | approval decision record | explicit human sign-off | evidence insufficient | premature promotion | 1 review session |
| 11. Deployment | Ship the approved model | Phase 10 | staged rollout, monitored | Phase 10 | live model, monitored | no production incidents, useful signal observed | any safety concern | operational risk | 1-2 days |
| 12. Monitoring and retraining | Sustain and improve | Phase 11 | drift monitoring, scheduled re-evaluation | Phase 11 | ongoing monitoring reports | drift detected and handled before it causes harm | N/A (ongoing) | model staleness | ongoing |

---

## 21. Final Decision

- **CURRENT PROJECT STATUS**: **YELLOW.**
- **CURRENT ML STATUS**: Strong, well-validated research results exist (Arabic, Instacart,
  Olist V3 research) but no validated HASEBHA-native fulfillment-risk model exists or can
  currently be built — blocked on data and a business decision, not on modeling capability.
- **CURRENT ENGINEERING STATUS**: Solid — hash-verified, fail-soft, tested (451/451),
  backward-compatible infrastructure exists and is reused across every model added so far.
  One small open item (same_zone resolution).
- **PRIMARY BLOCKER**: The combination of B-01 (no real fulfillment outcome data) and B-02
  (no real business-defined SLA) — neither solvable by engineering or modeling.
- **MOST IMPORTANT NEXT ACTION**: Obtain a real HASEBHA business decision on the shipping SLA
  (B-02) — this is the one blocker where a decision, not time, is the limiting factor, and it
  unblocks legitimate target construction the moment real fulfillment data starts arriving.
- **WHAT SHOULD NOT BE TOUCHED**: Olist V1 (production, frozen), all other frozen tracks
  (Arabic, Amazon, Instacart, Olist V2/V3, DataCo/EAGLE, Jumia), the existing shadow-mode
  infrastructure (functionally complete for its current purpose).
- **WHAT REQUIRES THE USER'S DECISION**: the HASEBHA shipping-SLA business definition (B-02);
  whether/when to invest in cleaning up the machine's disk space (B-00, outside this project);
  whether to prioritize the same_zone engineering item now or defer it.
- **WHAT I CAN EXECUTE WITHOUT ASKING**: see Execution Authorization Matrix below.
- **WHAT REQUIRES EXPLICIT APPROVAL**: see Execution Authorization Matrix below.
- **ESTIMATED PATH TO PROJECT COMPLETION**: Not estimable in calendar time — the critical path
  runs through a business decision (B-02) and organic order-volume growth (B-01), neither of
  which this project controls or can accelerate. Once both are resolved, the remaining
  technical path (Phases 6-11 above) is estimated at roughly 1-2 weeks of active engineering
  effort plus an unavoidable multi-week-to-multi-month waiting period for Phase 9 (shadow
  validation against real accruing outcomes).

---

## EXECUTION AUTHORIZATION MATRIX

| Task | Authorized Without Approval? | Requires Approval? | Reason |
|---|---|---|---|
| Implement same_zone StockLocation resolution | YES | — | Small, additive, backward-compatible, no frozen track touched, no business decision needed |
| Run a full deployment-readiness re-audit (read-only) | YES | — | Evidence-gathering only, no code change |
| Run a full security re-audit (read-only) | YES | — | Evidence-gathering only, no code change |
| Add HASEBHA data-quality tests once real data exists | YES | — | Additive testing, no production behavior change |
| Build the outcome-reconciliation job (once real outcomes exist) | YES | — | Additive, uses the already-approved `record_outcome()` contract |
| Run the real local E2E test (B-04) | YES | — | Read/verify only, no production behavior change, uses existing test infrastructure |
| Migrate PredictionFeedbackStore JSONL → Postgres | **NO** | YES | New infrastructure/schema decision with cost and maintenance implications; this project's own prior sessions explicitly deferred this pending volume justification |
| Define/implement the real `fulfillment_due_at` business SLA | **NO** | YES | Explicitly a business decision (B-02) — engineering must not invent it |
| Execute any stage of the Section 10 experiment ladder | **NO** | YES | This is new modeling on (eventually) real first-party data — requires explicit authorization per this project's own "code only after explicit authorization" rule, and is contingent on B-01/B-02 first |
| Retrain, modify, or replace the Olist V1 production model | **NO** | YES (and only with compelling new scientific evidence per Section 2) | Frozen, production, no defect found |
| Retrain any frozen research track (Arabic, Amazon, Instacart, Olist V2/V3, DataCo/EAGLE) | **NO** | YES (and only with compelling new scientific evidence) | Frozen per every prior session's explicit conclusion |
| Any Jumia-related work | **NO** | YES (and effectively never, per permanent project policy) | Permanently excluded |
| Promote any model from shadow to influencing a live decision/action | **NO** | YES | Safety-critical; requires the full Phase 10 approval process |
| Any automated customer-facing action (refund, cancellation, promise change) | **NO** | **PERMANENTLY NOT AUTHORIZED under any circumstance in this plan** | Explicit, permanent project rule, not a temporary gate |
| Clean up disk space / delete cached model checkpoints on C: | **NO** | YES | Outside this project's authority to decide what machine-level artifacts are safe to remove; flagged to user, not actioned |
| Update this execution plan as new evidence arrives | YES | — | Documentation maintenance, no production/model change |
