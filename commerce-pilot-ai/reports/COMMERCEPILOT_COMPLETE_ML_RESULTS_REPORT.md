# CommercePilot — Complete ML Results Reconstruction

Verified: 2026-08-21. Read-only reconstruction from actual repository artifacts. No new
training, HPO, or experiments were run. Every number below is traced to a source artifact;
where a number could not be traced, it is marked NOT VERIFIED rather than guessed.

**Classification key used throughout**: VERIFIED (I directly re-read the artifact and it
supports the claim) / REPRODUCED (an independent re-run this session matched a prior number) /
REPORTED (an artifact states it, not independently re-run this session) / NOT VERIFIED /
NOT RUN / REJECTED / FROZEN / PRODUCTION / RESEARCH ONLY / DIAGNOSTIC ONLY.

---

## 1. Executive Summary

Six ML/NLP tracks exist: Arabic sentiment (FROZEN), Amazon sentiment (FROZEN, partially
unwired), Instacart recommendation (FROZEN, production), Olist V1 (PRODUCTION), Olist
V2/V3 (RESEARCH), and DataCo/EAGLE (UNRESOLVED research reproduction). A seventh
"HASEBHA production-parity" track exists as an explicit attempt to transfer Olist V3 findings
to the real single-vendor Egyptian store; it produced a WEAK model wired only in shadow mode
and never yet executed against a real order. The project's single most defensible scientific
contribution is not any one model score — it is the rigorously demonstrated **gap between
research-feature performance and production-feature performance** (Section 12/17), and the
honestly-reported DataCo reproduction failure (Section 16).

## 2. Dataset-by-Dataset Inventory

| # | Track | Type | Status |
|---|---|---|---|
| 1 | Arabic Sentiment (MPOLD/ASTD/LABR) | NLP classification | FROZEN, PRODUCTION (wired) |
| 2 | Amazon NLP (Appliances reviews) | NLP classification | FROZEN, unwired at inference |
| 3 | Instacart Recommendation | Ranking/recsys | FROZEN, PRODUCTION (wired) |
| 4 | Olist V1 | Tabular classification | PRODUCTION (wired, live) |
| 5 | Olist V2 | Tabular classification | RESEARCH, STRESS_BENCHMARK |
| 6 | Olist V3 Seller-SLA (22-feature) | Tabular classification | RESEARCH_FULL_FEATURE_MODEL |
| 7 | Olist V3 Customer-Late T0 (stacked) | Tabular classification | REJECTED for flagship use |
| 8 | Olist V3 Customer-Late T1 | Tabular classification | RESEARCH, MODERATE/TRIAGE |
| 9 | Olist V3 Calibration | Post-hoc calibration | APPLIED to seller-SLA and production-parity models |
| 10 | DataCo/EAGLE | Graph/temporal reproduction | UNRESOLVED_PROTOCOL_MISMATCH |
| 11 | HASEBHA production-parity (13-feature) | Tabular classification | RESEARCH-TRAINED, SHADOW-WIRED, WEAK |
| 12 | HASEBHA shadow/production transfer | Engineering integration | IMPLEMENTED, NEVER EXECUTED AGAINST A REAL ORDER |

Engineering-only infrastructure (not ML tracks, labeled accordingly): FastAPI `ai_service`,
Decision Engine (deterministic rule evaluator, not a model), `PredictionFeedbackStore`
(JSONL), Medusa `order-placed.ts` subscriber.

---

## 3. Arabic Sentiment (Special Reconstruction, Section 8)

### 3.1 Dataset
Three tasks: MPOLD (offensive-language, task "E"), ASTD (4-class sentiment, task "B2"), LABR
(task "C"). Group-safe splits enforced (author/document grouping to prevent leakage across
train/val/test) — VERIFIED via `configs/nlp_champion_registry.yaml`'s reference to
`phase2c_nlp_transformer_finalist_confirmation_2026-08-11` (3-seed confirmation, seeds
101/202/303, fixed split seed 20260809).

### 3.2 Problem Definition
Per-task text classification; MPOLD binary (Offensive/Non-Offensive), ASTD 4-class
(NEG/NEUTRAL/OBJ/POS), LABR binary-ish sentiment. Standard supervised classification, no
target-definition controversy documented for this track (unlike Olist).

### 3.3 Champion selection (MPOLD, task E)
VERIFIED from `configs/nlp_champion_registry.yaml`: UBC-NLP/MARBERT, mean Macro-F1 **0.8280**
(std 0.0111) vs classical frozen baseline 0.6634. `champion_promotion_executed: false` at
registry-freeze time — this was a **recommendation**, not an executed artifact swap, per the
registry's own explicit note ("no champion artifact has been overwritten").

### 3.4 Post-freeze improvement candidates (this project's most recent Arabic work)
VERIFIED from `reports/generated/arabic_sota/ARABIC_FINAL_DECISION.json` (decision_date
2026-08-18):

| Experiment ID | Model/variant | Seed(s) | Macro-F1 | Delta vs baseline | Decision | Reason |
|---|---|---|---|---|---|---|
| baseline_seed42 | plain MARBERTv2, group-safe | 42 | 0.8130376 | — | CHAMPION | — |
| A_label_smoothing | label smoothing | 42 (screening) | 0.7944433 | -0.0186 | REJECTED | clear decrease |
| B_class_weighted_CE | class-weighted CE | 42 (screening) | 0.8147855 | +0.00175 | REJECTED | sub-threshold (+0.18pp < predeclared +0.3pp bar) |
| C_rdrop_alpha5 | R-Drop α=5 | 42 (screening) | 0.8041008 | -0.0089 | REJECTED | decrease AND higher cost (221.2s runtime, 4043MB VRAM) — worse on both axes |
| D_swa_or_ema | SWA/EMA | NOT ATTEMPTED | — | — | NOT RUN | 3/3 prior candidates failed; deliberate stop, not silent drop |
| E_supcon | Supervised contrastive | NOT ATTEMPTED | — | — | NOT RUN | same reasoning as D, plus highest implementation risk |

**FINAL_ARABIC_CHAMPION**: MARBERTv2, plain unweighted cross-entropy, group-safe split, seed
42, best checkpoint epoch 5.
**Single-seed headline**: Macro-F1 = **0.8130376** (seed 42 only).
**Multi-seed evidence**: 3-seed mean = **0.7905768**, std NOT independently re-quoted this
session beyond what's in the JSON (marked VERIFIED from artifact: the JSON reports the mean
directly; a per-seed breakdown for this specific 3-seed run was not re-extracted this session
— NOT VERIFIED at the individual-seed level for this exact metric, though the methodology
[seeds 101/202/303] is VERIFIED from the earlier registry).
**Neutral-class F1**: 0.7225806 (VERIFIED, same artifact).

### 3.5 SARF challenger
VERIFIED from the same artifact: 3-seed mean 0.790103 vs MARBERT's 0.790577 — **statistically
indistinguishable** (delta -0.0005) — **REJECTED on cost-effectiveness** (~1.93x runtime,
~1.83x VRAM for parity performance), explicitly NOT rejected for inferiority.

### 3.6 Defensible claim
"MARBERTv2 (plain, group-safe) is the best evidence-supported Arabic sentiment model under
the tested leakage-safe protocol and experiment budget." NOT defensible: "globally optimal,"
"production Egyptian e-commerce model" (no e-commerce-domain Arabic data was used — MPOLD/ASTD/
LABR are general Arabic NLP benchmarks, not HASEBHA product-review data — this distinction is
NOT explicitly flagged in the source artifacts I read, so I flag it here: **the domain
transfer from these benchmarks to actual HASEBHA customer text is UNKNOWN/NOT VERIFIED**).

### 3.7 Engineering test status
VERIFIED: `tests/test_arabic_sota_engineering.py` — 14/14 passing per this session's own
earlier direct run and the full-suite results throughout this session chain (447/447 overall,
Section 9 of the prior sprint's report).

---

## 4. Amazon NLP

### 4.1 Dataset
Amazon Appliances reviews. VERIFIED from `reports/generated/amazon/scope_decision.md`:
2,128,605 total rows in `reviews_text_ready.parquet`; verified-purchase scope selected as
primary modeling population (Negative 301,730 / Positive 1,640,151 / 3-star excluded 98,183
within verified purchases). Binary task: Negative (rating 1-2) vs Positive (rating 4-5).

### 4.2 Classical baseline/champion — VERIFIED
`reports/generated/amazon/metrics.json`: winner_model_type = `tfidf_wordchar_linearsvc`,
selected training size 100,000 (plateau rule: smallest size within 0.5pp of best macro-F1
across sizes 25k/50k/100k/200k, on validation only). Final model test_balanced (n=40,000):
**macro_f1 = 0.9453970, accuracy = 0.9454, ROC-AUC = 0.98639**. Dummy stratified baseline:
macro_f1 ≈ 0.5004 (near-trivial, as expected for balanced random guessing).

### 4.3 Transformer track — REPORTED, not the designated champion
`reports/generated/amazon/transformer_final_eval.json` shows a real evaluated DistilRoBERTa
run (test_balanced macro-F1 in the 0.95-0.97 range across threshold variants, temperature
1.1025, selected threshold 0.06). This is a **real, evaluated** result (REPORTED, artifact
exists), but `configs/nlp_champion_registry.yaml` explicitly notes: "Amazon transformer
confirmation (deferred, not cancelled)" — the transformer track was never promoted to champion
status. The classical LinearSVC remains the frozen champion per `metrics.json`.

### 4.4 CLAIM-VS-ARTIFACT DISCREPANCY (found this session, Section 25 audit item)
`src/ai_service/routers/nlp.py`'s docstring states Amazon "returns a structured
ARTIFACT_NOT_MATERIALIZED response: the frozen classical winner's hyperparameters/metrics are
evidenced, but the fitted model itself was never serialized to disk." **I directly checked**:
`artifacts/experiments/amazon/models/amazon_tfidf_wordchar_linearsvc_size100000.joblib`
**exists** (4,339,635 bytes, SHA256 `47ab5406717d7609d186f7084c0dc94756f4c6270de22345875774f2e5a903d9`,
which matches the hash recorded in `metrics.json`'s `final_model.artifact.sha256`). The
artifact is real and hash-verifiable. The accurate description is: the fitted classical model
DOES exist on disk, but `NlpInferenceService.keys_for_task(AMAZON_TASK)` in
`src/ai_service/services/nlp_inference.py` unconditionally returns `[]` — **the inference
service was simply never wired to load it**, an integration gap, not a missing artifact. This
is a real, minor documentation inaccuracy in the production code's own comment — flagged
honestly per the mission's audit requirement, not corrected (no code changes authorized this
session).

### 4.5 Defensible claim
"A classical TF-IDF+LinearSVC model achieves 94.5% macro-F1 on a held-out balanced Amazon
Appliances test set." NOT defensible: "Amazon sentiment is served in production" (it is not
currently loadable via the live API at all, regardless of artifact existence).

---

## 5. Instacart Recommendation

VERIFIED from `reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/` and
`reports/generated/instacart/protected_test_final_results.json`. Frozen candidate:
`hybrid_with_popularity_backfill`. Protected test: **26,314 users**, accessed exactly once
(`protected_test_access_count: 1`), stats refit on DEV_FIT+DEV_EVAL only, `post_test_tuning_
performed: false`, `candidate_switched_after_test: false` — a genuinely single-shot,
pre-committed protected-test protocol. Results at k=5 (VERIFIED): precision@5 = 0.36739,
recall@5 = 0.23783, NDCG@5 = 0.42978, hit_rate@5 = 0.78829, MAP@5 = 0.33143. Catalog coverage
32.25%; 92.17% of recommendations came from the top-20%-popular items — a real, disclosed
popularity-bias characteristic of the frozen candidate, not hidden. PRODUCTION status: wired
into the recommendation engine service (confirmed present in `main.py`'s lifespan), served via
`/v1/recommendations`.

---

## 6. Olist V1 (Section 9)

**Separate track from Olist V2/V3.** Predicts customer-late-delivery risk from exactly 2
features (`purchase_timestamp`, `approval_timestamp` → 9 derived strict-core timing features:
year/month/day-of-week/hour for each, plus purchase-to-approval seconds). Model: CatBoost.
VERIFIED: `src/ai_service/config.py` hard-pins `OLIST_MODEL_SHA256 =
"5a08ea55332332550a4436f87de91b479fab770a08ec232391d2141bc28a3b2c"`, and this session's own
earlier hash-verification run (Gate G3, prior sprint) confirmed the on-disk artifact at
`artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/models/catboost.cbm`
matches exactly. **This is the only model in the entire repository with real production
execution history**: I directly queried the live database this session and found 5 real
orders, all carrying `commercepilot_ai.fulfillment_risk.model_experiment_id =
"olist-phase2a-strict-core-v1"` — confirming Olist V1 actually scored real (if very few)
orders, including one legitimately-logged `AI_UNAVAILABLE` failure. **Untouched this entire
project (this session chain)** — confirmed via `git status` on
`src/modeling/olist/`, `configs/olist_phase2*.yaml`, `reports/generated/olist/` showing no
modifications across every session in this chain.

**Frozen test-set performance — VERIFIED this session**,
`reports/generated/olist/phase2a/final_test_metrics.json`: CatBoost ROC-AUC = **0.5634**
(dummy baseline 0.5, logistic regression and LightGBM also evaluated in the same artifact but
CatBoost is the deployed champion per `config.py`). This is notably weak — barely above random
— and worth stating plainly for a committee: **the only model with real HASEBHA production
execution history has a modest, frozen ROC-AUC of 0.56 on its own original test protocol**,
using only 2 raw input timestamps. Consistent with this, 4 of the 5 real orders in the live
database were scored `risk_class: "high"` at a fixed operating threshold of 0.1293
(`OLIST_RISK_THRESHOLD` in `config.py`) with scores clustering around 0.226-0.227 — i.e. a low
absolute threshold makes most orders register as "high risk," which is a calibration/threshold
characteristic worth the incoming supervisor's attention, not evidence of unusually risky real
orders (n=5 is far too small to conclude anything about real order risk distribution anyway).

---

## 7. Olist V2 — Failure As A Result (Section 10)

VERIFIED from `reports/generated/olist_v2/OLIST_V2_TEMPORAL_STABILITY_FINAL_REPORT.md` and
`FINAL_SCORECARD.json`, `CURRENT_STATE.json`/`.md`.

**Development**: rolling prequential dev-fold AUCs in the 0.72-0.76 range (static baseline
strategy).
**Protected-test collapse**: AUC = **0.5117** (near-random), reported in `CURRENT_STATE.json`
as the CRITICAL_FINDING.
**Investigation**: an earlier prior-session hypothesis blamed the May 2018 Brazilian trucker
strike. This was directly investigated and **REFUTED**: the true late-rate peak (21.4%,
n=7003) occurred in March 2018, two months BEFORE the strike window, which itself showed only
8.2% late rate (barely above the 8.1% average). `STRIKE_ASSOCIATION =
NOT_SUPPORTED_OR_INCONCLUSIVE` (VERIFIED, `FINAL_SCORECARD.json`).
**Temporal adaptation experiments** (max 3, historical-periods-only selection): static
baseline mean AUC 0.6529 (worst 0.5968) < recent-window 0.7054 (worst 0.6714) < expanding
0.7097 (worst 0.6714) < **recency-weighted 0.7126 (worst 0.6632), SELECTED**.
**Post-selection stress diagnostic**: recency-weighted strategy scores AUC = **0.5924** on the
same block that originally collapsed to 0.5117 — a real +0.0807 partial recovery, still well
below the historical dev range.
**Terminology correction (VERIFIED, explicit in the artifact)**: the block was renamed from
"protected final block" to **LATEST_TEMPORAL_STRESS_BLOCK**, with `LATEST_BLOCK_BLIND_STATUS
= EXPOSED` — it must never again be described as blind/protected, because it was used (post
hoc, for diagnostics only, never for selection) after being seen. This is the correct current
terminology; any document using "protected" for this block is describing outdated/incorrect
terminology.
**Why this is not automatically a leakage bug**: leakage tests passed cleanly (0/30 failures,
VERIFIED `CURRENT_STATE.json`) — the collapse is a genuine distributional/regime-shift
phenomenon, not a data-leakage artifact.
**Final role**: `FINAL_OLIST_ROLE = STRESS_BENCHMARK`, `ADVANCED_MODEL_NEXT_STAGE = NO`.
This track is scientifically valuable precisely because it demonstrates that naive static
temporal modeling is unsafe on this data — a lesson carried into Olist V3's evaluation design.

---

## 8. Olist V3 — Seller-SLA (Section 11, one of the largest tracks)

### 8.1 Target
`SELLER_HANDOFF_SLA_BREACH = order_delivered_carrier_date > shipping_limit_date`. VERIFIED
from `scripts/olist_v3_seller_sla_pipeline.py` and `reports/generated/olist_v3_multistage/
SELLER_SLA_DATASET_META.json`. Cohort: single-seller orders with a single distinct
`shipping_limit_date` (97,388 of ~98,666 item-joinable orders, 98.70%; VERIFIED
`MULTI_SELLER_TARGET_VALIDITY_AUDIT.json`).

### 8.2 Prediction point
T0 = `order_purchase_timestamp`. Justified (VERIFIED, `EVENT_SEMANTICS_AUDIT.json`):
`shipping_limit_date` never precedes purchase (0% of orders) or approval (0.12% of orders,
attributed to data noise), median gap purchase→limit = 6.0 days — a legitimately-known future
deadline, not a T0 violation.

### 8.3 Feature groups — CORRECTED COUNT (verified by direct Python import this session)
`from src.ai_service.services.seller_sla_risk import FEATURE_ORDER; len(FEATURE_ORDER)` =
**23**, not 22. Broken down and verified to sum exactly:
- **Order/product** (11): days_to_shipping_deadline, n_items, n_distinct_products,
  n_categories, total_price, total_freight, total_freight_over_price, weight_g, volume_cm3,
  payment_value, n_installments
- **Temporal** (3): purchase_weekday, purchase_hour, purchase_month
- **Geo** (1): same_state
- **Seller-history** (8, causal shift(1)+rolling): seller_past_order_count,
  seller_past_breach_rate_expanding, seller_past_handling_median_expanding,
  seller_past_handling_std_expanding, seller_breach_rate_30d, seller_breach_rate_90d,
  seller_handling_mean_30d, seller_recent_load_7d

11 + 3 + 1 + 8 = **23**, confirmed with zero unassigned/duplicate names.

**DISCREPANCY FLAGGED (Section 25 audit item)**: this project's own artifacts consistently
label this model "the 22-feature model" (`SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json`,
`OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json`, this report's own earlier drafting, and the
Gate 1 re-audit's "22 features" framing throughout the prior sprint). The actual live
inference feature list (`FEATURE_ORDER`) has **23** entries. This is a real, small,
previously-uncaught off-by-one labeling inconsistency across multiple artifacts, not a new
error introduced by this report — it is corrected here rather than silently propagated. It
does not change any reported metric (AUC/Brier/etc. were all computed against the actual
23-column training frame regardless of what the surrounding prose called it), so it is a
**documentation/terminology correction**, not a result-invalidating conflict.

### 8.4 Leakage audit
VERIFIED, `SELLER_SLA_LEAKAGE_TESTS.json`: **0/4 checks failed** — target-column exclusion
check, seller_past_order_count starts at 0 per seller, cold-start sentinel check, and a
30-comparison spot-check of the expanding breach-rate causal computation against manual
recomputation (0 mismatches).

### 8.5 Model development
LightGBM only (per the project's own "1 primary + 1 challenger only if justified" budget; no
CatBoost challenger was built for this track — VERIFIED, no CatBoost artifact exists in
`artifacts/experiments/olist_v3_multistage/`).

### 8.6 Temporal validation — VERIFIED, `SELLER_SLA_TEMPORAL_EVAL.json`
5 rolling-origin historical periods, mean AUC = **0.7702**, worst period AUC = **0.6762**
(period 2017-03..2017-05), std = 0.0483. Post-selection exposed stress-block diagnostic:
AUC = **0.7364** (2018-06..2018-08) — notably, this does NOT collapse the way Olist V2's
static model did, a real and meaningful contrast.

### 8.7 Interpretation
`STRONG_OPERATIONAL_MODEL` status applies **only to the Olist research population and
feature set**. This is a RESEARCH_FULL_FEATURE_MODEL classification, not a production
classification — see Section 12/17 for why.

---

## 9. Production-Availability Simulation (Section 12)

VERIFIED, `reports/generated/olist_v3_multistage/OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json`
and `SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json`.

The 22-feature model's 10 seller-history-dependent features (list VERIFIED in the simulation
script's `ALWAYS_UNAVAILABLE_SENTINEL` dict) plus `days_to_shipping_deadline` and
`n_installments` were forced to their exact real-production sentinel (never a fabricated
best-guess value — the SAME sentinel convention (-1.0 / 0) the live `SellerSlaRiskService`
cold-start path actually uses), and the SAME frozen model was re-scored on the SAME 5
historical temporal test periods.

**Original mean AUC**: 0.7702329747962686 (matches Section 8.6 exactly, VERIFIED same source
computation).
**Production-simulation mean AUC**: **0.518811509349124**.
**Worst-period simulation AUC**: **0.4949065094176262** (below 0.5 — worse than random on that
period).
**Delta**: **-0.2514214654471446**.
**Verdict** (VERIFIED, artifact's own field): `MATERIAL_COLLAPSE — frozen 22-feature model's
0.7702 status is RESEARCH_FULL_FEATURE_MODEL, NOT product performance.`

This is direct, quantitative evidence that the 0.77 headline result cannot be claimed as a
HASEBHA-relevant number.

---

## 10. Production-Parity Model (Section 13)

VERIFIED, `reports/generated/olist_v3_multistage/PRODUCTION_PARITY_MODEL_COMPARISON.json`.
Authorized as a genuine retraining (not model-zoo expansion) because the prediction
**contract** changed: 13 features restricted to those Gate 1's re-audit found genuinely
available in a single-vendor Medusa store (list VERIFIED,
`production_parity_seller_sla.py::FEATURE_ORDER`, confirmed exactly 13 entries by this
session's own earlier test `test_feature_order_has_exactly_13_features`).

**MODEL P (13-feature bootstrap-only)**: mean AUC = **0.5551351215216622**, worst = **0.528866096197271**.
**MODEL P+ (13 + 5 store-wide operational features)**: mean AUC = **0.5458327798711572** — a
**negative** gain of -0.0093 vs P. REJECTED (store-wide history across all Olist sellers
combined is a materially weaker/noisier signal than genuine per-seller history, exactly as the
parity re-audit's own semantic-honesty rule warned it likely would be).

**Comparison to research model**: 0.5551 vs 0.7702 — a 0.2151 AUC gap, consistent in
direction/magnitude with the Section 9 simulation's collapse.
**Calibration** (VERIFIED, `PRODUCTION_PARITY_CALIBRATION_REPORT.json`): ISOTONIC selected,
Brier 0.1255→0.0802 (held-out OOF), AUC preserved 0.5655→0.5699.
**Classification**: **WEAK** (explicit label in `production_parity_seller_sla.py`'s own
`PRODUCTION_PARITY_SIGNAL_STRENGTH = "WEAK"` config constant).

---

## 11. Customer-Late T0/T1 (Section 14)

VERIFIED, `reports/generated/olist_v3_multistage/TASK_B_C_RESULTS.json`.

**T0 stacked**: OLD (order/seller-history features, no realized-outcome leakage) mean AUC
**0.587347**, NEW (OLD + causal out-of-time predicted seller-SLA risk) mean AUC **0.590956**.
Mean delta = **+0.003609** — trivially small, ranging -0.0039 to +0.0083 across 4 periods.
**Decision**: REJECTED for flagship use / RESEARCH_ONLY — stacking adds negligible value on
this cohort. Note this differs from Olist V2's separately-computed customer-T0 estimate
(recency-weighted mean AUC 0.7126) because it uses a DIFFERENT, smaller cohort (single-seller
only) and a different feature set — the two numbers are **not directly comparable**, and this
report does not attempt to reconcile them into one "true" customer-T0 AUC (flagging this as a
genuine cross-track methodological difference, not a conflict requiring resolution, since both
are internally consistent within their own protocols).

**T1 dynamic** (post carrier-handoff): mean AUC **0.6890**, worst **0.6541**, mean PR-AUC
**0.2759**, mean recall@10% **0.2968**. Meaningfully stronger than T0 (both the OLD/NEW
versions and Olist V2's own T0 estimate) and does not exhibit Olist V2's severe stress-block
collapse pattern. Classified **MODERATE_EARLY_WARNING / TRIAGE-LEVEL** — real, useful for
human review prioritization, explicitly NOT strong enough for automated high-stakes action
(no automated cancellations/refunds authorized anywhere in this project).

---

## 12. Calibration (Section 15)

Two independent calibration exercises exist, for two different models — do not conflate them.

**Seller-SLA (22-feature research model)**, VERIFIED `SELLER_SLA_CALIBRATION_REPORT.json`:
ISOTONIC selected. RAW_BRIER 0.07171 → CALIBRATED_BRIER 0.07087. RAW_ECE 0.001134 →
CALIBRATED_ECE 0.000258. RAW_AUC 0.78855 → CALIBRATED_AUC 0.78820 (note: these AUC values are
computed on a held-out-half of the OOF calibration pool, NOT the same number as the 0.7702
5-period temporal mean in Section 8.6 — different evaluation slices, both legitimate, not a
conflict). Selection rule: lowest Brier among methods with AUC ≥ raw AUC - 0.005, calibrator
fit on one half of OOF dev predictions, evaluated on the other half, stress block never used.

**Production-parity (13-feature model)**: see Section 10 — ISOTONIC selected, same
methodology, smaller absolute improvement given the weaker underlying signal (higher thresholds
required: HIGH=0.513 vs the research model's HIGH=0.208, reflecting less separated score
distributions).

In both cases: ranking performance (AUC) was preserved within the predeclared 0.005 tolerance;
only probability calibration (Brier/ECE) improved.

---

## 13. DataCo/EAGLE Reproduction (Section 16)

### 13.1 Dataset identity — VERIFIED
DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS, Mendeley DOI 10.17632/8gx2fvg2k6.5, version 5.
Acquired via the Mendeley public content API; SHA256 of `DataCoSupplyChainDataset.csv`
(fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6) matched the API's recorded
hash exactly. **180,519 rows, 53 columns** — matches the EAGLE paper's stated
"180,519 order records with 53 features" exactly.

### 13.2 Graph reproduction — VERIFIED
46 nodes = 23 `Order Region` × 2 `Customer Country`, confirmed by direct groupby count against
the raw CSV — exact match to the paper's stated 46 nodes.

### 13.3 Paper reproduction (LSTM)
VERIFIED, `reports/generated/dataco/DATACO_LSTM_REPRODUCTION.json`, 4 predetermined seeds
(13, 42, 123, 2026 — chosen before any result was seen):

| Seed | Test AUC | Test Macro-F1 |
|---|---|---|
| 13 | 0.663224 | 0.611810 |
| 42 | 0.633431 | 0.599843 |
| 123 | 0.644292 | 0.464153 |
| 2026 | 0.640670 | 0.557961 |

Mean AUC = **0.645404**, std = **0.011006**. Published (paper-reported, REPORTED not
independently reproduced by the original authors here): LSTM AUC ≈ 0.9679, Macro-F1 ≈ 0.8095.
**Gap = 0.322**, far outside noise.

### 13.4 Forensic investigation — VERIFIED,
`DATACO_TARGET_FORENSIC_CORRECTION.json`. The paper's own quoted target formula
(`y_class = 1[d_next_mean > mu_v]`, mu_v = per-node train-only historical mean) was
re-implemented exactly as extracted from the paper's full text. Observed prevalence: 44.9-52.4%
across splits vs. the paper's reported 6.15%/2.84%/3.99%. **Mathematical root cause**: median
node-window order volume in this dataset is 174 orders; by the Central Limit Theorem, a
window-mean-vs-training-period-mean comparison at this sample size is expected to yield close
to 50% positive prevalence for a reasonably stationary process — quantitatively incompatible
with a 4-6% reported rate. One alternative (mu_v + k·sigma_v margin) was tested: it partially
fixes train prevalence (4.2% at k=0.5) but collapses val/test prevalence to near-zero
(0.1%/0.24%) — REJECTED, does not reproduce the paper's pattern. No further variants were
tried, per this project's own explicit anti-fishing rule (documented twice now — this is a
deliberate methodological stance, not a resource limitation).

### 13.5 EAGLE — NOT ATTEMPTED, methodological decision
VERIFIED, `LSTM_REPRODUCTION_GATE_DECISION.json`: the LSTM reproduction gate did not pass, so
per the project's own predeclared gating rule ("do not proceed to EAGLE if LSTM reproduction
is fundamentally broken"), EAGLE (the graph-attention architecture) was correctly never
implemented. This is a decision, not an incomplete task.

### 13.6 Status
`DATACO_REPRODUCTION_STATUS = UNRESOLVED_PROTOCOL_MISMATCH`. Structural reproduction (nodes,
windowing, split shape) PASSED; target-prevalence reproduction did not, for a mathematically
diagnosed reason. Never in scope as a HASEBHA production candidate regardless of outcome
(different domain — US-based DataCo logistics, not Egyptian e-commerce).

---

## 14. HASEBHA Transfer / Production Reality (Section 17)

```
Olist research (22-feature seller-SLA)         mean AUC 0.7702  [RESEARCH_FULL_FEATURE_MODEL]
        ↓ 10 seller-history + 2 other features have no HASEBHA analog
production-availability simulation             mean AUC 0.5188  [≈ random, MATERIAL_COLLAPSE]
        ↓ retrain on 13 genuinely-available features
production-parity model                         mean AUC 0.5551  [WEAK]
        ↓ wired in shadow mode (log-only, no automated action)
real-order execution                            0 of 5 real orders ever scored [NEVER RUN]
```

**Statement, directly supported by the artifacts above**: The project has validated
infrastructure (a working FastAPI/Medusa integration pattern, hash-verified model loading,
calibration methodology, a deterministic decision engine) and a research hypothesis (Olist
seller-history features carry real signal for the seller-SLA task, on Olist data), but has
**NOT** validated a useful HASEBHA first-party fulfillment-risk signal. The weak
production-parity model exists specifically to begin data collection, not because it is
believed to work.

---

## 15. Master Model Scorecard

| Track | Target | Prediction Time | Model | Features | Validation | Seeds | Primary Metric | Best | Worst | Status | Production Status | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Arabic MPOLD | offensive/non-offensive | N/A (text) | MARBERTv2 | text (BERT) | group-safe held-out | 1 (headline), 3 (confirmation) | Macro-F1 | 0.8130 (1-seed) / 0.7906 (3-seed mean) | — | FROZEN | PRODUCTION (wired) | ARABIC_FINAL_DECISION.json |
| Amazon Appliances | pos/neg sentiment | N/A (text) | TF-IDF+LinearSVC | word+char n-grams | held-out balanced (n=40,000) | 1 | Macro-F1 | 0.94540 | — | FROZEN | NOT wired at inference | metrics.json |
| Instacart | next-basket items | order time | hybrid + popularity backfill | user history | protected test (single access) | N/A | Precision@5 | 0.36739 | — | FROZEN | PRODUCTION (wired) | protected_test_final_results.json |
| Olist V1 | customer-late delivery | purchase/approval | CatBoost | 9 (timing-derived) | frozen test split (final_test_metrics.json) | 1 | ROC-AUC | 0.5634 | — | FROZEN | **PRODUCTION, live, 5 real orders scored** | reports/generated/olist/phase2a/final_test_metrics.json (VERIFIED, re-read this session), config.py hash, live DB |
| Olist V2 | customer-late delivery | purchase | LightGBM (recency-weighted) | ~similar to V1/V3 order features | rolling + exposed stress diagnostic | 1 | mean temporal AUC | 0.7126 | 0.6632 | RESEARCH, STRESS_BENCHMARK | NOT production | FINAL_SCORECARD.json |
| Olist V3 Seller-SLA | seller SLA breach | purchase (T0) | LightGBM | 22 | 5 rolling periods + exposed stress | 1 | mean temporal AUC | 0.7702 | 0.6762 | RESEARCH_FULL_FEATURE_MODEL | NOT production (feature parity FAIL) | SELLER_SLA_TEMPORAL_EVAL.json |
| Olist V3 Customer T0 stacked | customer-late delivery | purchase (T0) | LightGBM | order+seller-history+stacked risk | 4 rolling periods | 1 | mean AUC | 0.590956 | — | REJECTED (flagship) | NOT production | TASK_B_C_RESULTS.json |
| Olist V3 Customer T1 | customer-late delivery | carrier handoff | LightGBM | handling/slack/lane | 5 rolling periods | 1 | mean AUC | 0.6890 | 0.6541 | RESEARCH, MODERATE/TRIAGE | NOT production | TASK_B_C_RESULTS.json |
| Production-availability simulation | seller SLA breach | purchase (T0) | frozen 22-feature LightGBM, sentinel-substituted | 12 real + 10 sentinel | same 5 periods | 1 | mean AUC | 0.5188 | 0.4949 | DIAGNOSTIC ONLY | proves NOT production-usable as-is | OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json |
| Production-parity model | seller SLA breach | purchase (T0) | LightGBM | 13 (HASEBHA-derivable) | same 5 periods | 1 | mean AUC | 0.5551 | 0.5289 | RESEARCH-TRAINED, SHADOW-WIRED, WEAK | shadow-only, never scored a real order | PRODUCTION_PARITY_MODEL_COMPARISON.json |
| DataCo LSTM | node-window SLA deterioration | 14-day feature window | LSTM (paper-compatible) | 6 node features | chronological 70/15/15, 4 predetermined seeds | 4 | mean AUC | 0.645404 (std 0.011) | 0.633431 | RESEARCH, REPRODUCTION FAILURE (diagnosed cause) | NOT production, never in scope for HASEBHA | DATACO_LSTM_REPRODUCTION.json |
| DataCo EAGLE | same | same | graph-attention (paper) | — | — | 0 | — | NOT RUN | — | NOT RUN (methodological decision) | N/A | LSTM_REPRODUCTION_GATE_DECISION.json |

---

## 16. Experiment Decision Matrix

| Experiment | Hypothesis | Result | Threshold/Criterion | Decision | Reason |
|---|---|---|---|---|---|
| Arabic label smoothing | improves Macro-F1 | 0.7944 (-0.0186) | ≥ +0.3pp | REJECTED | clear decrease |
| Arabic class-weighted CE | improves Macro-F1 | 0.8148 (+0.0018) | ≥ +0.3pp | REJECTED | sub-threshold |
| Arabic R-Drop α=5 | improves Macro-F1 | 0.8041 (-0.0089), 221s/4043MB | ≥ +0.3pp, comparable cost | REJECTED | worse on both axes |
| Arabic SARF | matches/beats MARBERT at acceptable cost | 0.7901 vs 0.7906 (statistically tied), 1.9x cost | parity + acceptable cost | REJECTED | cost, not accuracy |
| Olist V2 static→adaptive temporal strategies | recency-weighting improves worst-case robustness | mean 0.6529→0.7126, worst 0.5968→0.6632 | best mean AUC among historical-only comparison | recency-weighted SELECTED | best mean among 3 tested |
| Olist trucker-strike hypothesis | explains the AUC collapse | true late-rate peak 2 months before strike window; strike window itself near-average | direct evidence check | REFUTED | contradicted by the data |
| Olist V3 T0 stacking (predicted seller risk) | improves customer-T0 AUC | +0.0036 mean delta | material improvement | REJECTED for flagship | trivial/inconsistent-sign delta |
| Olist V3 store-wide ops features (Model P+) | improve production-parity AUC | 0.5458 vs 0.5551 (P) | positive gain | REJECTED | negative gain |
| DataCo sigma-margin target alternative | reproduces paper's prevalence | fixes train (4.2%), breaks val/test (~0%) | reproduces train AND val AND test | REJECTED | val/test mismatch |
| DataCo EAGLE | reproduce paper's ~0.977 AUC | NOT RUN | LSTM gate must pass first | NOT ATTEMPTED | LSTM gate failed |
| Seller-SLA isotonic calibration | improves Brier/ECE without damaging AUC | Brier 0.0717→0.0709, ECE 0.00113→0.00026, AUC 0.7886→0.7882 | AUC drop ≤ 0.005 | ACCEPTED | criteria met |
| Production-parity isotonic calibration | same | Brier 0.1255→0.0802, AUC 0.5655→0.5699 | AUC drop ≤ 0.005 | ACCEPTED | criteria met (AUC actually improved slightly) |

---

## 17. What the Project Actually Contributed

**ML contributions**: a rigorously validated multi-task Arabic sentiment champion with honest
negative-result documentation for 5 rejected improvement attempts; a strong (on Olist data)
seller-SLA temporal model design (causal rolling seller-history features, leakage-tested); a
methodologically clean production-availability-simulation technique for quantifying feature
transfer risk before deployment.

**Data/validation contributions**: demonstration that a naive "protected test block" can
silently become non-blind if reused for diagnostics, and the correct terminology fix
(EXPOSED, not protected); demonstration that the DataCo/EAGLE paper's target formula is
mathematically inconsistent with its own reported prevalence at this dataset's scale — a real,
citable finding about that specific reproduction, not a defect in this project's own work.

**Engineering contributions**: a working hash-verified model-loading pattern reused across 3
services; a fail-soft Medusa integration (AI failures never block checkout, verified live in
the DB with a real `AI_UNAVAILABLE` record); a generic Decision Engine requiring no code
changes to accept new risk signals.

**Production-readiness findings**: the central one — Olist-trained seller-history features do
not transfer to a single-vendor store; production performance must be measured on the actual
deployable feature set, not the research feature set.

**Negative/reproducibility findings**: Arabic candidate rejections (Section 3.4); Olist V2
temporal collapse; Olist V3 T0-stacking's negligible gain; production-parity's negative gain
from store-wide operational features; DataCo LSTM reproduction failure with diagnosed cause.

This project has NOT contributed: a validated HASEBHA production ML signal beyond the
2-feature Olist V1 model; a resolved DataCo/EAGLE reproduction; any evidence about Arabic-text
performance specifically on HASEBHA's own customer text (general-benchmark Arabic NLP only).

---

## 18. What We Can Claim In Front Of A Committee

### Defensible claims
- MARBERTv2 group-safe is the best evidence-supported model for MPOLD/ASTD/LABR under this
  project's tested protocol and budget (not "globally optimal").
- The Instacart hybrid recommender achieves the reported protected-test metrics under a
  genuinely single-shot, pre-committed test-access protocol.
- The Olist V3 seller-SLA model achieves mean temporal AUC 0.7702 **on Olist research data,
  under the 22-feature research protocol** — this is a real, leakage-tested, temporally
  validated research result.
- A quantitative, causally-explained demonstration exists that this same model's performance
  collapses to near-random (AUC 0.52) when restricted to features realistically available in
  a single-vendor Medusa deployment.
- A retrained, feature-restricted "production-parity" model achieves AUC 0.5551 (weak) on the
  same protocol — the best currently-known HASEBHA-transferable estimate.
- DataCo dataset acquisition and graph-structure reproduction are independently verified
  (hash-matched, node-count-matched); LSTM reproduction failed for a mathematically diagnosed
  reason, not a bug; EAGLE was never attempted, correctly, per a predeclared gate.

### Claims we must NOT make
- "The project has a production HASEBHA fulfillment-risk model with 77% AUC" — FALSE; that
  number describes a model that cannot run on real HASEBHA orders.
- "Shadow mode has validated the production-parity model on real orders" — FALSE; zero real
  orders have ever been scored by it (directly verified against the live DB this session).
- "Arabic sentiment is validated for HASEBHA customer text" — NOT SUPPORTED; only
  general-benchmark Arabic NLP data was used.
- "DataCo/EAGLE reproduces the published 0.977 AUC" — FALSE; LSTM reproduction reached 0.645,
  and EAGLE was never run.
- "Amazon sentiment is served in production" — FALSE; the artifact exists but the inference
  service is not wired to load it.
- "The project has demonstrated real local E2E integration" — FALSE per the prior supervisor-
  handoff investigation (directly re-confirmed via the same DB evidence this session); only
  unit/integration tests against real artifacts have been demonstrated, never a live
  Medusa+Postgres+FastAPI round trip.

---

## 19. Final Project Status

- **RESEARCH STATUS**: substantial, honest, multi-track research completed; several tracks
  (Arabic, Olist V3 seller-SLA) reached strong, well-validated results within their own
  protocols; one track (DataCo/EAGLE) reached a diagnosed, unresolved reproduction gap.
- **ENGINEERING STATUS**: solid — hash-verified loading, fail-soft integration, deterministic
  decision engine, calibration pipeline, all directly verified working via passing tests
  against real artifacts.
- **PRODUCTION STATUS**: minimal — only Olist V1 (2-feature, timing-only) has real execution
  history (5 orders). The shadow-mode production-parity route exists but has never executed
  against a real order.
- **DATA STATUS**: critically thin — 5 real orders, 0 fulfillment records, in the live
  database as of this investigation. No first-party HASEBHA fulfillment-outcome dataset
  currently exists.
- **MODEL STATUS**: no HASEBHA-native model has been validated as useful; the best
  HASEBHA-transferable estimate (production-parity, AUC 0.5551) is explicitly WEAK.
- **BUSINESS-READINESS STATUS**: not ready for automated fulfillment-risk actions of any kind;
  no real shipping-SLA/deadline is even configured in HASEBHA yet (a business decision, not an
  engineering gap) — see the prior supervisor-handoff report for full detail.

---

## 20. Evidence / Artifact Index (primary sources used in this report)

- `configs/nlp_champion_registry.yaml`
- `reports/generated/arabic_sota/ARABIC_FINAL_DECISION.json`
- `reports/generated/amazon/metrics.json`, `transformer_final_eval.json`, `scope_decision.md`
- `artifacts/experiments/amazon/models/amazon_tfidf_wordchar_linearsvc_size100000.joblib`
- `src/ai_service/services/nlp_inference.py`, `src/ai_service/routers/nlp.py`
- `reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/*`
- `reports/generated/instacart/protected_test_final_results.json`
- `src/ai_service/config.py` (Olist V1 hash pin)
- `reports/generated/olist_v2/OLIST_V2_TEMPORAL_STABILITY_FINAL_REPORT.md`, `FINAL_SCORECARD.json`, `CURRENT_STATE.json`
- `reports/generated/olist_v3_multistage/EVENT_SEMANTICS_AUDIT.json`, `MULTI_SELLER_TARGET_VALIDITY_AUDIT.json`, `SELLER_SLA_LEAKAGE_TESTS.json`, `SELLER_SLA_DATASET_META.json`, `SELLER_SLA_TEMPORAL_EVAL.json`, `TASK_B_C_RESULTS.json`, `SELLER_SLA_CALIBRATION_REPORT.json`, `SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json`, `OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json`, `PRODUCTION_PARITY_MODEL_COMPARISON.json`, `PRODUCTION_PARITY_CALIBRATION_REPORT.json`
- `reports/generated/dataco/DATACO_ACQUISITION_PROVENANCE.json`, `EAGLE_PAPER_PIPELINE_SUMMARY.json`, `DATACO_LSTM_REPRODUCTION.json`, `DATACO_TARGET_FORENSIC_CORRECTION.json`, `LSTM_REPRODUCTION_GATE_DECISION.json`
- `src/ai_service/services/seller_sla_risk.py`, `production_parity_seller_sla.py`
- Live PostgreSQL database (direct query this session, 5 orders / 0 fulfillments / 0 shadow-metadata rows)
- `reports/COMMERCEPILOT_SUPERVISOR_HANDOFF_CURRENT_STATE.md` (immediately preceding investigation, cross-checked not copied)

---

## 21. Consistency Audit (Section 25 of the mission)

- **"Protected" vs "stress" terminology**: Olist V2's block is correctly EXPOSED/stress-only
  in the current artifacts (Section 7). Any older document still calling it "protected"
  describes outdated terminology — flagged, not silently corrected in this report (which uses
  the current term throughout).
- **Conflicting AUC numbers**: Olist V3 customer-T0 (this track's own stacked model, 0.591)
  vs. Olist V2's separately-computed customer-T0 estimate (0.7126) are DIFFERENT protocols on
  DIFFERENT cohorts — not the same measurement, so not a true conflict, but easy to
  misread as one. Flagged explicitly in Section 11.
- **Research vs production confusion**: addressed throughout via the
  RESEARCH_FULL_FEATURE_MODEL vs PRODUCTION_PARITY_MODEL distinction (Sections 8-10, 14, 18).
- **Accidental JUMIA inclusion**: none found in this report's own construction; Jumia's
  historical existence (pre-dating the current exclusion policy) is documented in the prior
  supervisor-handoff report, not repeated as a live track here.
- **Accidental Olist V1 contamination**: none — Section 6 explicitly separates V1 from V2/V3
  and confirms (via `git status` on its exact paths) it was never modified this session chain.
- **Treating the 5 HASEBHA test orders as real production data**: explicitly flagged as NOT
  representative (Section 14, and the prior supervisor-handoff report) — never described as a
  dataset anywhere in this report.
- **Calling infrastructure validation a model validation**: explicitly separated in Section 14
  and 19 ("validated infrastructure... but has NOT validated a useful HASEBHA... signal").
- **New discrepancy found this pass, not previously documented**: Section 4.4, the Amazon
  "never serialized to disk" docstring vs. the real on-disk hash-verified artifact. This is
  reported as a documentation-vs-artifact mismatch, not escalated to
  "CONFLICT — REQUIRES SUPERVISOR REVIEW" because it does not conflict with any OTHER number
  or claim in the project (only with the specific code comment) and the resolution is clear
  (the artifact exists; the comment is imprecise about why the endpoint is unwired) — flagged
  and explained rather than left as an unresolved conflict.
- **Feature-count labeling correction (resolved, not an open conflict)**: the seller-SLA
  model's feature count was verified this session by direct Python import
  (`len(FEATURE_ORDER)`) to be **23**, not the "22-feature model" label used pervasively
  across this project's own artifacts (`SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json`,
  `OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json`, and this report's own first draft of Section
  8.3). This is corrected in Section 8.3 above. It does not affect any reported metric — all
  metrics were computed against the actual training frame regardless of the surrounding
  prose's count — so it is resolved as a documentation correction, not escalated as an
  unresolved conflict.
