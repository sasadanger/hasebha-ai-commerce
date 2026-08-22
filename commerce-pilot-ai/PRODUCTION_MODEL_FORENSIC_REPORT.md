# PRODUCTION MODEL FORENSIC REPORT
**Project:** CommercePilot / HASEBHA — production ML fulfillment track
**Date:** 2026-08-22
**Investigator scope:** full repository forensics, target forensics, signal decomposition, controlled experiments, research-to-production ablation.
**Frozen tracks:** untouched. All experiments run by a NEW script: `scripts/forensics/production_model_forensics.py`. Raw results: `reports/generated/olist_v3_multistage/forensics/FORENSIC_EXPERIMENT_RESULTS.json`.

---

## 1. Executive verdict

**RED.** The production-parity signal (AUC 0.5551) is *genuinely weak*, not an artifact of weak
features-as-coded, wrong model family, bad calibration, or protocol mismatch. The 0.7702 research
AUC is driven almost entirely by seller-history features that are structurally unavailable in a
single-vendor HASEBHA deployment until first-party history accumulates. No legitimate feature
engineering or model-family change materially closes the gap on Olist. The honest path forward is
Option D (pilot/shadow/data-collection framing) with a quantified data-collection plan
(§8) and one conditional business decision (an explicit shipping SLA, §6).

---

## 2. Tracks audited (Phase 1)

| Track | Model | Features | Target | Protocol | AUC | Role |
|---|---|---|---|---|---|---|
| Olist V1 `olist-phase2a-strict-core-v1` | CatBoost | 9 timing features from 2 timestamps | customer late delivery (`delivered > estimated`) | locked manifest, one-shot frozen test | **0.5634** | LIVE production (order.placed wired; 5 real orders scored) |
| Olist V3 seller-SLA research | LightGBM | 23 | `SELLER_HANDOFF_SLA_BREACH` = `delivered_carrier > shipping_limit` | rolling prequential, 5 periods | **0.7702** mean | RESEARCH-ONLY (parity audit FAIL) |
| HASEBHA production-parity (MODEL P) | LightGBM + isotonic | 13 MODE_A | same as V3 | same protocol | **0.5551** mean | SHADOW-ONLY, never scored a real order |
| MODEL P+ (store ops features) | LightGBM | 18 | same | same | 0.5458 | REJECTED (negative gain) |

Protocol reproduction check (this investigation): FULL-23 → mean AUC **0.7686** (frozen: 0.7702);
MODEL P → **0.5540** (frozen: 0.5551). Both within run-to-run tolerance. **The frozen numbers
reproduce; the protocol is sound.**

## 3. Target forensics (Phase 3)

- V1 target (customer late delivery) is *observable* in HASEBHA (`order.delivered_at` vs a promise),
  but no promise/SLA field exists today (`HASEBHA_SHIPPING_SLA_PRODUCT_REQUIREMENT.md`, Gate 5).
- V3/P target (seller handoff SLA breach) requires `fulfillment.shipped_at > fulfillment_due_at`;
  `shipped_at` exists in Medusa, but `fulfillment_due_at` does not — no business SLA is configured.
- Conclusion: **both targets are legitimately aligned with the business problem** (late-fulfillment
  risk), but *neither is currently computable* in HASEBHA without one business decision. The
  weakness is NOT a wrong-target problem; it is a missing-upstream-contract problem.
- TASK_B_C artifacts confirm target timing matters (Customer T1 AUC 0.689 with later-observable
  features), but later observability violates the T0 = order-placed production constraint.

## 4. Decomposition of 0.5551 (Phase 4)

Univariate AUC (pooled) of the 13 MODEL-P features: max 0.540 (`volume_cm3`, `weight_g`), most
0.50–0.53. Permutation importance (final period, best legitimate model): `volume_cm3` 0.042,
`weight_g` 0.031, `item_price_mean` 0.028, `purchase_hour` 0.013 — the signal that exists is
almost entirely *physical product profile*. Temporal stability: per-period AUC 0.534/0.529/0.551/
0.586/0.576 — positive but fragile. Class prevalence ~9%; no pathological imbalance.

**Answer to "why 0.5551": category A+B combined — genuinely weak signal AND weak feature
representation forced by legitimate availability constraints.** It is NOT: protocol mismatch,
sample size, class imbalance, calibration, or model family (all ruled out experimentally below).

## 5. Ablation: where the 0.77 → 0.555 collapse comes from (Phase 7)

Same protocol, same data, feature removal only:

| Feature set | n feats | Mean AUC | Δ vs FULL |
|---|---|---|---|
| FULL 23 (frozen research) | 23 | **0.7686** | — |
| seller-history ONLY | 8 | **0.7568** | −0.012 |
| P + deadline + installments (no seller hist) | 15 | 0.5666 | −0.202 |
| P + installments (no deadline, no seller hist) | 14 | 0.5534 | −0.215 |
| **P baseline (13)** | 13 | **0.5540** | −0.215 |

Attribution of the −0.215 collapse:
- **Removing seller-history block: −0.202 (94% of the collapse).** Univariate AUCs: `seller_breach_rate_90d` 0.733, `seller_handling_mean_30d` 0.731, `seller_past_breach_rate_expanding` 0.729. The research model is essentially a *seller-behavior lookup*.
- **Removing `days_to_shipping_deadline` on top: −0.013.** Its univariate AUC is only 0.517; its value is interactional. It requires a business SLA decision (class D) — legitimate to add only after that decision.
- **Removing `n_installments`: −0.002.** Negligible.

Consistent with the earlier sentinel simulation (0.7702 → 0.5188, `OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json`).

## 6. New legitimate feature groups (Phase 5)

All point-in-time correct at T0; classification per RULE 8 in the Feature Inventory.
Paired per-period deltas vs P baseline:

| Candidate | New features | Mean AUC | Δ | Verdict |
|---|---|---|---|---|
| P + geography | haversine dist, lat/lng diff (zip centroids) | 0.5528 | **−0.001** | no signal |
| P + customer history | prior orders, repeat, tenure, prior-late-rate (strictly observable-only) | 0.5563 | +0.002 | no signal |
| P + product category | category code | 0.5678 | +0.014 | best single gain; unstable in last period (−0.001) |
| P + price structure | item price mean/max, freight/item | 0.5542 | +0.000 | no signal |
| P + all of the above | 24 features | 0.5647 | +0.011 | no net gain over category alone |

Conclusion: geography and customer history — the two most plausible unavailable-feature hopes —
**carry no incremental signal on Olist**. Category is worth ~+0.014 (0.555 → 0.568) at best and
does not survive the most recent period cleanly. Nothing here changes the WEAK classification.

## 7. Model family (Phase 6)

On the best legitimate numeric set, same protocol: LogReg 0.549, RandomForest 0.552,
HistGB 0.554, XGBoost 0.554, CatBoost 0.555, **LightGBM 0.565 (current choice, already best)**.
Model architecture is not the bottleneck.

## 8. First-party data requirements (Phase 9)

Prevalence assumption ~9% positives. Hanley–McNeil CI calculations:

| Precision goal | Positive events (breaches) | Total orders | Approx. time @ 30 orders/day |
|---|---|---|---|
| Minimum (CI ±0.05) | ~150 | ~1,650 | ~2 months |
| Recommended (CI ±0.03) | ~410 | ~4,500 | ~5 months |

Plus: ≥200–300 events before any store-level rolling feature (30d/90d breach rates) stops being
noise; required fields per the Gate-12 feedback contract (features + `fulfillment.shipped_at` +
`fulfillment_due_at`). The shadow loop already persists raw features — this is the correct
investment and should be presented as such.

## 9. Committee options ranking (Phase 10)

1. **OPTION D (recommended):** keep all infrastructure; label production ML as
   "pilot / shadow / data-collection phase"; present V3 as research with the rigorous
   research-to-production transfer analysis (§5) as a headline contribution.
2. **OPTION C (conditional):** adopt a business SLA (`promise_business_days` on shipping option,
   per Gate 5 doc). This makes the target *computable* and enables `days_to_shipping_deadline`
   (+~0.013 on Olist, more importantly makes first-party labels possible at all).
3. **OPTION B (micro):** add product-category feature (+0.014) in the next retrain — optional,
   honest, small.
4. **OPTION A:** acceptable but leaves the SLA decision unaddressed.
5. **OPTION E:** only if the committee rejects shadow data collection; loses the path to a
   future strong model.

## 10. What we claim / do NOT claim

CLAIM: research model 0.7702 (clearly labeled Olist, offline); production-parity 0.5551 shadow
pilot; full reproducible experimental protocol; quantified transfer-gap attribution; data
collection plan with sample-size math.
DO NOT CLAIM: 0.7702 as HASEBHA performance; production model as decision-grade; any
improvement beyond +0.014 from new features.
