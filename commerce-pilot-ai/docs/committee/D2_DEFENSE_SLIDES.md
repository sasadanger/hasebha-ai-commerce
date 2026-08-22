# D2 — Defense Slide Deck (22 slides)

Verified: 2026-08-22. Each `---` marks a new slide. Speaker notes in *italics* below each
slide's content. All numbers trace to D3/D4.

---

## Slide 1 — Title
**CommercePilot: An Applied AI Layer for HASEBHA**
*Evidence-based ML for a real, single-vendor e-commerce platform*
Subtitle: "What we validated, what transferred, what didn't — and why."

---

## Slide 2 — What is CommercePilot?
An AI layer alongside a real Medusa e-commerce backend: three production capabilities
(Arabic NLP, product recommendation, fulfillment-risk scoring), each independently
validated, plus a rigorous transfer study for the hardest one.
*Frame this as an engineering + science project, not a single model demo.*

---

## Slide 3 — Why multiple datasets?
Arabic/Amazon → text tasks. Instacart → recommendation. Olist → fulfillment-risk research
proxy (no first-party Egyptian order data existed in sufficient volume). DataCo → an
independent reproduction exercise, evaluated on its own merits.
*Anticipate: "why not just use your own data" — answer is the next slide.*

---

## Slide 4 — The core constraint
HASEBHA (live database, verified this cycle): **5 real orders. 0 fulfillment outcomes.**
No task in this project could be trained end-to-end on first-party data — this shaped
every methodological choice that follows.

---

## Slide 5 — Track 1: Arabic Sentiment
MARBERTv2, group-safe split. **0.8130 (1-seed) / 0.7906 (3-seed mean) Macro-F1.**
5 improvement candidates tested, all rejected (label smoothing, class weighting, R-Drop,
SARF cost-rejected at statistical parity). Frozen, live in production.

---

## Slide 6 — Track 2: Amazon Sentiment
TF-IDF + LinearSVC. **0.9454 Macro-F1**, 95% CI [0.9432, 0.9476] (n=40,000, bootstrap,
computed this cycle). Fitted artifact exists, hash-verified — but not currently wired to
serve live requests (an integration gap, honestly disclosed).

---

## Slide 7 — Track 3: Instacart Recommendation
Hybrid + popularity backfill. **Precision@5 = 0.3674**, single-shot protected test
(26,314 users, accessed exactly once, zero post-test tuning). Live in production.

---

## Slide 8 — Track 4: Olist V1 (the only live fulfillment model)
CatBoost, 2 raw timestamps → 9 features. **ROC-AUC 0.5634**, 95% CI [0.5483, 0.5774]
(n=24,744, bootstrap, computed this cycle, excludes 0.5). Wired live to `order.placed`;
5 real HASEBHA orders scored to date.
*This is weak but real, and the ONLY model with actual production execution history.*

---

## Slide 9 — Track 5: Olist V2 — a documented failure, on purpose
Static model: dev AUC 0.72–0.76 → protected-test AUC **0.5117** (near-random). NOT a
leakage bug (0/30 leakage checks failed). A genuine temporal regime shift.
*This slide exists to prove the team reports failures, not just successes.*

---

## Slide 10 — Olist V2, continued: the correction
An earlier hypothesis blamed a 2018 trucker strike — directly checked against the data and
**refuted** (true late-rate peak was 2 months before the strike window). Recency-weighted
adaptation partially recovers: 0.7126 historical mean, 0.5924 on the exposed stress block.
*Shows willingness to correct earlier, wrong internal claims — a strength, not a weakness.*

---

## Slide 11 — Track 6: Olist V3 Seller-SLA — the strongest research result
LightGBM, 23 features, leakage-tested (0/4 failures). **Mean temporal AUC 0.7702**
(worst period 0.6762), 5 independent rolling-origin periods.
*State clearly: this is a research number on Brazilian marketplace data.*

---

## Slide 12 — Where the 0.7702 comes from
8 of 23 features are causal, rolling **seller-history** statistics (each seller's own past
breach rate, handling duration). Real, temporally stable signal — because Olist has ~3,000
sellers with genuine behavioral heterogeneity.

---

## Slide 13 — The transfer test (this cycle's centerpiece)
HASEBHA has **no seller/vendor module** — confirmed by direct code inspection. Sentinel-
substituting the 10 unavailable features into the SAME frozen model: **AUC 0.7702 → 0.5188**
(worst period 0.4949, below random). This is the project's single most important finding.

---

## Slide 14 — The forensic ablation (independent confirmation)
A fresh, independent forensic study this cycle reproduced the frozen numbers under
identical protocol (0.7686 vs 0.7702; 0.5540 vs 0.5551) and isolated the cause with
feature-block ablation: **removing seller-history alone costs −0.202 AUC — 94% of the
research model's total advantage.** Spot-checked and independently reproduced by this
presenter directly from the raw parquet before use.

---

## Slide 15 — Did we try to fix it?
Yes — exhaustively. New feature groups tested this cycle: real geography (haversine
distance) **no signal** (Δ −0.001); strict point-in-time customer history **no signal**
(Δ +0.002); product category **+0.014, unstable**; price structure **no signal**. Six
model families compared: all land in **[0.549, 0.565]** — architecture is not the
bottleneck.

---

## Slide 15.5 — Did we leave any public data untested? (Marketing Funnel)
Predeclared hypothesis: +0.01 to +0.03 AUC from acquisition-channel/seller-onboarding data.
Result: **−0.0034 (negative)**, funnel features individually ≈0.498 AUC (noise). Explained by
a 4.5% coverage ceiling identified *before* the experiment ran. Reported exactly as measured.
*Use this slide only if time allows — it exists to preempt "did you try X" questions.*

---

## Slide 16 — Track 7/8: Customer-Late T0 and T1
T0 stacking (adding predicted seller risk): **+0.0036**, trivial, rejected. T1 (post
carrier-handoff): **mean AUC 0.689**, meaningfully stronger — real, moderate,
triage-level, not automated-decision quality.

---

## Slide 17 — Track 9: Calibration
Isotonic selected for both the research and production-parity models. Brier improved in
both (0.0717→0.0709; 0.1255→0.0802); ranking (AUC) preserved within a predeclared 0.005
tolerance in both cases.

---

## Slide 18 — Track 10: DataCo/EAGLE — a reproduction study
Dataset hash-verified, graph structure exactly matched (46 nodes). LSTM reproduction:
**0.6454 vs published 0.9679.** Root cause mathematically diagnosed (a sample-size/CLT
argument — not a shrug). EAGLE correctly never attempted, per a predeclared gate.

---

## Slide 19 — The honest production reality matrix
| Model | Ever scored a real order? |
|---|---|
| Olist V1 | **YES (n=5)** |
| Seller-SLA research | No (by design, research-only endpoint) |
| Production-parity shadow | **No (0 of 5 real orders)** |
*Binary, evidence-based — verified via direct database query, not inferred from code or logs.*

---

## Slide 20 — What we can and cannot claim
CAN: rigorous research results, a quantified transfer-gap explanation, a live (if thin)
production execution history, a data-collection plan with numbers.
CANNOT: 0.77 as HASEBHA performance; shadow-mode validation on real traffic; DataCo/EAGLE
matching the published benchmark; Amazon sentiment as "in production."

---

## Slide 21 — The path forward
Two blockers, neither solvable by more modeling: (1) a real business-defined shipping SLA,
(2) real order/fulfillment volume — quantified: **~1,650 orders minimum / ~4,500
recommended.** Everything engineering could safely close in advance has been closed
(raw-feature persistence, `same_zone` resolution, calibration, hash-verified loading).

---

## Slide 22 — Final status
**YELLOW** (project-level: infrastructure ready, ML blocked on data/business decision).
**RED** (signal-sufficiency-level, this cycle's forensic finding: no legitimate feature or
model-family change closes the gap with *currently available* information). Both are true
simultaneously and are not in conflict — see backup slide / Q&A brief for the distinction.
*Close with the one-sentence claim from D5, Q15.*
