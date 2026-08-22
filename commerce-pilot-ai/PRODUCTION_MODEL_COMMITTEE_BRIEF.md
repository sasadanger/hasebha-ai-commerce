# COMMITTEE BRIEF — HASEBHA PRODUCTION ML TRACK
**Bottom line:** The production model's weak AUC (0.555) is real and fully explained. The strong
research AUC (0.770) was never transferable with currently available data, and we can prove — to
the decimal — why. Recommended positioning: **Option D — pilot/shadow/data-collection phase**, with
a concrete, quantified path to a strong first-party model.

---

## What did we try?

A full forensic program (this investigation) on top of the existing V1/V3/parity work:
- Reproduced both frozen numbers under the identical temporal protocol (0.7686 vs 0.7702; 0.5540
  vs 0.5551) — our evaluation is sound.
- Ablated the research model feature-group by feature-group.
- Engineered and tested every plausible HASEBHA-derivable feature group: real geographic
  distance, strictly point-in-time customer history, product category, basket price structure.
- Compared six model families (LogReg, RF, HistGB, XGBoost, CatBoost, LightGBM) under the same
  protocol.
- Computed first-party data requirements with standard AUC sample-size math (Hanley–McNeil).

## Why was the production model weak? (the central scientific finding)

**94% of the research model's advantage comes from seller-history features** (rolling
seller breach rate, seller handling-time statistics). Those features alone reach AUC 0.757.
The research model is essentially a *seller behavior lookup table* — legitimate on Olist, where
3,000 sellers each have hundreds of past orders, and structurally impossible in a single-vendor
HASEBHA store with 5 orders and 0 fulfillment records. Everything else the order itself reveals
at placement time — value, weight, volume, freight, timing, geography, customer history —
carries only AUC ~0.55 of signal. **This is a property of the information available at
order-placement time, not of our engineering.** Model family, calibration, class balance,
protocol, and sample size were each tested and ruled out as causes.

## What improved (honestly)?

- Product category feature: 0.5551 → 0.5678 (+0.014) — real but marginal and not fully
  temporally stable; candidate for the next retrain, does not change the WEAK classification.
- Explicit shipping SLA (a business decision, not ours to invent) would enable
  `days_to_shipping_deadline` (+0.013 on Olist) **and**, more importantly, make any first-party
  target label computable at all.

## What cannot be improved without new data or a business decision?

Everything above ~0.60. Specifically: seller/store behavioral history (requires ~4,500 orders,
~410 breach events — about 5 months at 30 orders/day, or 2 months for a coarse minimum of
~1,650 orders / 150 events), and a defined `fulfillment_due_at` SLA contract.

## What is production-ready vs research-only?

- **Production-ready:** the *infrastructure* — live scoring wired to `order.placed`, shadow
  scoring loop, raw-feature persistence, hash-pinned artifacts, one-shot test ledger,
  447/447 passing tests, calibration pipeline, feedback dataset contract.
- **Research-only:** the 0.7702 seller-SLA model (feature-parity audit FAIL, correctly fenced
  behind an offline-only endpoint).

## What should we claim?

1. A rigorous research result: temporal AUC 0.7702 (worst 0.6762) with zero leakage failures,
   on a public benchmark dataset.
2. A rigorous *negative transfer result*: a quantified ablation showing exactly where
   0.77 → 0.55 comes from. Fewer projects can explain their production gap this precisely;
   this is a strength, not an embarrassment.
3. A live, honest pilot: shadow-mode scoring, explicitly labeled WEAK, collecting the exact
   features needed for the future first-party model.
4. A data plan with numbers: ~1,650 orders (minimum) to ~4,500 orders (recommended) before a
   validated HASEBHA-native model.

## What should we NOT claim?

- 0.7702 as HASEBHA performance (explicitly prohibited by our own parity audit).
- That the production model currently improves any business decision.
- Any AUC above 0.57 for order-placement-time features.

## Options (ranked)

1. **D** — pilot/shadow/data-collection framing (recommended; honest, preserves everything).
2. **C** — adopt a business SLA (`promise_business_days` per shipping option): one decision
   unlocks the target label, the strongest single feature, and the entire first-party path.
3. **B** — fold the +0.014 category feature into the next retrain.
4. **A** — status quo (acceptable, but dodges the SLA decision).
5. **E** — drop the predictive claim, keep research (only if shadow collection is rejected).

## Final decision

**RED** — no scientifically defensible way to materially strengthen the production model with
currently available data. The weakness is genuine, explained, quantified, and — critically —
*temporary*: the shadow loop is the correct instrument, and the exit criteria are now numeric.
