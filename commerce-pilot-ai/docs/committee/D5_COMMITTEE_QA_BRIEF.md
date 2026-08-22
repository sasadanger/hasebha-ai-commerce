# D5 — Committee Q&A Brief: The 15 Hardest Questions

Verified: 2026-08-22. Every answer traces to a real artifact (see D4). Numbers are never
softened; weaknesses are stated plainly.

**1. What is actually novel here?**
Not the individual model families (LightGBM, MARBERTv2, CatBoost — all standard). The novel
contribution is the *quantified research-to-production transfer analysis*: an explicit,
artifact-traced ablation showing exactly which features cause a public-dataset result to
collapse when moved to a real, structurally different production environment (0.7702 → 0.5188
sentinel simulation; 0.7686 → 0.5540 independent forensic reproduction; 94% of the gap
attributed to a single feature block by controlled ablation). Most projects either don't run
this test or don't publish it when it's unfavorable.

**2. Why is production AUC 0.56 (V1) / 0.555 (parity model)? Isn't that basically random?**
V1 is confirmed statistically distinguishable from random by bootstrap CI (95% CI [0.548,
0.577] on the raw saved test predictions, excluding 0.5). The production-parity model's
0.5551 does **not** yet have an equivalent bootstrap CI computed from raw per-order
predictions (D3 discloses this explicitly as NOT_COMPUTED) — its evidence for
non-randomness is the temporal fold dispersion across 5 independent periods (worst period
0.5289, still above 0.5 in every period) plus the independent forensic reproduction
(0.5540) landing in the same range under a fresh script, not a formal CI. Both signals point
the same direction, but only V1's claim currently has a computed confidence interval; this
is stated precisely rather than implied to be equally rigorous for both. Both are genuinely
weak in absolute terms regardless. This is not a training or engineering
failure — it is fully decomposed: for the parity model, univariate feature AUCs top out at
0.54 (`volume_cm3`), six model families all land in [0.549, 0.565], and adding every
plausible new feature group (geography, customer history, price structure) gains at most
+0.014. The ceiling is set by what information a single-vendor store genuinely has at order
time, not by modeling choices.

**3. Why so many datasets/tracks (Arabic, Amazon, Instacart, Olist, DataCo)?**
Each targets a different production capability the platform needs (text moderation/sentiment,
recommendation, fulfillment-risk) and no first-party HASEBHA dataset of adequate size exists
for any of them yet, so public benchmarks were used deliberately as research/validation
vehicles while a first-party data-collection pipeline (the shadow-mode loop) was built in
parallel.

**4. Published projects report higher numbers on the same/similar datasets — why are yours lower?**
Protocol, not modeling weakness. Three concrete differences, each independently checkable in
this repository: (a) **temporal vs. random splits** — this project used rolling-origin
temporal evaluation throughout (Olist V2/V3), which is harder and more realistic than a random
split; a random-split baseline would score higher and is deliberately not used as the headline
number. (b) **T0 availability discipline** — every production/production-parity feature is
enforced point-in-time (shift(1)/rolling only, leakage-tested 0/4 and 0/30 failures); many
published pipelines are less strict about what's "available" at prediction time. (c) The
project's own internal proof of exactly this effect exists: Olist V2's static model scored
0.7126 on historical rolling folds but collapsed to 0.5117 on a later exposed block — the same
phenomenon, demonstrated on our own data, that explains gaps versus looser published numbers.

**5. Did the project fail?**
No track failed due to a bug or broken process. Several tracks succeeded on their own terms
(Arabic 0.7906 3-seed mean, Amazon 0.9454, Instacart 0.367 P@5 under a genuinely single-shot
protected test, Olist V3 research 0.7702 leakage-tested). One central hypothesis (does the
Olist signal transfer to HASEBHA) was tested and answered "no," with a full quantitative
explanation — a valid, non-failing scientific outcome, and the forensic study now available
makes that "no" essentially airtight rather than just observed.

**6. What did the failed/rejected experiments teach you?**
15 formally registered negative results (`NEGATIVE_RESULTS_REGISTER.json`) span every track.
Highlights: adding store-wide operational features made the parity model worse, not better
(-0.0093, confirmed independently again in this cycle's forensic study at a similar
magnitude); stacking a weak upstream signal into a weaker downstream target added nothing
(+0.0036); a paper's own quoted target formula was mathematically incompatible with its
reported statistics (DataCo). Each taught something specific and is preserved, not hidden.

**7. What's actually running in production right now?**
Exactly one model: Olist V1 CatBoost, wired to Medusa's `order.placed` event, with 5 real
orders scored to date (verified by direct database query, not inferred from code or logs).
Arabic and Instacart are also live for their own (non-fulfillment) tasks.

**8. What's unvalidated?**
Everything fulfillment-related beyond V1. The shadow-mode production-parity route is fully
implemented, tested, and — as of this cycle — has a corrected `same_zone` feature resolution,
but has never scored a real order (0 of 5 real orders carry its metadata key).

**9. What business decision is missing, and why does it matter this much?**
A real shipping-time promise (`fulfillment_due_at`, per shipping option). Without it, "late"
has no defined meaning for HASEBHA — the current target is an Olist Brazilian proxy. The
forensic study quantifies its value precisely: adding `days_to_shipping_deadline` (which
requires this decision) is worth +0.013 AUC on Olist — small on its own, but it's also the
prerequisite for computing *any* real label at all, which is worth far more than 0.013.

**10. What data do you actually need, and how much?**
Quantified via Hanley-McNeil AUC confidence-interval sample-size math, assuming ~9.1%
prevalence (the Olist rate, used as a planning proxy, not a HASEBHA measurement — no HASEBHA
prevalence is known yet): minimum ~1,650 orders / ~150 breach events (±0.05 CI, ~2 months at
30 orders/day); recommended ~4,500 orders / ~410 breach events (±0.03 CI, ~5 months).

**11. Could you just retrain on more Olist data to get a better HASEBHA number?**
No — this was specifically tested. The problem is feature availability, not data quantity
within Olist. More Olist rows would not create the seller-heterogeneity signal a single-vendor
store structurally lacks. This is proven by the ablation (removing seller-history costs -0.202
AUC, i.e. 94% of the model's advantage) not merely assumed.

**12. Was the DataCo/EAGLE work a wasted effort?**
No — dataset acquisition and graph-structure reproduction were independently hash/count
verified; the LSTM reproduction gap has a mathematically diagnosed cause (a sample-size/CLT
argument: median node-window volume of 174 orders makes the paper's own quoted target formula
statistically incapable of reproducing its reported ~5% prevalence at this dataset's scale),
not a shrug. EAGLE was correctly never attempted per a predeclared gate rather than run on a
target already known to be broken.

**13. Is the shadow pipeline "working"?**
The code works (0 lint/type errors, passing unit/integration tests against the real loaded
model artifact). It has never processed a real order. These are two different, both-true
claims that must not be conflated — verified by direct database query, not inferred.

**14. What would change your assessment from RED/WEAK to something stronger?**
Two things, in this priority order: (1) the business SLA decision (unlocks a real target and
the deadline feature), (2) real order/fulfillment volume reaching the quantified minimums
above. Neither is achievable by more modeling. The forensic study explicitly ruled out every
lever available under the *current* data — six model families tested, four new feature groups
tested, all within [0.549, 0.568].

**15. What is the single most defensible sentence you can say about this project's ML results?**
"We built and rigorously validated several strong research/production models on public and
recommendation data, and for the one capability where we tested transfer to our real
production environment, we proved — quantitatively, not by assumption — that the public-data
signal does not survive real feature-availability constraints, and we specified exactly what
data and business decision are needed to close that gap."
