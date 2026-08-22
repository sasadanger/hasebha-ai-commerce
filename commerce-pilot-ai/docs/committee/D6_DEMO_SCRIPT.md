# D6 — Live Demo Script (3-5 minutes)

Verified: 2026-08-22. Strongest honest path, with fallbacks if any live component is
unavailable at defense time.

## Pre-flight checklist (do before walking into the room)
- [ ] `D:\ecommerce_medusa\commerce-pilot-ai\.venv` present and working (no rebuild needed —
      verified this session).
- [ ] `D3_RESULTS_TABLE_WITH_CI.json`/`.md` open in a tab (has the real bootstrap numbers).
- [ ] `PRODUCTION_MODEL_FORENSIC_REPORT.md` open in a tab (the strongest single artifact).
- [ ] Confirm whether Docker/Postgres will be running — if not, use Fallback Path B below.

## Primary path (if Docker/Postgres can be started safely)

**[0:00-0:30] Frame the story.** "We're not going to show you a magic AUC number — we're
going to show you a real prediction pipeline, and then show you exactly why its production
signal is honestly weak, with math, not a guess."

**[0:30-1:30] Show the research result.** Open `reports/generated/olist_v3_multistage/
SELLER_SLA_TEMPORAL_EVAL.json`. Point at mean AUC 0.7702, 5 independent temporal periods,
worst period 0.6762. Say: "This is real, leakage-tested (0 of 4 checks failed), and entirely
on Brazilian marketplace data — Olist. We do not claim this is our production number."

**[1:30-2:30] Show the transfer test — the centerpiece.** Open
`PRODUCTION_MODEL_FORENSIC_REPORT.md`, Section 5 (the ablation table). Say: "We removed only
the features a single-vendor store structurally cannot have — seller history — and re-scored
the *same frozen model* on the *same test periods*. AUC dropped from 0.7686 to 0.5540. That's
94% of this model's power, gone, because our store has one seller, not three thousand." This
is the single most defensible slide/moment in the whole defense — it is a controlled
experiment, not an assumption.

**[2:30-3:30] Show it live, briefly.** If Postgres is running: run the local FastAPI service,
POST a sample payload to `/v1/fulfillment/seller-sla-shadow`, show the JSON response
(`seller_sla_breach_probability`, `risk_level`, `automated_action_taken: false`). Say: "This is
running today, in shadow mode — it logs a prediction, it never blocks checkout, and it never
triggers an action. It exists to collect the very data we need to build the real thing."

**[3:30-4:15] The honest data plan.** Open the forensic scorecard's `first_party_data_
requirements` block. Say: "We know exactly how much real data we need — about 1,650 orders
minimum, 4,500 recommended — and we know the one business decision that unlocks it: a real
shipping-time promise. Neither of those is a modeling problem."

**[4:15-5:00] Close.** "Three things are already strong and already running: Arabic
sentiment, product recommendation, and this fulfillment pipeline's infrastructure. One thing —
does this specific model predict lateness for our specific store — we tested rigorously and
found: not yet, and we can tell you exactly why and exactly what closes the gap."

## Fallback Path B (no live services available)
Skip the [2:30-3:30] live-call step entirely. Instead show the `seller_sla_predictions.jsonl`
file (or the test suite passing: `pytest tests/test_production_parity_seller_sla.py -q`) as
proof the code path is real and tested, and say explicitly: "We are not going to fake a live
demo — this is the honest state: the code is tested and correct, it has simply never yet
scored a real customer order, because our store has only had 5 orders total to date."

## Fallback Path C (question derails the script)
If asked "why not just show us a higher number" mid-demo: do not improvise a new number. Say:
"Every number we're showing you today traces to a saved file we can open right now — that
discipline is worth more to us than a bigger number we couldn't defend," then return to the
script.

## What NOT to do in the demo
Do not claim the shadow prediction "worked" on a real order. Do not present 0.7702 without the
production-collapse context in the same breath. Do not improvise a live retraining run. Do not
promise automated actions (refunds/cancellations) as a future feature without qualification —
this project's own permanent rule is that no automated customer-facing action is ever
authorized regardless of model performance.
