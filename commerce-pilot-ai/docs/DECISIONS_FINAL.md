# Final Advisor Decisions — CommercePilot / HASEBHA

Verified: 2026-08-22. Role: FINAL ADVISOR — verdicts below are decisions, not options lists.
All work performed under `D:\ecommerce_medusa`; C: was never touched, read, or written by
this session.

---

## Item 1 — C: Deletion

**Status: RESOLVED, no action by me.** User has verbally approved; the coordinator's own
recursive-delete attempt was correctly blocked by their own safety classifier. This is now
a manual action for the user to perform themselves. I did not attempt it, will not attempt
it, and no tooling available to me is appropriate for a bulk delete of this scale on a path
outside my authorized working directory. **Nothing further required from this project.**

---

## Item 2 — D1-D6 Completion Review

Reviewed all six documents line-by-line against underlying artifacts (same discipline as the
funnel verification: re-derived numbers, checked file references, checked internal
consistency). Findings and actions:

### A1 (Related Work thin) — ACCEPTED, fixed
D1's Related Work section was indeed thin (self-admitted in the original draft). Verified
LABR's citation (arXiv:1411.6718, "LABR: A Large Scale Arabic Sentiment Analysis Benchmark,"
Aly & Atiya) via independent web search before citing it — confirmed real, not assumed from
the auditor's claim. Added a full paragraph naming LABR, MARBERT, AraBERT, DataCo (Mendeley
DOI), and EAGLE with their roles in this project, plus an explicit "positioning" statement
that this project applies established methods rather than proposing new architectures. Fixed
in `docs/committee/D1_ACADEMIC_DOCUMENT.md`, Section 2.

### A2 (D5 Q2 precision) — ACCEPTED, fixed
Confirmed the imprecision: D5's original answer claimed both V1 and the production-parity
model were "statistically distinguishable from random," but only V1 has a computed bootstrap
CI (D3 explicitly marks the parity model's CI as NOT_COMPUTED). Rewrote the answer to state
precisely which model has a formal CI (V1) versus which has only temporal-fold-dispersion and
independent-reproduction evidence (the parity model) — both point the same direction, but the
rigor is not equal, and now the text says so. Fixed in `docs/committee/D5_COMMITTEE_QA_BRIEF.md`, Q2.

### A3 (D2 is markdown, not a deck) — ACCEPTED the observation, DEFERRED the conversion
Verdict: correct observation, but conversion to PPTX/Google Slides format is a manual
formatting task with no scientific content to verify — it is not something a text-editing
session can meaningfully "get right" without the presenter's own visual judgment (slide
layout, institution template, font sizes for the room). **Decision: leave D2 as
presentation-ready structured markdown** (22 slides, clear breaks, speaker notes), which
converts cleanly via any markdown-to-slides tool (Marp, Pandoc, or manual copy-paste into
PowerPoint/Google Slides) in well under the 1-2 hours suggested for A1. Not doing the
mechanical conversion myself is a deliberate scope call, not an oversight — documented here
so it isn't silently dropped.

### Funnel result folded in — my editorial call: YES, presentation-worthy
Added the funnel negative result to D1 (new Section 8.5), D2 (new Slide 15.5), and D4 (new
artifact-index section). Reasoning: it directly strengthens the "we tested everything, we
don't hide negative results" narrative that is this project's core defensible claim, costs
one slide/paragraph, and is fully evidence-backed (own independent verification of the
forensic study, then this session's own execution of the funnel experiment).

### Package status: **LOCKED AS FINAL.**
`docs/committee/D1_ACADEMIC_DOCUMENT.md`, `D2_DEFENSE_SLIDES.md`,
`D3_RESULTS_TABLE_WITH_CI.md`/`.json`, `D4_ARTIFACT_INDEX.md`,
`D5_COMMITTEE_QA_BRIEF.md`, `D6_DEMO_SCRIPT.md` — all reviewed, all three auditor
observations addressed, funnel result folded in where valuable. No further edits
authorized without new evidence.

---

## Item 3 — Amazon Wiring Gap

**Verdict: DO NOT WIRE NOW. Documented as an honest integration gap — and a more serious,
previously-uncaught discrepancy was found underneath it.**

Before deciding, I re-verified which Amazon classical model is actually "the frozen
champion." Two different claims exist in the repository, from two different points in the
project's history, and they have **never been reconciled**:

1. The official NLP promotion registry (`reports/checkpoints/nlp_deployment_promotion_
   registry_v1_2026-08-14/nlp_deployment_promotion_registry_v1.json`) declares the frozen
   Amazon champion as `A::tfidf_word_bigram_logreg` — and the service code's own docstring
   (`src/ai_service/services/nlp_inference.py`) states this exact model **was never
   serialized to disk**.
2. A separate, later artifact (`reports/generated/amazon/metrics.json`) reports a DIFFERENT
   winning configuration, `tfidf_wordchar_linearsvc` (word+char n-grams, LinearSVC, not
   LogReg, not word-bigram-only) — and THIS model's fitted artifact DOES exist on disk,
   hash-verified (`amazon_tfidf_wordchar_linearsvc_size100000.joblib`).

These are not the same model under two names — different vectorizer configuration (word vs.
word+char n-grams) and different classifier family (LogReg vs. LinearSVC). Wiring the
LinearSVC artifact into `keys_for_task(AMAZON_TASK)` would silently promote a model that
never went through the same promotion-registry confirmation process the other three NLP
tasks did — that is not a small additive fix, it is an unreviewed champion substitution
dressed up as one.

**Action taken**: left `keys_for_task` unchanged (0 code touched). Documented this
discrepancy explicitly here and it should be added to the negative-results/known-issues
list before the next NLP-track session — resolving it requires a human decision (which
experiment is actually authoritative) or a fresh, properly-registered promotion confirmation
for the LinearSVC candidate, neither of which is a "wire it in" action.

---

## Item 4 — Shadow E2E

**Verdict: DEFER AGAIN, with an exact, concrete trigger — not an indefinite deferral.**

The "irreplaceable first-party data" argument is real and was weighed seriously, not waved
away. Counter-considerations that tipped the decision:

1. **A latent-bug risk exists on both sides of this decision.** Running the E2E now, with
   Docker containers that have been stopped for most of a week and a `.venv`/dependency set
   that has never been exercised together as a live multi-service stack since the D: migration,
   carries its own non-trivial risk of consuming the FIRST real order attempt on an
   environment-configuration problem (port conflicts, stale connection strings, migration
   drift) rather than a genuine app-logic bug — which would waste the same irreplaceable data
   point for a different, less useful reason.
2. **Zero real orders are imminent.** The database has had exactly 5 orders total across the
   entire project's history to date; there is no evidence of active real traffic that makes
   "first real order arrives any minute" a live risk today.
3. Both of this project's actual production blockers (no real SLA, ~0 fulfillment data) are
   unaffected either way — running the E2E today validates plumbing, not the underlying
   scientific readiness.

**Exact trigger that changes this decision**: run the full E2E test (Docker Postgres+Redis on
D:-backed storage, Medusa backend, FastAPI service, one real test order through the actual
`order.placed` event) **immediately before, and as a precondition of, either (a) the business
SLA decision being implemented in code (Item 5 unlocks this), or (b) any real customer
traffic being pointed at this store for the first time — whichever comes first.** This ties
the E2E to a concrete, meaningful checkpoint instead of an open-ended "later." I did not
execute it this session under the current constraints.

---

## Item 5 — Business Decision Memo

Produced as the standalone deliverable: `docs/HASEBHA_SLA_BUSINESS_DECISION_MEMO.md`.

---

## Item 6 — Funnel Closure

**Confirmed complete.** `reports/generated/olist_funnel/` contains the full report, scorecard,
and three supporting JSON audits (acquisition/quality, leakage check, experiment results).
The negative result is now formally entry **NR-16** in
`reports/generated/committee_defense/NEGATIVE_RESULTS_REGISTER.json` (register `count` field
updated from 15 to 16, verified by direct re-parse of the JSON after editing). Committee-
package updates from Item 2 (D1/D2/D4) are in place, referencing the funnel result by its
real numbers (mean AUC delta −0.0034, 4.5% coverage, 0/4,384 leakage violations) — no new
number was introduced anywhere that isn't already in `OLIST_FUNNEL_SCORECARD.json`.

---

## Git Scope (before / after this session)

**Before**: 6 modified tracked files (the standing production diff) + 1 additional tracked
file already modified by the funnel mission (`docs/data_provenance.md`) = 7 modified tracked
files; untracked additions from all prior sessions.

**After this session**: still exactly the same 7 modified tracked files — `config.py`,
`main.py`, `fulfillment.py`, `health.py`, `schemas.py`, `order-placed.ts`, and
`docs/data_provenance.md` — **zero additional tracked files modified**, zero production code
touched, zero frozen-track files touched. New untracked additions: `docs/committee/` (edited
in place, no new files), `docs/DECISIONS_FINAL.md`,
`docs/HASEBHA_SLA_BUSINESS_DECISION_MEMO.md`, and the edit to
`reports/generated/committee_defense/NEGATIVE_RESULTS_REGISTER.json` (already untracked from
the prior committee-defense session, edited in place).

**Project state at the end of this session**: nothing left to close on the engineering side
except the two items explicitly out of engineering's authority — the business SLA decision
(Item 5) and real order volume — plus the two deliberately-deferred items above (Amazon
wiring, pending a human reconciliation decision; Shadow E2E, pending its stated trigger).
