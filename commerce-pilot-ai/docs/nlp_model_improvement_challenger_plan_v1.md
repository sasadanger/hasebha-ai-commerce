# NLP Model-Improvement Challenger Plan v1 (prospective; screening only)

Status: `MODEL_IMPROVEMENT_PHASE = ACTIVE`. Batch 1 classical baselines are
`OFFICIAL_FROZEN_CLASSICAL_BASELINE` (do not modify). This document records, prospectively,
the challenger and transformer-screening work layered on top of Batch 1, consistent with
`docs/nlp_experiment_methodology_policies.md` Step 19's baseline progression (majority-class →
stratified → classical TF-IDF → **transformer**, item 6-7) and Step 20's transformer candidate
policy, which explicitly requires its own training gate (already satisfied: the user
authorized real training, including transformer screening, for this session).

This is not a new roadmap invented from a single prompt — it extends the already-documented
baseline progression that Batch 1 only partially executed (classical tier only).

## Scope boundary (unchanged from every prior Phase 2C gate)

- `INTERNAL_TEST_STATUS = CLOSED`. No internal_test row is read, counted, predicted on, or
  used for calibration/threshold-tuning at any point in this plan. All work below is
  train+validation only.
- Batch 1's 20 classical configurations, their winners, and their artifacts under
  `artifacts/experiments/nlp/phase2c/batch1/` are frozen and are not retrained, edited, or
  overwritten by anything in this plan. Every challenger below is a **new**, separately
  artifacted experiment.
- No task definition, label mapping, or split methodology changes. LABR remains five-class
  rating prediction (never collapsed to binary). Amazon/LABR "binned rating" values are the
  raw 1–5 integer labels per the active `configs/nlp_label_ontology_v2.yaml`
  (`derived_sentiment_mapping: NONE`) — unchanged.
- No claim of Egyptian-market or production readiness is made anywhere in this plan. Per
  `docs/nlp_experiment_methodology_policies.md` Step 33, results here belong to Layer 1
  (language benchmark) and Layer 2 (commerce-task relevance) only. Layer 3 (Egyptian-market
  validation) requires Experiment H (first-party Egyptian commerce data), which does not exist
  (`docs/nlp_egyptian_first_party_data_gap_analysis.md`).

## Phase A — Layered Batch 1 baseline analysis (completes existing work)

Majority-class and stratified-random baselines computed against the same deterministic
validation split each winner was selected on (`scripts/compute_batch1_baselines.py`), reported
per Step 37 (success relative to majority-class baseline, never as a cross-dataset ranking).

## Phase B — Validation-only diagnostics (A/B2/C/E)

Per experiment: class distribution, confusion matrix, per-class precision/recall/F1/support,
macro-F1, balanced accuracy, accuracy, most-confused label pairs. For the two 1–5 star tasks
(A, C): mean absolute rating error, distribution of absolute rating distance, adjacent-class
error rate, and severe-error rate (`|prediction - truth| >= 2`) as supplemental diagnostics —
macro-F1 remains the primary model-selection metric, unchanged.

## Phase C — Classical challengers (new artifacts, Batch 1 untouched)

Motivation: word-level TF-IDF may under-serve Arabic's rich morphology and MPOLD/ASTD's
noisy, code-switched, elongated-character social text; character n-grams are a standard,
cheap, evidence-backed classical technique for exactly this failure mode before reaching for a
transformer.

- **MPOLD** (first — smallest, fastest signal): char/`char_wb` TF-IDF (~2–5 grams) + LinearSVC;
  then word+char combined if it helps.
- **ASTD**: char/`char_wb` TF-IDF (~3–5 grams) + LinearSVC; then word(1–2)+char(3–5) combined.
  Highest priority among Arabic classical challengers — ASTD currently has the weakest
  classical macro-F1 (0.394) among the Arabic benchmark tasks.
- **LABR**: char TF-IDF (~3–5 grams) + LinearSVC; then word(1–3)+char(3–5) combined. Remains
  five-class throughout.

`class_weight=balanced` used only where validation diagnostics justify it (class imbalance
observed in Phase B), not applied blindly.

## Phase D — Transformer screening (one seed, screening only — not final confirmation)

Order is deliberate: smallest datasets first, to maximize the number of independent completed
experiments before the largest dataset (Amazon) can monopolize the GPU.

| Order | Experiment | Model | Rationale |
|---|---|---|---|
| T1 | MPOLD | MARBERT | smallest (4K rows); doubles as the transformer-pipeline smoke test |
| T2 | ASTD | MARBERT | weakest classical macro-F1 among Arabic tasks — highest-value screen |
| T3 | LABR | AraBERT | largest Arabic dataset; needs its own tokenizer-length calibration first |
| T4 | ASTD | AraBERT | cross-check against T2's model family, if time remains |
| T5 | LABR | MARBERT | cross-check against T3's model family, if time remains |
| T6 | Amazon | ONE English encoder (ModernBERT-base or DeBERTa-v3-base, evidence-based single choice) | last — largest dataset (1.5M train rows); throughput-calibrated GO/DEFER decision before committing, to avoid consuming the entire compute window on one experiment |

Each transformer run: one fixed seed, up to 4 epochs, `save_total_limit=1`, validation-only
selection, full config/seed/revision/hash recorded. This is **screening**, not final
multi-seed confirmation — that only happens for serious finalists in a future session, after
today's screening narrows the field.

Amazon transformer training is authorized tonight as **one** controlled screening run, gated
on a short throughput/memory calibration (not itself a model-selection result) that estimates
examples/sec and expected total runtime before committing. If the estimate implies the run
would not complete within the available window, the correct decision is
`AMAZON_TRANSFORMER_EXECUTION = DEFER` with the reason and prepared configuration preserved
for a future dedicated compute window — not a forced partial/rushed run.

## Selection philosophy (Step 21 of the overnight brief, preserved here)

A transformer does not automatically win. Compared dimensions: macro-F1, balanced accuracy,
per-class F1 (especially minority classes), stability, inference latency, model size,
deployment requirements. A roughly +0.02 absolute macro-F1 improvement is treated as a useful
engineering materiality threshold, not a universal law — smaller deltas require stronger
justification, and a cheap classical challenger remains preferable when quality is comparable.

## Egyptian e-commerce domain target (documented, not executed)

CommercePilot's eventual NLP layer must support business-facing labels beyond sentiment:
delivery issue, damaged product, wrong/missing item, size/fit, quality, description mismatch,
return/refund, customer service, payment, price/value, warranty, authenticity, urgency,
resolution intent — over language actually observed in the target market (Egyptian Arabic,
MSA, English, Arabizi, Arabic-English code-switching). None of A/B2/C/E, nor any challenger or
transformer screen in this plan, constitutes evidence toward this — they are public benchmark
datasets (`DEVELOPMENT_BENCHMARK` role per `docs/phase2c_egyptian_market_evidence_requirements.md`),
not Egyptian commerce data. Domain-adaptive pretraining is explicitly not begun before
appropriate domain data with documented provenance/consent exists.

## What happens after tonight

Serious finalists (classical or transformer) identified tonight receive multi-seed
confirmation in a future session. Internal-test release remains a separate, explicitly
authorized gate, entered only after champion selection is frozen on validation evidence —
never opened to "check" a promising result during screening.
