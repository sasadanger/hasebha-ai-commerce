# NLP Experiment Methodology Policies

`PHASE2C_AUTHORIZATION_SCOPE = NLP_EXPERIMENT_DEFINITION_ONLY`. Nothing in this document authorizes training, fine-tuning, or embedding generation. All content is design-only.

## Step 19 — Baseline model policy (no training performed)

Future baseline classes, weakest to strongest, for every classification task in `configs/nlp_experiment_manifest.yaml`:

1. **Majority-class baseline** — reports the metric floor every real model must beat.
2. **Stratified/random baseline** — useful mainly for multi-class tasks (ArSAS speech act, ArSAS sentiment) where majority-class alone under-characterizes difficulty.
3. **TF-IDF + Logistic Regression** — first real baseline; fast, interpretable, well-suited to Arabic and English short text alike.
4. **TF-IDF + Linear SVM** — typically strong on sparse bag-of-words text; standard second baseline.
5. **Naive Bayes** — considered only where class-conditional word-independence is a reasonable approximation (short social-media posts); not proposed for the longer Amazon/LABR review text.
6. **Arabic-capable transformer baseline** (see Step 20) — for Arabic-language tasks (B1/B2/B3/C/D1/D2/E) once a training gate authorizes it.
7. **Multilingual transformer baseline** (see Step 20) — cross-checks whether a multilingual model matches or beats an Arabic-specific one.
8. **English e-commerce model** — for Experiment A only.

No final champion is selected in this gate. `training_authorized: false` on every entry.

## Step 20 — Transformer candidate policy (design only; no download/train in this gate)

| Candidate family | Applies to | Required checks before any future use |
|---|---|---|
| Arabic-focused pretrained transformer | B1, B2, B3, C, D1, D2, E | license, model-card provenance, language/dialect coverage claims, parameter size, tokenizer suitability for Arabic script + code-switching, commercial-use terms |
| Multilingual transformer | B1–E (comparison arm) | same checks; explicit language-coverage table for Arabic/Egyptian dialect |
| English e-commerce-tuned model | A | license, provenance, commercial-use terms |
| Egyptian-dialect-oriented candidate | B1/B2/B3/F (if one becomes available) | same checks; must not be assumed to exist — search and document at authorization time, not now |

No transformer is downloaded or executed in this gate.

## Step 30 — Cross-dataset generalization plan

- **Train Egyptian Tweets 40K → external-test ASTD**: measures whether a large, balanced, binary-sentiment model generalizes to a smaller, 4-class, differently-collected Egyptian corpus.
- **Train ASTD → external-test Egyptian Tweets 40K**: the reverse direction; measures whether a smaller corpus's signal generalizes to a larger one, and exposes any 4-class→binary label-mapping assumptions explicitly (per the merge policy, `MERGE_ALLOWED_WITH_RELABELLING_RULE` only).
- **Train LABR (Arabic review) → do NOT assume Twitter generalization.** LABR is intentionally *not* paired with any Twitter-sourced external test in this gate; domain shift (book reviews vs. social media) is treated as an open question requiring its own dedicated experiment, not an assumption.

## Step 31 — Platform generalization

Planned per-platform breakdown, never pooled invisibly:

- E-commerce reviews (Amazon, and LABR-as-reviews though not e-commerce)
- Twitter/X (Egyptian Tweets 40K, ASTD, ArSAS)
- Multi-platform social (MPOLD: Twitter+Facebook+YouTube)
- YouTube specifically (EESA, once acquired)
- Customer service (ADAB, once acquired; first-party data, once it exists)

Every experiment's reporting must break out platform as a column, per `configs/nlp_split_policy.yaml`'s `source_platform_awareness` requirement.

## Step 32 — Language generalization

Distinct evaluation required for: English (Amazon); Arabic general/MSA-leaning (LABR); Egyptian Arabic (Egyptian Tweets 40K, ASTD); mixed Arabic varieties (ArSAS, MPOLD); Arabic-English code-switch (EESA, once acquired). No single aggregate score may be reported as covering all of these.

## Step 33 — Business relevance layer

Three distinct, non-substitutable layers, to be labeled explicitly in every future report:

1. **Language benchmark success** — e.g. "Egyptian Tweets sentiment model reaches macro F1 = X." Says nothing about commerce.
2. **Commerce task success** — e.g. "review-rating model correctly identifies dissatisfied Amazon customers." Says nothing about Egypt.
3. **Egyptian market validation** — requires genuine Egyptian commerce/customer-service data (Experiment H), which does not yet exist. A model succeeding at (1) and/or (2) has proven neither (3) nor readiness for it.

## Step 34 — Future NLP signals (conceptual only; none created in this gate)

`sentiment_score`, `complaint_probability`, `intent_class`, `speech_act`, `politeness`, `toxicity_offensive_flag`, `aspect_topic`, `emotion`. No live signal is produced by this gate.

## Step 35 — Structured + NLP integration boundary (reaffirmed, not modified)

Unchanged from `docs/structured_nlp_integration_design.md`: text may only enter an earlier predictive model if it existed before that model's decision timestamp. Post-delivery review text must never enter the late-delivery prediction model (decision timestamp `order_approved_at`). Permitted uses for post-outcome text remain VOC analytics, product/seller quality, customer satisfaction, complaint analysis, and future support automation — none of which are executed in this gate.

## Step 36 — Future first-party Egyptian data contract (schema only; no data created)

Recommended future schema: `text`, `channel`, `event_timestamp`, `ingestion_timestamp`, `customer_pseudonymous_id`, `order_id` (where lawful), `product_id` (where lawful), `seller_id` (where lawful), `reason_code`, `human_label`, `resolution`, `pii_redaction_status`, `language`, `dialect` (if annotated), `consent_legal_basis`. No synthetic or fabricated data is created to populate this schema.

## Step 37 — Success criteria per experiment (evidence-grounded, not arbitrary)

| Experiment | Primary metric | Baseline-to-beat | Quality gates |
|---|---|---|---|
| A | macro F1 (binned rating) | majority-class | duplicate gate, license gate (deferred pending rights review) |
| B1 | macro F1 | majority-class | duplicate gate (**blocked until reaudit**), license gate (CC0 data license, platform terms pending) |
| B2 | macro F1 | majority-class | duplicate gate, license gate (GPLv2 + platform terms pending) |
| B3 | macro F1 on external test | in-domain B1/B2 result | cross_dataset_exact_overlap check required before running |
| C | macro F1 (binned rating) | majority-class | duplicate gate, license gate (GPLv2 + Goodreads rights pending) |
| D1/D2 | macro F1 | majority-class | duplicate gate (**blocked until reaudit**), license gate (no license stated — research-only) |
| E | macro F1 | majority-class | duplicate gate, license gate (Apache 2.0 + platform-text scope pending) |

No arbitrary absolute F1 target (e.g. "0.90") is set for any task; success is defined relative to the majority-class baseline and, where available, published results for the same public dataset, not an invented number.

## Step 38 — Failure conditions (any one blocks a future training run)

License contradiction; unresolved dataset identity; split leakage (cross-split duplicate found); duplicate contamination (reaudit not completed for Egyptian Tweets 40K / ArSAS); label-mapping ambiguity (e.g. attempting to merge ASTD/Egyptian-Tweets without the documented relabelling rule); class collapse (a class effectively disappears after preprocessing); insufficient samples for a class below `min_class_count_for_stratification`; unreproducible preprocessing (normalization contract not followed exactly); any claim of Egyptian-market or commerce readiness not backed by Experiment H.

## Step 39 — Experiment priority order (independently justified, not copied verbatim from the prompt's example)

1. **Amazon baseline (A)** — largest, cleanest, most licensable-pending-review dataset; establishes pipeline correctness before touching harder Arabic text.
2. **Egyptian Tweets 40K (B1)** — largest Egyptian-specific asset, but *blocked* until its duplicate reaudit completes; listed second so the reaudit can start immediately in parallel with (1).
3. **ASTD (B2)** — no reaudit blocker, ready now; provides a second, independent Egyptian-dialect signal while B1's reaudit is pending.
4. **B3 cross-domain generalization** — depends on both B1 and B2 being clean; naturally follows.
5. **LABR (C)** — general Arabic-language robustness check, useful once Egyptian-specific baselines exist for comparison.
6. **MPOLD (E)** — no reaudit blocker, ready now, but lower business relevance (safety, not sentiment) than the Egyptian-sentiment work above it.
7. **ArSAS sentiment (D1)** and **speech act (D2)** — *blocked* until reaudit; also lowest Egypt-relevance among active datasets (mixed dialects, not Egypt-specific), so placed after MPOLD despite being "active."
8. **EESA (F)** — highest-value pending acquisition (Egyptian code-switch, direct relevance to CommercePilot's stated NLP emphasis); should be actively pursued for acquisition in parallel with 1–7.
9. **ADAB (G)** — second-highest-value pending acquisition (explicit e-commerce/customer-service domain); access-channel resolution should be pursued in parallel.
10. **First-party Egyptian commerce data (H)** — longest lead time, requires business/legal work outside this project's current scope; last in sequence but should be scoped early given its lead time.

## Step 40 — Compute/hardware feasibility design (inspection only; no training)

| Tier | Feasible on this environment? | Applies to |
|---|---|---|
| CPU-friendly baseline (majority-class, TF-IDF+LogReg/SVM/NB) | Yes — no GPU required, small-to-moderate datasets (max ~63K rows for LABR) | A, B1, B2, B3, C, D1, D2, E |
| Single-GPU transformer baseline | Not verified in this gate — no GPU inventory check was performed; a future training-authorization gate must confirm hardware availability before selecting this tier | B1–E transformer candidates |
| Transformer fine-tuning | Same caveat as above, plus dataset-size adequacy (smallest active dataset, ArSAS, is 19,897 rows — plausible for light fine-tuning if hardware exists) | B1–E |
| Larger-model research (optional) | Not assumed available; explicitly optional and lowest priority | none currently planned |

Every experiment (A–E) has a CPU-friendly baseline tier defined, so no experiment is designed without a feasible fallback.
