# Model Card: Amazon Appliances Sentiment Classifier

**Model:** `distilroberta-base`, fine-tuned end-to-end for binary sequence
classification. **Status: frozen primary model** (baseline: TF-IDF+LinearSVC,
also frozen, kept for comparison).

## Intended use

Classifying the sentiment (Negative / Positive) of English-language Amazon
product review text **in the Appliances category**, for review-analytics or
triage use cases (e.g. surfacing likely-negative reviews for attention).
Designed and evaluated for **verified-purchase** reviews.

## Non-intended use

- **Not** a 3-class (Negative/Neutral/Positive) classifier — it has never
  seen a 3★ ("neutral") training example and will force any such review into
  Negative or Positive.
- **Not** validated on any product category other than Appliances, any
  language other than English, or any platform other than Amazon.
- **Not** validated on unverified-purchase reviews as a primary population
  (they were excluded from training/evaluation; see `FINAL_MODELING_REPORT.md` §5).
- **Not** a rating predictor — it never sees `rating` and does not
  reconstruct star counts, only Negative/Positive sentiment.
- Raw model output (pre-calibration, threshold 0.5) reflects a **balanced
  50/50 training prior**, not the real deployment class balance — do not use
  raw scores or the 0.5 default threshold operationally; use the calibrated
  score and the 0.06 selected threshold instead (see below).

## Training data

2,128,605-row Amazon Appliances review corpus, filtered to `verified_purchase
= True` and rating ∈ {1,2,4,5} (3★ excluded), deduplicated (exact + near),
then a class-balanced 50,000-row sample (25,000 Negative / 25,000 Positive,
preserving natural 1★-vs-2★ and 4★-vs-5★ sub-mixes) drawn via a
product-group-disjoint split so no training product overlaps any evaluation
set. Full detail: `FINAL_MODELING_REPORT.md` §§1–9.

## Label semantics

`0` = Negative (source rating 1★ or 2★). `1` = Positive (source rating 4★ or
5★). Rating 3★ is out of scope for this model entirely — there is no
"Neutral" output class.

## The class-prior issue caused by balanced training

The model was trained on an artificially balanced 50/50 sample, but real
Amazon Appliances traffic is approximately 15.5% Negative / 84.5% Positive
(the true verified-purchase population ratio). A model trained this way
tends to over-predict the minority (Negative) class relative to its true
frequency if used with the naive 0.5 decision threshold — this is a known,
expected effect of class-balanced training, not a bug.

## Why calibration and threshold selection are required before natural deployment

Raw model probabilities were **not** trustworthy as real-world probabilities
out of the box (this is exactly what class-balanced training predicts).
Temperature scaling (T=1.10246741771698) was fit on a held-out,
naturally-distributed validation set (`val_natural`, 15,000 rows, never used
for anything else) to partially correct this, and the operational decision
threshold was moved from 0.5 to **0.06** — also selected purely from
`val_natural`, never from any test set. **Any deployment of this model must
use the calibrated score and the 0.06 threshold, not the raw score and 0.5**
— using the raw/default combination will materially misjudge the real-world
Negative/Positive balance.

## Expected inputs

Free-text English product review, ideally with both a title and a body
(the model concatenates `title + ". " + body`; body alone if no title).
Tokenized to a maximum of 128 tokens (distilroberta-base tokenizer);
longer reviews are truncated for the model but the original full text
should always be preserved outside the model input (this repo's pipeline
does this — see `src/nlp/amazon/data.py`).

## Inference output

A calibrated probability of Positive sentiment (`calibrated_score`), plus a
binary label at the selected 0.06 threshold. Both raw and calibrated scores
are available; only the calibrated score + 0.06 threshold pairing is
recommended for operational use (see above).

## Limitations

- No support for 3★/neutral sentiment.
- English only; Amazon-Appliances-review-domain only.
- A measurable rate (0.36–0.60%, from a 100k audit sample) of training
  labels show apparent rating-vs-text-sentiment conflict — genuine label
  noise inherent to using star ratings as a sentiment proxy, which caps
  achievable accuracy regardless of model quality.
- Slice-level macro-F1 by individual star rating (Section 19 of
  `FINAL_MODELING_REPORT.md`) is **not directly interpretable** as
  per-rating quality — those slices are effectively single-true-class
  subsets, where macro-F1 is not a meaningful discrimination measure; use
  accuracy for those specific slices instead (0.86–0.98 across ratings).
- No hyperparameter search was performed; the reported result reflects one
  documented configuration, not an optimum found by search.

## Language / domain limitations

Trained and evaluated exclusively on English-language Amazon Appliances
reviews. No claim of transfer to other languages, dialects, product
categories, or review platforms is made or implied.

## Distribution-shift risks and generalization evidence

Two stress tests were run specifically to probe distribution shift, both on
data disjoint from training in the relevant axis:

- **Product-holdout** (products never seen in training): macro-F1 0.9642 —
  no meaningful drop from the in-distribution balanced test (0.9663).
- **Chronological** (the most recent reviews, i.e. simulating deployment on
  future/unseen-at-training-time data): macro-F1 0.9678 — also no drop.

Both results beat the TF-IDF baseline on the same rows. This is real,
verified evidence of reasonable generalization across products and time
within the trained domain — it is not evidence of generalization outside
that domain (different product category, language, or platform).

## Leakage protections

- `rating`/`verified_purchase`/`helpful_vote` never enter the model's text
  input — enforced in code and unit-tested.
- Exact- and near-duplicate review text removed globally before any split.
- All evaluation sets are product-group-disjoint from training (except
  `chronological_stress`, which deliberately is not, by design — see
  `FINAL_MODELING_REPORT.md` §8).
- Calibration/threshold selection used only `val_natural`, never any test set.
- Training-size and epoch selection used only the balanced `val` set, never
  any test set.

## Evaluation summary

| Eval set | n | Macro-F1 | vs. frozen TF-IDF baseline |
|---|---:|---:|---|
| Balanced test | 40,000 | 0.9663 | +0.0209 |
| Representative test (natural ratio, calibrated threshold) | 50,000 | 0.9499 | +0.0301 (default-threshold basis) |
| Chronological stress | 5,000 | 0.9678 | +0.0168 |
| Product-holdout stress | 5,002 | 0.9642 | +0.0199 |

Transformer wins on all four independently-verified sets. Full metrics,
confusion matrices, and slice breakdowns: `FINAL_MODELING_REPORT.md`,
`reports/generated/amazon/transformer_final_eval.json`.
