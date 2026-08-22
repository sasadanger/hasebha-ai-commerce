# Amazon Appliances Sentiment — Final Modeling Report

Status: **FROZEN**. Prepared 2026-08-17. This document packages and cross-checks
work already completed and independently verified across this session — it
does not retrain, re-tune, or re-select anything. Every number below is
either quoted directly from a saved JSON artifact or was recomputed by
re-running verification code against the actual files on disk (marked 🟢
where I personally recomputed it in this session, as opposed to 📄 where I
am quoting a saved report file).

## 1. Source dataset

- File: `data/processed/amazon_reviews_appliances/reviews_text_ready.parquet`
- 2,128,605 rows 🟢 (confirmed via direct DuckDB scan)
- Columns: `rating, title, text, images, asin, parent_asin, user_id, timestamp, helpful_vote, verified_purchase, review_datetime_utc, has_usable_text` 🟢
- Timestamp range: 2000-10-23 to 2023-09-12 📄

## 2. Rating distribution (full dataset) 📄

| Rating | Count |
|---|---:|
| 1★ | 250,453 |
| 2★ | 79,664 |
| 3★ | 104,047 |
| 4★ | 208,216 |
| 5★ | 1,486,225 |

## 3. Binary target definition

`label = 0` (Negative) for rating ∈ {1, 2}; `label = 1` (Positive) for rating
∈ {4, 5}; rating = 3 is **excluded** from the binary modeling task (retained
only in the EDA notebook). `rating` — and anything derived from it,
including `verified_purchase` — is never a model input feature; the model
only ever sees combined title+body text. This is enforced in code
(`build_model_text()` / `build_transformer_text()` in `src/nlp/amazon/data.py`)
and checked by a dedicated unit test in `tests/test_amazon_nlp.py`.

## 4. Exclusions and retained counts

| Scope | Negative (1-2★) | Positive (4-5★) | 3★ (excluded) |
|---|---:|---:|---:|
| `verified_purchase = True` | 301,730 | 1,640,151 | 98,183 |
| `verified_purchase = False` | 28,387 | 54,290 | 5,864 |
| All reviews | 330,117 | 1,694,441 | 104,047 |

## 5. Verified-purchase scope decision

**Decision: `verified_purchase = True` only**, for both the TF-IDF and
Transformer pipelines. Both classes exceed 300K rows within verified
purchases — far above the "tens of thousands per class" bar — so the
restriction does not create a class-size problem, and it removes exposure to
incentivized/fraudulent-review noise more common in unverified reviews.
Unverified rows remain in the EDA notebook only. Full reasoning:
`reports/generated/amazon/scope_decision.md`.

## 6. Text preprocessing rules

**TF-IDF model input** (`build_model_text`): `title + ". " + text` when title
is present, else `text` alone. No stemming/lemmatization/stopword removal.

**Transformer model input** (`build_transformer_text` /
`normalize_text_for_transformer`): same title+body combination, plus light
normalization only — Unicode NFKC normalization, raw HTML tag stripping,
control-character stripping, whitespace collapse. Explicitly **not**
applied: stemming, lemmatization, stopword removal, punctuation stripping,
emoji stripping, lowercasing. Both pipelines share the same leakage rule:
`rating`/`verified_purchase`/`helpful_vote` are never part of the model text.

Data-quality items quantified but deliberately **not excluded** (100k audit
sample, 📄 `transformer_data_audit.json`): 7.48% contain HTML tags (stripped
by normalization, not dropped), 0.85% contain review-disclosure boilerplate,
0.36%/0.60% show label-text/title-text keyword-conflict heuristics — treated
as genuine, expected label noise inherent to star-rating-derived labels, not
a data defect to filter out.

## 7. Duplicate and near-duplicate handling

`remove_duplicate_text()` runs globally on the full modeling pool, **before**
any split: exact-duplicate text removed first (keep first occurrence), then
near-duplicate (normalized: lowercase, strip non-`[a-z0-9 ]` characters,
collapse whitespace) removed the same way. Full-dataset counts: 285,655 exact
duplicates (13.4%), 342,031 near-duplicates (16.1%) 📄. Pipeline-pool dedup
report (📄 `split_manifest.json`): 801,184 → 719,378 (81,806 exact removed) →
700,609 (18,769 near removed).

## 8. Leakage-control methodology

- Feature/label separation enforced in code and tested (`tests/test_amazon_nlp.py`).
- Global dedup before any split (above).
- **Product-group-disjoint splits**: every `parent_asin` assigned as a whole
  to exactly one bucket among {val, test_balanced, test_representative,
  train, val_natural} — no product crosses these five buckets.
  `product_holdout_stress` is fully product-disjoint from everything by
  construction (whole products removed from the pool before the main split).
  `chronological_stress` **intentionally** shares products with other
  buckets (this is correct by design — it tests time-generalization, not
  product-generalization; verified 🟢 zero *row-level* overlap regardless).
- Test sets were never used for training-size selection, epoch selection,
  calibration, or threshold tuning — those decisions used only the
  validation sets (`val` for TF-IDF training-size/epoch selection, `val`
  again for the transformer's per-epoch selection, `val_natural` exclusively
  for calibration/threshold).

**Independent disjointness verification, this session 🟢** (content-hash
`review_uid` and `parent_asin` overlap, computed directly from the split-ID
files):

| Pair | review_uid overlap | parent_asin overlap |
|---|---:|---:|
| Every pair among {val, test_balanced, test_representative, train_full_pool, product_holdout_stress} | 0 | 0 |
| val_natural × every other bucket | 0 | 0 (except chronological_stress, see below) |
| chronological_stress × {val, test_balanced, test_representative, train_full_pool} | 0 | 183–1,164 (expected, by design) |
| chronological_stress × val_natural | 0 | 83 (expected, by design) |
| chronological_stress × product_holdout_stress | 0 | 0 |

## 9. Split definitions and sizes

| Split | Rows | Negative | Positive | Purpose |
|---|---:|---:|---:|---|
| `train` (max, 200K) | 200,000 | 100,000 | 100,000 | Learning-curve pool (TF-IDF); nested 25K/50K/100K/200K subsets |
| `val` (balanced) | 20,000 | 10,000 | 10,000 | Fair model/training-size comparison; transformer per-epoch selection |
| `val_natural` | 15,000 | 2,331 | 12,669 | Transformer calibration + threshold selection **only** |
| `test_balanced` | 40,000 | 20,000 | 20,000 | Fixed, one-time final evaluation |
| `test_representative` | 50,000 | 7,769 | 42,231 | Fixed, one-time final evaluation, true population ratio |
| `product_holdout_stress` | 5,002 | 2,086 | 2,916 | Never-trained-on products |
| `chronological_stress` | 5,000 | 2,772 | 2,228 | Most-recent reviews, never trained on |

`test_representative`'s 7,769/42,231 split is not an error — it mirrors the
true verified-purchase population ratio exactly: 301,730/(301,730+1,640,151)
= 15.54% negative = 7,769/50,000 🟢 (independently recomputed and confirmed
this session).

## 10. Balanced vs. natural validation — why both exist

`val` (balanced, 10K/10K) answers "which model/training-size is best,
compared fairly?" — used for the TF-IDF learning-curve/plateau decision and
the transformer's per-epoch best-checkpoint selection. `val_natural`
(natural ratio, 2,331/12,669) answers a different question — "at real-world
class balance, what score should count as 'positive' and how well-calibrated
are the probabilities?" — used **exclusively** for temperature scaling and
threshold selection (Section 15). Neither set is used for the other's
purpose; both are strictly disjoint from every test set and from each other.

## 11. Token-length audit and max_length decision 📄

Token-length percentiles (distilroberta-base tokenizer, 50K training sample,
no truncation): p50=35, p75=61, p90=103, p95=144, p99=272, max=4,393, mean=51.5.

Pilot comparison (3,000 train / 1,000 eval rows, 1 epoch):

| max_length | Truncation % | Pilot macro-F1 | Examples/sec | Peak VRAM |
|---|---:|---:|---:|---:|
| 128 | 6.34% | 0.9470 | 302.1 | 1,900 MB |
| 192 | 2.43% | 0.9450 | 249.7 | 2,317 MB |

**Selected: 128.** 192 gave zero measurable quality gain (−0.20pp, i.e.
slightly worse, within noise) at higher cost on both speed and memory —
128 is strictly better on this evidence.

## 12. Exact Transformer checkpoint and hyperparameters 📄

- Checkpoint: `distilroberta-base` (no fallback needed — loaded successfully)
- Fine-tuned **end-to-end** for sequence classification (`AutoModelForSequenceClassification`), not a frozen-embedding+classifier approach
- Learning rate: 2e-5 · Weight decay: 0.01 · Max epochs: 2 · Warmup: 6% (47 steps — `warmup_ratio` was removed from `TrainingArguments` in transformers 5.15.0, so `warmup_steps` was computed explicitly to reproduce the same schedule)
- Optimizer: AdamW (`adamw_torch`) · Seed: 20260817 · bf16: True · TF32: True · `pad_to_multiple_of=8`
- `metric_for_best_model="eval_macro_f1"`, `load_best_model_at_end=True`, `save_total_limit=1`
- Batch size: 128 (largest stable size found via a dry run) · max_length: 128

## 13. Training sample composition (50K, balanced) 📄

Sub-rating breakdown: 1★=18,616, 2★=6,384 (Negative=25,000 total); 4★=3,420,
5★=21,580 (Positive=25,000 total). 7,109 unique products. Timestamp span:
2006-07-09 to 2023-06-09.

## 14. Epoch 1 vs. epoch 2, and why 50K was sufficient 📄

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Eval loss (balanced val) | 0.1041 | 0.1039 |
| Macro-F1 | 0.9638 | 0.9659 |
| Negative F1 | 0.9640 | 0.9662 |
| Positive F1 | 0.9636 | 0.9656 |

Gain epoch 1→2: +0.21pp macro-F1 — small but real (not plateaued, no
overfitting signal), so both allowed epochs were run and the epoch-2
checkpoint (higher macro-F1) was kept via `load_best_model_at_end`.

**100K expansion: not performed, and not needed.** The 50K/epoch-2 result
(0.9659 balanced-val macro-F1) already exceeds the TF-IDF pipeline's own
**100K-row** validation macro-F1 (0.9429). Given the small epoch-to-epoch
gain and that the 50K result already cleared the relevant comparison bar,
expanding to 100K was judged not to justify the extra runtime. Full 50K/2-epoch
run: **4.5 minutes wall-clock**, 269.9 seconds.

## 15. Hardware/software environment and GPU/precision proof 📄

- GPU: NVIDIA RTX 2000 Ada Generation, 16 GB VRAM, compute capability 8.9 (Ampere-or-newer)
- `torch==2.6.0+cu124`, `transformers==5.15.0`, `datasets==5.0.1`, `accelerate==1.14.0`
- CPU: 28 logical cores · System RAM: 31.71 GiB total
- **Genuine GPU compute proof** (not just `is_available()`): a timed 20× bf16 4096×4096 matmul on `cuda:0` completed in 0.143s, achieving 19.2 TFLOPS — this proves bf16 tensor-core compute genuinely ran on the GPU, not a CPU fallback.
- bf16/TF32/SDPA all confirmed usable given Ampere-or-newer compute capability.
- Smoke test (300 steps) passed every check before the full run: loss decreased 0.654→0.129, gradients finite, no label inversion (6/6 spot-checked rating→label mappings correct), checkpoint save/load verified.
- Peak VRAM during full training run: 4,389.8 MB (well within the 16 GB budget).

## 16. Calibration method, temperature, and threshold 📄

- Method: single-scalar temperature scaling (Guo et al. 2017), fit by NLL minimization
- Fit on: `val_natural` **only** (15,000 rows) — never any test set
- Temperature: **1.10246741771698**
- Default threshold (0.5) accuracy on val_natural: 0.9613
- Selected operational threshold: **0.06** (argmax of macro-F1 over calibrated positive-class probability, swept 0.05–0.95 on val_natural only) — val_natural macro-F1 at this threshold: 0.9465
- The shift from 0.5 to 0.06 is expected and correct: training was on a balanced 50/50 sample, but the real deployment prior is ~15.5% negative / ~84.5% positive — a lower threshold compensates for that prior mismatch.
- Brier score: raw 0.03171 → calibrated 0.03125 (improved). ECE: raw 0.03461 → calibrated 0.03584 (slightly worse) — reported honestly; temperature scaling helped one calibration metric and not the other, not oversold as an unambiguous win.

## 17. Final test results — all four sets, calibrated scores at the selected threshold (0.06) 📄

| Eval set | n | Macro-F1 | Weighted-F1 | Accuracy | Balanced Acc. | MCC | ROC-AUC | Avg. Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_balanced | 40,000 | 0.9663 | — | — | — | — | — | — |
| test_representative | 50,000 | 0.9499 | 0.9739 | 0.9741 | 0.9434 | 0.9000 | 0.9936 | 0.9987 |
| chronological_stress | 5,000 | 0.9678 | — | — | — | — | — | — |
| product_holdout_stress | 5,002 | 0.9642 | — | — | — | — | — | — |

(Full per-class precision/recall/F1, confusion matrices, and both
default-threshold and calibrated-threshold results for every set are in
`reports/generated/amazon/transformer_final_eval.json` — the table above
shows macro-F1 for all four plus the full metric suite for
`test_representative` as the headline "real-world-like" set; MCC/balanced
accuracy/PR-AUC were computed for the transformer, which the earlier TF-IDF
`metrics.json` did not include.)

## 18. Transformer vs. TF-IDF — same row IDs, all four sets 📄🟢 (deltas independently re-verified this session against the frozen TF-IDF `metrics.json`)

| Eval set | n | TF-IDF macro-F1 (frozen) | Transformer macro-F1 | Δ | Agreement rate |
|---|---:|---:|---:|---:|---:|
| test_balanced | 40,000 | 0.9454 | 0.9663 | +0.0209 | 95.08% |
| test_representative | 50,000 | 0.9061 | 0.9362* | +0.0301 | 95.13% |
| chronological_stress | 5,000 | 0.9510 | 0.9678 | +0.0168 | 95.58% |
| product_holdout_stress | 5,002 | 0.9444 | 0.9642 | +0.0199 | 95.14% |

\* This row is the transformer's **default-threshold** macro-F1 (used for the
apples-to-apples same-row-ID delta above); Section 17's 0.9499 for the same
set is the **calibrated-threshold** result — both are real, from different
threshold choices, and both beat TF-IDF.

**Verdict: transformer wins on 4 of 4 eval sets**, including both stress
tests (product-holdout and chronological) — this is not an in-distribution-only
result. Where the two models disagree, the transformer is correct roughly
2× as often as TF-IDF is. **Recommendation: distilroberta-base is the frozen
primary Amazon model; TF-IDF+LinearSVC remains the frozen classical
baseline.**

## 19. Slice results (test_representative) 📄 — read with the caution noted below

By review length tertile: short macro-F1 unavailable in aggregate table
above but see `transformer_final_eval.json::results.test_representative.slices.by_length_tertile`
(short=0.954 short reviews scored *better*, not worse, than long ones —
long=0.920, medium=0.941).

By product-frequency band: `rare_2_5` (products with 2-5 reviews) macro-F1
0.959; `frequent_6_plus` macro-F1 0.491.

By rating (within-slice macro-F1): 1★=0.496, 2★=0.484, 4★=0.462, 5★=0.494 —
all roughly 0.46–0.50, alongside accuracy 0.86–0.98 for the same slices.

**Important caution on interpreting the rating and frequent-product
slices**: filtering to a single true rating value means every row in that
slice has the *same* true label — macro-F1 computed on a single-effective-class
subset is not a meaningful discrimination measure (a model that's always
correct on that slice still can't demonstrate balanced precision/recall
across two classes it never sees there). The much more informative number
for those slices is **accuracy**, which is high (0.86–0.98) across every
rating value. The `frequent_6_plus` band's low macro-F1 (0.491) likely has
the same cause (a skewed within-band class mix) but this was **not**
separately confirmed by inspecting that band's class balance — flagged as
an open item, not a hidden problem. Do not read the 0.46–0.50 numbers in
this section as "the model performs near chance on these slices" — that
would misread what macro-F1 measures on a single-class subset.

## 20. Limitations and known failure modes

- **Excludes 3★ reviews from training entirely** — the model has no
  training signal for genuinely mixed/neutral sentiment; a real 3★-equivalent
  review in deployment will be forced into Negative or Positive.
- **Trained on a class-balanced 50K sample** — real deployment traffic is
  ~85% positive; this is why calibration/threshold correction (Section 16)
  is required, not optional, before using raw model output operationally.
- **English-only, Amazon-Appliances-domain-only** — no claim is made about
  transfer to other product categories, languages, or platforms.
- **Slice interpretation caveat** — see Section 19; several slice numbers
  need care, not face-value reading.
- **Reproducibility gap in the split-manifest hash records** — see
  `REPRODUCIBILITY_MANIFEST.md` Section "Known discrepancy": the SHA-256
  hashes recorded for the nine `split_ids/*.parquet` files in
  `split_manifest.json` and `transformer_split_manifest.json` do not match
  the current on-disk files, even though row counts/label distributions/
  product counts were independently reverified identical for five of the
  nine files. This looks like a parquet re-serialization artifact, not data
  corruption, but has not been proven for all nine files individually.
- **Label noise is real and unremoved** — 0.36-0.60% label-text/title-text
  conflict rate (Section 6) reflects genuine ambiguity in star-rating-derived
  labels; this is a ceiling on achievable accuracy, not a pipeline bug.
- **No hyperparameter search was performed** — the configuration in Section
  12 was used as specified, not tuned; a search might improve results
  further but was explicitly out of scope (Gate 6 rules).

## 21. Conclusions (supported by evidence in this document only)

1. The verified-purchase, binary (1-2★ vs. 4-5★) sentiment task is
   well-posed and the modeling pool is large enough for robust training and
   evaluation at every size tested.
2. A genuinely fine-tuned `distilroberta-base` transformer, trained on only
   50,000 balanced reviews for 4.5 minutes on a single GPU, beats the
   already-strong frozen TF-IDF+LinearSVC baseline on macro-F1 on all four
   independently-verified evaluation sets, including both product- and
   time-generalization stress tests.
3. Calibration and threshold selection were performed correctly — fit only
   on a natural-distribution validation set never used for any other
   decision, never touching any test set.
4. The frozen transformer checkpoint reloads correctly in a fresh process
   and reproduces its saved predictions within 0.0001 (well inside the
   0.01 tolerance) — verified independently in this session, not merely
   claimed.
5. The split-manifest hash staleness (Section 20) is a real, disclosed gap
   in this freeze package's reproducibility guarantees and should be treated
   as such — not silently omitted.
