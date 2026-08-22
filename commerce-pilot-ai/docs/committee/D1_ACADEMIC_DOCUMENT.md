# CommercePilot: An Applied Machine Learning Layer for a Single-Vendor E-Commerce Platform — Evidence, Transfer Analysis, and Production Reality

Prepared: 2026-08-22. Language: English (technical/scientific main text — see editorial note
below). All figures traced to artifacts in `docs/committee/D4_ARTIFACT_INDEX.md`.

**Editorial note on language**: the source mission suggested Arabic main text with English
technical terms as a plausible default for an Egyptian committee. Given this document's role
as the primary evidence-tracing record (every number must map exactly to a source artifact
written in English, using English field names and English metric conventions), the editorial
decision made here is to keep the full technical document in English to eliminate any risk of
translation drift in a number or claim, and to rely on the presenter's own spoken Arabic
during the defense (supported by D5's plain-language answers and D6's demo script) for
audience accessibility. This is a deliberate choice, stated explicitly rather than left
implicit.

---

## Abstract

CommercePilot is an applied machine learning layer built for HASEBHA, a real, currently
single-vendor Medusa-based e-commerce platform. This work reports results across six
independent ML tracks (Arabic sentiment, Amazon Appliances sentiment, Instacart
recommendation, and three Olist Brazilian e-commerce fulfillment-risk tracks), plus a
reproduction study of a published graph-learning method (DataCo/EAGLE). The central
scientific contribution is a rigorous, quantitatively-traced investigation of whether a
strong research result (mean temporal AUC 0.7702 for a seller-handoff SLA-breach classifier)
transfers to a structurally different production environment. It does not: forcing the
model's unavailable features to their honest production-unavailable state collapses
performance to near-random (AUC 0.5188), and a fresh, independently-reproduced forensic
ablation study attributes 94% of this collapse to a single feature block (per-seller
historical behavior) that has no analog in a single-vendor store. A feature-restricted
retrain recovers only a weak signal (AUC 0.5551), and this ceiling is shown, by controlled
experiment across six model families and four new feature groups, not to be a modeling
limitation. The paper concludes with an honest classification of project status
(engineering: production-ready; ML: blocked on first-party data and a business decision,
not on technique) and a quantified data-collection plan for closing the gap.

---

## 1. Introduction

E-commerce platforms increasingly rely on machine learning for fulfillment-risk prediction,
recommendation, and customer-facing text understanding. A common and rarely examined
assumption in applied ML work is that a model validated on a public benchmark dataset will
transfer, with acceptable degradation, to a real production environment sharing the same
general task. This project set out to build several such capabilities for HASEBHA and, for
the fulfillment-risk task specifically, to test that assumption directly rather than assume
it. The result is both a set of validated applied ML capabilities and a negative,
quantitatively explained transfer finding that is, in the authors' assessment, the project's
most defensible scientific contribution.

## 2. Related Work

This document does not survey the broader ML literature exhaustively, but grounds its two
externally-sourced tracks (Arabic sentiment, DataCo/EAGLE reproduction) in their primary
sources rather than citing them only by internal filename.

**Arabic sentiment benchmarks.** LABR (Aly & Atiya, 2014, arXiv:1411.6718, "LABR: A Large
Scale Arabic Sentiment Analysis Benchmark," >63,000 Arabic book reviews) is one of the three
benchmark tasks used in this project's Arabic track and is evaluated with both frozen
finalists (no champion selected — bootstrap CI includes zero). The champion model family for
all three Arabic tasks, MARBERT (Abdul-Mageed et al., UBC-NLP), is an Arabic-specific BERT
pretrained on dialectal and Modern Standard Arabic tweets; the challenger family, AraBERT
(Antoun et al., AUB-MIND), is a separately pretrained Arabic BERT variant. Both are used here
strictly as fine-tuning backbones via their public Hugging Face checkpoints (revision hashes
recorded in `configs/nlp_champion_registry.yaml`), not re-pretrained.

**DataCo / EAGLE.** The DataCo Smart Supply Chain dataset (Constante, Silva & Pereira,
Mendeley Data, DOI 10.17632/8gx2fvg2k6.5, v5) is the canonical source for the reproduction
study in Section 10. EAGLE is the target published method for that reproduction (an
edge-aware graph learning architecture for delivery-delay prediction on this dataset); its
node/edge/window construction and reported baseline figures are the reproduction targets,
extracted and independently re-verified from the paper's own text before implementation
(see `reports/generated/dataco/EAGLE_PAPER_PIPELINE_SUMMARY.json`).

**Positioning.** This project is not proposing a new model architecture in any track,
including Arabic and DataCo — it applies established, published methods and benchmarks
(LABR, MARBERT, AraBERT, EAGLE) and, for the fulfillment-risk tracks, contributes a
methodology (the production-availability simulation and its independent forensic
confirmation) for quantifying research-to-production transfer, rather than a new model.

## 3. Methodology — The Unified Protocol

Three methodological commitments were applied uniformly across every track that involved a
temporal/sequential prediction problem:

1. **Temporal prequential validation.** Rolling-origin (walk-forward) evaluation over
   multiple independent historical periods, never a single random train/test split, for
   every fulfillment-risk model (Olist V1/V2/V3, the production-parity model).
2. **Protected, one-shot evaluation.** Where a genuinely held-out test partition existed
   (Instacart's protected test, the Olist V1 frozen test set), it was accessed at most once
   for final reporting, with no post-access tuning — verified directly (`protected_test_
   access_count=1`, `post_test_tuning_performed=false` for Instacart).
3. **Multi-seed confirmation where the decision mattered.** The Arabic champion selection
   used a 3-seed (101/202/303) confirmation; the DataCo LSTM reproduction used 4
   predetermined seeds (13/42/123/2026), chosen before any result was observed.

Leakage was audited explicitly wherever a causal/point-in-time feature contract was claimed:
the Olist V3 seller-SLA feature pipeline passed 0/4 targeted leakage checks with failures
(including a 30-comparison spot-check of the expanding breach-rate computation against
manual recomputation), and the Olist V2 pipeline passed 0/30. A stress/diagnostic block
(the "exposed" Olist V2 block) was, after an early-project terminology correction, never
again described as blind or protected, and was used only for post-selection diagnostics,
never for model selection.

## 4. Results Per Track

### 4.1 Arabic Sentiment (MPOLD/ASTD/LABR)
Champion: MARBERTv2, plain cross-entropy, group-safe split. Single-seed headline Macro-F1
0.8130 (seed 42); 3-seed confirmation mean 0.7906. Five post-freeze improvement candidates
were tested and rejected: label smoothing (Δ −0.0186), class-weighted cross-entropy
(Δ +0.0018, sub-threshold against a predeclared +0.3pp bar), R-Drop α=5 (Δ −0.0089, and
higher cost), and an alternative architecture SARF (statistically indistinguishable at
3-seed mean, rejected on ~1.9x runtime/VRAM cost, not on accuracy). Two further candidates
(SWA/EMA, supervised contrastive learning) were deliberately not attempted after three
consecutive negative results — a documented stopping decision, not an oversight.

### 4.2 Amazon Appliances Sentiment
Champion: TF-IDF (word+char n-grams) + LinearSVC, selected via a plateau rule over four
training-set sizes (25k/50k/100k/200k, validation-only). Macro-F1 0.9454, 95% bootstrap CI
[0.9432, 0.9476] (n=40,000, computed this cycle from the raw saved prediction file, point
estimate reproduced exactly). A DistilRoBERTa transformer variant was also evaluated
(reported, comparable range) but was never promoted to champion status
(`configs/nlp_champion_registry.yaml`: "Amazon transformer confirmation deferred, not
cancelled"). Neither model is currently wired to serve live inference requests — the
classical artifact exists on disk, hash-verified, but `NlpInferenceService.keys_for_task`
returns an empty list for this task unconditionally, an integration gap distinct from an
absent artifact.

### 4.3 Instacart Recommendation
Candidate: hybrid recommender with popularity backfill. Protected test: 26,314 users,
accessed exactly once. Precision@5 0.3674, Recall@5 0.2378, NDCG@5 0.4298, HitRate@5 0.7883,
MAP@5 0.3314. 92.17% of recommendations drew from the top-20%-popularity band — a disclosed
characteristic of the frozen candidate, not concealed. Egyptian-catalog transfer is
explicitly marked `NOT_YET_VALIDATED` in the service's own response schema.

### 4.4 Olist V1 (Production)
CatBoost, trained on 9 features derived from exactly 2 raw order timestamps (purchase,
payment approval). Frozen test-set ROC-AUC 0.5634, 95% bootstrap CI [0.5483, 0.5774]
(n=24,744, computed this cycle from the raw saved prediction file, point estimate reproduced
exactly). This is the only model in the project with confirmed real HASEBHA production
execution history: a direct query of the live database found 5 real orders, all carrying
this model's prediction metadata, including one correctly-logged `AI_UNAVAILABLE` failure
record proving the fail-soft integration path genuinely executed, not merely existed in
code.

### 4.5 Olist V2 (Research, Stress-Benchmark)
A static (non-adaptive) temporal model showed dev-fold AUC in the 0.72–0.76 range but
collapsed to 0.5117 on a later, out-of-sample block — not attributable to leakage (0/30
checks failed). An earlier internal hypothesis attributing this to a May 2018 Brazilian
trucker strike was directly checked against the data and refuted: the true late-rate peak
(21.4%, n=7,003) occurred in March 2018, two months before the strike window, which itself
showed a near-average 8.2% late rate. Three temporal-adaptation strategies were compared
using only historical periods for selection; the best (recency-weighting) achieved
historical mean AUC 0.7126 and, when subsequently checked (diagnostically only) against the
originally-collapsed block, scored 0.5924 — a real but partial recovery. This track's role is
explicitly `STRESS_BENCHMARK`, not a production candidate.

### 4.6 Olist V3 Seller-SLA (Research)
Target: `SELLER_HANDOFF_SLA_BREACH` (seller hands the order to the carrier after its
item-level deadline). LightGBM, 23 features (11 order/product/payment, 3 temporal, 1
geography, 8 causal seller-history), on a 96,380-row single-seller cohort (98.70% of
item-joinable orders). Leakage-tested, 0/4 checks failed. Rolling-origin evaluation over 5
historical periods: mean AUC 0.7702, worst period 0.6762, standard deviation across periods
0.0483. A post-selection diagnostic on an exposed later block scored 0.7364 — notably, this
model does not exhibit the severe collapse pattern seen in Olist V2's static model.

### 4.7 Olist V3 Customer-Late T0 (Stacked)
A causal, out-of-time predicted seller-SLA-risk feature was stacked into a customer-late-T0
model on the same cohort. Mean AUC moved from 0.5873 (order+seller-history features) to
0.5910 (+0.0036), an inconsistent-sign, trivial delta across the 4 evaluated periods.
Rejected for flagship use.

### 4.8 Olist V3 Customer-Late T1 (Dynamic)
Using only information legitimately known after carrier handoff (handling duration,
remaining promise slack, handoff timing, lane/geography, seller history), mean AUC reached
0.6890 (worst period 0.6541), meaningfully stronger than either T0 variant and stable across
5 periods. Classified moderate/early-warning/triage-level — genuinely useful for human review
prioritization, not sufficient for automated action.

### 4.9 Calibration
Isotonic regression was selected, over Platt scaling and raw scores, for both the Olist V3
research model and the production-parity model, using a predeclared rule (lowest Brier score
among methods with AUC degradation ≤ 0.005 versus raw, evaluated on a held-out half of
temporal out-of-fold dev predictions, never the exposed stress block). Research model: Brier
0.07171 → 0.07087, ECE 0.001134 → 0.000258, AUC 0.78855 → 0.78820. Production-parity model:
Brier 0.1255 → 0.0802, AUC 0.5655 → 0.5699 (calibration slightly *improved* ranking here,
within noise).

### 4.10 HASEBHA Production-Parity Model and the Transfer Analysis
See Section 5 (dedicated, given its centrality).

## 5. The HASEBHA Feature-Transfer Gap — The Project's Central Finding

Direct inspection of the HASEBHA Medusa codebase confirmed no seller/vendor module exists
anywhere — the platform is architecturally single-vendor. Of the Olist V3 model's 23
features, 12 (the 8-feature seller-history block plus `days_to_shipping_deadline` and
`n_installments`) have no legitimate online analog in this architecture.

**Production-availability simulation.** The same frozen 23-feature model, evaluated on the
same 5 historical periods, with those 12 features forced to the exact sentinel value the live
service would honestly produce (never a fabricated best-guess), collapsed from mean AUC
0.7702 to **0.5188** (worst period **0.4949**, below random). This is a direct measurement,
not an extrapolation.

**Production-parity retrain.** A fresh model, trained (not fine-tuned — the prediction
contract itself changed, an explicitly authorized reason for retraining) on only the 13
genuinely-available features, reached mean AUC **0.5551** (worst period 0.5289). An
ops-feature variant (adding store-wide, non-per-seller rolling statistics) was tested and
made results *worse* (−0.0093), consistent with the semantic-honesty finding that store-wide
history is a materially different, noisier statistical object than genuine per-seller
history.

**Independent forensic confirmation (this cycle).** A newly-written, independently-executed
forensic script (`scripts/forensics/production_model_forensics.py`) reproduced both frozen
numbers under an identical protocol (full-23: 0.7686 vs. 0.7702; production-parity: 0.5540
vs. 0.5551 — small, expected run-to-run drift) and performed a controlled feature-block
ablation, attributing **94% of the 0.7686→0.5540 collapse (−0.202 of −0.215 total AUC) to the
seller-history block alone**. Four new legitimate feature groups were engineered and tested
under the same protocol: geography (haversine ship-to/ship-from distance, Δ −0.001, no
signal), strict point-in-time customer history (Δ +0.002, no signal), product category
(Δ +0.014, the best single gain, but unstable in the most recent period), and basket price
structure (Δ +0.000, no signal). Six model families were compared on the best legitimate
feature set: Logistic Regression 0.5487, Random Forest 0.5521, HistGradientBoosting 0.5537,
XGBoost 0.5538, CatBoost 0.5547, LightGBM (incumbent) 0.5647 — architecture is not the
limiting factor.

**This forensic study's own numbers were independently spot-checked by the author of this
document** before inclusion: four univariate feature AUCs were recomputed directly from the
raw canonical parquet and matched the reported JSON values to full floating-point precision;
the script's use of real raw Olist CSVs (orders, customers, sellers, geolocation) and a
genuine haversine distance function was confirmed by direct code reading; file timestamps
were checked for internal consistency (a 50-second gap between script and results,
consistent with real model training on ~96,000 rows, not an instantaneous or suspiciously
absent computation).

## 6. Discussion

The transfer-gap finding reframes what "0.7702" and "0.5551" mean. 0.7702 measures whether
seller-level historical reliability is a learnable signal in a marketplace with genuine
seller heterogeneity — it is, robustly. 0.5551 measures whether the residual information
available at order-placement time in a single-vendor store, absent that heterogeneity,
carries useful signal for the same target — it does not, materially. These are not
comparable numbers describing the same underlying capability; conflating them would
misrepresent both.

## 7. Threats to Validity

- **Proxy dataset.** All fulfillment-risk numbers (Sections 4.6–4.10) are computed on Olist
  (Brazilian, 2016–2018) data, used as a research/methodology testbed, not as a measurement
  of HASEBHA behavior. No claim in this document asserts otherwise.
- **Extremely limited first-party data.** The live HASEBHA database contains 5 real orders
  and 0 fulfillment outcome records as of this writing — insufficient for any conclusion
  about real-world HASEBHA order behavior, positively or negatively.
- **Target validity.** The current fulfillment-risk target itself (`SELLER_HANDOFF_SLA_
  BREACH`, and its production-parity analog) is not yet a legitimate HASEBHA business target,
  because no real shipping-time promise is configured anywhere in the platform. This is a
  business-decision gap, not a statistical one, and is stated as such rather than worked
  around.
- **Forensic study reproducibility gap.** The forensic reproduction numbers (0.7686, 0.5540)
  differ slightly from the originally-frozen numbers (0.7702, 0.5551) due to expected
  run-to-run stochastic variation (same protocol, independently re-executed) — reported as
  such, not silently reconciled to a single "true" number.
- **Bootstrap CI coverage gap.** Of six headline metrics, two (Arabic, Instacart) do not yet
  have a bootstrap confidence interval computed from raw per-example predictions within this
  reporting cycle; this is disclosed explicitly (Section 9/D3) rather than approximated.

## 8. Limitations

No model in this project has ever been trained on, or validated against, real HASEBHA
outcome data. No production automated action (of any kind) is authorized or implemented
anywhere in this system. The shadow-mode production-parity pipeline, while fully implemented
and tested, has never scored a real customer order.

## 8.5 Enrichment Test: Olist Marketing Funnel Data

As a further, predeclared test of whether any additional legitimate public-data signal
remains untapped for the seller-SLA research model, the Olist Marketing Funnel dataset
(acquisition-channel, lead-type, and seller-onboarding-timing data, Kaggle
`olistbr/marketing-funnel-olist`, CC BY-NC-SA 4.0) was acquired, leakage-audited (0 of 4,384
joined rows violated point-in-time correctness), and tested under the identical rolling-
origin protocol used throughout this document. Result: mean AUC moved from 0.7692 (baseline,
reproduced) to 0.7658 (+funnel), a small **negative** delta of −0.0034, against a predeclared
hypothesis of +0.01 to +0.03. Every individual funnel feature's univariate AUC fell between
0.4977 and 0.4979 (indistinguishable from noise). This null result is well-explained by a
coverage ceiling identified before the experiment ran: only 12.8% of the canonical cohort's
sellers, covering 4.5% of its rows, appear in the funnel dataset at all. Reported here in
full, exactly as measured (`reports/generated/olist_funnel/`), as a further demonstration
that this project's negative results are preserved and reported, not selectively omitted.

## 9. Conclusion and Future Work

CommercePilot demonstrates: (a) several independently strong, validated ML capabilities on
public/first-party-adjacent data; (b) a rigorous, now doubly-confirmed (original simulation
+ independent forensic ablation) negative finding that a strong fulfillment-risk research
result does not transfer to HASEBHA's real architecture, with the cause quantitatively
attributed rather than assumed; and (c) production-ready supporting infrastructure (hash-
verified model loading, fail-soft integration, calibration, raw-feature persistence for
future first-party training) built specifically to make the eventual, currently-blocked
next step possible. Future work requires two things outside this project's engineering
authority: a real HASEBHA shipping-SLA business decision, and roughly 1,650 (minimum) to
4,500 (recommended) real orders with linked fulfillment outcomes. No further modeling on
Olist data is expected to change this picture — this has now been tested directly, twice.

## 10. DataCo/EAGLE Reproduction Study (Independent Track)

Dataset: DataCo SMART SUPPLY CHAIN, Mendeley DOI 10.17632/8gx2fvg2k6.5 v5, 180,519 rows, 53
columns, SHA256-verified against the publisher's own recorded hash. Graph structure (46
nodes = 23 regions × 2 countries) reproduced exactly. LSTM baseline, 4 predetermined seeds:
mean AUC 0.6454 (std 0.011) versus the published ~0.9679. Root cause: the paper's own quoted
target formula, implemented exactly as specified, produces ~50% positive prevalence at this
dataset's per-node-window sample size (median 174 orders), which is mathematically
incompatible — via a Central Limit Theorem argument, independently verified — with the
paper's reported 2.84–6.15% rate. One alternative threshold formulation was tested and also
rejected. EAGLE (the more expensive graph-attention architecture) was correctly never
attempted, following a predeclared reproduction gate, rather than run on a target already
known to be broken.

## References

Primary references are internal repository artifacts (see `D4_ARTIFACT_INDEX.md` for the
complete, path-traced list) plus the two external sources directly engaged with: the
Mendeley-hosted DataCo SMART SUPPLY CHAIN dataset (DOI 10.17632/8gx2fvg2k6.5) and the EAGLE
paper's publicly stated methodology (as extracted and independently verified against the
paper's own text during the DataCo reproduction study).

## Appendices

See `D3_RESULTS_TABLE_WITH_CI.md`/`.json` (unified results with confidence intervals),
`D4_ARTIFACT_INDEX.md` (complete artifact index), `D5_COMMITTEE_QA_BRIEF.md` (anticipated
questions and answers), `PRODUCTION_MODEL_FEATURE_INVENTORY.md` (full feature-by-feature
availability classification), `PRODUCTION_MODEL_EXPERIMENT_MATRIX.json` (every forensic
experiment, structured).
