# Code Completeness Gate

Prepared 2026-08-17. Every row below was verified by directly opening the
file, running the relevant script/test, or inspecting the actual data/
artifact on disk — not inferred from filenames or documentation claims.
Status legend: **complete** (exists, runs, evidence on disk) /
**partial** (some pieces exist, others don't) / **missing**.

## Olist fulfillment-risk (the only currently deployed model)

| Component | File | Status | Evidence |
|---|---|---|---|
| Dataset loader | `src/data_pipeline/clean_olist.py` | complete | Verified: 9 processed parquet files exist with correct row counts |
| Schema/eligibility validation | `src/modeling/olist/strict_feature_builder.py` | complete | Read in full; re-ran its exact SQL myself against raw data, got identical counts |
| Target construction | same file, line 32 | complete | `late_delivery = delivered > estimated`, verified by independent recomputation |
| Leakage checks | `docs/olist_asof_feature_contract.md` + code-enforced column allowlist | complete | Code raises an error on any non-approved column; 15 "expanded" features formally rejected as unprovable |
| Train/val/test split | `configs/olist_phase2a_benchmark.yaml`, `src/modeling/olist/temporal_validation.py` | complete | Time-based, independently reproduced (43,516 / 26,822 / 24,744 rows) |
| Baseline training | `phase2a_benchmark.py::make_model("dummy",...)` | complete | Saved model + predictions on disk |
| Main-model training | same file, CatBoost/LightGBM/LogReg | complete | Saved models + predictions on disk |
| Evaluation | `src/modeling/olist/evaluation.py` | complete | Independently recomputed AP/ROC-AUC from stored predictions, exact match |
| Confusion matrix / per-class metrics | not pre-saved as a report artifact | **partial** | I computed this myself from stored predictions (see verification report above); no saved confusion-matrix JSON exists yet — will add in the audit notebook |
| Model persistence | `artifacts/.../models/catboost.cbm` | complete | Hash-verified against `config.py` |
| Saved predictions | `artifacts/.../predictions/test_*.parquet` | complete | Used directly for independent metric recomputation |
| Inference function | `src/ai_service/services/fulfillment_risk.py` | complete | Read in full; hash-verifies model at load |
| Configuration | `configs/olist_phase2a_benchmark.yaml`, `src/ai_service/config.py` | complete | Read in full |
| Reproducibility/seeds | `random_seed: 42`, determinism block, `reproducibility_report.json` | complete | Verified PASS, bit-exact refit hashes |
| Tests | `tests/test_olist_phase2a.py` + 4 others | complete | **Ran them: 66 passed, 0 failed** |
| Notebook integration | — | **missing** | `notebooks/` is empty |

**Required action:** build `01_olist_eda_and_model_audit.ipynb` (notebook only — the underlying code is sound). Confusion matrix should be computed in-notebook from the existing saved predictions rather than duplicating training logic.

## Instacart recommender

| Component | Status | Evidence |
|---|---|---|
| Loader, split, ranking logic, evaluation, "frozen" artifact, inference wiring | complete | Verified in the earlier research pass: `scripts/instacart_recsys_lib.py`, hash-verified artifact, live in `src/ai_service/services/recommendation_engine.py`, protected-test results on disk |
| Notebook integration | missing | Not in this session's priority scope per your instructions (Amazon/Arabic/Olist only) |

Not touched further in this session — already functionally complete, and out of your stated 5-hour scope.

## Arabic NLP core tasks (ASTD=B2, MPOLD=E, LABR=C)

| Component | File(s) | Status | Evidence |
|---|---|---|---|
| Dataset loaders | raw files present under `data/quarantine/nlp/{astd,labr,mpold}/` | complete | Directly read and parsed all three myself this session (real columns confirmed) |
| Preprocessing/normalization | `src/nlp/text_normalization.py` | complete | Exists, referenced by config `nlp-text-normalization-contract-v2` |
| Split | `src/nlp/splitting.py`, `split_preparation.py` | complete | 70/15/15 group-safe split, documented and code-enforced |
| Classical baselines | `reports/generated/nlp/challengers/classical_challengers_{B2,C,E}.json` | complete | Results on disk |
| Transformer training | `reports/generated/nlp/transformer_confirmation/confirm_*.json` (9 files, 3 seeds × 3 tasks) | complete | Results on disk |
| Model persistence | `artifacts/experiments/nlp/inference_exports/{B2_MARBERT, C_AraBERT, C_MARBERT, E_MARBERT}/model.safetensors` | complete | Real files, 500-622MB each, confirmed present |
| Evaluation / confusion matrix / per-class | in the same JSON files | complete | Macro-F1 and per-class F1 present (verified in earlier deep research pass) |
| Inference function | `src/ai_service/services/nlp_inference.py` | complete | Hash-verifies + lazy-loads weights |
| Notebook integration | — | missing | `notebooks/` empty |

**Required action:** build `02_arabic_nlp_eda_and_analysis.ipynb` from real data + these existing results. No retraining needed for ASTD/MPOLD baselines-vs-transformer comparison — the results already exist and are strong (statistically significant gains, CIs excluding zero). LABR stays "inconclusive" (both finalists, no forced champion) — report honestly, not resolved by picking one arbitrarily.

## Jumia (Arabic e-commerce domain-shift study)

| Component | Status | Evidence |
|---|---|---|
| Loader, target, split, classical baseline, direct transfer, transformer adaptation, class-2 remediation, protected test, evaluation | complete | Fully verified in earlier research pass — real 3,809-row dataset, real protected-test macro-F1 0.3741 |
| Production wiring | **intentionally not wired** | Champion status is `PARTIAL`, correctly excluded from the live service by the project's own decision |
| Notebook integration | missing | Covered as a section of `02_arabic_nlp_eda_and_analysis.ipynb` |

## Amazon Appliances NLP — the actual gap

| Component | File | Status | Evidence |
|---|---|---|---|
| Dataset loader | `src/data_pipeline/clean_amazon_appliances.py` | **partial — code/data mismatch found** | The script (current version) renames `rating→overall, title→review_title, text→review_text`. The actual file on disk, `data/processed/amazon_reviews_appliances/reviews_text_ready.parquet` (I loaded it directly), still has the **original** names `rating`, `title`, `text` plus `has_usable_text`, `verified_purchase`, `parent_asin`, `user_id`, `timestamp`, `helpful_vote`, `images`, `asin`. Either the file predates a later script edit, or the rename isn't taking effect. **This needs to be resolved by using the real on-disk column names (not the script's claimed names) for all new Amazon code** — flagging, not silently fixing the older script. |
| Schema adapter | `src/nlp/amazon_adapter.py` | complete but unused by the actual file | `adapt_amazon_record()` expects physical `rating/title/text` — that part matches the real file; downstream code should call this rather than assume renamed columns |
| Data-quality checks | none Amazon-specific beyond the cleaning script's row-count/empty-text counts | partial | `cleaning_summary.json` exists with basic counts; no duplicate/near-duplicate audit exists yet |
| Target construction | **does not exist** | missing | No rating→sentiment mapping has ever been implemented in code (only "documented as a candidate, not adopted" per `docs/nlp_rating_mapping_and_reproducibility_policy.md`) |
| Leakage checks | none | missing | N/A — no model exists yet to check |
| Train/val/test split | none | missing | No split of any kind exists for Amazon |
| Baseline training | `reports/generated/nlp/batch1_baseline_A.json` | **partial, and for the wrong task** | Only majority-class/stratified-random baselines for a **5-class star-rating task** (macro-F1 0.161 / 0.199) — no baseline exists for the binary sentiment task you've specified |
| Main-model training | none | **missing** | Confirmed directly in `src/ai_service/routers/nlp.py`: task A returns a hardcoded `ARTIFACT_NOT_MATERIALIZED` response — there has never been a trained, saved Amazon classifier |
| Evaluation | none | missing | Nothing to evaluate yet |
| Confusion matrix/per-class | none | missing | — |
| Model persistence | none | missing | `artifacts/experiments/` has no `amazon` subdirectory at all |
| Saved predictions | none | missing | `reports/generated/` has no `amazon` subdirectory at all |
| Inference function | none | missing | — |
| Configuration | none Amazon-model-specific | missing | — |
| Reproducibility/seeds | N/A | missing | — |
| Tests | none exercise real Amazon training | missing | `grep` for "amazon" across `tests/` hits only cleaning/registry/provenance tests — none test a trained classifier, because none exists |
| Notebook integration | none | missing | — |

**This confirms your prioritization is correct, not arbitrary**: Amazon is the only component in this repository with a real dataset (2,128,605 verified rows) and literally zero modeling code, zero saved model, zero evaluation. Every other component (Olist, Instacart, ASTD/MPOLD/LABR, Jumia) has at least a complete, evaluated pipeline even where the science says "weak" or "inconclusive." Amazon is the one place where "weak" isn't even measured yet.

## Scope decision (per your instructions, not asked as a question — flagging as executed)

Given the above, and your explicit priority order, I'm proceeding now to build, in order:
1. Full Amazon pipeline (source modules → training → evaluation → 2 notebooks) — the largest real gap.
2. Arabic NLP EDA notebook + priority baselines confirmation (ASTD, MPOLD; LABR/Jumia included since their results already exist).
3. Olist audit notebook (code is complete; only the notebook is missing) — I'm doing this one directly myself since I already hold the deepest independently-verified numbers for it in this session.
4. Documentation suite, once real results from 1–3 exist to report honestly.

I'm delegating the large, well-specified implementation work (Amazon pipeline; Arabic notebook + baselines) to parallel background workers so they can run simultaneously rather than serially — each is being given the exact facts I've already verified in this session so nothing gets re-guessed or invented. I'll report back with real results, file paths, and test outcomes as each piece completes rather than going silent.
