# Project Audit & Notebook Plan (Stage 1)

Prepared 2026-08-17. Scope: `commerce-pilot-ai/` only (the ML/data-science
side of the repository), per the agreed scope. Nothing has been moved,
renamed, or deleted — this is the audit deliverable, matching Stage 1 of
the requested workflow.

---

## 1. Summary of the current project

`commerce-pilot-ai` contains **three independent ML components**, each
taken through the same disciplined process: raw data → documented
cleaning → leakage/eligibility audit → feature engineering → model
comparison → a **locked, one-time protected test evaluation** → an
honest model card. All three are exposed by one FastAPI service
(`src/ai_service/`) and one of them (fulfillment risk) is wired live
into the Medusa storefront.

| Component | What it predicts | Method | Headline result | Status |
|---|---|---|---|---|
| **Fulfillment risk** (Olist) | Whether a delivered order arrives after Olist's own recorded delivery estimate | CatBoost, 9 calendar/duration features only | Test Average Precision 0.0795 vs. 0.0633 baseline prevalence (a real but modest signal); ROC-AUC 0.563 | Live in production |
| **Product recommendation** (Instacart) | Which products a user will buy next | Hand-crafted recency/frequency heuristic + popularity backfill (not a trained model) | Protected-test Precision@10 0.288, Recall@10 0.340, NDCG@10 0.412 | Exposed via API, not wired into checkout |
| **Arabic text classification** (NLP) | Offensive language (MPOLD), sentiment (ASTD), star rating (LABR, Jumia) | Fine-tuned MARBERT/AraBERT transformers vs. classical TF-IDF baselines | MPOLD/ASTD: strong, statistically significant gains. Jumia (Egyptian e-commerce): protected-test macro-F1 0.374, only marginally beats the classical baseline (+0.0013); a fix that worked in validation did not hold on the final test | MPOLD/ASTD live; Jumia explicitly research-only, not deployed |

This already matches the top-level `README.md`'s own reported numbers
almost exactly — the project's public-facing claims are consistent with
what the underlying evidence files actually say. That is a genuinely
good sign for scientific integrity and something to preserve, not "fix."

## 2. What's already good (do not change)

- **Code quality is already high.** Small, focused modules (biggest file
  in `src/ai_service/` is 192 lines), clear docstrings, hash-verified
  model loading, dataclasses for structured results. The original brief
  assumed messy code needing simplification — that assumption doesn't
  hold here. No rewrite is planned.
- **Leakage discipline is unusually rigorous.** Every raw column is
  classified `ALLOWED`/`FORBIDDEN`/`REQUIRES VERIFICATION`; a dedicated
  audit found all 15 "expanded" Olist features could never be proven
  safe to use at prediction time, and the project's own conclusion was
  to **not use them in production** rather than quietly ship them.
- **Test sets are formally "frozen" and used exactly once**, with a
  written pre-commitment record, a hash-verified ledger, and code that
  refuses to run a second time (`FileExistsError` on re-access). This is
  a stronger safeguard against p-hacking/test-set leakage than most
  academic projects implement.
- **Weak or inconclusive results are reported as such**, not hidden:
  the Jumia classifier's own class-2 fix reverting on protected test,
  the LABR MARBERT-vs-AraBERT tie staying statistically inconclusive,
  and the Olist champion actually losing to Logistic Regression on the
  final test split (but correctly *not* switching champions after the
  fact, since that would itself be a leakage violation).

## 3. Problems found

These are the things a committee reviewer would flag, and what I
recommend doing about each. None require deleting anything.

1. **No notebooks exist.** `commerce-pilot-ai/notebooks/` is empty.
   There is no committee-facing walkthrough of any of the three
   components — only scripts and ~130 markdown/JSON files. *Fix: build
   the 9 notebooks (Section 5).*
2. **`commerce-pilot-ai/README.md` is stale and says so itself** — it's
   explicitly marked "Phase 1 foundation... not rewritten retroactively"
   and describes Olist as a **technical NO-GO**, which was true in
   early Phase 1 but is no longer the current state. A reader who opens
   this file first gets a wrong first impression. *Fix: see Section 4.*
3. **Reproducibility gap.** `data/raw/`, `data/processed/`,
   `artifacts/` (trained models), `reports/generated/`, and
   `reports/checkpoints/` are all git-ignored. A fresh `git clone` of
   this repository gets code and docs only — no data, no trained
   models, no results. The committee needs to know this up front rather
   than discover it by trying to run something that silently has
   nothing to load. *Fix: see Section 6 (needs your decision — some of
   these files are 500MB+ and genuinely shouldn't go into git).*
4. **One superseded/contradicted result exists in the raw files**,
   already caught and documented by the project itself:
   `reports/generated/olist/phase2b/paired_comparisons.json` originally
   over-claimed `SUPPORTED_INCREMENTAL_SIGNAL` for the "expanded"
   Olist features; a later correction found the labeling logic ignored
   the leakage-safety question and replaced it with
   `INCONCLUSIVE_INCREMENTAL_SIGNAL` in
   `correction_v3/corrected_paired_comparisons.json`. The original file
   is kept (with a `SUPERSEDED_NOTICE.md` next to it) rather than
   deleted — correct practice — but any notebook must cite the
   corrected file, not the original. *Fix: notebooks will cite
   `correction_v3/` only; flagged here so it's explicit.*
5. **One unresolved internal inconsistency, documented but not fixed:**
   the Olist production risk threshold (0.1293) was validated using a
   confusion matrix that turns out to be from the *validation* split,
   not the *test* split; on the actual test split that threshold flags
   100% of orders as high-risk. The project's own model card already
   flags this as "a genuine documentation inconsistency, flagged not
   resolved" rather than quietly picking whichever number looks better.
   *This is a real scientific finding to present honestly in the
   notebook — not something to smooth over.*
6. **70 files in `docs/` with no index**, and 52 dated folders in
   `reports/checkpoints/` (a development session log, not meant for the
   committee). A newcomer can't tell which of the 70 docs matter most.
   *Fix: add a `docs/README.md` index grouped by component (pure
   addition, no files moved).*

## 4. `commerce-pilot-ai/README.md`: proposed change (needs your approval)

This is the one **existing file** I'm proposing to change, so per your
instructions, here's exactly what I intend to do — nothing happens
until you approve:

- Move the current content, unchanged, to
  `commerce-pilot-ai/docs/archive/phase1_original_charter_readme.md`
  (preserved exactly, plus it's already fully recoverable via `git log`
  regardless).
- Replace `commerce-pilot-ai/README.md` with a short, current
  file that states the project's real status today and points to the
  new `notebooks/` folder and the top-level `README.md` for full detail
  — not a duplicate of either.

## 5. Proposed folder structure (additions only)

```
commerce-pilot-ai/
├── notebooks/                          <- currently empty
│   ├── README.md                        execution order + what each notebook covers
│   ├── 01_project_overview.ipynb
│   ├── 02_olist_data_understanding.ipynb
│   ├── 03_olist_cleaning_and_leakage_audit.ipynb
│   ├── 04_olist_modeling_and_evaluation.ipynb
│   ├── 05_instacart_recommender.ipynb
│   ├── 06_nlp_arabic_classification.ipynb
│   ├── 07_nlp_jumia_domain_shift.ipynb
│   ├── 08_production_integration.ipynb
│   └── 09_final_presentation.ipynb
├── docs/
│   ├── README.md                        <- NEW: index of the 70 existing docs, grouped by topic
│   └── archive/
│       └── phase1_original_charter_readme.md   <- moved from commerce-pilot-ai/README.md
└── README.md                            <- NEW: short, current, points to notebooks/
```

Nothing under `data/`, `src/`, `artifacts/`, `reports/`, `scripts/`,
`configs/`, or `tests/` is being moved or renamed. Those are already
organized sensibly (this is not a project that needs its source tree
rebuilt) and the notebooks will `import` from `src/` and read from
`reports/`/`artifacts/` rather than duplicating logic.

## 6. Reproducibility gap — needs your decision

Right now, cloning this repo fresh gives no data, no trained models, no
results. Three options, not mutually exclusive:

- **(a) Do nothing** — document the gap clearly in the README and
  notebooks ("results shown were computed locally; re-running requires
  the raw datasets, which are large third-party downloads"). Zero risk.
- **(b) Track the small result/evidence files in git** — the metrics
  JSON files (`reports/generated/**`, currently a few hundred KB
  total), `cleaning_summary.json` files, and the production
  `catboost.cbm` model (12KB) are all small. Committing these means the
  notebooks' numbers are verifiable from the repo alone, without
  re-running anything. Requires editing `.gitignore`.
- **(c) Track large files too** — the NLP transformer weights are
  500–622MB *each* (4 of them, ~2.3GB total) and the Instacart raw zip
  is 198MB; committing these to a normal git repo is a bad idea (GitHub
  hard-blocks files over 100MB) and would need Git LFS, a bigger change
  I'd want explicit sign-off on before touching.

My recommendation is **(a) + (b)**: cheap, safe, and it means the
notebooks' claims are checkable from the repository itself. I'll wait
for your decision before touching `.gitignore`.

## 7. Notebooks recommended for removal

**None.** No notebooks exist in this repository — there is nothing to
remove or replace. (Confirmed by searching the entire repo for `.ipynb`
files and for any mention of "Antigravity"; both searches returned zero
results.)

## 8. Files recommended for deletion

**None**, at this stage. Everything found is either current source
material or a documented historical record with its own supersession
notice. The only *change* proposed to an existing file is the
`commerce-pilot-ai/README.md` update in Section 4, which preserves the
old content rather than deleting it.

## 9. Notebook-by-notebook content plan

Each notebook will cite real source files (scripts, configs, JSON
metrics) rather than re-deriving or re-typing numbers. Where a notebook
needs to *run* code, it will import from `src/` — no logic duplicated
into notebook cells.

1. **`01_project_overview.ipynb`** — the business problem (ops teams
   see order data and fulfillment problems as separate systems), what
   HASEBHA/CommercePilot does about it, the three ML components at a
   glance, how this notebook set is organized. No code, mostly markdown
   + one architecture diagram.
2. **`02_olist_data_understanding.ipynb`** — where the Olist dataset
   comes from, its 9 tables and what each represents, load via
   `src/data_pipeline` output, show real row counts/date range/status
   breakdown, license caveat (CC BY-NC-SA 4.0, non-commercial).
3. **`03_olist_cleaning_and_leakage_audit.ipynb`** — the cleaning
   script's actual behavior (typed Parquet conversion, zero rows
   dropped), the eligibility waterfall (99,441 → 95,082 orders, with
   the exact table from the research above), and the leakage audit
   story: why 15 "expanded" features were investigated and why all 15
   were ultimately rejected as unverifiable — presented as a case study
   in good practice, not a failure.
4. **`04_olist_modeling_and_evaluation.ipynb`** — the four models
   compared (dummy/logistic regression/CatBoost/LightGBM), the
   validation-then-locked-test protocol, the real numbers (CatBoost
   Test AP 0.0795, ROC-AUC 0.563), the honest discussion of CatBoost
   *not* winning on the test set but remaining champion by design (why
   that's correct, not a bug), and the documented threshold
   inconsistency.
5. **`05_instacart_recommender.ipynb`** — the recommendation problem,
   why it's a heuristic ranking system rather than a trained model (and
   why that's a legitimate, explained choice — a tested collaborative-
   filtering alternative underperformed it), the frozen/protected-test
   methodology, and the real precision/recall/NDCG numbers at K=5/10/20.
6. **`06_nlp_arabic_classification.ipynb`** — the three "core" NLP
   tasks (MPOLD offensive-language, ASTD sentiment, LABR rating), why
   they're kept as separate label schemes rather than merged, the
   classical-vs-transformer comparison, and the real statistically-
   significant results for MPOLD/ASTD plus the still-inconclusive
   LABR tie.
7. **`07_nlp_jumia_domain_shift.ipynb`** — the most nuanced story: can
   a model trained on other Arabic text work on real Egyptian
   e-commerce reviews? Zero-shot transfer, classical baseline,
   fine-tuned adaptation, the class-2 (2-star) minority-class collapse
   and its partial, non-generalizing fix. Presented honestly as
   "PARTIAL — not production-ready," matching the project's own
   verdict.
8. **`08_production_integration.ipynb`** — how the winning Olist model
   actually gets called from a live Medusa order (hash-verified model
   loading, the Decision Engine's rule-based layer, what's live vs.
   API-only vs. research-only), grounded in the real `src/ai_service`
   code.
9. **`09_final_presentation.ipynb`** — the committee-facing summary:
   problem, data, methods, results comparison table, best result and
   why, limitations, future work, conclusion. Short, visual, links out
   to notebooks 2–8 and the source code rather than repeating detail.

## 10. What I need from you before Stage 2 starts

1. Approve (or adjust) the `commerce-pilot-ai/README.md` change in
   Section 4.
2. Pick an option in Section 6 (reproducibility gap) — I recommend (a)
   + (b).
3. Confirm the 9-notebook plan in Section 9, or tell me what to change.

Once confirmed, Stage 2 is: build the notebooks, make the two
documentation additions, run everything top-to-bottom, and report back
with what was tested and the results — exactly as your original
workflow describes.
