# Phase 2C NLP Batch 1 Training Authorization — V5 Remediation Summary

```text
schema_version: phase2c-nlp-remediation-v5-summary
checkpoint: reports/checkpoints/phase2c_nlp_training_authorization_remediation_v5/
supersedes_review: reports/checkpoints/phase2c_nlp_end_of_day_2026-08-09/ (REJECT_NLP_BATCH1_TRAINING_AUTHORIZATION_V4)
next_gate: PHASE2C_NLP_TRAINING_AUTHORIZATION_INDEPENDENT_REVIEW_V5
nlp_batch1_training_authorized: false
```

## 1. Independent verification performed before any change

Before writing any code, the saved end-of-day state was independently re-verified, read-only:

- `sha256sum` of `CURRENT_STATE.md`, `CURRENT_STATE.json`, `NEXT_AGENT_HANDOFF.md` in
  `reports/checkpoints/phase2c_nlp_end_of_day_2026-08-09/` matched
  `FILE_MANIFEST.sha256` exactly (all three hashes byte-identical).
- Full non-training test suite re-run: **179 passed, 0 failed, 0 warnings, 0 skipped,
  0 xfailed** (matches the recorded end-of-day figure exactly; the local pytest-cache
  permission errors seen with the default temp directory were an environment artifact,
  resolved with `--basetemp`, not a code issue).
- Acquisition manifest (`reports/generated/nlp/acquisition_manifest_v2.json`)
  independently re-hashed: **19/19 files verified, 0 missing, 0 mismatched**.
- Amazon raw file hash independently recomputed:
  `sha256sum data/raw/amazon_reviews_appliances/Appliances.jsonl.gz` =
  `150f209befceaa6f837abc997065b2d251034bbbda19bebc4ad56dac779730c2`, matching the
  handoff's declared value exactly.
- `artifacts/experiments/nlp/` does not exist — 0 unauthorized NLP artifacts, confirming
  the handoff's claim.
- The active contracts and modules listed in the handoff (`configs/nlp_split_policy.yaml`,
  `configs/nlp_duplicate_control_contract_v2.yaml`, `configs/nlp_execution_contract_v4.yaml`,
  `configs/nlp_experiment_manifest.yaml`, `configs/nlp_training_batch_authorization_v2.yaml`,
  `configs/nlp_metric_contract_v2.yaml`, `configs/nlp_label_ontology_v2.yaml`,
  `src/nlp/splitting.py`, `src/nlp/duplicate_control.py`, `src/nlp/execution_control.py`,
  `src/nlp/configuration.py`, `src/nlp/amazon_adapter.py`, `src/nlp/text_normalization.py`,
  `tests/test_phase2c_remediation_v4.py`) were read in full and independently confirmed
  to exhibit exactly the four flaws described in the handoff (see §3).

## 2. Design decision: additive, non-breaking remediation

All four workstreams are implemented as **new modules** that supersede the flawed V4 APIs
for any future real execution, rather than as edits to the existing V4 evidence files:

| New module | Supersedes (left byte-for-byte unmodified) |
|---|---|
| `src/nlp/split_preparation.py` | `src/nlp/splitting.py` (still the correct low-level "no group crosses a partition" primitive; kept as-is) |
| `src/nlp/winner_selection.py` | `src/nlp/execution_control.select_winner` |
| `src/nlp/release_control.py` | `src/nlp/execution_control.PartitionStore` / `ReleaseAuthorization` |
| `src/nlp/batch1_executor.py` | (no prior integrated executor existed) |

**Why additive, not a rewrite of `execution_control.py`:** the four V4-IR findings require
several new *mandatory* fields (e.g. `configuration_fingerprint`, `split_hash`,
`metric_provenance`, `execution_order` cross-checked against a known registry) that the V4
evidence tests in `tests/test_phase2c_remediation_v4.py` do not supply. Changing
`execution_control.select_winner`'s contract in place would have broken that historical
V4-evidence test file to "make tests pass" — exactly what the handoff prohibits. Instead,
`execution_control.py` and `tests/test_phase2c_remediation_v4.py` are preserved byte-identical
(verified: `git`-less repo, so verified via re-read + the unified diff of file lists below),
and all 179 original tests still pass unchanged. `src/nlp/batch1_executor.py` — the one
piece of code intended to represent how a future real run would actually work — calls
**only** the new hardened modules (`split_preparation`, `winner_selection`) and never the
old `execution_control` APIs.

No active `configs/*.yaml` file was edited. No historical checkpoint, review package, or
V1–V4 manifest was modified.

## 3. Closure matrix

| Finding | Status | Evidence |
|---|---|---|
| **V4-IR-01** — duplicate-control / split preparation | **CLOSED** | `src/nlp/split_preparation.py`; 19 test functions (21 collected cases with parametrization) in `tests/test_phase2c_remediation_v5.py` (`test_task_split_policy_*`, `test_*duplicate*`, `test_conflicting_groups_*`, `test_no_group_ever_crosses_a_partition`, `test_deterministic_repeated_execution`, `test_shuffled_input_is_deterministic_by_content`, `test_very_small_dataset_fails_closed`, `test_imbalanced_group_case_*`, `test_malformed_split_policy_fails_closed`, `test_unsupported_conflict_action_fails_closed`, `test_unsupported_same_label_action_fails_closed`, `test_contradictory_active_policy_is_detected_and_stopped`, `test_unrecognized_schema_version_fails_closed`) |
| **V4-IR-02** — internal-test release / leakage | **CLOSED** | `src/nlp/release_control.py`; 24 test functions (29 collected cases with parametrization) covering authorized release, self-asserted/forged authorization, missing/malformed/empty-field authorization files, wrong winner/fingerprint/split-hash/experiment, second release (same instance and a fresh instance via the durable ledger), ledger path traversal (both `run_id` and `experiment_id`), malformed `FrozenWinnerRecord`, evaluator exception handling, evaluator row-retention, TOCTOU evidence-file tampering, and a concurrent-release race closed by a pre-evaluation ledger reservation |
| **V4-IR-03** — integrated executor | **CLOSED** | `src/nlp/batch1_executor.py`; 11 test functions (14 collected cases with parametrization) covering a full successful run, train-only-fit/validation-only-eval proof via spy vectorizer/estimator, optional-stage enforcement, unsupported model family, missing configuration field, tampered fingerprint, wrong dataset provenance, output collision, malformed run_id (4 variants), unsupported experiment_id, and deterministic repeated execution |
| **V4-IR-04** — winner-selection validation | **CLOSED** | `src/nlp/winner_selection.py`; 14 test functions (18 collected cases with parametrization) covering NaN/+Inf/-Inf, missing field, malformed numeric, unknown compound_id, wrong fingerprint, missing/wrong split_hash, non-validation provenance (5 variants), cross-experiment, duplicate identity, empty results, execution_order-vs-registry mismatch, and exact tie-break ordering (two scenarios) |

All four are backed by **executable, currently-passing tests** (73 test functions, 87 collected
cases total after the independent-review addendum in §11), not narrative claims alone.

### V4-IR-01 in detail

`src/nlp/splitting.py#materialize_group_split` never deduplicated same-label rows and took
conflict handling as a caller-supplied string. `src/nlp/split_preparation.py` adds:

- `resolve_task_split_policy(task_type, ...)` — loads `configs/nlp_split_policy.yaml` and
  `configs/nlp_duplicate_control_contract_v2.yaml` at call time (never cached/hardcoded),
  validates `schema_version` on both, maps each Batch 1 `task_type` to the split policy's
  conflict category via an evidence-based restatement (§4), resolves the category's
  conflict action to one of `REMOVE_FROM_ALL_SPLITS` / `KEEP_IN_TRAIN_ONLY_WITH_FLAG` /
  `FAIL`, cross-checks the resolution against `duplicate_control_contract_v2`'s
  `sentiment_conflict_action` / `speech_act_conflict_action` fields where they overlap, and
  raises `SplitPolicyError` on any malformed, unsupported, or contradictory combination.
- `prepare_task_bound_split(...)` — groups by `NORMALIZED_EXACT_KEY`, deterministically keeps
  exactly one row per same-label duplicate group (content-based tiebreak, independent of
  input order), applies the resolved conflict action to conflicting-label groups (remove /
  force-to-train-with-flag / fail), then delegates the surviving single-label eligible groups
  to the unmodified, already-tested `materialize_group_split` for the two-stage stratified
  split. Returns a full audit record (`SplitPreparationAudit`) with counts for every category
  the handoff required, plus per-row `PreparedRowAssignment` including `flagged`/`flag_reason`.

### V4-IR-02 in detail

`src/nlp/release_control.py` replaces self-asserted authorization with:

- `load_external_authorization(path)` — the only supported way to obtain an
  `ExternalAuthorizationEvidence`; schema-validates every field, rejects empty strings,
  rejects non-hex fingerprint/split-hash fields, and records the evidence file's own
  sha256 for later tamper detection.
- `FrozenWinnerRecord` — immutable, `__post_init__`-validated (safe run_id/experiment_id
  tokens, non-empty compound_id, 64-hex fingerprint and split_hash).
- `ProtectedInternalTestPartition.release_once(...)` — cross-checks the partition's own
  `(run_id, experiment_id, split_hash)` against the frozen winner, re-verifies the
  authorization evidence file's hash at call time (TOCTOU defense), computes the ledger path
  **internally** from validated tokens (callers cannot choose a ledger location), denies if
  a ledger already exists, **writes a `RESERVED_PENDING_EVALUATION` marker before invoking
  the evaluator** (closing a race where a second release could slip in if the final ledger
  write failed after a successful evaluation), clears its own row reference before the
  evaluator runs, and unconditionally records the true outcome (`RELEASED_ONCE_SUCCESS` or
  `RELEASE_FAILED_EVALUATOR_EXCEPTION`) in a `finally` block.
- **Documented residual limitation** (not a bug, a CPython fact): once row data is handed to
  an evaluator callback in-process, nothing in pure Python can prevent that specific callback
  from copying the data into its own memory. `test_evaluator_row_retention_does_not_defeat_single_release_or_store_clearing`
  proves this honestly — the evaluator *does* see the rows — while also proving what IS
  enforced regardless: the store's own reference is cleared before the evaluator runs, no
  other API path returns rows, and release happens at most once per run/experiment (durably,
  via the ledger file, even across a fresh partition instance).

### V4-IR-03 in detail

`src/nlp/batch1_executor.py#execute_batch1_experiment` enforces the full declared pipeline in
one fail-closed function: output-path safety and non-overwrite, `optional_stages` re-checked
live against `configs/nlp_execution_contract_v4.yaml` (all `DISABLED` except
`class_weighting`), dataset-provenance equality check, configuration-set validation
(experiment match, allowed model family, **independently recomputed** fingerprint match
against `src/nlp/configuration.py`'s own canonical hashing — catching tampering between
resolution and execution), `prepare_task_bound_split` for duplicate/conflict/group-split,
train-only TF-IDF fit and classifier fit (proved train-only / validation-only by a spy
vectorizer/estimator test that reconstructs the exact expected train/validation text sets
from the executor's own returned split audit and asserts byte-for-byte equality),
`winner_selection`'s full schema/finiteness/provenance validation on every candidate result,
`select_validated_winner`, a `FrozenWinnerRecord`, and deterministic canonical-JSON artifact
writing (`resolved_config.json`, `dataset_hashes.json`, `prepared_split_hashes.json`,
`environment.json`, `development_metrics.json`, `candidate_results.json`, `winner.json`,
`run_manifest.json`). It deliberately **stops there** — it never calls `release_control`,
keeping internal-test release a separately controlled, separately authorized step as required.

It is data-source-agnostic: it never opens a dataset file itself. All 12 executor tests in
V5 pass fabricated, in-memory records and either a `_SpyVectorizer`/`_SpyEstimator` pair or
real (but tiny, synthetic-text) `TfidfVectorizer`/`LogisticRegression` instances via
`configuration.instantiate_configuration`. No project dataset file was read by any V5 test.

### V4-IR-04 in detail

`src/nlp/winner_selection.py#select_validated_winner` validates, per candidate result, before
any comparison: all four string identity fields non-empty, `macro_f1`/`balanced_accuracy`/
`accuracy` all `math.isfinite` (rejecting NaN, +Inf, -Inf, and non-numeric/bool values),
`execution_order` a non-negative int, `metric_provenance == "NLP_DEV_VALIDATION"` (rejecting
`NLP_DEV_TRAIN` / `NLP_INTERNAL_TEST` / `NLP_EXTERNAL_TEST` / missing / empty), the result's
`experiment_id` matching the registry's single experiment, `compound_id` known to a
`KnownConfigurationRegistry` built from the actual resolved configuration set,
`configuration_fingerprint` matching that registry's recorded fingerprint for the compound_id,
`execution_order` matching that registry's recorded execution order for the compound_id
(closing a gap found during V5's own adversarial self-review, where a forged `execution_order`
could otherwise manipulate the tie-break), and `split_hash` matching the caller-supplied
`expected_split_hash`. Duplicate `compound_id` identities across the result set are rejected.
Tie-break order is unchanged: macro-F1 desc, balanced accuracy desc, accuracy desc,
execution order asc — proved by two dedicated tests.

## 4. Task → conflict-category resolution (not invented; restated from existing evidence)

`configs/nlp_split_policy.yaml`'s `conflicting_label_policy.per_task` is keyed by four
semantic categories, not by experiment id. The mapping used by
`split_preparation.TASK_TYPE_TO_CONFLICT_CATEGORY` restates, without alteration, the
`task_type` values already declared per Batch 1 experiment in
`configs/nlp_metric_contract_v2.yaml` / `configs/nlp_label_ontology_v2.yaml`:

| Batch 1 `task_type` | Split-policy category | Resolved conflict action |
|---|---|---|
| `FIVE_CLASS_RATING_CLASSIFICATION` (A, C) | `REVIEW_RATING` | `KEEP_IN_TRAIN_ONLY_WITH_FLAG` |
| `FOUR_CLASS_SENTIMENT_CLASSIFICATION` (B2) | `SENTIMENT` | `REMOVE_FROM_ALL_SPLITS` |
| `BINARY_OFFENSIVE_LANGUAGE_CLASSIFICATION` (E) | `OFFENSIVE_LANGUAGE_SAFETY` | `FAIL` (no automatic resolution is authorized; MPOLD is independently documented as having 0 known duplicates, so this is not expected to trigger in a real run, but must fail closed if it ever does) |

`SPEECH_ACT` has no Batch 1 mapping because no active Batch 1 experiment is a speech-act
task. `DO_NOT_AUTO_RESOLVE` and `DATASET_SPECIFIC_REVIEW` are both treated as `FAIL` because
neither authorizes an automatic resolution. This mapping is enforced in code and covered by
`test_task_split_policy_resolves_from_active_contracts_not_caller_strings`; it is not something
a future agent chooses at call time.

## 5. Residual, explicitly unresolved scientific ambiguities (per the handoff's own instruction)

These are **not** closed by V5 and are called out rather than silently decided, because
deciding them would be a scientific choice outside this remediation's authority:

1. **Zero-division handling** for precision/recall on absent/unseen classes is not pinned by
   `configs/nlp_metric_contract_v2.yaml`. `batch1_executor.Batch1ExecutorInputs.metric_function`
   is therefore a **required** (no-default) injected callable — the executor refuses to guess
   a value on the contract's behalf. V5's own tests use `zero_division=0` only as an explicit,
   commented, test-fixture choice, not a production default.
2. **Per-class label ordering** for `confusion_matrix` / per-class secondary metrics is not
   pinned anywhere in the active contracts. This does not block V5's winner-selection metrics
   (`macro_f1`, `balanced_accuracy`, `accuracy` are all order-invariant), but must be resolved
   before a future run reports the metric contract's secondary per-class metrics.
3. The V5 executor computes and validates only the three primary comparison metrics plus
   provenance; full `nlp_metric_contract_v2.yaml` secondary-metric computation
   (`per_class_precision`, `per_class_recall`, `per_class_f1`, `confusion_matrix`,
   `minority_class_recall` for Experiment E) is left to the future authorized execution agent,
   using the same required-injection pattern (no silent default).
4. **Real dataset-acquisition-hash wiring**: `batch1_executor` requires the caller to supply
   both the actual and expected `dataset_acquisition_sha256` and fails closed on mismatch, but
   V5 does not wire it to actually read/hash the real Amazon/ASTD/LABR/MPOLD files (that would
   mean opening real project data files from executor code, which V5 deliberately avoids). A
   future authorized agent must source the expected hash from
   `reports/generated/nlp/acquisition_manifest_v2.json` and the actual hash from re-hashing the
   real acquired file before calling the executor.

## 6. Architectural residual observations (not blocking; documented per the handoff's adversarial-review requirement)

Found during V5's own "attack your own implementation" pass:

1. `src/nlp/splitting.py#materialize_group_split` (the low-level, unmodified V4 primitive)
   remains directly callable and does **not** itself deduplicate or resolve conflict policy —
   by design, since V4-IR-01's fix lives one layer up in `split_preparation.py`. Any future
   code must go through `split_preparation.prepare_task_bound_split` (or the integrated
   `batch1_executor`, which does), not call `splitting.py` directly for a real run. This is a
   discipline requirement to record for the next agent, not a code defect.
2. `metric_provenance` in a result record is a **declared, trusted tag**. `winner_selection`
   validates the tag's presence/allowed-value but has no way to cryptographically verify that
   metrics tagged `NLP_DEV_VALIDATION` were actually computed from the validation partition —
   that guarantee comes from `batch1_executor` always setting the tag itself (never
   caller-suppliable) and always computing metrics from the properly-separated validation
   rows, which is proved by `test_executor_fits_train_only_and_evaluates_validation_only`. Any
   future code that constructs result records outside the executor must preserve this
   invariant manually; `winner_selection` alone cannot enforce computational lineage.
3. `output_root` (executor) and `artifact_root` (release control) are trusted, caller-supplied
   parameters — a legitimate future agent supplies the real contract paths
   (`artifacts/experiments/nlp/phase2c/batch1/` for the executor). The path-safety guarantees
   proved here are that `run_id`/`experiment_id` cannot escape *whatever root is given* (path
   traversal), not that the root itself is restricted to one location.

## 7. Files changed / added

**Added (all new, purely additive):**

- `src/nlp/split_preparation.py` (358 lines)
- `src/nlp/winner_selection.py` (160 lines, after review fixes)
- `src/nlp/release_control.py` (325 lines, after review fixes)
- `src/nlp/batch1_executor.py` (374 lines, after review fixes)
- `tests/test_phase2c_remediation_v5.py` (87 collected cases, 73 test functions, after review additions)
- `reports/checkpoints/phase2c_nlp_training_authorization_remediation_v5/` (this checkpoint)
- `reports/review_packages/olist/phase2c/remediation_v5_summary.md` (this file)

**Modified:** none. `src/nlp/splitting.py`, `src/nlp/duplicate_control.py`,
`src/nlp/execution_control.py`, `src/nlp/configuration.py`, `src/nlp/amazon_adapter.py`,
`src/nlp/text_normalization.py`, every `configs/*.yaml`, every prior checkpoint/review
package, and `tests/test_phase2c_remediation_v4.py` are byte-identical to their state at the
start of this session.

## 8. Tests run and results

```text
Targeted V5 suite:  .venv\Scripts\python.exe -m pytest -q -ra tests/test_phase2c_remediation_v5.py
  -> 87 passed, 0 failed  (after the independent-review addendum in §11; was 82 at first implementation)

Full non-training suite: .venv\Scripts\python.exe -m pytest -q -ra
  -> 266 passed, 0 failed, 0 warnings, 0 skipped, 0 xfailed
  (179 pre-existing + 87 new V5 tests; 0 regressions)
```

## 9. Integrity re-verification after implementation

```text
Acquisition manifest (reports/generated/nlp/acquisition_manifest_v2.json): 19/19 verified, 0 missing, 0 mismatched
Amazon raw sha256: 150f209befceaa6f837abc997065b2d251034bbbda19bebc4ad56dac779730c2 (matches)
Unauthorized NLP artifacts: 0 (artifacts/experiments/nlp/ does not exist)
Historical checkpoints/review packages modified: NO
Datasets modified: NO
```

## 10. Authorization boundary (unchanged, restated)

No real Phase 2C NLP training was executed. No TF-IDF or estimator was fit on project data
at any point in this session — every fit/transform call in every V5 test operates on
fabricated in-memory strings (e.g. `"synthetic class_0 review body number 12 filler words..."`)
never on `data/raw/amazon_reviews_appliances/`, `data/quarantine/nlp/**`, or any other project
dataset. No internal-test content was released, evaluated, or accessed — all
`ProtectedInternalTestPartition` tests use synthetic integer payloads. No Phase 2A protected
content was accessed. No transformer, embedding, or additional search dimension was added.
The approved matrix remains exactly A=4/B2=6/C=4/E=6, total 20 (unchanged; not touched by V5).

**`NLP_BATCH1_TRAINING_AUTHORIZED = NO`.** This remediation does not authorize training.
It prepares the repository for `PHASE2C_NLP_TRAINING_AUTHORIZATION_INDEPENDENT_REVIEW_V5`.

## 11. Addendum — fixes made during the V5 independent review (same session)

An independent review of this V5 package was conducted immediately after implementation
(same session, role switched from implementer to reviewer at the user's explicit direction).
The review reproduced all hashes/tests fresh, then critically re-read every new module hunting
for bypasses. It found and fixed four additional issues before rendering a decision — full
detail in `reports/checkpoints/phase2c_nlp_training_authorization_remediation_v5_independent_review_2026-08-09/`:

1. **Missing executor↔release connector**: nothing built the provenance-tagged internal-test
   rows `release_control.ProtectedInternalTestPartition` expects after a run completed — a
   future agent would have had to hand-roll that step. **Fixed**: added
   `batch1_executor.build_protected_internal_test_rows(inputs, result)`, the sole sanctioned
   connector; proven with a full executor→release_once synthetic end-to-end test.
2. **Leaky error type**: `KnownConfigurationRegistry.from_resolved_configurations` did bare
   `c["execution_order"]` dict lookups, so a configuration missing that field (fingerprinting
   excludes it, so a matching fingerprint doesn't guarantee its presence) would raise a raw
   `KeyError` instead of failing closed uniformly. **Fixed**: wrapped in `WinnerSelectionRejected`.
3. **Ledger check-then-write race**: `release_once` checked `ledger_path.exists()` then wrote
   the reservation marker as two separate steps, leaving a narrow window where two concurrent
   `release_once` calls (from two independently constructed partition instances for the same
   run/experiment) could both pass the check before either wrote. **Fixed**: replaced with an
   atomic exclusive-create (`open(path, "x")`), closing the race at the OS level.
4. **Unwired schema-adaptation stage**: the executor's own flow diagram promised
   `adapter → normalization → ...`, but the code silently assumed the caller had already
   adapted records — it never called `amazon_adapter.adapt_amazon_record` or any adapter
   itself. Cross-checked against `configs/nlp_experiment_manifest.yaml`: only Experiment A
   (Amazon) declares a `physical_to_canonical_adapter`; B2/C/E already use native canonical
   field names directly, so no adapter exists or is needed for them. **Fixed**: added an
   optional `schema_adapter` field to `Batch1ExecutorInputs` (default `None`, fully backward
   compatible), wired through both `execute_batch1_experiment` and
   `build_protected_internal_test_rows` via one shared helper so internal-test payloads are
   always in the same adapted schema as train/validation rows. Proven with a synthetic,
   Amazon-physical-schema (`rating`/`title`/`text`) end-to-end test using
   `amazon_adapter.adapt_amazon_record` on fabricated data.

After these fixes: `tests/test_phase2c_remediation_v5.py` now collects **87 test cases**
(73 test functions); full suite = 179 pre-existing + 87 new = **266 passed, 0 failed**. All
checkpoint/acquisition hashes re-verified clean, zero real data touched. See the independent
review report for the full verdict.
