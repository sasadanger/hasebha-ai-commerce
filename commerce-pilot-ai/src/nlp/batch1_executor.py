"""Integrated, fail-closed Batch 1 executor architecture (V4-IR-03).

Binds together the already-authorized contracts and modules into a single
enforced pipeline so that a future authorized training agent never has to
make a scientifically material decision after authorization:

    authorized data source (caller-supplied records)
    -> schema adaptation (optional injected `schema_adapter`, e.g.
       amazon_adapter.adapt_amazon_record for Experiment A; a no-op passthrough
       for B2/C/E, which already use native canonical field names per
       configs/nlp_experiment_manifest.yaml)
    -> authorized text normalization (text_normalization.normalize_text)
    -> task policy resolution (split_preparation.resolve_task_split_policy)
    -> duplicate/conflict preparation + group-aware splitting
       (split_preparation.prepare_task_bound_split, which itself delegates the
       eligible groups to splitting.materialize_group_split)
    -> resolved configuration set (validated against configuration.py's own
       fingerprinting, not re-derived)
    -> train-only TF-IDF fitting -> train-only classifier fitting
    -> validation-only evaluation -> validated result record
       (winner_selection's schema/finiteness/provenance validation)
    -> deterministic winner selection (winner_selection.select_validated_winner)
    -> deterministic, non-overwriting artifact writing
    -> frozen winner record (release_control.FrozenWinnerRecord)
    -> (a separately controlled internal-test release: this executor
        deliberately stops before that point; see release_control.py)

This executor is data-source agnostic: it never reads a dataset file itself
and never invents which classical model families are permitted (it reuses
`configuration.FAMILY_CLASS`). It is exercised in V5 exclusively with
synthetic, in-memory records and dependency-injected dummy/spy
vectorizers+estimators; it must never be pointed at real project data files
during V5 remediation.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .configuration import FAMILY_CLASS, _canonical_json
from .release_control import FrozenWinnerRecord, ProvenancedRow
from .split_preparation import (
    DEFAULT_DUPLICATE_CONTRACT_PATH,
    DEFAULT_SPLIT_POLICY_PATH,
    SplitPreparationResult,
    prepare_task_bound_split,
)
from .text_normalization import normalize_text
from .winner_selection import KnownConfigurationRegistry, _validate_result_schema, select_validated_winner

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "artifacts" / "experiments" / "nlp" / "phase2c" / "batch1"
DEFAULT_EXECUTION_CONTRACT_PATH = _REPO_ROOT / "configs" / "nlp_execution_contract_v4.yaml"

ALLOWED_EXPERIMENT_IDS = frozenset({"A", "B2", "C", "E"})
ALLOWED_MODEL_FAMILIES = frozenset(FAMILY_CLASS)

RUN_ID_PATTERN = re.compile(r"^nlp-batch1-\d{8}T\d{6}Z-[0-9a-f]{8}$")

_OPTIONAL_STAGES_THAT_MUST_BE_DISABLED_EXCEPT = {"class_weighting"}


class Batch1ExecutionError(RuntimeError):
    """Raised whenever the integrated executor refuses to proceed; fail closed."""


def _validate_no_optional_stages(execution_contract_path: Path) -> None:
    try:
        contract = yaml.safe_load(execution_contract_path.read_text(encoding="utf-8"))
        optional_stages = contract["optional_stages"]
    except (OSError, KeyError, TypeError) as exc:
        raise Batch1ExecutionError(f"cannot load optional_stages from execution contract: {exc}") from exc
    for stage, state in optional_stages.items():
        if stage in _OPTIONAL_STAGES_THAT_MUST_BE_DISABLED_EXCEPT:
            continue
        if state != "DISABLED":
            raise Batch1ExecutionError(f"optional stage {stage!r} is not DISABLED in the active execution contract: {state!r}")


def _recompute_fingerprint(configuration: Mapping[str, Any]) -> str:
    payload_keys = (
        "experiment_id", "canonical_experiment_id", "configuration_id", "compound_id",
        "model_family", "vectorizer", "classifier", "seed", "experiment_order", "configuration_order",
    )
    payload = {key: configuration[key] for key in payload_keys}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_configuration_set(
    configurations: Sequence[Mapping[str, Any]], experiment_id: str,
) -> list[dict[str, Any]]:
    if not configurations:
        raise Batch1ExecutionError("no configurations supplied")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for configuration in configurations:
        if configuration.get("experiment_id") != experiment_id:
            raise Batch1ExecutionError(
                f"configuration experiment_id {configuration.get('experiment_id')!r} does not match run experiment_id {experiment_id!r}"
            )
        family = configuration.get("model_family")
        if family not in ALLOWED_MODEL_FAMILIES:
            raise Batch1ExecutionError(f"unsupported model family: {family!r}")
        try:
            recomputed = _recompute_fingerprint(configuration)
        except KeyError as exc:
            raise Batch1ExecutionError(f"configuration is missing a required field: {exc}") from exc
        if recomputed != configuration.get("fingerprint_sha256"):
            raise Batch1ExecutionError(
                f"configuration fingerprint mismatch for {configuration.get('compound_id')!r} (tampered or stale configuration)"
            )
        compound_id = configuration["compound_id"]
        if compound_id in seen:
            raise Batch1ExecutionError(f"duplicate compound_id in configuration set: {compound_id!r}")
        seen.add(compound_id)
        validated.append(dict(configuration))
    return validated


def _validate_output_path(run_id: str, experiment_id: str, output_root: Path) -> Path:
    if experiment_id not in ALLOWED_EXPERIMENT_IDS:
        raise Batch1ExecutionError(f"unsupported experiment_id: {experiment_id!r}")
    if not RUN_ID_PATTERN.match(run_id):
        raise Batch1ExecutionError(f"run_id does not match the required format: {run_id!r}")
    root = output_root.resolve()
    run_dir = (root / experiment_id / run_id).resolve()
    if root not in run_dir.parents:
        raise Batch1ExecutionError("resolved run output path escapes the allowed output root")
    if run_dir.exists():
        raise Batch1ExecutionError(f"run output directory already exists (non-overwrite forbidden): {run_dir}")
    return run_dir


@dataclass(frozen=True)
class Batch1ExecutorInputs:
    run_id: str
    experiment_id: str
    task_type: str
    canonical_records: Sequence[Mapping[str, Any]]
    text_key: str
    label_key: str
    seed: int
    resolved_configurations: Sequence[Mapping[str, Any]]
    estimator_factory: Callable[[Mapping[str, Any]], tuple[Any, Any]]
    metric_function: Callable[[Sequence[Any], Sequence[Any]], Mapping[str, float]]
    dataset_acquisition_sha256: str
    expected_dataset_acquisition_sha256: str
    output_root: Path = DEFAULT_OUTPUT_ROOT
    execution_contract_path: Path = DEFAULT_EXECUTION_CONTRACT_PATH
    split_policy_path: Path = DEFAULT_SPLIT_POLICY_PATH
    duplicate_contract_path: Path = DEFAULT_DUPLICATE_CONTRACT_PATH
    # Only Amazon (Experiment A) declares a physical_to_canonical_adapter in
    # configs/nlp_experiment_manifest.yaml; B2/C/E already use native
    # canonical field names directly per that manifest, so this is optional.
    # When set, `canonical_records` is treated as pre-adapter (physical)
    # records and this callable is applied to each one, first, before
    # normalization -- wiring the "adapter" stage the V4-IR-03 flow requires
    # instead of silently assuming the caller already ran it.
    schema_adapter: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class Batch1ExecutionResult:
    run_id: str
    experiment_id: str
    results: tuple[dict[str, Any], ...]
    winner: FrozenWinnerRecord
    split_hash: str
    prep_result: SplitPreparationResult
    artifact_paths: Mapping[str, Path]


def _effective_canonical_records(inputs: Batch1ExecutorInputs) -> list[dict[str, Any]]:
    """Apply `inputs.schema_adapter` (the "adapter" pipeline stage) if provided.

    Shared by `execute_batch1_experiment` and `build_protected_internal_test_rows`
    so that internal-test row payloads are always in the same (adapted,
    canonical) schema as the train/validation rows actually fit and evaluated
    on -- never the raw pre-adapter schema.
    """
    if inputs.schema_adapter is None:
        return [dict(record) for record in inputs.canonical_records]
    return [dict(inputs.schema_adapter(record)) for record in inputs.canonical_records]


def execute_batch1_experiment(inputs: Batch1ExecutorInputs) -> Batch1ExecutionResult:
    """Run the full authorized pipeline for one experiment against in-memory records.

    Every stage fails closed: unsafe/duplicate output paths, unsupported
    model families, tampered configuration fingerprints, dataset-provenance
    mismatches, non-DISABLED optional stages, and malformed/non-finite
    candidate results are all rejected before any artifact is written.
    """
    run_dir = _validate_output_path(inputs.run_id, inputs.experiment_id, inputs.output_root)
    _validate_no_optional_stages(inputs.execution_contract_path)

    if inputs.dataset_acquisition_sha256 != inputs.expected_dataset_acquisition_sha256:
        raise Batch1ExecutionError("dataset acquisition provenance mismatch: refusing to execute on unverified data")

    configurations = _validate_configuration_set(inputs.resolved_configurations, inputs.experiment_id)
    registry = KnownConfigurationRegistry.from_resolved_configurations(configurations)

    source_records = _effective_canonical_records(inputs)                    # schema adaptation
    normalized_records = [
        {**record, "__normalized_text__": normalize_text(record[inputs.text_key])}
        for record in source_records
    ]

    prep = prepare_task_bound_split(
        normalized_records, text_key=inputs.text_key, label_key=inputs.label_key,
        task_type=inputs.task_type, seed=inputs.seed,
        split_policy_path=inputs.split_policy_path, duplicate_contract_path=inputs.duplicate_contract_path,
    )
    split_hash = _compute_split_hash(prep, seed=inputs.seed)

    train_rows = [normalized_records[a.row_index] for a in prep.assignments if a.split == "train"]
    validation_rows = [normalized_records[a.row_index] for a in prep.assignments if a.split == "validation"]
    if not validation_rows:
        raise Batch1ExecutionError("no rows were assigned to the validation partition")

    train_text = [row["__normalized_text__"] for row in train_rows]
    validation_text = [row["__normalized_text__"] for row in validation_rows]
    train_labels = [row[inputs.label_key] for row in train_rows]
    validation_labels = [row[inputs.label_key] for row in validation_rows]

    results: list[dict[str, Any]] = []
    for configuration in configurations:
        vectorizer, estimator = inputs.estimator_factory(configuration)
        train_features = vectorizer.fit_transform(train_text)                 # train-only TF-IDF fitting
        validation_features = vectorizer.transform(validation_text)
        estimator.fit(train_features, train_labels)                          # train-only classifier fitting
        predictions = estimator.predict(validation_features)                 # validation-only evaluation
        metrics = dict(inputs.metric_function(validation_labels, predictions))
        result = {
            "experiment_id": inputs.experiment_id,
            "compound_id": configuration["compound_id"],
            "configuration_fingerprint": configuration["fingerprint_sha256"],
            "execution_order": configuration["execution_order"],
            "split_hash": split_hash,
            "metric_provenance": "NLP_DEV_VALIDATION",
            **metrics,
        }
        _validate_result_schema(result, registry, split_hash)               # validated result record
        results.append(result)

    winner = select_validated_winner(results, registry=registry, expected_split_hash=split_hash)
    frozen_winner = FrozenWinnerRecord(
        run_id=inputs.run_id, experiment_id=inputs.experiment_id,
        compound_id=winner["compound_id"], configuration_fingerprint=winner["configuration_fingerprint"],
        split_hash=split_hash,
    )

    artifact_paths = _write_run_artifacts(
        run_dir=run_dir, inputs=inputs, configurations=configurations,
        results=results, winner=winner, prep=prep, split_hash=split_hash,
    )

    return Batch1ExecutionResult(
        run_id=inputs.run_id, experiment_id=inputs.experiment_id, results=tuple(results),
        winner=frozen_winner, split_hash=split_hash, prep_result=prep, artifact_paths=artifact_paths,
    )


def build_protected_internal_test_rows(
    inputs: Batch1ExecutorInputs, result: Batch1ExecutionResult,
) -> tuple[ProvenancedRow, ...]:
    """Construct the provenance-tagged internal-test rows for a completed run.

    This is the sanctioned connector between `execute_batch1_experiment` and
    `release_control.ProtectedInternalTestPartition`: it reads only the
    original caller-supplied `canonical_records` for rows `prepare_task_bound_split`
    assigned to `internal_test`, tags each with this run's own provenance
    (`run_id`, `experiment_id`, `split_hash`), and returns them ready to hand to
    `ProtectedInternalTestPartition`. No other function in this module
    constructs or returns internal-test row content; `execute_batch1_experiment`
    itself never reads, counts, or iterates internal_test rows.
    """
    if result.run_id != inputs.run_id or result.experiment_id != inputs.experiment_id:
        raise Batch1ExecutionError("result does not match the supplied inputs")
    indices = [a.row_index for a in result.prep_result.assignments if a.split == "internal_test"]
    if not indices:
        raise Batch1ExecutionError("no rows were assigned to the internal_test partition")
    source_records = _effective_canonical_records(inputs)
    return tuple(
        ProvenancedRow(
            run_id=result.run_id, experiment_id=result.experiment_id,
            partition="internal_test", split_hash=result.split_hash,
            payload=source_records[index],
        )
        for index in indices
    )


def _compute_split_hash(prep: SplitPreparationResult, *, seed: int) -> str:
    payload = {
        "seed": seed,
        "policy": {
            "task_type": prep.policy.task_type,
            "conflict_category": prep.policy.conflict_category,
            "resolved_conflict_action": prep.policy.resolved_conflict_action,
            "same_label_action": prep.policy.same_label_action,
            "split_policy_schema": prep.policy.split_policy_schema,
            "duplicate_contract_schema": prep.policy.duplicate_contract_schema,
        },
        "assignments": [
            {
                "row_index": a.row_index, "group_key": a.group_key, "split": a.split,
                "excluded": a.excluded, "exclusion_reason": a.exclusion_reason,
                "flagged": a.flagged, "flag_reason": a.flag_reason,
            }
            for a in prep.assignments
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_run_artifacts(
    *, run_dir: Path, inputs: Batch1ExecutorInputs, configurations: list[dict[str, Any]],
    results: list[dict[str, Any]], winner: dict[str, Any], prep: SplitPreparationResult, split_hash: str,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=False)
    written: dict[str, Path] = {}

    def _dump(name: str, payload: Any) -> None:
        path = run_dir / name
        if path.exists():
            raise Batch1ExecutionError(f"refusing to overwrite existing artifact: {path}")
        path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
        written[name] = path

    audit = prep.audit
    _dump("resolved_config.json", {"experiment_id": inputs.experiment_id, "configurations": configurations})
    _dump("dataset_hashes.json", {"dataset_acquisition_sha256": inputs.dataset_acquisition_sha256})
    _dump("prepared_split_hashes.json", {
        "split_hash": split_hash,
        "policy": {
            "task_type": prep.policy.task_type,
            "conflict_category": prep.policy.conflict_category,
            "conflict_action": prep.policy.conflict_action,
            "resolved_conflict_action": prep.policy.resolved_conflict_action,
            "same_label_action": prep.policy.same_label_action,
        },
        "audit": {
            "total_input_rows": audit.total_input_rows,
            "unique_normalized_groups": audit.unique_normalized_groups,
            "same_label_duplicate_groups": audit.same_label_duplicate_groups,
            "same_label_duplicate_rows_removed": audit.same_label_duplicate_rows_removed,
            "conflicting_label_groups": audit.conflicting_label_groups,
            "conflicting_label_rows_removed": audit.conflicting_label_rows_removed,
            "conflicting_label_rows_flagged": audit.conflicting_label_rows_flagged,
            "eligible_groups_after_preparation": audit.eligible_groups_after_preparation,
            "final_split_counts": audit.final_split_counts,
        },
    })
    _dump("environment.json", {"executor": "batch1_executor_v5_synthetic", "seed": inputs.seed})
    _dump("development_metrics.json", {"results": results})
    _dump("candidate_results.json", {"results": results})
    _dump("winner.json", {
        "winner": winner,
        "selection_rule": "macro_f1_desc,balanced_accuracy_desc,accuracy_desc,execution_order_asc",
        "selected_utc": datetime.now(timezone.utc).isoformat(),
    })
    _dump("run_manifest.json", {
        "run_id": inputs.run_id, "experiment_id": inputs.experiment_id,
        "configuration_count": len(configurations), "result_count": len(results),
        "artifact_files": sorted(written) + ["run_manifest.json"],
    })
    return written
