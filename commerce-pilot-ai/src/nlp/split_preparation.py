"""Task-bound, policy-resolved duplicate/conflict split preparation (V4-IR-01).

`src/nlp/splitting.py#materialize_group_split` is a pure group-level, no-crossing
splitter: it partitions NORMALIZED_EXACT_KEY groups across train/validation/
internal_test without ever splitting a group across partitions, but it does not
deduplicate same-label duplicate rows and it takes conflict handling as a caller
string rather than resolving it from the active contracts. This module is the
missing layer described in configs/nlp_duplicate_control_contract_v2.yaml and
configs/nlp_split_policy.yaml: it resolves same-label and conflicting-label
behaviour from those two active contracts (never from a caller-chosen string),
deterministically deduplicates same-label groups, applies the resolved
conflict policy, and then delegates the remaining single-label eligible groups
to the existing, already-tested `materialize_group_split` for the two-stage
stratified split.

`src/nlp/splitting.py` is left unmodified: it remains the correct low-level
"no group crosses a partition" primitive, exercised directly by
`tests/test_phase2c_remediation_v4.py`. This module is purely additive.

Task -> conflict-category resolution
-------------------------------------
`configs/nlp_split_policy.yaml`'s `conflicting_label_policy.per_task` is keyed
by four semantic categories (SENTIMENT, SPEECH_ACT, OFFENSIVE_LANGUAGE_SAFETY,
REVIEW_RATING), not by experiment id or task_type string. The mapping below is
not an invented scientific rule: it is a direct, unambiguous restatement of the
`task_type` values already declared for each Batch 1 experiment in
`configs/nlp_metric_contract_v2.yaml` / `configs/nlp_label_ontology_v2.yaml`
(FIVE_CLASS_RATING_CLASSIFICATION is manifestly a REVIEW_RATING task,
FOUR_CLASS_SENTIMENT_CLASSIFICATION is manifestly a SENTIMENT task,
BINARY_OFFENSIVE_LANGUAGE_CLASSIFICATION is manifestly an
OFFENSIVE_LANGUAGE_SAFETY task). SPEECH_ACT has no Batch 1 mapping because no
active Batch 1 experiment is a speech-act task (D2 is BLOCKED).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .duplicate_control import normalized_exact_key, raw_exact_key
from .splitting import SplitAssignment, materialize_group_split

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_SPLIT_POLICY_PATH = CONFIG_ROOT / "nlp_split_policy.yaml"
DEFAULT_DUPLICATE_CONTRACT_PATH = CONFIG_ROOT / "nlp_duplicate_control_contract_v2.yaml"

EXPECTED_SPLIT_POLICY_SCHEMA = "nlp-split-policy-v2"
EXPECTED_DUPLICATE_CONTRACT_SCHEMA = "nlp-duplicate-control-contract-v2"

# Evidence-based restatement of each Batch 1 task_type as a split-policy
# conflict category; see module docstring.
TASK_TYPE_TO_CONFLICT_CATEGORY: Mapping[str, str] = {
    "FIVE_CLASS_RATING_CLASSIFICATION": "REVIEW_RATING",
    "FOUR_CLASS_SENTIMENT_CLASSIFICATION": "SENTIMENT",
    "BINARY_OFFENSIVE_LANGUAGE_CLASSIFICATION": "OFFENSIVE_LANGUAGE_SAFETY",
}

# The split policy's per-task conflict actions are free-text category policies;
# deterministic_materialization.conflicting_group_actions only recognizes three
# literal tokens. DO_NOT_AUTO_RESOLVE and DATASET_SPECIFIC_REVIEW both mean "no
# automatic resolution is authorized", which this implementation treats as
# fail-closed (FAIL) rather than silently picking a default.
CONFLICT_ACTION_TO_SPLIT_TOKEN: Mapping[str, str] = {
    "REMOVE_FROM_ALL_SPLITS": "REMOVE_FROM_ALL_SPLITS",
    "KEEP_IN_TRAIN_ONLY_WITH_FLAG": "KEEP_IN_TRAIN_ONLY_WITH_FLAG",
    "DO_NOT_AUTO_RESOLVE": "FAIL",
    "DATASET_SPECIFIC_REVIEW": "FAIL",
}

# duplicate_control_contract_v2 fields that must agree with split_policy's
# per_task resolution for the categories it explicitly names, as a
# contradiction check (see resolve_task_split_policy).
_DUPLICATE_CONTRACT_CATEGORY_FIELD = {
    "SENTIMENT": "sentiment_conflict_action",
    "SPEECH_ACT": "speech_act_conflict_action",
}
_DUPLICATE_CONTRACT_EQUIVALENCE = {
    "REMOVE_CONFLICTING_GROUP_FROM_ALL_SPLITS": "REMOVE_FROM_ALL_SPLITS",
    "BLOCKED_PENDING_DATASET_SPECIFIC_POLICY": "FAIL",
}

SAME_LABEL_ACTION_DEDUPLICATE_KEEP_ONE = "DEDUPLICATE_KEEP_ONE_PER_GROUP"


class SplitPolicyError(ValueError):
    """Raised for malformed, unsupported, or contradictory active-contract policy."""


@dataclass(frozen=True)
class TaskSplitPolicy:
    task_type: str
    conflict_category: str
    conflict_action: str
    resolved_conflict_action: str  # REMOVE_FROM_ALL_SPLITS | KEEP_IN_TRAIN_ONLY_WITH_FLAG | FAIL
    same_label_action: str
    split_policy_schema: str
    duplicate_contract_schema: str


@dataclass(frozen=True)
class PreparedRowAssignment:
    row_index: int
    group_key: str
    split: str | None  # None when excluded
    excluded: bool
    exclusion_reason: str | None
    flagged: bool
    flag_reason: str | None


@dataclass(frozen=True)
class SplitPreparationAudit:
    total_input_rows: int
    unique_normalized_groups: int
    same_label_duplicate_groups: int
    same_label_duplicate_rows_removed: int
    conflicting_label_groups: int
    conflicting_label_rows_removed: int
    conflicting_label_rows_flagged: int
    eligible_groups_after_preparation: int
    final_split_counts: Mapping[str, int]


@dataclass(frozen=True)
class SplitPreparationResult:
    policy: TaskSplitPolicy
    assignments: tuple[PreparedRowAssignment, ...]
    audit: SplitPreparationAudit


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SplitPolicyError(f"cannot read policy file {path}: {exc}") from exc
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise SplitPolicyError(f"policy file {path} did not parse to a mapping")
    return loaded


def resolve_task_split_policy(
    task_type: str, *,
    split_policy_path: Path = DEFAULT_SPLIT_POLICY_PATH,
    duplicate_contract_path: Path = DEFAULT_DUPLICATE_CONTRACT_PATH,
) -> TaskSplitPolicy:
    """Resolve duplicate/conflict behaviour for `task_type` from active contracts only.

    Fails closed (SplitPolicyError) on: an unsupported/unknown task_type, a
    missing or malformed contract, an unrecognized schema_version, an
    unrecognized conflict-action token, an unsupported same-label action, or a
    detected contradiction between the split policy and the duplicate-control
    contract.
    """
    if not isinstance(task_type, str) or not task_type:
        raise SplitPolicyError("task_type must be a non-empty string")

    split_policy = _load_yaml(split_policy_path)
    duplicate_contract = _load_yaml(duplicate_contract_path)

    if split_policy.get("schema_version") != EXPECTED_SPLIT_POLICY_SCHEMA:
        raise SplitPolicyError(
            f"unsupported split policy schema_version: {split_policy.get('schema_version')!r}"
        )
    if duplicate_contract.get("schema_version") != EXPECTED_DUPLICATE_CONTRACT_SCHEMA:
        raise SplitPolicyError(
            f"unsupported duplicate contract schema_version: {duplicate_contract.get('schema_version')!r}"
        )

    category = TASK_TYPE_TO_CONFLICT_CATEGORY.get(task_type)
    if category is None:
        raise SplitPolicyError(f"unsupported task_type for conflict-policy resolution: {task_type!r}")

    try:
        per_task = split_policy["conflicting_label_policy"]["per_task"]
        conflict_action = per_task[category]["policy"]
    except (KeyError, TypeError) as exc:
        raise SplitPolicyError(f"malformed split policy: missing per_task[{category!r}].policy") from exc

    resolved_conflict_action = CONFLICT_ACTION_TO_SPLIT_TOKEN.get(conflict_action)
    if resolved_conflict_action is None:
        raise SplitPolicyError(f"unsupported conflict action in active policy: {conflict_action!r}")

    try:
        same_label_action = duplicate_contract["split_requirements"]["same_label_action"]
    except (KeyError, TypeError) as exc:
        raise SplitPolicyError("malformed duplicate contract: missing split_requirements.same_label_action") from exc

    if same_label_action != SAME_LABEL_ACTION_DEDUPLICATE_KEEP_ONE:
        raise SplitPolicyError(f"unsupported same-label action in active contract: {same_label_action!r}")

    contract_field = _DUPLICATE_CONTRACT_CATEGORY_FIELD.get(category)
    if contract_field is not None:
        try:
            declared = duplicate_contract["split_requirements"][contract_field]
        except (KeyError, TypeError) as exc:
            raise SplitPolicyError(f"malformed duplicate contract: missing split_requirements.{contract_field}") from exc
        equivalent = _DUPLICATE_CONTRACT_EQUIVALENCE.get(declared)
        if equivalent is None:
            raise SplitPolicyError(f"unrecognized duplicate contract action: {declared!r}")
        if equivalent != resolved_conflict_action:
            raise SplitPolicyError(
                "contradictory active policy: "
                f"split_policy.conflicting_label_policy.per_task[{category!r}]={conflict_action!r} "
                f"resolves to {resolved_conflict_action!r}, but "
                f"duplicate_control_contract.split_requirements.{contract_field}={declared!r} "
                f"resolves to {equivalent!r}"
            )

    return TaskSplitPolicy(
        task_type=task_type,
        conflict_category=category,
        conflict_action=conflict_action,
        resolved_conflict_action=resolved_conflict_action,
        same_label_action=same_label_action,
        split_policy_schema=split_policy["schema_version"],
        duplicate_contract_schema=duplicate_contract["schema_version"],
    )


def _record_tiebreak_signature(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Content-only signature used to make survivor selection independent of input order."""
    return tuple(sorted((str(key), repr(value)) for key, value in record.items()))


def _select_survivor(records: Sequence[Mapping[str, Any]], indices: Sequence[int], text_key: str) -> int:
    def sort_key(index: int) -> tuple[bytes, tuple[tuple[str, str], ...], int]:
        raw = raw_exact_key(records[index].get(text_key))
        return (raw, _record_tiebreak_signature(records[index]), index)

    return sorted(indices, key=sort_key)[0]


def prepare_task_bound_split(
    records: Sequence[Mapping[str, Any]], *, text_key: str, label_key: str, task_type: str, seed: int,
    train_pct: int = 70, validation_pct: int = 15, test_pct: int = 15,
    min_class_count_for_stratification: int = 30,
    split_policy_path: Path = DEFAULT_SPLIT_POLICY_PATH,
    duplicate_contract_path: Path = DEFAULT_DUPLICATE_CONTRACT_PATH,
) -> SplitPreparationResult:
    """Resolve policy, deduplicate/handle conflicts, then materialize a group split.

    Deterministic under shuffled input: every grouping/selection decision is a
    pure function of record *content* (via NORMALIZED_EXACT_KEY, sorted key
    iteration, and a content-only survivor tiebreak), never of input position.
    """
    if not records:
        raise ValueError("records must not be empty")

    policy = resolve_task_split_policy(
        task_type, split_policy_path=split_policy_path, duplicate_contract_path=duplicate_contract_path,
    )

    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if text_key not in record or label_key not in record:
            raise ValueError(f"record {index} lacks required field {text_key!r} or {label_key!r}")
        key = normalized_exact_key(record[text_key])
        groups[key].append(index)

    assignments: dict[int, PreparedRowAssignment] = {}
    eligible_groups: list[tuple[str, int]] = []  # (group_key, survivor_original_index)
    fail_conflict_keys: list[str] = []

    same_label_duplicate_groups = 0
    same_label_duplicate_rows_removed = 0
    conflicting_label_groups = 0
    conflicting_label_rows_removed = 0
    conflicting_label_rows_flagged = 0

    for key in sorted(groups):
        member_indices = groups[key]
        labels_seen = {records[index][label_key] for index in member_indices}
        if len(labels_seen) > 1:
            conflicting_label_groups += 1
            if policy.resolved_conflict_action == "REMOVE_FROM_ALL_SPLITS":
                conflicting_label_rows_removed += len(member_indices)
                for index in member_indices:
                    assignments[index] = PreparedRowAssignment(
                        row_index=index, group_key=key, split=None, excluded=True,
                        exclusion_reason="CONFLICTING_LABEL_GROUP_REMOVED", flagged=False, flag_reason=None,
                    )
            elif policy.resolved_conflict_action == "KEEP_IN_TRAIN_ONLY_WITH_FLAG":
                conflicting_label_rows_flagged += len(member_indices)
                for index in member_indices:
                    assignments[index] = PreparedRowAssignment(
                        row_index=index, group_key=key, split="train", excluded=False,
                        exclusion_reason=None, flagged=True, flag_reason="KEEP_IN_TRAIN_ONLY_WITH_FLAG_CONFLICT",
                    )
            else:  # FAIL
                fail_conflict_keys.append(key)
            continue

        if len(member_indices) > 1:
            same_label_duplicate_groups += 1
            survivor = _select_survivor(records, member_indices, text_key)
            for index in member_indices:
                if index == survivor:
                    continue
                same_label_duplicate_rows_removed += 1
                assignments[index] = PreparedRowAssignment(
                    row_index=index, group_key=key, split=None, excluded=True,
                    exclusion_reason="SAME_LABEL_DUPLICATE_REMOVED", flagged=False, flag_reason=None,
                )
        else:
            survivor = member_indices[0]
        eligible_groups.append((key, survivor))

    if fail_conflict_keys:
        raise SplitPolicyError(
            "unresolved conflicting-label groups under a FAIL-closed policy "
            f"(task_type={task_type!r}, category={policy.conflict_category!r}, "
            f"conflict_action={policy.conflict_action!r}): {sorted(fail_conflict_keys)}"
        )

    if len(eligible_groups) < 4:
        raise ValueError(
            "at least four eligible (non-conflicting, deduplicated) groups are required after preparation; "
            f"got {len(eligible_groups)}"
        )

    prepared = [
        {"__group_key__": key, "__label__": records[survivor][label_key]}
        for key, survivor in eligible_groups
    ]
    split_assignments = materialize_group_split(
        prepared, group_key="__group_key__", label_key="__label__", seed=seed,
        train_pct=train_pct, validation_pct=validation_pct, test_pct=test_pct,
        min_class_count_for_stratification=min_class_count_for_stratification,
        conflicting_group_policy="FAIL",  # conflicts were already fully resolved above; any hit here is a bug
    )
    for split_assignment in split_assignments:
        key, survivor = eligible_groups[split_assignment.row_index]
        assignments[survivor] = PreparedRowAssignment(
            row_index=survivor, group_key=key, split=split_assignment.split, excluded=False,
            exclusion_reason=None, flagged=False, flag_reason=None,
        )

    final_split_counts = Counter(
        assignment.split for assignment in assignments.values() if not assignment.excluded
    )
    audit = SplitPreparationAudit(
        total_input_rows=len(records),
        unique_normalized_groups=len(groups),
        same_label_duplicate_groups=same_label_duplicate_groups,
        same_label_duplicate_rows_removed=same_label_duplicate_rows_removed,
        conflicting_label_groups=conflicting_label_groups,
        conflicting_label_rows_removed=conflicting_label_rows_removed,
        conflicting_label_rows_flagged=conflicting_label_rows_flagged,
        eligible_groups_after_preparation=len(eligible_groups),
        final_split_counts=dict(final_split_counts),
    )
    ordered_assignments = tuple(assignments[index] for index in sorted(assignments))
    return SplitPreparationResult(policy=policy, assignments=ordered_assignments, audit=audit)
