"""V5 remediation tests for V4-IR-01..04. Synthetic/in-memory data only.

No project dataset file is read anywhere in this module. All records, texts,
and hashes below are fabricated for testing. No real NLP Batch 1 training,
TF-IDF fit on project data, estimator fit on project data, or internal-test
access occurs anywhere in this file.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.nlp.batch1_executor import (
    ALLOWED_EXPERIMENT_IDS,
    Batch1ExecutionError,
    Batch1ExecutorInputs,
    build_protected_internal_test_rows,
    execute_batch1_experiment,
)
from src.nlp.configuration import resolve_batch1_configurations
from src.nlp.release_control import (
    AUTHORIZATION_EVIDENCE_SCHEMA,
    ExternalAuthorizationEvidence,
    FrozenWinnerRecord,
    ProtectedInternalTestPartition,
    ProvenancedRow,
    ReleaseDenied,
    ReleaseEvaluationFailed,
    _safe_ledger_path,
    load_external_authorization,
)
from src.nlp.split_preparation import (
    SplitPolicyError,
    prepare_task_bound_split,
    resolve_task_split_policy,
)
from src.nlp.text_normalization import normalize_text
from src.nlp.winner_selection import (
    KnownConfigurationRegistry,
    WinnerSelectionRejected,
    select_validated_winner,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_POLICY_PATH = ROOT / "configs/nlp_split_policy.yaml"
DUPLICATE_CONTRACT_PATH = ROOT / "configs/nlp_duplicate_control_contract_v2.yaml"


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

def _rating_records(per_class=40, n_classes=2, seed=0):
    """Fabricated FIVE_CLASS_RATING_CLASSIFICATION-shaped rows (REVIEW_RATING policy)."""
    classes = [f"class_{i}" for i in range(n_classes)]
    records = []
    for label in classes:
        for i in range(per_class):
            records.append({"text": f"synthetic {label} review body number {i} filler words vary token{i % 7}", "rating": label})
    return records


def _write_synthetic_authorization_yaml(tmp_path, experiment_id="A", canonical_id="EXPERIMENT_A_synthetic",
                                          n_configs=4, family="TFIDF_LOGISTIC_REGRESSION"):
    configs = [
        {"configuration_id": f"cfg_{i}", "model_family": family, "vectorizer": {}, "classifier": {}, "seed": 42}
        for i in range(n_configs)
    ]
    data = {
        "execution_defaults": {
            "tfidf_vectorizer": {}, "logistic_regression": {}, "linear_svm": {}, "multinomial_naive_bayes": {},
        },
        "BATCH_1_CANDIDATE": [
            {"experiment_id": experiment_id, "canonical_id": canonical_id, "learned_configurations": configs}
        ],
    }
    path = tmp_path / "synthetic_authorization.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _metric_function(y_true, y_pred):
    # Test-fixture-only choice: nlp_metric_contract_v2.yaml does not pin a
    # zero-division policy, so this is not a production default (see V5
    # summary residual-ambiguity note).
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


class _SpyVectorizer:
    def __init__(self):
        self.fit_text = None
        self.transform_calls = []

    def fit_transform(self, text):
        self.fit_text = list(text)
        return list(range(len(self.fit_text)))

    def transform(self, text):
        call = list(text)
        self.transform_calls.append(call)
        return list(range(len(call)))


class _SpyEstimator:
    def __init__(self):
        self.fit_features = None
        self.fit_labels = None

    def fit(self, features, labels):
        self.fit_features = list(features)
        self.fit_labels = list(labels)
        return self

    def predict(self, features):
        return [self.fit_labels[0]] * len(features)


# ===========================================================================
# V4-IR-01: split_preparation
# ===========================================================================

def test_task_split_policy_resolves_from_active_contracts_not_caller_strings():
    policy = resolve_task_split_policy("FIVE_CLASS_RATING_CLASSIFICATION")
    assert policy.conflict_category == "REVIEW_RATING"
    assert policy.resolved_conflict_action == "KEEP_IN_TRAIN_ONLY_WITH_FLAG"
    assert policy.same_label_action == "DEDUPLICATE_KEEP_ONE_PER_GROUP"

    policy_sentiment = resolve_task_split_policy("FOUR_CLASS_SENTIMENT_CLASSIFICATION")
    assert policy_sentiment.conflict_category == "SENTIMENT"
    assert policy_sentiment.resolved_conflict_action == "REMOVE_FROM_ALL_SPLITS"

    policy_offensive = resolve_task_split_policy("BINARY_OFFENSIVE_LANGUAGE_CLASSIFICATION")
    assert policy_offensive.conflict_category == "OFFENSIVE_LANGUAGE_SAFETY"
    assert policy_offensive.resolved_conflict_action == "FAIL"


def test_unsupported_task_type_fails_closed():
    with pytest.raises(SplitPolicyError, match="unsupported task_type"):
        resolve_task_split_policy("SPEECH_ACT_CLASSIFICATION")


def test_malformed_split_policy_fails_closed(tmp_path):
    data = yaml.safe_load(SPLIT_POLICY_PATH.read_text(encoding="utf-8"))
    del data["conflicting_label_policy"]["per_task"]["REVIEW_RATING"]["policy"]
    path = tmp_path / "malformed_split_policy.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SplitPolicyError, match="malformed split policy"):
        resolve_task_split_policy(
            "FIVE_CLASS_RATING_CLASSIFICATION", split_policy_path=path, duplicate_contract_path=DUPLICATE_CONTRACT_PATH,
        )


def test_unsupported_conflict_action_fails_closed(tmp_path):
    data = yaml.safe_load(SPLIT_POLICY_PATH.read_text(encoding="utf-8"))
    data["conflicting_label_policy"]["per_task"]["REVIEW_RATING"]["policy"] = "MAJORITY_VOTE"
    path = tmp_path / "unsupported_action.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SplitPolicyError, match="unsupported conflict action"):
        resolve_task_split_policy(
            "FIVE_CLASS_RATING_CLASSIFICATION", split_policy_path=path, duplicate_contract_path=DUPLICATE_CONTRACT_PATH,
        )


def test_unsupported_same_label_action_fails_closed(tmp_path):
    data = yaml.safe_load(DUPLICATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    data["split_requirements"]["same_label_action"] = "KEEP_ALL_COPIES"
    path = tmp_path / "unsupported_dedup.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SplitPolicyError, match="unsupported same-label action"):
        resolve_task_split_policy(
            "FIVE_CLASS_RATING_CLASSIFICATION", split_policy_path=SPLIT_POLICY_PATH, duplicate_contract_path=path,
        )


def test_contradictory_active_policy_is_detected_and_stopped(tmp_path):
    data = yaml.safe_load(DUPLICATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    # split_policy says SENTIMENT -> REMOVE_FROM_ALL_SPLITS; contradict it here.
    data["split_requirements"]["sentiment_conflict_action"] = "BLOCKED_PENDING_DATASET_SPECIFIC_POLICY"
    path = tmp_path / "contradictory_dedup.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SplitPolicyError, match="contradictory active policy"):
        resolve_task_split_policy(
            "FOUR_CLASS_SENTIMENT_CLASSIFICATION", split_policy_path=SPLIT_POLICY_PATH, duplicate_contract_path=path,
        )


def test_unrecognized_schema_version_fails_closed(tmp_path):
    data = yaml.safe_load(SPLIT_POLICY_PATH.read_text(encoding="utf-8"))
    data["schema_version"] = "nlp-split-policy-v1"
    path = tmp_path / "wrong_schema.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(SplitPolicyError, match="unsupported split policy schema_version"):
        resolve_task_split_policy(
            "FIVE_CLASS_RATING_CLASSIFICATION", split_policy_path=path, duplicate_contract_path=DUPLICATE_CONTRACT_PATH,
        )


def test_same_label_duplicates_are_deterministically_deduplicated():
    records = _rating_records(per_class=40)
    records.extend([{"text": "  DUPLICATE!!!  ", "rating": "class_0"} for _ in range(5)])
    result = prepare_task_bound_split(
        records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=20260809,
    )
    assert result.audit.same_label_duplicate_groups == 1
    assert result.audit.same_label_duplicate_rows_removed == 4  # keep exactly one of the five
    group_members = [a for a in result.assignments if a.group_key == normalize_text("  DUPLICATE!!!  ")]
    survivors = [a for a in group_members if not a.excluded]
    assert len(group_members) == 5
    assert len(survivors) == 1
    excluded = [a for a in result.assignments if a.excluded and a.exclusion_reason == "SAME_LABEL_DUPLICATE_REMOVED"]
    assert len(excluded) == 4


def test_unequal_size_duplicate_groups_all_deduplicated_to_one():
    records = _rating_records(per_class=40)
    records.extend([{"text": "pair dup text", "rating": "class_0"} for _ in range(2)])
    records.extend([{"text": "quad dup text", "rating": "class_1"} for _ in range(4)])
    result = prepare_task_bound_split(
        records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=1,
    )
    assert result.audit.same_label_duplicate_groups == 2
    assert result.audit.same_label_duplicate_rows_removed == (2 - 1) + (4 - 1)


def test_conflicting_groups_remove_from_all_splits_for_sentiment_task():
    records = _rating_records(per_class=40)
    records.extend([
        {"text": "ambiguous sentiment text", "rating": "class_0"},
        {"text": "ambiguous sentiment text", "rating": "class_1"},
    ])
    result = prepare_task_bound_split(
        records, text_key="text", label_key="rating", task_type="FOUR_CLASS_SENTIMENT_CLASSIFICATION", seed=1,
    )
    assert result.audit.conflicting_label_groups == 1
    assert result.audit.conflicting_label_rows_removed == 2
    assert result.audit.conflicting_label_rows_flagged == 0
    conflict_key = normalize_text("ambiguous sentiment text")
    conflict_assignments = [a for a in result.assignments if a.group_key == conflict_key]
    assert len(conflict_assignments) == 2
    assert all(a.excluded and a.exclusion_reason == "CONFLICTING_LABEL_GROUP_REMOVED" for a in conflict_assignments)


def test_conflicting_groups_keep_in_train_only_with_flag_for_rating_task():
    records = _rating_records(per_class=40)
    records.extend([
        {"text": "ambiguous rating text", "rating": "class_0"},
        {"text": "ambiguous rating text", "rating": "class_1"},
    ])
    result = prepare_task_bound_split(
        records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=1,
    )
    assert result.audit.conflicting_label_rows_flagged == 2
    conflict_key = normalize_text("ambiguous rating text")
    conflict_assignments = [a for a in result.assignments if a.group_key == conflict_key]
    assert len(conflict_assignments) == 2
    assert all(a.split == "train" and a.flagged and a.flag_reason == "KEEP_IN_TRAIN_ONLY_WITH_FLAG_CONFLICT" for a in conflict_assignments)


def test_conflicting_groups_fail_closed_for_offensive_language_task():
    records = _rating_records(per_class=40)
    records.extend([
        {"text": "ambiguous safety text", "rating": "class_0"},
        {"text": "ambiguous safety text", "rating": "class_1"},
    ])
    with pytest.raises(SplitPolicyError, match="unresolved conflicting-label groups"):
        prepare_task_bound_split(
            records, text_key="text", label_key="rating", task_type="BINARY_OFFENSIVE_LANGUAGE_CLASSIFICATION", seed=1,
        )


def test_no_group_ever_crosses_a_partition():
    records = _rating_records(per_class=40)
    records.extend([{"text": "grouped duplicate text", "rating": "class_0"} for _ in range(3)])
    result = prepare_task_bound_split(
        records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=20260809,
    )
    by_group = {}
    for a in result.assignments:
        if a.excluded:
            continue
        by_group.setdefault(a.group_key, set()).add(a.split)
    assert all(len(splits) == 1 for splits in by_group.values())


def test_deterministic_repeated_execution():
    records = _rating_records(per_class=40)
    first = prepare_task_bound_split(records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=20260809)
    second = prepare_task_bound_split(records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=20260809)
    assert first.assignments == second.assignments
    assert first.audit == second.audit


def test_shuffled_input_is_deterministic_by_content():
    records = _rating_records(per_class=40)
    records.extend([{"text": "shuffle duplicate text", "rating": "class_0"} for _ in range(3)])
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)

    def _by_content(result, source_records):
        out = {}
        for a in result.assignments:
            record = source_records[a.row_index]
            key = (a.group_key, record["rating"])
            out.setdefault(key, []).append((str(a.split), a.excluded, str(a.exclusion_reason), a.flagged, str(a.flag_reason)))
        return {k: sorted(v) for k, v in out.items()}

    first = prepare_task_bound_split(records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=20260809)
    second = prepare_task_bound_split(shuffled, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=20260809)
    assert first.audit.same_label_duplicate_rows_removed == second.audit.same_label_duplicate_rows_removed
    assert first.audit.final_split_counts == second.audit.final_split_counts
    assert _by_content(first, records) == _by_content(second, shuffled)


def test_very_small_dataset_fails_closed():
    records = [
        {"text": "a1", "rating": "class_0"}, {"text": "a2", "rating": "class_0"},
        {"text": "b1", "rating": "class_1"},
    ]
    with pytest.raises(ValueError, match="at least four eligible"):
        prepare_task_bound_split(records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=1)


def test_empty_records_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        prepare_task_bound_split([], text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=1)


def test_imbalanced_group_case_is_still_deterministic_and_unstratified():
    records = _rating_records(per_class=40, n_classes=1)
    records.extend([{"text": f"minor {i}", "rating": "rare_class"} for i in range(4)])
    result = prepare_task_bound_split(records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=1)
    assert result.audit.eligible_groups_after_preparation == 44
    assert sum(result.audit.final_split_counts.values()) == 44


def test_missing_required_field_fails_closed():
    records = [{"text": "x", "rating": "a"}, {"rating": "a"}]
    with pytest.raises(ValueError, match="lacks required field"):
        prepare_task_bound_split(records, text_key="text", label_key="rating", task_type="FIVE_CLASS_RATING_CLASSIFICATION", seed=1)


# ===========================================================================
# V4-IR-04: winner_selection
# ===========================================================================

def _registry(execution_order_by_compound_id=None):
    return KnownConfigurationRegistry(
        experiment_id="A",
        compound_ids=frozenset({"A::cfg0", "A::cfg1", "A::cfg2"}),
        fingerprints_by_compound_id={"A::cfg0": "a" * 64, "A::cfg1": "b" * 64, "A::cfg2": "c" * 64},
        execution_order_by_compound_id=execution_order_by_compound_id or {"A::cfg0": 0, "A::cfg1": 1, "A::cfg2": 2},
    )


def _base_row(**overrides):
    row = {
        "experiment_id": "A", "compound_id": "A::cfg0", "configuration_fingerprint": "a" * 64,
        "split_hash": "d" * 64, "execution_order": 0, "metric_provenance": "NLP_DEV_VALIDATION",
        "macro_f1": 0.8, "balanced_accuracy": 0.7, "accuracy": 0.9,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metrics_rejected(bad_value):
    row = _base_row(macro_f1=bad_value)
    with pytest.raises(WinnerSelectionRejected, match="finite number"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


def test_missing_field_rejected():
    row = _base_row()
    del row["balanced_accuracy"]
    with pytest.raises(WinnerSelectionRejected, match="missing required fields"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


def test_malformed_numeric_value_rejected():
    row = _base_row(accuracy="0.9")
    with pytest.raises(WinnerSelectionRejected, match="finite number"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


def test_unknown_compound_id_rejected():
    row = _base_row(compound_id="A::not_in_registry")
    with pytest.raises(WinnerSelectionRejected, match="unknown compound_id"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


def test_wrong_configuration_fingerprint_rejected():
    row = _base_row(configuration_fingerprint="f" * 64)
    with pytest.raises(WinnerSelectionRejected, match="fingerprint mismatch"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


def test_missing_split_hash_rejected():
    row = _base_row()
    del row["split_hash"]
    with pytest.raises(WinnerSelectionRejected, match="missing required fields"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


def test_wrong_split_hash_rejected():
    row = _base_row(split_hash="e" * 64)
    with pytest.raises(WinnerSelectionRejected, match="split_hash does not match"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


@pytest.mark.parametrize("provenance", ["NLP_DEV_TRAIN", "NLP_INTERNAL_TEST", "NLP_EXTERNAL_TEST", "", None])
def test_non_validation_provenance_rejected(provenance):
    row = _base_row(metric_provenance=provenance)
    with pytest.raises(WinnerSelectionRejected, match="metric_provenance"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


def test_cross_experiment_results_rejected():
    rows = [_base_row(), _base_row(compound_id="A::cfg1", configuration_fingerprint="b" * 64, experiment_id="B2")]
    with pytest.raises(WinnerSelectionRejected, match="cross-experiment result"):
        select_validated_winner(rows, registry=_registry(), expected_split_hash="d" * 64)


def test_duplicate_result_identity_rejected():
    rows = [_base_row(execution_order=0), _base_row(execution_order=0)]
    with pytest.raises(WinnerSelectionRejected, match="duplicate result identity"):
        select_validated_winner(rows, registry=_registry(), expected_split_hash="d" * 64)


def test_execution_order_mismatch_against_registry_rejected():
    row = _base_row(compound_id="A::cfg1", configuration_fingerprint="b" * 64, execution_order=99)
    with pytest.raises(WinnerSelectionRejected, match="execution_order mismatch"):
        select_validated_winner([row], registry=_registry(), expected_split_hash="d" * 64)


def test_missing_results_rejected():
    with pytest.raises(WinnerSelectionRejected, match="must not be empty"):
        select_validated_winner([], registry=_registry(), expected_split_hash="d" * 64)


def test_exact_tie_breaking_order_preserved():
    registry = _registry({"A::cfg0": 2, "A::cfg1": 1, "A::cfg2": 0})
    rows = [
        _base_row(compound_id="A::cfg0", configuration_fingerprint="a" * 64, execution_order=2),
        _base_row(compound_id="A::cfg1", configuration_fingerprint="b" * 64, execution_order=1),
        _base_row(compound_id="A::cfg2", configuration_fingerprint="c" * 64, execution_order=0, balanced_accuracy=0.6),
    ]
    winner = select_validated_winner(rows, registry=registry, expected_split_hash="d" * 64)
    assert winner["compound_id"] == "A::cfg1"  # equal macro_f1/balanced_accuracy/accuracy -> lower execution_order wins


def test_higher_macro_f1_wins_regardless_of_other_metrics():
    registry = _registry({"A::cfg0": 5, "A::cfg1": 0, "A::cfg2": 2})
    rows = [
        _base_row(compound_id="A::cfg0", configuration_fingerprint="a" * 64, macro_f1=0.95, balanced_accuracy=0.1, accuracy=0.1, execution_order=5),
        _base_row(compound_id="A::cfg1", configuration_fingerprint="b" * 64, macro_f1=0.5, balanced_accuracy=0.99, accuracy=0.99, execution_order=0),
    ]
    winner = select_validated_winner(rows, registry=registry, expected_split_hash="d" * 64)
    assert winner["compound_id"] == "A::cfg0"


# ===========================================================================
# V4-IR-02: release_control
# ===========================================================================

def _write_evidence(path, **overrides):
    data = {
        "schema_version": AUTHORIZATION_EVIDENCE_SCHEMA, "authorized": True, "run_id": "run-1",
        "experiment_id": "A", "winning_compound_id": "A::cfg0", "winning_configuration_fingerprint": "a" * 64,
        "split_hash": "b" * 64, "reason": "synthetic test authorization", "metric_contract": "nlp-metric-contract-v2",
        "split_policy": "nlp-split-policy-v2",
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _partition():
    rows = tuple(
        ProvenancedRow(run_id="run-1", experiment_id="A", partition="internal_test", split_hash="b" * 64, payload=i)
        for i in range(5)
    )
    return ProtectedInternalTestPartition(rows)


def _winner():
    return FrozenWinnerRecord(run_id="run-1", experiment_id="A", compound_id="A::cfg0", configuration_fingerprint="a" * 64, split_hash="b" * 64)


def test_valid_release_succeeds_and_writes_success_ledger(tmp_path):
    evidence_path = _write_evidence(tmp_path / "evidence.json")
    evidence = load_external_authorization(evidence_path)
    partition = _partition()
    result = partition.release_once(
        evidence=evidence, winner=_winner(), evaluator=lambda rows: {"n": len(rows)},
        artifact_root=tmp_path / "artifacts",
    )
    assert result == {"n": 5}
    ledger = tmp_path / "artifacts" / "A" / "run-1" / "internal_test_release.json"
    event = json.loads(ledger.read_text(encoding="utf-8"))
    assert event["release_state"] == "RELEASED_ONCE_SUCCESS"
    with pytest.raises(ReleaseDenied, match="not accessible"):
        partition.peek_rows()


def test_self_asserted_authorization_object_is_rejected(tmp_path):
    # Even if a caller manages to hand-build an ExternalAuthorizationEvidence
    # (bypassing load_external_authorization), it must still fail cross-checks
    # unless it exactly matches the frozen winner and a genuine evidence file.
    fake = ExternalAuthorizationEvidence(
        schema_version=AUTHORIZATION_EVIDENCE_SCHEMA, authorized=True, run_id="run-1", experiment_id="A",
        winning_compound_id="A::cfg0", winning_configuration_fingerprint="a" * 64, split_hash="b" * 64,
        reason="self asserted", metric_contract="x", split_policy="y",
        evidence_path=str(tmp_path / "nonexistent.json"), evidence_sha256="0" * 64,
    )
    partition = _partition()
    with pytest.raises((ReleaseDenied, OSError)):
        partition.release_once(evidence=fake, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")


def test_authorized_false_denied(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json", authorized=False))
    with pytest.raises(ReleaseDenied, match="does not grant release"):
        _partition().release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")


def test_missing_authorization_file_denied(tmp_path):
    with pytest.raises(ReleaseDenied, match="unavailable"):
        load_external_authorization(tmp_path / "does_not_exist.json")


def test_malformed_authorization_json_denied(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ReleaseDenied, match="not valid JSON"):
        load_external_authorization(path)


def test_wrong_schema_version_denied(tmp_path):
    path = _write_evidence(tmp_path / "evidence.json", schema_version="nlp-internal-test-authorization-v0")
    with pytest.raises(ReleaseDenied, match="unsupported authorization evidence schema_version"):
        load_external_authorization(path)


@pytest.mark.parametrize("field", ["run_id", "winning_compound_id", "winning_configuration_fingerprint", "split_hash", "reason", "experiment_id"])
def test_empty_required_field_denied(tmp_path, field):
    path = _write_evidence(tmp_path / "evidence.json", **{field: ""})
    with pytest.raises(ReleaseDenied, match="non-empty string"):
        load_external_authorization(path)


def test_wrong_winner_denied(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json", winning_compound_id="A::other"))
    with pytest.raises(ReleaseDenied, match="winning_compound_id"):
        _partition().release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")


def test_wrong_fingerprint_denied(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json", winning_configuration_fingerprint="f" * 64))
    with pytest.raises(ReleaseDenied, match="fingerprint"):
        _partition().release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")


def test_wrong_split_hash_denied(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json", split_hash="e" * 64))
    with pytest.raises(ReleaseDenied, match="split_hash"):
        _partition().release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")


def test_wrong_experiment_denied(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json", experiment_id="B2"))
    with pytest.raises(ReleaseDenied, match="experiment_id"):
        _partition().release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")


def test_second_release_is_denied(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json"))
    partition = _partition()
    partition.release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")
    with pytest.raises(ReleaseDenied, match="one-time and already consumed"):
        partition.release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")


def test_second_release_denied_even_from_a_fresh_partition_instance_due_to_ledger(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json"))
    root = tmp_path / "artifacts"
    _partition().release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=root)
    fresh = _partition()
    with pytest.raises(ReleaseDenied, match="already exists"):
        fresh.release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=root)


def test_ledger_path_traversal_rejected_for_run_id(tmp_path):
    with pytest.raises(ReleaseDenied, match="unsafe run_id"):
        _safe_ledger_path("../evil", "A", tmp_path / "root")


def test_ledger_path_traversal_rejected_for_experiment_id(tmp_path):
    with pytest.raises(ReleaseDenied, match="unsafe experiment_id"):
        _safe_ledger_path("run-1", "../../evil", tmp_path / "root")


def test_winner_record_rejects_unsafe_run_id_at_construction():
    with pytest.raises(ReleaseDenied, match="run_id"):
        FrozenWinnerRecord(run_id="../evil", experiment_id="A", compound_id="A::cfg0", configuration_fingerprint="a" * 64, split_hash="b" * 64)


def test_winner_record_rejects_malformed_fingerprint():
    with pytest.raises(ReleaseDenied, match="fingerprint"):
        FrozenWinnerRecord(run_id="run-1", experiment_id="A", compound_id="A::cfg0", configuration_fingerprint="not-hex", split_hash="b" * 64)


def test_evaluator_exception_writes_failure_ledger_and_still_denies_further_release(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json"))
    partition = _partition()

    def bad_evaluator(rows):
        raise RuntimeError("synthetic evaluator failure")

    with pytest.raises(ReleaseEvaluationFailed, match="synthetic evaluator failure"):
        partition.release_once(evidence=evidence, winner=_winner(), evaluator=bad_evaluator, artifact_root=tmp_path / "artifacts")

    ledger = tmp_path / "artifacts" / "A" / "run-1" / "internal_test_release.json"
    event = json.loads(ledger.read_text(encoding="utf-8"))
    assert event["release_state"] == "RELEASE_FAILED_EVALUATOR_EXCEPTION"
    assert "synthetic evaluator failure" in event["error_summary"]

    with pytest.raises(ReleaseDenied, match="already consumed"):
        partition.release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")
    with pytest.raises(ReleaseDenied):
        partition.peek_rows()


def test_evaluator_row_retention_does_not_defeat_single_release_or_store_clearing(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json"))
    partition = _partition()
    leaked = {}

    def retaining_evaluator(rows):
        leaked["stolen_reference"] = rows  # adversarial: attempt to retain protected rows
        return "ok"

    result = partition.release_once(evidence=evidence, winner=_winner(), evaluator=retaining_evaluator, artifact_root=tmp_path / "artifacts")
    assert result == "ok"
    assert leaked["stolen_reference"] == tuple(range(5))  # documented residual limitation: the evaluator DID see the rows
    # what IS enforced regardless of evaluator behavior:
    with pytest.raises(ReleaseDenied, match="already consumed"):
        partition.release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")
    with pytest.raises(ReleaseDenied):
        partition.peek_rows()


def test_reservation_blocks_a_concurrent_second_release_before_final_ledger_write(tmp_path):
    evidence = load_external_authorization(_write_evidence(tmp_path / "evidence.json"))
    root = tmp_path / "artifacts"
    outer_partition = _partition()

    def evaluator_that_races(rows):
        # Simulate a concurrent second attempt for the same run/experiment
        # while the first release is still mid-evaluation (before the
        # `finally` block writes the final ledger state).
        racing_partition = _partition()
        with pytest.raises(ReleaseDenied, match="already exists"):
            racing_partition.release_once(evidence=evidence, winner=_winner(), evaluator=lambda r: r, artifact_root=root)
        return "primary result"

    result = outer_partition.release_once(evidence=evidence, winner=_winner(), evaluator=evaluator_that_races, artifact_root=root)
    assert result == "primary result"
    ledger = root / "A" / "run-1" / "internal_test_release.json"
    assert json.loads(ledger.read_text(encoding="utf-8"))["release_state"] == "RELEASED_ONCE_SUCCESS"


def test_authorization_file_tampered_after_load_is_detected(tmp_path):
    evidence_path = _write_evidence(tmp_path / "evidence.json")
    evidence = load_external_authorization(evidence_path)
    evidence_path.write_text(json.dumps({**json.loads(evidence_path.read_text()), "reason": "tampered"}), encoding="utf-8")
    with pytest.raises(ReleaseDenied, match="changed after it was loaded"):
        _partition().release_once(evidence=evidence, winner=_winner(), evaluator=lambda rows: rows, artifact_root=tmp_path / "artifacts")


def test_partition_requires_uniform_provenance():
    rows = (
        ProvenancedRow(run_id="run-1", experiment_id="A", partition="internal_test", split_hash="b" * 64, payload=1),
        ProvenancedRow(run_id="run-2", experiment_id="A", partition="internal_test", split_hash="b" * 64, payload=2),
    )
    with pytest.raises(ValueError, match="share exactly one"):
        ProtectedInternalTestPartition(rows)


def test_partition_requires_internal_test_partition_tag():
    rows = (ProvenancedRow(run_id="run-1", experiment_id="A", partition="train", split_hash="b" * 64, payload=1),)
    with pytest.raises(ValueError, match="partition='internal_test'"):
        ProtectedInternalTestPartition(rows)


def test_empty_partition_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        ProtectedInternalTestPartition(())


# ===========================================================================
# V4-IR-03: batch1_executor (synthetic end-to-end)
# ===========================================================================

def test_synthetic_end_to_end_execution_succeeds(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=2)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-deadbeef", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=20260809, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="synthetic-hash-value", expected_dataset_acquisition_sha256="synthetic-hash-value",
        output_root=tmp_path / "artifacts_root",
    )
    result = execute_batch1_experiment(inputs)
    assert result.run_id == inputs.run_id
    assert len(result.results) == 2
    for row in result.results:
        assert math.isfinite(row["macro_f1"])
        assert row["metric_provenance"] == "NLP_DEV_VALIDATION"
    assert result.winner.compound_id in {c["compound_id"] for c in configurations}
    run_dir = tmp_path / "artifacts_root" / "A" / inputs.run_id
    for name in ("resolved_config.json", "winner.json", "candidate_results.json", "run_manifest.json"):
        assert (run_dir / name).exists()


def test_executor_fits_train_only_and_evaluates_validation_only(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)

    created = []

    def factory(configuration):
        pair = (_SpyVectorizer(), _SpyEstimator())
        created.append(pair)
        return pair

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-cafebabe", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=20260809, resolved_configurations=configurations,
        estimator_factory=factory, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root",
    )
    result = execute_batch1_experiment(inputs)
    vectorizer, estimator = created[0]
    train_idx = {a.row_index for a in result.prep_result.assignments if a.split == "train"}
    validation_idx = {a.row_index for a in result.prep_result.assignments if a.split == "validation"}
    assert train_idx and validation_idx and train_idx.isdisjoint(validation_idx)
    expected_train_text = sorted(normalize_text(records[i]["text"]) for i in train_idx)
    expected_validation_text = sorted(normalize_text(records[i]["text"]) for i in validation_idx)
    assert sorted(vectorizer.fit_text) == expected_train_text
    assert len(vectorizer.transform_calls) == 1
    assert sorted(vectorizer.transform_calls[0]) == expected_validation_text
    assert sorted(estimator.fit_labels) == sorted(records[i]["rating"] for i in train_idx)


def test_optional_stages_must_remain_disabled(tmp_path):
    data = yaml.safe_load((ROOT / "configs/nlp_execution_contract_v4.yaml").read_text(encoding="utf-8"))
    data["optional_stages"]["calibration"] = "ENABLED"
    contract_path = tmp_path / "tampered_execution_contract.yaml"
    contract_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-11112222", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=1, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root", execution_contract_path=contract_path,
    )
    with pytest.raises(Batch1ExecutionError, match="calibration"):
        execute_batch1_experiment(inputs)
    assert not (tmp_path / "artifacts_root" / "A" / inputs.run_id).exists()


def test_unsupported_model_family_is_rejected(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    configurations[0] = {**configurations[0], "model_family": "BERT_TRANSFORMER"}
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-33334444", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=1, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root",
    )
    with pytest.raises(Batch1ExecutionError, match="unsupported model family"):
        execute_batch1_experiment(inputs)


def test_configuration_missing_required_field_is_rejected(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    del configurations[0]["vectorizer"]
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-55556666", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=1, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root",
    )
    with pytest.raises(Batch1ExecutionError, match="missing a required field"):
        execute_batch1_experiment(inputs)


def test_tampered_fingerprint_is_rejected(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    configurations[0] = {**configurations[0], "fingerprint_sha256": "0" * 64}
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-77778888", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=1, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root",
    )
    with pytest.raises(Batch1ExecutionError, match="fingerprint mismatch"):
        execute_batch1_experiment(inputs)


def test_wrong_dataset_provenance_is_rejected(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-99990000", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=1, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="actual-hash", expected_dataset_acquisition_sha256="expected-hash",
        output_root=tmp_path / "artifacts_root",
    )
    with pytest.raises(Batch1ExecutionError, match="provenance mismatch"):
        execute_batch1_experiment(inputs)
    assert not (tmp_path / "artifacts_root").exists()


def test_output_collision_is_rejected(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    def build_inputs():
        return Batch1ExecutorInputs(
            run_id="nlp-batch1-20260809T120000Z-aaaabbbb", experiment_id="A",
            task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
            text_key="text", label_key="rating", seed=1, resolved_configurations=configurations,
            estimator_factory=instantiate_configuration, metric_function=_metric_function,
            dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
            output_root=tmp_path / "artifacts_root",
        )

    execute_batch1_experiment(build_inputs())
    with pytest.raises(Batch1ExecutionError, match="already exists"):
        execute_batch1_experiment(build_inputs())


@pytest.mark.parametrize("bad_run_id", ["not-a-run-id", "nlp-batch1-2026-08-09-deadbeef", "../evil", ""])
def test_malformed_run_id_is_rejected(tmp_path, bad_run_id):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id=bad_run_id, experiment_id="A", task_type="FIVE_CLASS_RATING_CLASSIFICATION",
        canonical_records=records, text_key="text", label_key="rating", seed=1,
        resolved_configurations=configurations, estimator_factory=instantiate_configuration,
        metric_function=_metric_function, dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root",
    )
    with pytest.raises(Batch1ExecutionError, match="run_id"):
        execute_batch1_experiment(inputs)


def test_unsupported_experiment_id_is_rejected(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, experiment_id="B1", n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-ccccdddd", experiment_id="B1",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=1, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root",
    )
    with pytest.raises(Batch1ExecutionError, match="unsupported experiment_id"):
        execute_batch1_experiment(inputs)
    assert "B1" not in ALLOWED_EXPERIMENT_IDS


def test_repeated_synthetic_runs_are_deterministic(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=2)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    def run(run_id):
        inputs = Batch1ExecutorInputs(
            run_id=run_id, experiment_id="A", task_type="FIVE_CLASS_RATING_CLASSIFICATION",
            canonical_records=records, text_key="text", label_key="rating", seed=20260809,
            resolved_configurations=configurations, estimator_factory=instantiate_configuration,
            metric_function=_metric_function, dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
            output_root=tmp_path / "artifacts_root",
        )
        return execute_batch1_experiment(inputs)

    first = run("nlp-batch1-20260809T120000Z-11111111")
    second = run("nlp-batch1-20260809T130000Z-22222222")
    assert first.split_hash == second.split_hash
    assert first.winner.compound_id == second.winner.compound_id
    assert first.winner.configuration_fingerprint == second.winner.configuration_fingerprint
    first_by_id = {r["compound_id"]: {k: v for k, v in r.items()} for r in first.results}
    second_by_id = {r["compound_id"]: {k: v for k, v in r.items()} for r in second.results}
    assert first_by_id == second_by_id


# ===========================================================================
# Independent-review follow-up fixes: executor <-> release_control connector,
# and registry error-type hardening
# ===========================================================================

def test_build_protected_internal_test_rows_matches_prep_assignment_and_releases_successfully(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-eeeeffff", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=records,
        text_key="text", label_key="rating", seed=20260809, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root",
    )
    result = execute_batch1_experiment(inputs)

    internal_test_indices = {a.row_index for a in result.prep_result.assignments if a.split == "internal_test"}
    assert internal_test_indices  # sanity: this synthetic run does produce an internal_test partition

    rows = build_protected_internal_test_rows(inputs, result)
    assert len(rows) == len(internal_test_indices)
    expected_payloads = sorted((records[i]["text"], records[i]["rating"]) for i in internal_test_indices)
    actual_payloads = sorted((row.payload["text"], row.payload["rating"]) for row in rows)
    assert actual_payloads == expected_payloads
    assert all(row.run_id == inputs.run_id and row.experiment_id == "A" and row.partition == "internal_test" for row in rows)
    assert all(row.split_hash == result.split_hash for row in rows)

    partition = ProtectedInternalTestPartition(rows)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps({
        "schema_version": AUTHORIZATION_EVIDENCE_SCHEMA, "authorized": True, "run_id": inputs.run_id,
        "experiment_id": "A", "winning_compound_id": result.winner.compound_id,
        "winning_configuration_fingerprint": result.winner.configuration_fingerprint,
        "split_hash": result.split_hash, "reason": "synthetic end-to-end release test",
        "metric_contract": "nlp-metric-contract-v2", "split_policy": "nlp-split-policy-v2",
    }), encoding="utf-8")
    evidence = load_external_authorization(evidence_path)
    released = partition.release_once(
        evidence=evidence, winner=result.winner, evaluator=lambda payload_rows: len(payload_rows),
        artifact_root=tmp_path / "release_artifacts",
    )
    assert released == len(internal_test_indices)


def test_build_protected_internal_test_rows_rejects_mismatched_result(tmp_path):
    records = _rating_records(per_class=40)
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    from src.nlp.configuration import instantiate_configuration

    def build_inputs(run_id):
        return Batch1ExecutorInputs(
            run_id=run_id, experiment_id="A", task_type="FIVE_CLASS_RATING_CLASSIFICATION",
            canonical_records=records, text_key="text", label_key="rating", seed=20260809,
            resolved_configurations=configurations, estimator_factory=instantiate_configuration,
            metric_function=_metric_function, dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
            output_root=tmp_path / "artifacts_root",
        )

    real_inputs = build_inputs("nlp-batch1-20260809T120000Z-11112222")
    real_result = execute_batch1_experiment(real_inputs)
    other_inputs = build_inputs("nlp-batch1-20260809T130000Z-33334444")
    with pytest.raises(Batch1ExecutionError, match="does not match"):
        build_protected_internal_test_rows(other_inputs, real_result)


def test_registry_missing_execution_order_fails_closed_not_keyerror():
    configuration = {
        "experiment_id": "A", "compound_id": "A::cfg0", "fingerprint_sha256": "a" * 64,
        # execution_order intentionally omitted
    }
    with pytest.raises(WinnerSelectionRejected, match="missing a required field"):
        KnownConfigurationRegistry.from_resolved_configurations([configuration])


def test_schema_adapter_stage_is_actually_wired_for_amazon_shaped_input(tmp_path):
    # Fabricated Amazon-physical-schema records (rating/title/text), never real
    # project data, proving the executor's optional schema_adapter stage
    # actually runs adapt_amazon_record before normalization/splitting.
    from src.nlp.amazon_adapter import adapt_amazon_record
    from src.nlp.configuration import instantiate_configuration

    physical_records = []
    for label, adjective in (("class_0", "good"), ("class_1", "bad")):
        for i in range(40):
            physical_records.append({
                "rating": label, "title": f"{adjective} title {i}",
                "text": f"synthetic {adjective} amazon-shaped body {i} filler token{i % 7}",
            })

    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)

    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-a1a1a1a1", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=physical_records,
        text_key="review_text", label_key="overall", seed=20260809, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root", schema_adapter=adapt_amazon_record,
    )
    result = execute_batch1_experiment(inputs)
    assert result.winner.compound_id in {c["compound_id"] for c in configurations}

    # Internal-test payloads must be in the adapted (canonical) schema too.
    internal_test_indices = [a.row_index for a in result.prep_result.assignments if a.split == "internal_test"]
    assert internal_test_indices
    rows = build_protected_internal_test_rows(inputs, result)
    assert all("review_text" in row.payload and "overall" in row.payload for row in rows)
    assert all("rating" not in row.payload and "text" not in row.payload for row in rows)


def _raising_schema_adapter(record):
    raise ValueError("missing physical fields")


def test_missing_schema_adapter_field_fails_closed(tmp_path):
    from src.nlp.configuration import instantiate_configuration

    physical_records = _rating_records(per_class=40)  # has "text"/"rating", not "review_text"/"overall"
    auth_path = _write_synthetic_authorization_yaml(tmp_path, n_configs=1)
    configurations = resolve_batch1_configurations(auth_path)
    inputs = Batch1ExecutorInputs(
        run_id="nlp-batch1-20260809T120000Z-b2b2b2b2", experiment_id="A",
        task_type="FIVE_CLASS_RATING_CLASSIFICATION", canonical_records=physical_records,
        text_key="review_text", label_key="overall", seed=1, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=_metric_function,
        dataset_acquisition_sha256="h", expected_dataset_acquisition_sha256="h",
        output_root=tmp_path / "artifacts_root", schema_adapter=_raising_schema_adapter,
    )
    with pytest.raises(ValueError, match="missing physical fields"):
        execute_batch1_experiment(inputs)


def test_instantiate_configuration_actually_fits_with_real_yaml_ngram_range():
    # Regression test for a real-execution failure: TfidfVectorizer's fit-time
    # parameter validation requires ngram_range to be a literal tuple, but YAML
    # parses `[1, 2]` as a list. instantiate_configuration only *constructs*
    # the vectorizer/estimator -- sklearn's validation runs lazily inside
    # fit()/fit_transform(), not at construction -- so a construction-only
    # check (as in test_phase2c_remediation_v4.py's
    # test_all_20_configs_resolve_in_order_and_instantiate_without_fit) cannot
    # catch this; only an actual fit_transform call does, as this test does
    # with fabricated in-memory text (never real project data).
    from src.nlp.configuration import instantiate_configuration

    configurations = resolve_batch1_configurations(ROOT / "configs" / "nlp_training_batch_authorization_v2.yaml")
    bigram_configs = [c for c in configurations if c["vectorizer"].get("ngram_range") == [1, 2]]
    assert bigram_configs, "expected at least one real bigram configuration to exercise the fix"
    # Bigram configs require min_df=5, so at least 5 fabricated documents must
    # share a bigram for fit_transform to succeed; this is expected sklearn
    # behavior given too little data, not a defect.
    synthetic_documents = ["good product review", "bad product review", "good bad review"] * 3
    for configuration in bigram_configs:
        vectorizer, estimator = instantiate_configuration(configuration)
        assert isinstance(vectorizer.ngram_range, tuple)
        features = vectorizer.fit_transform(synthetic_documents)
        assert features.shape[0] == len(synthetic_documents)
