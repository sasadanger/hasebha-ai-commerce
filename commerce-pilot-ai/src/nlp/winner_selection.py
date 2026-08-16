"""Fail-closed, provenance-validated winner selection (V4-IR-04).

`execution_control.select_winner` (V4 evidence, left unmodified) accepts any
dict with the four bare fields it required; nothing prevented NaN/Infinity
metrics, an unknown compound_id, a mismatched configuration fingerprint, a
mismatched split hash, or a duplicate identity from winning. This module
supersedes it for any future real execution: every candidate result must pass
full schema, finiteness, and provenance validation before it is even eligible
to compete, and only then is the same tie-break order applied:
macro-F1 descending, balanced accuracy descending, accuracy descending,
execution order ascending.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_NUMERIC_FIELDS = ("macro_f1", "balanced_accuracy", "accuracy")
REQUIRED_STRING_FIELDS = ("experiment_id", "compound_id", "configuration_fingerprint", "split_hash")
REQUIRED_FIELDS = REQUIRED_STRING_FIELDS + REQUIRED_NUMERIC_FIELDS + ("execution_order", "metric_provenance")

# Only validation-derived metrics may compete for winner selection; see
# execution_contract_v4.yaml `development_execution.configuration_selection_partition:
# development_validation_only`. NLP_DEV_TRAIN / NLP_INTERNAL_TEST / NLP_EXTERNAL_TEST
# results must never be compared here.
VALID_METRIC_PROVENANCE = frozenset({"NLP_DEV_VALIDATION"})

_FINGERPRINT_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


class WinnerSelectionRejected(ValueError):
    """Raised whenever a candidate result fails validation; fail closed."""


@dataclass(frozen=True)
class KnownConfigurationRegistry:
    """The exact resolved configuration set a result is allowed to belong to."""
    experiment_id: str
    compound_ids: frozenset[str]
    fingerprints_by_compound_id: Mapping[str, str]
    execution_order_by_compound_id: Mapping[str, int]

    @staticmethod
    def from_resolved_configurations(configurations: Sequence[Mapping[str, Any]]) -> "KnownConfigurationRegistry":
        if not configurations:
            raise WinnerSelectionRejected("configuration registry must not be empty")
        try:
            experiment_ids = {c["experiment_id"] for c in configurations}
            fingerprints = {c["compound_id"]: c["fingerprint_sha256"] for c in configurations}
            execution_orders = {c["compound_id"]: c["execution_order"] for c in configurations}
        except KeyError as exc:
            raise WinnerSelectionRejected(f"configuration registry input is missing a required field: {exc}") from exc
        if len(experiment_ids) != 1:
            raise WinnerSelectionRejected("configuration registry must belong to exactly one experiment")
        if len(fingerprints) != len(configurations):
            raise WinnerSelectionRejected("duplicate compound_id in configuration registry")
        return KnownConfigurationRegistry(
            experiment_id=next(iter(experiment_ids)),
            compound_ids=frozenset(fingerprints),
            fingerprints_by_compound_id=fingerprints,
            execution_order_by_compound_id=execution_orders,
        )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_well_formed_hex_id(value: Any, expected_length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == expected_length
        and set(value.lower()) <= _HEX_DIGITS
    )


def _validate_result_schema(
    row: Mapping[str, Any], registry: KnownConfigurationRegistry, expected_split_hash: str,
) -> None:
    if not isinstance(row, Mapping):
        raise WinnerSelectionRejected(f"result must be a mapping, got {type(row).__name__}")
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise WinnerSelectionRejected(f"result missing required fields: {sorted(missing)}")

    for field in REQUIRED_STRING_FIELDS:
        if not isinstance(row[field], str) or not row[field]:
            raise WinnerSelectionRejected(f"result field {field!r} must be a non-empty string")

    for field in REQUIRED_NUMERIC_FIELDS:
        if not _is_finite_number(row[field]):
            raise WinnerSelectionRejected(f"result field {field!r} must be a finite number, got {row[field]!r}")

    execution_order = row["execution_order"]
    if isinstance(execution_order, bool) or not isinstance(execution_order, int) or execution_order < 0:
        raise WinnerSelectionRejected("execution_order must be a non-negative int")

    if row["metric_provenance"] not in VALID_METRIC_PROVENANCE:
        raise WinnerSelectionRejected(
            f"metric_provenance must be one of {sorted(VALID_METRIC_PROVENANCE)}, got {row['metric_provenance']!r}"
        )

    if row["experiment_id"] != registry.experiment_id:
        raise WinnerSelectionRejected(
            f"cross-experiment result: expected experiment_id {registry.experiment_id!r}, got {row['experiment_id']!r}"
        )

    if row["compound_id"] not in registry.compound_ids:
        raise WinnerSelectionRejected(f"unknown compound_id: {row['compound_id']!r}")

    expected_execution_order = registry.execution_order_by_compound_id[row["compound_id"]]
    if execution_order != expected_execution_order:
        raise WinnerSelectionRejected(
            f"execution_order mismatch for {row['compound_id']!r}: expected {expected_execution_order}, got {execution_order}"
        )

    if not _is_well_formed_hex_id(row["configuration_fingerprint"], _FINGERPRINT_LENGTH):
        raise WinnerSelectionRejected("configuration_fingerprint must be a 64-character hex string")
    expected_fingerprint = registry.fingerprints_by_compound_id[row["compound_id"]]
    if row["configuration_fingerprint"] != expected_fingerprint:
        raise WinnerSelectionRejected(f"configuration fingerprint mismatch for {row['compound_id']!r}")

    if not _is_well_formed_hex_id(row["split_hash"], _FINGERPRINT_LENGTH):
        raise WinnerSelectionRejected("split_hash must be a 64-character hex string")
    if row["split_hash"] != expected_split_hash:
        raise WinnerSelectionRejected("split_hash does not match the authorized split for this experiment")


def select_validated_winner(
    results: Sequence[Mapping[str, Any]], *, registry: KnownConfigurationRegistry, expected_split_hash: str,
) -> dict[str, Any]:
    """Validate every candidate result, then select the winner under a fixed tie order.

    Rejects (raising WinnerSelectionRejected): empty input, non-finite metrics,
    missing/malformed fields, unknown compound_id, mismatched configuration
    fingerprint, mismatched split hash, non-validation metric provenance,
    cross-experiment results, and duplicate result identities.
    """
    if not results:
        raise WinnerSelectionRejected("results must not be empty")
    if not _is_well_formed_hex_id(expected_split_hash, _FINGERPRINT_LENGTH):
        raise WinnerSelectionRejected("expected_split_hash must be a 64-character hex string")

    seen_compound_ids: set[str] = set()
    for row in results:
        _validate_result_schema(row, registry, expected_split_hash)
        if row["compound_id"] in seen_compound_ids:
            raise WinnerSelectionRejected(f"duplicate result identity: {row['compound_id']!r}")
        seen_compound_ids.add(row["compound_id"])

    winner = max(
        results,
        key=lambda row: (row["macro_f1"], row["balanced_accuracy"], row["accuracy"], -row["execution_order"]),
    )
    return dict(winner)
