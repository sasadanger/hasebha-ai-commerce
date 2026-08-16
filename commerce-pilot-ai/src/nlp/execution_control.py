"""Leakage-safe development boundaries and one-time internal-test release."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class InternalTestAccessDenied(PermissionError):
    pass


@dataclass(frozen=True)
class ReleaseAuthorization:
    authorized: bool
    run_id: str
    winning_compound_id: str
    winning_configuration_fingerprint: str
    reason: str
    metric_contract: str
    split_policy: str


class PartitionStore:
    def __init__(self, train: list[Any], validation: list[Any], internal_test: list[Any]):
        self._train = tuple(train)
        self._validation = tuple(validation)
        self._internal_test = tuple(internal_test)
        self._released = False

    def development_partitions(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        return self._train, self._validation

    def internal_test(self) -> tuple[Any, ...]:
        raise InternalTestAccessDenied("internal test is unavailable before an audited release")

    def evaluate_internal_test_once(
        self, authorization: ReleaseAuthorization, ledger_path: Path,
        evaluator: Callable[[tuple[Any, ...]], Any],
    ) -> Any:
        """Release rows only to one evaluator call; never return the rows themselves."""
        if not authorization.authorized:
            raise InternalTestAccessDenied("explicit independent authorization is required")
        if self._released or ledger_path.exists():
            raise InternalTestAccessDenied("internal test release is one-time and already recorded")
        event = {
            "schema_version": "nlp-internal-test-release-v1",
            **asdict(authorization),
            "release_state": "RELEASED_ONCE",
            "released_utc": datetime.now(timezone.utc).isoformat(),
        }
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._released = True
        result = evaluator(self._internal_test)
        self._internal_test = ()
        return result


def fit_train_only_transform(
    store: PartitionStore, transformer: Any,
    text_getter: Callable[[Any], str] = str,
) -> tuple[Any, Any]:
    """Fit only on train; transform validation separately; internal test is unreachable."""
    train, validation = store.development_partitions()
    train_text = [text_getter(row) for row in train]
    validation_text = [text_getter(row) for row in validation]
    train_features = transformer.fit_transform(train_text)
    validation_features = transformer.transform(validation_text)
    return train_features, validation_features


def fit_model_train_only(estimator: Any, train_features: Any, train_labels: list[Any]) -> Any:
    """Fit a candidate only on explicitly supplied train features and labels."""
    if len(train_features) != len(train_labels):
        raise ValueError("train feature/label length mismatch")
    estimator.fit(train_features, train_labels)
    return estimator


def select_winner(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select per experiment: macro-F1, balanced accuracy, accuracy, then order."""
    if not results:
        raise ValueError("results must not be empty")
    experiment_ids = {row["experiment_id"] for row in results}
    if len(experiment_ids) != 1:
        raise ValueError("winner selection cannot compare different experiments")
    required = {"macro_f1", "balanced_accuracy", "accuracy", "execution_order"}
    for row in results:
        if not required <= row.keys():
            raise ValueError(f"result missing fields: {sorted(required - row.keys())}")
    return max(results, key=lambda row: (
        row["macro_f1"], row["balanced_accuracy"], row["accuracy"], -row["execution_order"],
    ))
