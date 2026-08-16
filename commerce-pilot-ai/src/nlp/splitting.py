"""Deterministic group-aware split materialization; no project data is loaded here."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import train_test_split

from .duplicate_control import normalized_exact_key


@dataclass(frozen=True)
class SplitAssignment:
    row_index: int
    group_key: str
    split: str


def materialize_text_group_split(
    records: Sequence[Mapping[str, Any]], *, text_key: str, label_key: str, seed: int,
    **kwargs: Any,
) -> list[SplitAssignment]:
    """Construct NORMALIZED_EXACT_KEY groups and materialize their assignments."""
    prepared = []
    for index, record in enumerate(records):
        if text_key not in record:
            raise ValueError(f"record {index} lacks {text_key!r}")
        prepared.append({**record, "NORMALIZED_EXACT_KEY": normalized_exact_key(record[text_key])})
    return materialize_group_split(
        prepared, group_key="NORMALIZED_EXACT_KEY", label_key=label_key, seed=seed, **kwargs,
    )


def materialize_group_split(
    records: Sequence[Mapping[str, Any]], *, group_key: str, label_key: str,
    seed: int, train_pct: int = 70, validation_pct: int = 15, test_pct: int = 15,
    min_class_count_for_stratification: int = 30,
    conflicting_group_policy: str = "FAIL",
) -> list[SplitAssignment]:
    """Split normalized-text groups in two exact, deterministic stages.

    Stage 1 retains 30% after selecting 70% train. Stage 2 divides that
    remainder at 15/30 = 0.5 validation and 15/30 = 0.5 internal test.
    """
    if not records:
        raise ValueError("records must not be empty")
    if (train_pct, validation_pct, test_pct) != (70, 15, 15):
        raise ValueError("V4 authorizes exactly 70/15/15")
    if not isinstance(seed, int):
        raise TypeError("seed must be int")
    groups: dict[str, list[int]] = defaultdict(list)
    group_labels: dict[str, set[Any]] = defaultdict(set)
    for index, record in enumerate(records):
        if group_key not in record or label_key not in record:
            raise ValueError(f"record {index} lacks {group_key!r} or {label_key!r}")
        key = str(record[group_key])
        groups[key].append(index)
        group_labels[key].add(record[label_key])

    forced_train: set[str] = set()
    eligible: list[str] = []
    labels: list[Any] = []
    for key in sorted(groups):
        values = group_labels[key]
        if len(values) != 1:
            if conflicting_group_policy == "KEEP_IN_TRAIN_ONLY_WITH_FLAG":
                forced_train.add(key)
                continue
            if conflicting_group_policy == "REMOVE_FROM_ALL_SPLITS":
                continue
            raise ValueError(f"conflicting labels in group {key!r}")
        eligible.append(key)
        labels.append(next(iter(values)))
    if len(eligible) < 4:
        raise ValueError("at least four eligible groups are required")

    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= min_class_count_for_stratification else None
    train_groups, remainder_groups = train_test_split(
        eligible, train_size=0.70, random_state=seed, shuffle=True, stratify=stratify,
    )
    remainder_label_map = {key: labels[eligible.index(key)] for key in remainder_groups}
    remainder_labels = [remainder_label_map[key] for key in remainder_groups]
    remainder_counts = Counter(remainder_labels)
    stage2_stratify = (
        remainder_labels
        if remainder_counts and min(remainder_counts.values()) >= 2
        else None
    )
    validation_groups, test_groups = train_test_split(
        remainder_groups, train_size=0.5, random_state=seed + 1,
        shuffle=True, stratify=stage2_stratify,
    )
    split_by_group = {key: "train" for key in train_groups}
    split_by_group.update({key: "train" for key in forced_train})
    split_by_group.update({key: "validation" for key in validation_groups})
    split_by_group.update({key: "internal_test" for key in test_groups})
    return [
        SplitAssignment(index, key, split_by_group[key])
        for key in sorted(split_by_group)
        for index in groups[key]
    ]
