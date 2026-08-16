"""Explicit Amazon Reviews 2023 physical-to-canonical schema adapter."""
from __future__ import annotations

from collections.abc import Mapping

PHYSICAL_TO_CANONICAL = {
    "rating": "overall",
    "title": "review_title",
    "text": "review_text",
}
REQUIRED_PHYSICAL_FIELDS = frozenset(PHYSICAL_TO_CANONICAL)
CANONICAL_FIELDS = tuple(PHYSICAL_TO_CANONICAL.values())


def adapt_amazon_record(record: Mapping[str, object]) -> dict[str, object]:
    """Return canonical NLP fields, failing closed on schema ambiguity or absence."""
    missing = sorted(REQUIRED_PHYSICAL_FIELDS - record.keys())
    collisions = sorted(set(CANONICAL_FIELDS) & record.keys())
    if missing:
        raise ValueError(f"Amazon physical record missing required fields: {missing}")
    if collisions:
        raise ValueError(f"Amazon physical record unexpectedly contains canonical fields: {collisions}")
    return {canonical: record[physical] for physical, canonical in PHYSICAL_TO_CANONICAL.items()}
