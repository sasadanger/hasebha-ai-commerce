"""Deterministic exact duplicate keys; no fuzzy or model-derived matching."""
from __future__ import annotations

import hashlib
from .text_normalization import normalize_text


def raw_exact_key(value: object) -> bytes:
    """UTF-8 bytes of the unmodified text-field value."""
    if isinstance(value, bytes):
        return value
    return ("" if value is None else str(value)).encode("utf-8")


def normalized_exact_key(value: object) -> str:
    return normalize_text(value)


def stable_key_digest(key: bytes | str) -> str:
    data = key if isinstance(key, bytes) else key.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
