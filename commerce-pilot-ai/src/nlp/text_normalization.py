"""Canonical implementation of nlp-text-normalization-contract-v2."""
from __future__ import annotations

import re
import unicodedata

INVISIBLES = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\ufeff"), None)
DIACRITICS = re.compile(r"[\u064b-\u0652\u0670\u06d6-\u06ed]")
ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
MENTION = re.compile(r"(?<!\w)@[\w_]+", re.UNICODE)
WHITESPACE = re.compile(r"\s+", re.UNICODE)


def _coerce_text(value: object) -> str:
    return "" if value is None else str(value)


def _strip_invisibles_and_collapse_whitespace(value: object) -> str:
    """Apply only the V2 minimal emptiness-normalization stage."""
    text = unicodedata.normalize("NFC", _coerce_text(value))
    text = text.translate(INVISIBLES)
    return WHITESPACE.sub(" ", text).strip()


def _lowercase_latin_only(text: str) -> str:
    """Locale-independent lowercase of Unicode characters named LATIN only."""
    return "".join(
        char.lower()
        if "LATIN" in unicodedata.name(char, "") and unicodedata.category(char) in {"Lu", "Lt"}
        else char
        for char in text
    )


def _normalize_repeated_punctuation(text: str) -> str:
    """Replace every 3+ identical Unicode P* run without a regex prefilter."""
    output: list[str] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and text[end] == text[index]:
            end += 1
        run = text[index:end]
        if len(run) >= 3 and unicodedata.category(text[index]).startswith("P"):
            output.append("[REPEAT_PUNCT]")
        else:
            output.append(run)
        index = end
    return "".join(output)


def normalize_text(value: object) -> str:
    """Return the contract's NORMALIZED value; RAW remains caller-owned."""
    text = _strip_invisibles_and_collapse_whitespace(value)
    text = text.replace("ـ", "")
    text = DIACRITICS.sub("", text)
    text = text.translate(ALEF_VARIANTS)
    text = URL.sub("[URL]", text)
    text = MENTION.sub("[MENTION]", text)
    text = re.sub(r"#(?=\w)", "", text)
    text = _normalize_repeated_punctuation(text)
    text = _lowercase_latin_only(text)
    # Contract names these placeholders in uppercase. Preserve those literals
    # despite the later general Latin-lowercasing step.
    return text.replace("[url]", "[URL]").replace("[mention]", "[MENTION]").replace("[repeat_punct]", "[REPEAT_PUNCT]")


def is_empty_text(value: object) -> bool:
    """V2 EMPTY rule: minimal normalization only, separate from dedup."""
    return _strip_invisibles_and_collapse_whitespace(value) == ""
