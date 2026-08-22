"""Light Arabic text normalization for the sentiment foundation task.

Design principle (per task brief): LIGHT normalization only. We deliberately preserve
punctuation, emoji, negation particles, English/code-switched words, spelling variation,
letter-repetition (elongation used for emphasis), and Arabizi (Arabic written in Latin script).
We do NOT stem, lemmatize, remove stopwords, or aggressively fold dialectal forms into MSA.

Two independent, documented (not assumed) decisions:

1. Tatweel (kashida, U+0640) removal: ON by default. Tatweel is a purely decorative elongation
   character with no lexical/semantic content (e.g. inserted for justification/calligraphy);
   stripping it cannot change meaning and is considered "light" cleanup, not dialectal
   normalization.

2. Diacritics (tashkeel) removal: OFF by default. Justification is DATA-DRIVEN, not assumed --
   see reports/generated/arabic_foundation/diacritics_pilot.json, produced by
   scripts/arabic_foundation_build_splits.py, which measures the fraction of Arabic-script
   characters in a LABR sample that are combining diacritics. LABR is casual/MSA book-review
   text where diacritics are rare; if the measured rate is negligible (<0.5%), removal is
   skipped as unnecessary rather than applied automatically. If a future corpus shows a
   materially higher rate, this default should be revisited with an actual pilot comparison,
   not flipped blindly.

3. Alef/Ya/Ta-Marbuta unification: OFF by default (not applied anywhere in this module). Per
   instruction this is only to be applied if a small pilot shows evidence for it; no such pilot
   was run for LABR because the light-normalization-only default already performs adequately for
   the primary task (see Gate 12 loss pilot and Gate 15 training results) -- documented as a
   deliberate non-default, not an oversight.
"""
from __future__ import annotations

import html
import re
import unicodedata

TATWEEL = "ـ"
ARABIC_DIACRITICS_RE = re.compile(
    "[ؐ-ًؚ-ٟۖ-ۜ۟-ۤۧ-۪ۨ-ۭࣣ-ࣿ]"
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
# Strip C0/DEL control chars (built from codepoints, not literal bytes, so the source file never
# embeds an actual NUL/control byte); \t \n \r are left alone since WHITESPACE_RE collapses them.
_CONTROL_CODEPOINTS = [c for c in range(0x00, 0x20) if chr(c) not in ("\t", "\n", "\r")] + [0x7F]
CONTROL_CHAR_RE = re.compile("[" + "".join(re.escape(chr(c)) for c in _CONTROL_CODEPOINTS) + "]")


def diacritic_fraction(text: str) -> float:
    """Fraction of characters in `text` that are Arabic combining diacritics -- used to
    data-drive the diacritics-removal decision rather than assuming it."""
    if not text:
        return 0.0
    n_diac = len(ARABIC_DIACRITICS_RE.findall(text))
    return n_diac / max(len(text), 1)


def normalize_text(text: str, remove_tatweel: bool = True, remove_diacritics: bool = False) -> str:
    """Light, documented normalization. Preserves punctuation/emoji/negation/English/Arabizi/
    repetition/spelling-variation. See module docstring for the tatweel/diacritics defaults and
    their justification.
    """
    if text is None:
        return ""
    t = str(text)
    t = html.unescape(t)
    t = HTML_TAG_RE.sub(" ", t)
    t = unicodedata.normalize("NFKC", t)
    t = CONTROL_CHAR_RE.sub(" ", t)
    if remove_tatweel:
        t = t.replace(TATWEEL, "")
    if remove_diacritics:
        t = ARABIC_DIACRITICS_RE.sub("", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def is_empty_or_whitespace(text: str) -> bool:
    return text is None or str(text).strip() == ""


LABR_RATING_TO_3CLASS = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
LABEL_NAMES_3CLASS = {0: "Negative", 1: "Neutral/Mixed", 2: "Positive"}
LABEL_NAME_TO_ID_3CLASS = {v: k for k, v in LABEL_NAMES_3CLASS.items()}


def labr_rating_to_3class(rating: int) -> int:
    """Primary label contract (Gate 4): LABR 1-2 -> Negative(0), 3 -> Neutral/Mixed(1), 4-5 ->
    Positive(2). The Neutral/Mixed class name is kept deliberately verbose to flag the documented
    rating-text ambiguity caveat: a 3-star LABR review may reflect genuinely mixed sentiment, or a
    lukewarm-but-directionally-positive/negative opinion that the rating scale compresses -- this
    is a known property of star-rating-derived sentiment labels, not a modeling defect."""
    if rating not in LABR_RATING_TO_3CLASS:
        raise ValueError(f"rating must be an integer in [1,5], got {rating!r}")
    return LABR_RATING_TO_3CLASS[rating]


def astd_label_to_3class(label: str):
    """Gate 6 ASTD robustness-eval mapping. Returns None for OBJ (objective) rows, which must be
    EXCLUDED from the 3-class sentiment robustness eval (documented choice, see astd_audit.json)."""
    m = {"POS": 2, "NEG": 0, "NEUTRAL": 1, "MIXED": 1}
    key = str(label).strip().upper()
    if key == "OBJ":
        return None
    if key not in m:
        raise ValueError(f"unrecognized ASTD label: {label!r}")
    return m[key]


def arsas_label_to_3class(label: str) -> int:
    """Gate 8 ArSAS mapping: Negative->0, Neutral->1, Positive->2, Mixed->1 (Neutral/Mixed)."""
    m = {"Negative": 0, "Neutral": 1, "Positive": 2, "Mixed": 1}
    key = str(label).strip()
    if key not in m:
        raise ValueError(f"unrecognized ArSAS label: {label!r}")
    return m[key]
