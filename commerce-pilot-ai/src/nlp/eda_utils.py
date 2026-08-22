"""Reusable exploratory-analysis helpers for the Arabic-NLP dataset portfolio.

These functions back `notebooks/02_arabic_nlp_eda_and_analysis.ipynb`. They are
deliberately dataset-agnostic (ASTD / LABR / MPOLD / Jumia all call the same
functions with their own text/label columns) so the notebook never duplicates
analysis logic inline.

Two things this module intentionally does NOT do:
  1. Silently mutate text. `safe_normalize_arabic` always returns the
     normalized string alongside the untouched original; callers decide
     whether/when to use the normalized form.
  2. Merge label schemes across datasets. Each dataset's labels stay in their
     own space (sentiment vs. rating vs. offensive/not) -- see
     `docs/nlp_label_ontology_mapping.md` for why that is a hard rule here.

Normalization itself is not reimplemented here: `safe_normalize_arabic` wraps
the project's single canonical implementation in `src/nlp/text_normalization.py`
(nlp-text-normalization-contract-v2) rather than duplicating its rules.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from .duplicate_control import normalized_exact_key
from .text_normalization import normalize_text

__all__ = [
    "text_length_stats",
    "script_ratio",
    "detect_text_artifacts",
    "word_ngram_frequencies",
    "char_ngram_frequencies",
    "find_exact_duplicates",
    "find_near_duplicate_groups",
    "safe_normalize_arabic",
    "TfidfBaselineResult",
    "train_tfidf_baseline",
    "plot_class_distribution",
    "plot_length_histogram",
    "plot_confusion_matrix",
]

# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_MENTION_RE = re.compile(r"(?<!\w)@[\w_]+", re.UNICODE)
_HASHTAG_RE = re.compile(r"(?<!\w)#[\w_]+", re.UNICODE)
_DIACRITICS_RE = re.compile(r"[ً-ْٰۖ-ۭ]")
_TATWEEL_RE = re.compile(r"ـ")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]",
    flags=re.UNICODE,
)


def text_length_stats(texts: Iterable[object]) -> dict[str, float]:
    """Return char- and word-length summary statistics for a collection of texts.

    Non-string / missing values are coerced to "" (contribute a length of 0)
    rather than raising, so this can be run directly over a raw dataframe
    column that may contain NaN.
    """
    char_lens: list[int] = []
    word_lens: list[int] = []
    for value in texts:
        text = "" if value is None else str(value)
        if isinstance(value, float) and np.isnan(value):
            text = ""
        char_lens.append(len(text))
        word_lens.append(len(text.split()))
    char_arr = np.asarray(char_lens, dtype=float)
    word_arr = np.asarray(word_lens, dtype=float)

    def _summary(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "n": len(char_lens),
        "chars": _summary(char_arr),
        "words": _summary(word_arr),
    }


def script_ratio(text: object) -> dict[str, float]:
    """Return the Arabic / Latin / other character-share of one text string.

    Ratios are computed over non-whitespace characters and sum to ~1.0.
    Empty text returns all-zero ratios rather than dividing by zero.
    """
    value = "" if text is None else str(text)
    chars = [c for c in value if not c.isspace()]
    total = len(chars)
    if total == 0:
        return {"arabic_ratio": 0.0, "latin_ratio": 0.0, "other_ratio": 0.0, "n_chars": 0}
    arabic = sum(1 for c in chars if _ARABIC_RE.match(c))
    latin = sum(1 for c in chars if _LATIN_RE.match(c))
    other = total - arabic - latin
    return {
        "arabic_ratio": arabic / total,
        "latin_ratio": latin / total,
        "other_ratio": other / total,
        "n_chars": total,
    }


def detect_text_artifacts(text: object) -> dict[str, Any]:
    """Detect common social/web text artifacts: URLs, @mentions, #hashtags,
    emoji, Arabic diacritics (tashkeel), and tatweel (kashida, ـ).

    Returns counts (not just booleans) so callers can e.g. report "3.2 URLs
    per 1000 tweets" rather than only "X% of tweets contain a URL".
    """
    value = "" if text is None else str(text)
    return {
        "n_urls": len(_URL_RE.findall(value)),
        "n_mentions": len(_MENTION_RE.findall(value)),
        "n_hashtags": len(_HASHTAG_RE.findall(value)),
        "n_emoji": len(_EMOJI_RE.findall(value)),
        "n_diacritics": len(_DIACRITICS_RE.findall(value)),
        "n_tatweel": len(_TATWEEL_RE.findall(value)),
        "has_url": bool(_URL_RE.search(value)),
        "has_mention": bool(_MENTION_RE.search(value)),
        "has_hashtag": bool(_HASHTAG_RE.search(value)),
        "has_emoji": bool(_EMOJI_RE.search(value)),
        "has_diacritics": bool(_DIACRITICS_RE.search(value)),
        "has_tatweel": bool(_TATWEEL_RE.search(value)),
    }


# ---------------------------------------------------------------------------
# N-gram frequency counts
# ---------------------------------------------------------------------------

def word_ngram_frequencies(
    texts: Iterable[object], *, n: int = 1, top_k: int = 25
) -> list[tuple[str, int]]:
    """Most common whitespace-token n-grams across a text collection.

    A light non-destructive tokenization (whitespace split, no stemming or
    stopword removal) is used deliberately -- this is for descriptive EDA,
    not a modeling pipeline.
    """
    counter: Counter[str] = Counter()
    for value in texts:
        tokens = ("" if value is None else str(value)).split()
        for i in range(len(tokens) - n + 1):
            counter[" ".join(tokens[i : i + n])] += 1
    return counter.most_common(top_k)


def char_ngram_frequencies(
    texts: Iterable[object], *, n: int = 3, top_k: int = 25
) -> list[tuple[str, int]]:
    """Most common character n-grams across a text collection (whitespace kept)."""
    counter: Counter[str] = Counter()
    for value in texts:
        text = "" if value is None else str(value)
        for i in range(len(text) - n + 1):
            counter[text[i : i + n]] += 1
    return counter.most_common(top_k)


# ---------------------------------------------------------------------------
# Duplicate / near-duplicate detection
# ---------------------------------------------------------------------------

def find_exact_duplicates(texts: Sequence[object]) -> dict[str, Any]:
    """Group rows by the project's canonical NORMALIZED_EXACT_KEY
    (`src/nlp/duplicate_control.py`), i.e. the same key used by the official
    70/15/15 split (`src/nlp/splitting.py`) to keep duplicate text out of
    more than one split.

    Returns group sizes and the row-index groups with size > 1, so callers
    can report both a duplicate-row count and a "how many of those would
    collide across train/validation/test under the real split policy" count.
    """
    keys = [normalized_exact_key(t) for t in texts]
    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(keys):
        groups.setdefault(key, []).append(idx)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    n_dup_rows = sum(len(v) for v in dup_groups.values())
    return {
        "n_rows": len(texts),
        "n_unique_normalized": len(groups),
        "n_duplicate_groups": len(dup_groups),
        "n_rows_in_duplicate_groups": n_dup_rows,
        "duplicate_row_share": (n_dup_rows / len(texts)) if texts else 0.0,
        "example_groups": list(dup_groups.items())[:5],
    }


_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_PLACEHOLDER_RE = re.compile(r"\[URL\]|\[MENTION\]|\[REPEAT_PUNCT\]")


def _shingle_normalize(text: str) -> str:
    """Normalization used ONLY for near-duplicate shingle comparison.

    Deliberately lighter than `normalize_text`: punctuation differences
    (e.g. a trailing "!!!") should make two rows look MORE similar for
    near-dup purposes, not less -- but `normalize_text`'s [REPEAT_PUNCT]
    placeholder is 13 characters long and would dilute short texts' n-gram
    overlap. So punctuation/placeholders and whitespace are stripped
    entirely here, after reusing `normalize_text` for alef-folding and
    diacritic stripping. This function is not exposed for any other use.
    """
    text = normalize_text(text)
    text = _PLACEHOLDER_RE.sub("", text)
    text = _NON_WORD_RE.sub("", text)
    return text


def _char_shingles(text: str, n: int = 5) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def find_near_duplicate_groups(
    texts: Sequence[object],
    *,
    ngram_n: int = 5,
    jaccard_threshold: float = 0.85,
    max_bucket_size: int = 400,
) -> dict[str, Any]:
    """Heuristic near-duplicate detector, scalable to tens of thousands of rows.

    This is NOT an exhaustive all-pairs comparison (that is O(n^2) and
    infeasible for LABR's 63k rows). Instead, rows are bucketed by their
    first two whitespace tokens -- genuinely near-duplicate texts almost
    always share their opening words even when they diverge later (extra
    punctuation, a trailing word, etc.) -- and only within-bucket pairs are
    compared via character 5-gram Jaccard similarity. Buckets larger than
    `max_bucket_size` are sampled, not skipped, and this is reported so the
    result is honestly labeled as a sample-based estimate for large buckets.

    Returns a summary dict, not a full pairwise result set (this is for EDA
    reporting, not for materializing dedup decisions).
    """
    normalized = [normalize_text(t) for t in texts]
    shingle_texts = [_shingle_normalize(t) for t in texts]
    buckets: dict[tuple[str, ...], list[int]] = {}
    for idx, text in enumerate(normalized):
        tokens = tuple(text.split()[:2])
        buckets.setdefault(tokens, []).append(idx)

    near_dup_pairs: list[tuple[int, int, float]] = []
    sampled_buckets = 0
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        if len(indices) > max_bucket_size:
            indices = indices[:max_bucket_size]
            sampled_buckets += 1
        shingles = {i: _char_shingles(shingle_texts[i], ngram_n) for i in indices}
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                i, j = indices[a_pos], indices[b_pos]
                s_i, s_j = shingles[i], shingles[j]
                if not s_i or not s_j:
                    continue
                union = s_i | s_j
                jaccard = len(s_i & s_j) / len(union) if union else 0.0
                if jaccard >= jaccard_threshold:
                    near_dup_pairs.append((i, j, jaccard))

    rows_involved = {i for pair in near_dup_pairs for i in pair[:2]}
    return {
        "n_rows": len(texts),
        "n_buckets": len(buckets),
        "n_buckets_sampled_for_size": sampled_buckets,
        "n_near_duplicate_pairs": len(near_dup_pairs),
        "n_rows_involved": len(rows_involved),
        "near_duplicate_row_share": (len(rows_involved) / len(texts)) if texts else 0.0,
        "example_pairs": near_dup_pairs[:5],
        "method": (
            f"bucketed char-{ngram_n}gram Jaccard >= {jaccard_threshold}, "
            f"bucket = (len//10, first_word), sampled at {max_bucket_size}/bucket"
        ),
    }


# ---------------------------------------------------------------------------
# Safe normalization (never silent, never destructive of the original)
# ---------------------------------------------------------------------------

def safe_normalize_arabic(text: object) -> dict[str, str]:
    """Apply the project's canonical text-normalization contract (V2) and
    return BOTH the original and normalized text, plus whether anything
    changed. Never mutates in place and never discards the original --
    callers must explicitly choose to use `normalized` instead of `original`.

    The underlying rules (see `src/nlp/text_normalization.py`) are: NFC
    normalize, strip invisible/zero-width chars, collapse whitespace, strip
    tatweel, strip diacritics, fold alef variants (أ/إ/آ/ٱ -> ا), mask URLs
    and @mentions with placeholders, strip leading '#' from hashtags,
    collapse 3+ repeated punctuation runs, and lowercase Latin letters only.
    """
    original = "" if text is None else str(text)
    normalized = normalize_text(original)
    return {
        "original": original,
        "normalized": normalized,
        "changed": original != normalized,
    }


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression baseline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TfidfBaselineResult:
    """JSON-serializable result of a TF-IDF + LogisticRegression baseline run."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    n_train: int
    n_test: int
    n_features: int
    vectorizer_params: dict[str, Any] = field(default_factory=dict)
    classifier_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix,
            "class_labels": self.class_labels,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "vectorizer_params": self.vectorizer_params,
            "classifier_params": self.classifier_params,
        }


def train_tfidf_baseline(
    train_texts: Sequence[str],
    train_labels: Sequence[Any],
    test_texts: Sequence[str],
    test_labels: Sequence[Any],
    *,
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int | None = 50_000,
    C: float = 1.0,
    class_weight: str | None = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
) -> TfidfBaselineResult:
    """Train and evaluate a TF-IDF + multinomial LogisticRegression baseline.

    Dataset-agnostic by design: pass in whatever text/label columns a given
    dataset uses (4-class sentiment, binary offensive/not, 5-class rating,
    ...) and get back macro-F1, per-class F1, and a confusion matrix in a
    uniform shape. This is a fast, independent sanity-check baseline -- it is
    a separate, freshly-run classifier from the classical challengers already
    recorded in `reports/generated/nlp/challengers/`, not a replacement for them.
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=False,  # Arabic has no case; avoid mangling Latin fragments unexpectedly.
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
    )
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    class_labels = sorted({*train_labels, *test_labels}, key=str)
    per_class = f1_score(test_labels, preds, labels=class_labels, average=None, zero_division=0)
    cm = confusion_matrix(test_labels, preds, labels=class_labels)

    return TfidfBaselineResult(
        accuracy=float(accuracy_score(test_labels, preds)),
        balanced_accuracy=float(balanced_accuracy_score(test_labels, preds)),
        macro_f1=float(f1_score(test_labels, preds, average="macro", zero_division=0)),
        per_class_f1={str(lbl): float(score) for lbl, score in zip(class_labels, per_class)},
        confusion_matrix=cm.tolist(),
        class_labels=[str(lbl) for lbl in class_labels],
        n_train=len(train_texts),
        n_test=len(test_texts),
        n_features=X_train.shape[1],
        vectorizer_params={
            "analyzer": analyzer,
            "ngram_range": list(ngram_range),
            "min_df": min_df,
            "max_features": max_features,
        },
        classifier_params={
            "model": "LogisticRegression",
            "C": C,
            "class_weight": class_weight,
            "random_state": random_state,
        },
    )


def quick_stratified_split(
    texts: Sequence[str], labels: Sequence[Any], *, test_size: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str], list[Any], list[Any]]:
    """A simple stratified train/test split for fast, independent sanity-check
    baselines only. This is NOT the project's official group-aware 70/15/15
    split (`src/nlp/splitting.py` + `configs/nlp_split_policy.yaml`), which
    additionally dedupes by NORMALIZED_EXACT_KEY before splitting. Any number
    produced with this helper should be labeled as a quick sanity check, not
    compared 1:1 against the authoritative confirmed baselines.
    """
    return train_test_split(
        list(texts), list(labels), test_size=test_size, random_state=seed, stratify=list(labels)
    )


# ---------------------------------------------------------------------------
# Plotting helpers (consistent style across every dataset section)
# ---------------------------------------------------------------------------

_BAR_COLOR = "#2b6cb0"
_HIST_COLOR = "#2c7a7b"


def plot_class_distribution(
    label_counts: dict[str, int],
    *,
    title: str,
    xlabel: str = "Class",
    ylabel: str = "Number of rows",
    ax: "plt.Axes | None" = None,
) -> "plt.Axes":
    """Bar chart of real class counts, annotated with the count on each bar."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    bars = ax.bar(labels, counts, color=_BAR_COLOR)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, count in zip(bars, counts):
        ax.annotate(
            f"{count:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.figure.tight_layout()
    return ax


def plot_length_histogram(
    lengths: Sequence[float],
    *,
    title: str,
    xlabel: str = "Length (characters)",
    ylabel: str = "Number of rows",
    bins: int = 50,
    ax: "plt.Axes | None" = None,
) -> "plt.Axes":
    """Histogram of a length distribution (e.g. char or word counts per row)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(lengths, bins=bins, color=_HIST_COLOR, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.figure.tight_layout()
    return ax


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_labels: Sequence[str],
    *,
    title: str,
    ax: "plt.Axes | None" = None,
) -> "plt.Axes":
    """Annotated confusion-matrix heatmap: rows = true label, columns = predicted label."""
    cm_arr = np.asarray(cm)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, label="Number of rows")
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=12, fontweight="bold")
    max_val = cm_arr.max() if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            value = cm_arr[i, j]
            color = "white" if max_val and value > max_val / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)
    ax.figure.tight_layout()
    return ax
