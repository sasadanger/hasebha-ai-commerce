"""Gate 24: post-hoc error analysis on the frozen MARBERT primary model's test_natural
predictions. >=100 errors, stratified by true class, tagged with heuristic pattern flags. Post-hoc
reporting only -- no retraining happens after this (predictions were already frozen at Gate 21).

Run (after arabic_foundation_final_eval.py has produced test_natural predictions):
  .venv/Scripts/python.exe scripts/arabic_foundation_error_analysis.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nlp.arabic_foundation import transformer as af_tf  # noqa: E402
from nlp.arabic_foundation.normalization import LABEL_NAMES_3CLASS  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
PRED_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "primary_model" / "predictions"
SEED = 42

NEGATION_WORDS = ["مش", "مو", "لا ", "ليس", "ماكانش", "مافيش", "ولا", "بدون"]
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[؀-ۿ]")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")


def tag_row(text: str, rating: int, true_label: int, pred_label: int) -> list:
    tags = []
    has_ar = bool(ARABIC_RE.search(text))
    has_la = bool(LATIN_RE.search(text))
    if has_ar and has_la:
        tags.append("english_code_switch")
    if not has_ar and has_la:
        tags.append("arabizi_or_latin_only")
    if EMOJI_RE.search(text):
        tags.append("emoji_present")
    if REPEAT_CHAR_RE.search(text):
        tags.append("letter_repetition")
    if any(w in text for w in NEGATION_WORDS):
        tags.append("negation_present")
    n_tokens = len(text.split())
    if n_tokens <= 3:
        tags.append("very_short")
    if n_tokens >= 120:
        tags.append("long_possibly_truncated")
    if true_label == 1:
        tags.append("neutral_mixed_true_class")
    if rating == 3 and true_label == 1:
        tags.append("rating3_text_ambiguity_candidate")
    if abs(true_label - pred_label) == 2:
        tags.append("extreme_miss_negative_vs_positive")
    if not tags:
        tags.append("no_heuristic_pattern_matched")
    return tags


def main() -> None:
    preds = pd.read_parquet(PRED_DIR / "test_natural_predictions.parquet")
    test_split = af_tf.load_split("test_natural")
    merged = preds.merge(test_split[["review_uid", "text", "rating"]], on="review_uid", validate="one_to_one")

    errors = merged[merged["true_label"] != merged["pred_label"]].copy()
    print(f"Total test_natural rows: {len(merged)}, errors: {len(errors)} ({len(errors)/len(merged)*100:.1f}%)")

    target_n = min(150, len(errors))
    # stratify by true class
    parts = []
    per_class_n = max(1, target_n // 3)
    for cls in [0, 1, 2]:
        cls_errors = errors[errors["true_label"] == cls]
        n = min(per_class_n, len(cls_errors))
        if n > 0:
            parts.append(cls_errors.sample(n=n, random_state=SEED))
    sample = pd.concat(parts, axis=0) if parts else errors.head(0)
    # top up to target_n if some classes were short
    if len(sample) < target_n:
        remaining = errors.drop(sample.index)
        extra_n = min(target_n - len(sample), len(remaining))
        if extra_n > 0:
            sample = pd.concat([sample, remaining.sample(n=extra_n, random_state=SEED)], axis=0)

    sample["tags"] = sample.apply(
        lambda r: tag_row(r["text"], r["rating"], int(r["true_label"]), int(r["pred_label"])), axis=1
    )
    sample["true_label_name"] = sample["true_label"].map(LABEL_NAMES_3CLASS)
    sample["pred_label_name"] = sample["pred_label"].map(LABEL_NAMES_3CLASS)

    tag_counts: dict = {}
    for tags in sample["tags"]:
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    by_true_class = {
        LABEL_NAMES_3CLASS[c]: int((sample["true_label"] == c).sum()) for c in [0, 1, 2]
    }

    out_records = sample[[
        "review_uid", "rating", "true_label", "true_label_name", "pred_label", "pred_label_name",
        "proba_negative", "proba_neutral_mixed", "proba_positive", "tags", "text",
    ]].to_dict(orient="records")

    report = {
        "total_test_rows": int(len(merged)),
        "total_errors": int(len(errors)),
        "error_rate": float(len(errors) / len(merged)),
        "n_sampled_for_analysis": int(len(sample)),
        "sample_stratified_by_true_class": by_true_class,
        "tag_frequency_in_sample": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])),
        "note": "Post-hoc reporting only (Gate 24) -- no retraining or threshold changes follow from this analysis. Tags are heuristic pattern flags (regex-based), not human-verified labels.",
        "sample": out_records,
    }
    (REPORTS_DIR / "error_analysis.json").write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote error_analysis.json with {len(sample)} tagged error examples")
    print("Tag frequency:", json.dumps(dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])), indent=2))


if __name__ == "__main__":
    main()
