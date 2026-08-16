"""Small, evidence-backed classical challengers (char n-gram TF-IDF + LinearSVC).

New experiments only -- Batch 1's frozen artifacts under
artifacts/experiments/nlp/phase2c/batch1/ are never read for writing, never
modified. Uses the same deterministic train/validation split each Batch 1
winner was selected on (same seed, same split_preparation call). Train-only
fit, validation-only evaluation. internal_test is never touched.

Not a grid search: one char-only variant, then (if useful) one word+char
combined variant, per experiment, matching the challenger plan in
docs/nlp_model_improvement_challenger_plan_v1.md.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.split_preparation import prepare_task_bound_split
from src.nlp.text_normalization import normalize_text
from scripts.run_nlp_batch1_real import EXPERIMENT_SPECS

CHALLENGER_ROOT = ROOT / "reports" / "generated" / "nlp" / "challengers"

CHAR_NGRAM_RANGE = {"E": (2, 5), "B2": (3, 5), "C": (3, 5)}
WORD_NGRAM_RANGE = {"E": (1, 1), "B2": (1, 2), "C": (1, 3)}
CLASS_WEIGHT = {"E": "balanced", "B2": None, "C": None}  # E: known ~83/17 imbalance from Batch 1 baseline


def _metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def run_challengers(experiment_id: str) -> dict:
    spec = EXPERIMENT_SPECS[experiment_id]
    print(f"[{experiment_id}] loading real data...", flush=True)
    records, actual_sha256, expected_sha256 = spec["loader"]()
    if actual_sha256 != expected_sha256:
        raise SystemExit(f"[{experiment_id}] acquisition hash mismatch")

    adapted = [spec["schema_adapter"](r) if spec["schema_adapter"] else r for r in records]
    normalized = [{**r, "__normalized_text__": normalize_text(r[spec["text_key"]])} for r in adapted]

    print(f"[{experiment_id}] re-deriving the same deterministic split...", flush=True)
    prep = prepare_task_bound_split(
        normalized, text_key=spec["text_key"], label_key=spec["label_key"],
        task_type=spec["task_type"], seed=20260809,
    )
    train_rows = [normalized[a.row_index] for a in prep.assignments if a.split == "train"]
    validation_rows = [normalized[a.row_index] for a in prep.assignments if a.split == "validation"]
    train_text = [r["__normalized_text__"] for r in train_rows]
    validation_text = [r["__normalized_text__"] for r in validation_rows]
    train_labels = [r[spec["label_key"]] for r in train_rows]
    validation_labels = [r[spec["label_key"]] for r in validation_rows]

    class_weight = CLASS_WEIGHT[experiment_id]
    char_range = CHAR_NGRAM_RANGE[experiment_id]
    word_range = WORD_NGRAM_RANGE[experiment_id]

    results = {}

    print(f"[{experiment_id}] challenger 1/2: char_wb TF-IDF {char_range} + LinearSVC (class_weight={class_weight})...", flush=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=char_range, min_df=2, lowercase=False)
    char_train = char_vec.fit_transform(train_text)
    char_val = char_vec.transform(validation_text)
    char_clf = LinearSVC(C=1.0, class_weight=class_weight, random_state=42)
    char_clf.fit(char_train, train_labels)
    char_pred = char_clf.predict(char_val)
    results["char_tfidf_linear_svm"] = {
        "vectorizer": {"analyzer": "char_wb", "ngram_range": list(char_range), "min_df": 2, "lowercase": False},
        "classifier": {"model": "LinearSVC", "C": 1.0, "class_weight": class_weight, "random_state": 42},
        "n_features": char_train.shape[1],
        **_metrics(validation_labels, char_pred),
    }
    print(f"[{experiment_id}] char_tfidf_linear_svm: {results['char_tfidf_linear_svm']}", flush=True)

    print(f"[{experiment_id}] challenger 2/2: word {word_range} + char {char_range} combined TF-IDF + LinearSVC...", flush=True)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=word_range, min_df=2, lowercase=False)
    word_train = word_vec.fit_transform(train_text)
    word_val = word_vec.transform(validation_text)
    combined_train = hstack([word_train, char_train]).tocsr()
    combined_val = hstack([word_val, char_val]).tocsr()
    combined_clf = LinearSVC(C=1.0, class_weight=class_weight, random_state=42)
    combined_clf.fit(combined_train, train_labels)
    combined_pred = combined_clf.predict(combined_val)
    results["word_char_combined_linear_svm"] = {
        "word_vectorizer": {"analyzer": "word", "ngram_range": list(word_range), "min_df": 2, "lowercase": False},
        "char_vectorizer": {"analyzer": "char_wb", "ngram_range": list(char_range), "min_df": 2, "lowercase": False},
        "classifier": {"model": "LinearSVC", "C": 1.0, "class_weight": class_weight, "random_state": 42},
        "n_features": combined_train.shape[1],
        **_metrics(validation_labels, combined_pred),
    }
    print(f"[{experiment_id}] word_char_combined_linear_svm: {results['word_char_combined_linear_svm']}", flush=True)

    return {
        "experiment_id": experiment_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "results": results,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_classical_challengers.py <B2|C|E>")
    exp = sys.argv[1]
    out = run_challengers(exp)
    CHALLENGER_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = CHALLENGER_ROOT / f"classical_challengers_{exp}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{exp}] written to {out_path}", flush=True)
