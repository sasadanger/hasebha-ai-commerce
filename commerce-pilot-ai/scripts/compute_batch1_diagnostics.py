"""Validation-only diagnostics for the frozen Batch 1 winners.

Refits ONLY the already-frozen winning configuration for one experiment (never
re-selects a winner, never touches Batch 1's saved artifacts) on the same
deterministic train partition, predicts on the same validation partition, and
reports confusion matrix / per-class precision/recall/F1/support / most-confused
pairs, plus (for the two 1-5 star tasks) mean absolute rating error, absolute
rating-distance distribution, adjacent-class error rate, and severe-error rate.

internal_test is never read, counted, or predicted on anywhere in this script.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.configuration import instantiate_configuration, resolve_batch1_configurations
from src.nlp.split_preparation import prepare_task_bound_split
from src.nlp.text_normalization import normalize_text
from scripts.run_nlp_batch1_real import EXPERIMENT_SPECS

WINNER_COMPOUND_ID = {
    "A": "A::tfidf_word_bigram_logreg",
    "B2": "B2::tfidf_word_unigram_linear_svm",
    "C": "C::tfidf_word_bigram_logreg",
    "E": "E::tfidf_word_unigram_linear_svm",
}

RATING_TASKS = {"A", "C"}


def run_diagnostics(experiment_id: str) -> dict:
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

    all_configs = resolve_batch1_configurations(ROOT / "configs" / "nlp_training_batch_authorization_v2.yaml")
    winner_config = next(c for c in all_configs if c["compound_id"] == WINNER_COMPOUND_ID[experiment_id])

    print(f"[{experiment_id}] refitting frozen winner {winner_config['compound_id']} (train-only)...", flush=True)
    vectorizer, estimator = instantiate_configuration(winner_config)
    train_features = vectorizer.fit_transform(train_text)
    validation_features = vectorizer.transform(validation_text)
    estimator.fit(train_features, train_labels)
    predictions = estimator.predict(validation_features)

    labels_sorted = sorted(set(train_labels) | set(validation_labels), key=str)
    cm = confusion_matrix(validation_labels, predictions, labels=labels_sorted)
    report = classification_report(validation_labels, predictions, labels=labels_sorted, output_dict=True, zero_division=0)

    confusions = []
    for i, true_label in enumerate(labels_sorted):
        for j, pred_label in enumerate(labels_sorted):
            if i != j and cm[i][j] > 0:
                confusions.append({"true": str(true_label), "predicted": str(pred_label), "count": int(cm[i][j])})
    confusions.sort(key=lambda x: -x["count"])

    result = {
        "experiment_id": experiment_id,
        "winner_compound_id": winner_config["compound_id"],
        "labels": [str(l) for l in labels_sorted],
        "validation_class_distribution": {str(k): v for k, v in Counter(validation_labels).items()},
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
        "macro_f1": f1_score(validation_labels, predictions, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(validation_labels, predictions),
        "accuracy": accuracy_score(validation_labels, predictions),
        "top_confused_pairs": confusions[:10],
    }

    if experiment_id in RATING_TASKS:
        true_int = [int(v) for v in validation_labels]
        pred_int = [int(v) for v in predictions]
        abs_errors = [abs(t - p) for t, p in zip(true_int, pred_int)]
        result["rating_diagnostics"] = {
            "mean_absolute_error": sum(abs_errors) / len(abs_errors),
            "abs_error_distribution": dict(Counter(abs_errors)),
            "adjacent_class_error_rate": sum(1 for e in abs_errors if e == 1) / len(abs_errors),
            "severe_error_rate_ge2": sum(1 for e in abs_errors if e >= 2) / len(abs_errors),
        }

    print(f"[{experiment_id}] macro_f1={result['macro_f1']:.4f} balanced_accuracy={result['balanced_accuracy']:.4f}", flush=True)
    print(f"[{experiment_id}] top confused pairs: {confusions[:5]}", flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: compute_batch1_diagnostics.py <A|B2|C|E>")
    exp = sys.argv[1]
    out = run_diagnostics(exp)
    out_path = ROOT / "reports" / "generated" / "nlp" / f"batch1_diagnostics_{exp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{exp}] written to {out_path}", flush=True)
