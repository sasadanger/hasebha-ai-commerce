"""Compute majority-class and stratified baselines for the real Batch 1 winners.

Per docs/nlp_experiment_methodology_policies.md Step 19/37: every classification
task requires a majority-class baseline (and, for multi-class tasks, a
stratified/random baseline) to beat, and success is defined *relative to that
baseline*, not an absolute macro-F1 target or a cross-dataset ranking. This
was never computed for the real Batch 1 run in
reports/checkpoints/phase2c_nlp_batch1_real_execution_2026-08-10/.

This script only re-derives the same deterministic train/validation split
(via prepare_task_bound_split, same seed) to obtain label sequences, then
scores two trivial baselines against the validation labels. It fits no
TF-IDF vectorizer and no real classifier, and it never reads, counts, or
touches internal_test rows. It reuses the exact loaders from
scripts/run_nlp_batch1_real.py (no new data-loading logic).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.split_preparation import prepare_task_bound_split
from src.nlp.text_normalization import normalize_text
from scripts.run_nlp_batch1_real import EXPERIMENT_SPECS


def compute_baselines(experiment_id: str) -> dict:
    spec = EXPERIMENT_SPECS[experiment_id]
    print(f"[{experiment_id}] loading real data (same loader as the real run)...", flush=True)
    records, actual_sha256, expected_sha256 = spec["loader"]()
    if actual_sha256 != expected_sha256:
        raise SystemExit(f"[{experiment_id}] acquisition hash mismatch")

    adapted = [spec["schema_adapter"](r) if spec["schema_adapter"] else r for r in records]
    normalized = [{**r, "__normalized_text__": normalize_text(r[spec["text_key"]])} for r in adapted]

    print(f"[{experiment_id}] re-deriving the same deterministic split (train/validation labels only)...", flush=True)
    prep = prepare_task_bound_split(
        normalized, text_key=spec["text_key"], label_key=spec["label_key"],
        task_type=spec["task_type"], seed=20260809,
    )
    train_labels = [normalized[a.row_index][spec["label_key"]] for a in prep.assignments if a.split == "train"]
    validation_labels = [normalized[a.row_index][spec["label_key"]] for a in prep.assignments if a.split == "validation"]

    majority_label = Counter(train_labels).most_common(1)[0][0]
    majority_predictions = [majority_label] * len(validation_labels)
    majority_metrics = {
        "macro_f1": f1_score(validation_labels, majority_predictions, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(validation_labels, majority_predictions),
        "accuracy": accuracy_score(validation_labels, majority_predictions),
    }

    dummy = DummyClassifier(strategy="stratified", random_state=20260809)
    dummy.fit([[0]] * len(train_labels), train_labels)
    stratified_predictions = dummy.predict([[0]] * len(validation_labels))
    stratified_metrics = {
        "macro_f1": f1_score(validation_labels, stratified_predictions, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(validation_labels, stratified_predictions),
        "accuracy": accuracy_score(validation_labels, stratified_predictions),
    }

    result = {
        "experiment_id": experiment_id,
        "train_label_distribution": dict(Counter(train_labels)),
        "validation_label_distribution": dict(Counter(validation_labels)),
        "majority_class_baseline": {"predicted_label": str(majority_label), **majority_metrics},
        "stratified_random_baseline": stratified_metrics,
    }
    print(f"[{experiment_id}] majority-class: {majority_metrics}", flush=True)
    print(f"[{experiment_id}] stratified-random: {stratified_metrics}", flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: compute_batch1_baselines.py <A|B2|C|E>")
    exp = sys.argv[1]
    out = compute_baselines(exp)
    out_path = ROOT / "reports" / "generated" / "nlp" / f"batch1_baseline_{exp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{exp}] written to {out_path}", flush=True)
