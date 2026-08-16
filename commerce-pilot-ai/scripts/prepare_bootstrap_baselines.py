"""Refit each frozen Batch 1 classical winner and save validation predictions.

Needed because compute_batch1_diagnostics.py (already run for B2/C/E) saved
per-class reports and confusion matrices but not raw per-row predictions,
which a paired bootstrap comparison against transformer confirmation runs
requires (same validation examples, resampled together). Refits ONLY the
already-frozen winning configuration -- no new winner is selected, Batch 1
artifacts are untouched. Train-only fit, validation-only prediction.
internal_test is never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.configuration import instantiate_configuration, resolve_batch1_configurations
from src.nlp.split_preparation import prepare_task_bound_split
from src.nlp.text_normalization import normalize_text
from scripts.run_nlp_batch1_real import EXPERIMENT_SPECS

WINNER_COMPOUND_ID = {
    "B2": "B2::tfidf_word_unigram_linear_svm",
    "C": "C::tfidf_word_bigram_logreg",
    "E": "E::tfidf_word_unigram_linear_svm",
}


def run(experiment_id: str) -> None:
    spec = EXPERIMENT_SPECS[experiment_id]
    print(f"[{experiment_id}] loading real data...", flush=True)
    records, actual_sha256, expected_sha256 = spec["loader"]()
    if actual_sha256 != expected_sha256:
        raise SystemExit(f"[{experiment_id}] acquisition hash mismatch")
    adapted = [spec["schema_adapter"](r) if spec["schema_adapter"] else r for r in records]
    normalized = [{**r, "__normalized_text__": normalize_text(r[spec["text_key"]])} for r in adapted]
    prep = prepare_task_bound_split(
        normalized, text_key=spec["text_key"], label_key=spec["label_key"],
        task_type=spec["task_type"], seed=20260809,
    )
    train_rows = [normalized[a.row_index] for a in prep.assignments if a.split == "train"]
    validation_rows = [normalized[a.row_index] for a in prep.assignments if a.split == "validation"]
    train_text = [r["__normalized_text__"] for r in train_rows]
    validation_text = [r["__normalized_text__"] for r in validation_rows]
    train_labels = [str(r[spec["label_key"]]) for r in train_rows]
    validation_labels = [str(r[spec["label_key"]]) for r in validation_rows]

    all_configs = resolve_batch1_configurations(ROOT / "configs" / "nlp_training_batch_authorization_v2.yaml")
    winner_config = next(c for c in all_configs if c["compound_id"] == WINNER_COMPOUND_ID[experiment_id])

    print(f"[{experiment_id}] refitting frozen winner {winner_config['compound_id']}...", flush=True)
    vectorizer, estimator = instantiate_configuration(winner_config)
    train_features = vectorizer.fit_transform(train_text)
    validation_features = vectorizer.transform(validation_text)
    estimator.fit(train_features, train_labels)
    predictions = estimator.predict(validation_features).tolist()

    label_set = sorted(set(train_labels) | set(validation_labels))
    label2id = {label: i for i, label in enumerate(label_set)}

    result = {
        "experiment_id": experiment_id,
        "winner_compound_id": winner_config["compound_id"],
        "labels": label_set,
        "validation_true_labels": [label2id[l] for l in validation_labels],
        "validation_predictions": [label2id[p] for p in predictions],
    }
    out_dir = ROOT / "reports" / "generated" / "nlp" / "transformer_confirmation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"classical_baseline_predictions_{experiment_id}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{experiment_id}] written to {out_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare_bootstrap_baselines.py <B2|C|E>")
    run(sys.argv[1])
