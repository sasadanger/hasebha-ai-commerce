"""One-seed transformer screening for a Batch 1 Arabic/English task (validation only).

Reuses the exact same real-data loaders and deterministic split_preparation
call as the classical Batch 1 run (same seed 20260809) -- train/validation
partitions only. internal_test is never read, tokenized, or evaluated.

This is SCREENING, not final multi-seed confirmation: one fixed seed, up to
4 epochs, save_total_limit=1, validation-only model selection. Does not
modify any Batch 1 artifact; writes to a new, separate location.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/commercepilot_ml_cache/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "D:/commercepilot_ml_cache/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/commercepilot_ml_cache/hf/hub")
os.environ.setdefault("TORCH_HOME", "D:/commercepilot_ml_cache/torch")
os.environ.setdefault("TMPDIR", "D:/commercepilot_ml_cache/tmp")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.nlp.split_preparation import prepare_task_bound_split
from src.nlp.text_normalization import normalize_text
from scripts.run_nlp_batch1_real import EXPERIMENT_SPECS

SEED = 20260809
OUTPUT_ROOT = Path("D:/commercepilot_ml_cache/checkpoints")


class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_split(experiment_id: str):
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
        task_type=spec["task_type"], seed=SEED,
    )
    train_rows = [normalized[a.row_index] for a in prep.assignments if a.split == "train"]
    validation_rows = [normalized[a.row_index] for a in prep.assignments if a.split == "validation"]
    train_text = [r["__normalized_text__"] for r in train_rows]
    validation_text = [r["__normalized_text__"] for r in validation_rows]
    train_labels_raw = [str(r[spec["label_key"]]) for r in train_rows]
    validation_labels_raw = [str(r[spec["label_key"]]) for r in validation_rows]
    return train_text, train_labels_raw, validation_text, validation_labels_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", choices=["A", "B2", "C", "E"])
    parser.add_argument("model_name")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--calibrate_only", action="store_true", help="only compute tokenizer length percentiles, no training")
    args = parser.parse_args()

    set_seed(SEED)
    train_text, train_labels_raw, validation_text, validation_labels_raw = load_split(args.experiment_id)

    label_set = sorted(set(train_labels_raw) | set(validation_labels_raw))
    label2id = {label: i for i, label in enumerate(label_set)}
    id2label = {i: label for label, i in label2id.items()}
    train_labels = [label2id[l] for l in train_labels_raw]
    validation_labels = [label2id[l] for l in validation_labels_raw]

    print(f"[{args.experiment_id}] loading tokenizer {args.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="D:/commercepilot_ml_cache/hf/hub")

    if args.calibrate_only:
        sample = train_text[:5000] if len(train_text) > 5000 else train_text
        lengths = [len(tokenizer.encode(t, truncation=False)) for t in sample]
        percentiles = {p: float(np.percentile(lengths, p)) for p in (50, 90, 95, 99, 100)}
        print(f"[{args.experiment_id}] tokenizer length percentiles (n={len(sample)}): {percentiles}", flush=True)
        return

    print(f"[{args.experiment_id}] tokenizing (max_length={args.max_length})...", flush=True)
    train_enc = tokenizer(train_text, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    val_enc = tokenizer(validation_text, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    train_ds = SimpleTextDataset(train_enc, train_labels)
    val_ds = SimpleTextDataset(val_enc, validation_labels)

    print(f"[{args.experiment_id}] loading model {args.model_name} ({len(label_set)} labels)...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_set), id2label=id2label, label2id=label2id,
        cache_dir="D:/commercepilot_ml_cache/hf/hub",
    )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    run_name = f"{args.experiment_id}_{args.model_name.replace('/', '__')}"
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "accuracy": accuracy_score(labels, preds),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        seed=SEED,
        bf16=bf16_ok,
        fp16=not bf16_ok and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"[{args.experiment_id}] training {args.model_name} for up to {args.epochs} epochs...", flush=True)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    train_start = time.time()
    train_result = trainer.train()
    train_runtime = time.time() - train_start

    eval_start = time.time()
    eval_result = trainer.evaluate()
    eval_runtime = time.time() - eval_start

    logits_output = trainer.predict(val_ds)
    preds = np.argmax(logits_output.predictions, axis=-1)
    report = classification_report(validation_labels, preds, target_names=[id2label[i] for i in range(len(label_set))], output_dict=True, zero_division=0)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None

    result = {
        "experiment_id": args.experiment_id,
        "model_name": args.model_name,
        "seed": SEED,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs_configured": args.epochs,
        "learning_rate": args.lr,
        "bf16": bf16_ok,
        "labels": label_set,
        "train_rows": len(train_text),
        "validation_rows": len(validation_text),
        "macro_f1": eval_result.get("eval_macro_f1"),
        "balanced_accuracy": eval_result.get("eval_balanced_accuracy"),
        "accuracy": eval_result.get("eval_accuracy"),
        "per_class_report": report,
        "train_runtime_sec": train_runtime,
        "eval_runtime_sec": eval_runtime,
        "peak_vram_mb": peak_vram_mb,
        "best_epoch": trainer.state.best_model_checkpoint,
        "log_history": trainer.state.log_history,
    }

    out_dir = ROOT / "reports" / "generated" / "nlp" / "transformer_screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"transformer_{args.experiment_id}_{args.model_name.replace('/', '__')}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"[{args.experiment_id}] macro_f1={result['macro_f1']} balanced_accuracy={result['balanced_accuracy']} peak_vram_mb={peak_vram_mb}", flush=True)
    print(f"[{args.experiment_id}] written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
