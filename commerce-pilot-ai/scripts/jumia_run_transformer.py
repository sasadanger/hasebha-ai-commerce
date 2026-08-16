"""Phase 7 -- Jumia transformer adaptation (fine-tuning). Train/validation
only; internal_test/protected_test is never read, tokenized, or evaluated
here. Structurally mirrors scripts/run_transformer_confirmation.py (same
Trainer conventions, same HF cache paths) applied to Jumia's own
train/validation split instead of Batch 1 loaders.
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
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

from src.nlp.text_normalization import normalize_text

CSV_PATH = ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
SPLIT_PATH = ROOT / "artifacts" / "experiments" / "jumia" / "phase1_split" / "jumia_split_assignments.parquet"
LABELS = ["1", "2", "3", "4", "5"]


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


def load_split():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["customer_rating"] = df["customer_rating"].astype(str)
    df["_norm"] = df["review"].map(normalize_text)
    split_df = pd.read_parquet(SPLIT_PATH)
    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].tolist()
    val_idx = split_df.loc[split_df["split"] == "validation", "row_index"].tolist()
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    return train["_norm"].tolist(), train["customer_rating"].tolist(), val["_norm"].tolist(), val["customer_rating"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--training_seed", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--run_name_suffix", default="")
    args = parser.parse_args()

    set_seed(args.training_seed)
    train_text, train_labels_raw, val_text, val_labels_raw = load_split()

    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    train_labels = [label2id[l] for l in train_labels_raw]
    val_labels_int = [label2id[l] for l in val_labels_raw]

    print(f"loading tokenizer {args.model_name}@{args.revision}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision, cache_dir="D:/commercepilot_ml_cache/hf/hub")

    train_enc = tokenizer(train_text, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    val_enc = tokenizer(val_text, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    train_ds = SimpleTextDataset(train_enc, train_labels)
    val_ds = SimpleTextDataset(val_enc, val_labels_int)

    print(f"loading model {args.model_name}@{args.revision} ({len(LABELS)} labels)...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, revision=args.revision, num_labels=len(LABELS), id2label=id2label, label2id=label2id,
        cache_dir="D:/commercepilot_ml_cache/hf/hub",
    )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    run_name = f"jumia_{args.model_name.replace('/', '__')}_seed{args.training_seed}{args.run_name_suffix}"
    output_dir = Path("D:/commercepilot_ml_cache/checkpoints") / run_name

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
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.training_seed,
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

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)

    print(f"training seed={args.training_seed} for up to {args.epochs} epochs...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    trainer.train()
    train_runtime = time.time() - train_start

    eval_result = trainer.evaluate()
    logits_output = trainer.predict(val_ds)
    preds = np.argmax(logits_output.predictions, axis=-1)
    report = classification_report(val_labels_int, preds, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(val_labels_int, preds, labels=list(range(len(LABELS))))
    worst_class_f1 = min(report[name]["f1-score"] for name in LABELS)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None

    true_int = [int(id2label[i]) for i in val_labels_int]
    pred_int = [int(id2label[i]) for i in preds]
    abs_errors = [abs(t - p) for t, p in zip(true_int, pred_int)]

    result = {
        "dataset": "jumia_egypt_reviews_jerd",
        "model_name": args.model_name,
        "revision": args.revision,
        "training_seed": args.training_seed,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs_configured": args.epochs,
        "learning_rate": args.lr,
        "bf16": bf16_ok,
        "labels": LABELS,
        "train_rows": len(train_text),
        "validation_rows": len(val_text),
        "macro_f1": eval_result.get("eval_macro_f1"),
        "balanced_accuracy": eval_result.get("eval_balanced_accuracy"),
        "accuracy": eval_result.get("eval_accuracy"),
        "worst_class_f1": worst_class_f1,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABELS,
        "rating_diagnostics": {
            "mean_absolute_error": sum(abs_errors) / len(abs_errors),
            "adjacent_class_error_rate": sum(1 for e in abs_errors if e == 1) / len(abs_errors),
            "severe_error_rate_ge2": sum(1 for e in abs_errors if e >= 2) / len(abs_errors),
        },
        "train_runtime_sec": train_runtime,
        "peak_vram_mb": peak_vram_mb,
        "internal_test_accessed": False,
        "checkpoint_dir": str(output_dir),
    }

    out_dir = ROOT / "reports" / "generated" / "jumia" / "transformer_adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"seed={args.training_seed} macro_f1={result['macro_f1']} balanced_accuracy={result['balanced_accuracy']} peak_vram_mb={peak_vram_mb}", flush=True)
    print(f"written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
