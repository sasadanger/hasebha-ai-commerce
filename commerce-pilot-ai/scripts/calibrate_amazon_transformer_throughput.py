"""Short throughput calibration for an Amazon transformer screen (NOT a training result).

Uses a small real-Amazon sample (fast to load/normalize) to measure actual
examples/sec for the chosen model+batch_size on this GPU, then extrapolates to
the full known train partition size (1,513,171 rows, from the frozen Batch 1
split -- see reports/checkpoints/phase2c_nlp_batch1_real_execution_2026-08-10/)
to estimate total runtime before committing to a full run. This produces no
model-selection result and writes no transformer_screening artifact.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/commercepilot_ml_cache/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "D:/commercepilot_ml_cache/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/commercepilot_ml_cache/hf/hub")
os.environ.setdefault("TMPDIR", "D:/commercepilot_ml_cache/tmp")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

from src.nlp.amazon_adapter import adapt_amazon_record
from src.nlp.text_normalization import normalize_text

SEED = 20260809
MODEL_NAME = "microsoft/deberta-v3-base"
KNOWN_TRAIN_ROWS = 1513171  # from the frozen Batch 1 A split, artifacts/experiments/nlp/phase2c/batch1/A/...


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main() -> None:
    set_seed(SEED)
    batch_size = 16
    calibration_steps = 60
    sample_size = batch_size * (calibration_steps + 5)

    print(f"[calibration] loading {sample_size} real Amazon rows for throughput measurement only...", flush=True)
    source = ROOT / "data" / "raw" / "amazon_reviews_appliances" / "Appliances.jsonl.gz"
    con = duckdb.connect()
    relation = f"read_json_auto('{source.as_posix()}', format='newline_delimited', maximum_object_size=33554432)"
    rows = con.execute(f"SELECT rating, title, text FROM {relation} LIMIT {sample_size}").fetchall()
    con.close()
    physical_records = [{"rating": r[0], "title": r[1], "text": r[2]} for r in rows]
    adapted = [adapt_amazon_record(r) for r in physical_records]
    texts = [normalize_text(r["review_text"]) for r in adapted]
    labels = [int(r["overall"]) - 1 for r in adapted]  # 5-class 0..4 for calibration purposes only

    print(f"[calibration] loading tokenizer/model {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="D:/commercepilot_ml_cache/hf/hub")
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    ds = TinyDataset(enc, labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=5, cache_dir="D:/commercepilot_ml_cache/hf/hub",
    )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir="D:/commercepilot_ml_cache/checkpoints/amazon_calibration",
        per_device_train_batch_size=batch_size,
        max_steps=calibration_steps,
        learning_rate=2e-5,
        seed=SEED,
        bf16=bf16_ok,
        fp16=not bf16_ok and torch.cuda.is_available(),
        logging_steps=1000,  # suppress noisy logging during calibration
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)

    print(f"[calibration] running {calibration_steps} steps to measure throughput...", flush=True)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    examples_processed = calibration_steps * batch_size
    examples_per_sec = examples_processed / elapsed
    steps_per_sec = calibration_steps / elapsed
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None

    steps_per_epoch = KNOWN_TRAIN_ROWS / batch_size
    sec_per_epoch = steps_per_epoch / steps_per_sec
    estimates = {f"{e}_epoch(s)_estimated_hours": round(sec_per_epoch * e / 3600, 2) for e in (1, 2, 3, 4)}

    result = {
        "model_name": MODEL_NAME,
        "batch_size": batch_size,
        "max_length": 256,
        "calibration_steps": calibration_steps,
        "calibration_elapsed_sec": elapsed,
        "examples_per_sec": examples_per_sec,
        "steps_per_sec": steps_per_sec,
        "peak_vram_mb_at_calibration": peak_vram_mb,
        "known_amazon_train_rows": KNOWN_TRAIN_ROWS,
        "steps_per_epoch_full_run": steps_per_epoch,
        **estimates,
    }
    print(json.dumps(result, indent=2), flush=True)
    out_path = ROOT / "reports" / "generated" / "nlp" / "transformer_screening" / "amazon_throughput_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[calibration] written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
