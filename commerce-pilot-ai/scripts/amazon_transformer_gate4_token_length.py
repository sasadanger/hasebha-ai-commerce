"""Gate 4: token-length audit and max_length decision (128 vs 192).

Uses the real distilroberta-base fast tokenizer on the 50k balanced training sample
(reports/generated/amazon/split_ids/learning_curve_50000.parquet, rejoined to text) to compute
token-length percentiles, then runs a SMALL fixed pilot (a few thousand rows, 1 epoch) comparing
max_length=128 vs 192 on validation macro-F1, throughput, and peak VRAM. Selects 128 unless 192
shows a meaningful macro-F1 gain.

Run once from repo root:
  .venv/Scripts/python.exe scripts/amazon_transformer_gate4_token_length.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.amazon import transformer as amz_tf  # noqa: E402

CHECKPOINT = "distilroberta-base"
SEED = amz_tf.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "amazon"
PILOT_TRAIN_N = 3_000
PILOT_EVAL_N = 1_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log("Reproducing full pool with text and rejoining learning_curve_50000 + val...")
    full_pool = amz_tf.reproduce_full_pool_with_text(seed=SEED)
    train_50k = amz_tf.load_split_with_text("learning_curve_50000", full_pool)
    val = amz_tf.load_split_with_text("val", full_pool)
    log(f"train_50k n={len(train_50k)}, val n={len(val)}")

    log(f"Loading tokenizer for {CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # ---- token length percentiles on the 50k training sample ------------------------------
    log("Computing token-length percentiles on the 50k training sample (no truncation)...")
    lengths = []
    batch_size = 1000
    texts = train_50k[amz_tf.TEXT_COL].tolist()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=False, padding=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    lengths = np.array(lengths)
    percentiles = [50, 75, 90, 95, 99, 100]
    length_percentiles = {f"p{p}": float(np.percentile(lengths, p)) for p in percentiles}
    length_percentiles["mean"] = float(lengths.mean())
    length_percentiles["n"] = int(len(lengths))
    log(f"Token length percentiles: {length_percentiles}")

    truncation_pct = {}
    for max_len in (128, 192):
        truncation_pct[str(max_len)] = float((lengths > max_len).mean() * 100)
    log(f"Truncation pct at 128/192: {truncation_pct}")

    # ---- small fixed pilot: 128 vs 192 -----------------------------------------------------
    rng = np.random.default_rng(SEED)
    pilot_train_idx = rng.choice(len(train_50k), size=min(PILOT_TRAIN_N, len(train_50k)), replace=False)
    pilot_train = train_50k.iloc[pilot_train_idx].reset_index(drop=True)
    pilot_eval_idx = rng.choice(len(val), size=min(PILOT_EVAL_N, len(val)), replace=False)
    pilot_eval = val.iloc[pilot_eval_idx].reset_index(drop=True)
    log(f"Pilot: train n={len(pilot_train)}, eval n={len(pilot_eval)}")

    pilot_results = {}
    for max_len in (128, 192):
        log(f"--- pilot run: max_length={max_len} ---")
        torch.manual_seed(SEED)
        train_ds = amz_tf.tokenize_dataframe(pilot_train, tokenizer, max_len)
        eval_ds = amz_tf.tokenize_dataframe(pilot_eval, tokenizer, max_len)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

        model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
        model.to("cuda")

        out_dir = REPO_ROOT / "artifacts" / "experiments" / "amazon" / "transformer" / f"_pilot_{max_len}"
        total_steps = (len(pilot_train) // 32) * 1
        warmup_steps = max(1, round(0.06 * total_steps))
        args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=1,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",
            bf16=True,
            tf32=True,
            report_to=[],
            seed=SEED,
            data_seed=SEED,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds.remove_columns(["review_uid"]),
            eval_dataset=eval_ds.remove_columns(["review_uid"]),
            data_collator=collator,
            compute_metrics=amz_tf.compute_hf_metrics,
        )

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        t0 = time.time()
        eval_metrics = trainer.evaluate()
        eval_seconds = time.time() - t0
        examples_per_sec = len(pilot_eval) / eval_seconds if eval_seconds > 0 else float("nan")

        pilot_results[str(max_len)] = {
            "train_seconds": train_seconds,
            "train_examples_per_sec": len(pilot_train) / train_seconds,
            "eval_macro_f1": eval_metrics["eval_macro_f1"],
            "eval_negative_f1": eval_metrics["eval_negative_f1"],
            "eval_positive_f1": eval_metrics["eval_positive_f1"],
            "eval_examples_per_sec": examples_per_sec,
            "peak_vram_allocated_mb": peak_vram_mb,
            "truncation_pct_at_this_length": truncation_pct[str(max_len)],
        }
        log(f"  max_length={max_len}: {pilot_results[str(max_len)]}")

        del model, trainer
        torch.cuda.empty_cache()

    f1_128 = pilot_results["128"]["eval_macro_f1"]
    f1_192 = pilot_results["192"]["eval_macro_f1"]
    gain = f1_192 - f1_128
    # Select 192 only if it shows a meaningful gain (>0.5pp macro-F1) that would justify its
    # materially higher compute/memory cost per step (roughly 1.5x tokens at 192 vs 128).
    MEANINGFUL_GAIN_THRESHOLD = 0.005
    selected = 192 if gain > MEANINGFUL_GAIN_THRESHOLD else 128
    decision = {
        "macro_f1_gain_192_over_128": gain,
        "meaningful_gain_threshold": MEANINGFUL_GAIN_THRESHOLD,
        "selected_max_length": selected,
        "reasoning": (
            f"192 pilot macro-F1 ({f1_192:.4f}) vs 128 pilot macro-F1 ({f1_128:.4f}): gain="
            f"{gain:+.4f}. "
            + (
                "Below the 0.5pp meaningful-gain threshold, so 128 is selected -- it truncates "
                f"only {truncation_pct['128']:.2f}% of the 50k training sample (vs "
                f"{truncation_pct['192']:.2f}% at 192) and is materially cheaper per step."
                if selected == 128
                else "Exceeds the 0.5pp threshold, so 192 is selected despite its extra cost."
            )
        ),
    }
    log(f"DECISION: {decision}")

    audit = {
        "generated_at": "2026-08-17",
        "checkpoint": CHECKPOINT,
        "tokenizer_is_fast": tokenizer.is_fast,
        "token_length_percentiles_full_50k_no_truncation": length_percentiles,
        "truncation_pct_by_max_length": truncation_pct,
        "pilot": {
            "train_n": len(pilot_train),
            "eval_n": len(pilot_eval),
            "epochs": 1,
            "batch_size": 32,
            "results_by_max_length": pilot_results,
        },
        "decision": decision,
    }
    OUT = REPORTS_DIR / "transformer_token_length_audit.json"
    OUT.write_text(json.dumps(audit, indent=2, default=str))
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
