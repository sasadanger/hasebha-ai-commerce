"""Gate 11: tokenizer/max-length audit for the MARBERT fine-tune.

Real UBC-NLP/MARBERT fast tokenizer, token-length percentiles on the full train split, then a
small FIXED pilot (a few thousand rows, 1 epoch) comparing max_length=128 vs 192 on validation
macro-F1 + throughput + peak VRAM. Selects 128 unless 192 shows a meaningful gain. Methodology
reused from reports/generated/amazon/transformer_token_length_audit.json (same repo, prior
session) as the template, adapted to 3-class MARBERT.

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_token_length_audit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from nlp.arabic_foundation import transformer as af_tf  # noqa: E402

CHECKPOINT = "UBC-NLP/MARBERT"
SEED = af_tf.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
PILOT_TRAIN_N = 3_000
PILOT_EVAL_N = 1_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

    log(f"Loading tokenizer for {CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    log(f"tokenizer class: {type(tokenizer).__name__}, is_fast={tokenizer.is_fast}, vocab_size={tokenizer.vocab_size}")

    train = af_tf.load_split("train")
    val = af_tf.load_split("val_natural")
    log(f"train n={len(train)}, val n={len(val)}")

    log("Computing token-length percentiles on the full train split (no truncation)...")
    lengths = []
    batch_size = 512
    texts = train[af_tf.TEXT_COL].tolist()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    lengths = np.array(lengths)
    percentiles = {str(p): float(np.percentile(lengths, p)) for p in [50, 75, 90, 95, 99]}
    percentiles["max"] = int(lengths.max())
    percentiles["mean"] = float(lengths.mean())
    log(f"token length percentiles: {percentiles}")

    pct_over_128 = float((lengths > 128).mean() * 100)
    pct_over_192 = float((lengths > 192).mean() * 100)

    # ---- small fixed pilot: 128 vs 192 ----
    rng = np.random.RandomState(SEED)
    train_idx = rng.choice(len(train), size=min(PILOT_TRAIN_N, len(train)), replace=False)
    val_idx = rng.choice(len(val), size=min(PILOT_EVAL_N, len(val)), replace=False)
    pilot_train = train.iloc[train_idx].reset_index(drop=True)
    pilot_val = val.iloc[val_idx].reset_index(drop=True)

    pilot_results = {}
    for max_len in [128, 192]:
        log(f"--- pilot max_length={max_len} ---")
        model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=3)
        ds_train = af_tf.tokenize_dataframe(pilot_train, tokenizer, max_len)
        ds_val = af_tf.tokenize_dataframe(pilot_val, tokenizer, max_len)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

        out_dir = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "_pilot_tmp" / f"maxlen{max_len}"
        args_kwargs = dict(
            output_dir=str(out_dir),
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=1,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=int(0.06 * (len(ds_train) / 32)),
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            seed=SEED,
            report_to=[],
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            tf32=torch.cuda.is_available(),
            dataloader_num_workers=0,
        )
        training_args = TrainingArguments(**args_kwargs)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
            compute_metrics=af_tf.compute_hf_metrics,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0
        metrics = trainer.evaluate()
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
        pilot_results[str(max_len)] = {
            "train_seconds": train_seconds,
            "eval_macro_f1": metrics.get("eval_macro_f1"),
            "eval_neutral_mixed_f1": metrics.get("eval_neutral_mixed_f1"),
            "peak_vram_mb": peak_vram_mb,
        }
        log(f"max_length={max_len} results: {pilot_results[str(max_len)]}")
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    f1_128 = pilot_results["128"]["eval_macro_f1"]
    f1_192 = pilot_results["192"]["eval_macro_f1"]
    gain = f1_192 - f1_128
    decision = "192" if gain >= 0.01 else "128"  # meaningful gain threshold documented: >=1.0pp macro-F1
    decision_reasoning = (
        f"macro-F1 gain from 192 vs 128 = {gain:+.4f} ({gain*100:+.2f}pp). Decision rule (defined "
        "before comparing cost): select 192 only if it improves validation macro-F1 by >=1.0pp; "
        f"otherwise keep 128 as the cheaper default. Result: selected max_length={decision}."
    )
    log(decision_reasoning)

    result = {
        "checkpoint": CHECKPOINT,
        "tokenizer_is_fast": tokenizer.is_fast,
        "token_length_percentiles_full_train": percentiles,
        "pct_train_rows_exceeding_128_tokens": pct_over_128,
        "pct_train_rows_exceeding_192_tokens": pct_over_192,
        "pilot_train_n": len(pilot_train),
        "pilot_val_n": len(pilot_val),
        "pilot_results": pilot_results,
        "decision": decision,
        "decision_reasoning": decision_reasoning,
    }
    (REPORTS_DIR / "token_length_audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("\nWrote token_length_audit.json, decision:", decision)


if __name__ == "__main__":
    main()
