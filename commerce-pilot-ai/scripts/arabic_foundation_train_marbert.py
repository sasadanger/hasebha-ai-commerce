"""Gates 13-15: MARBERT primary fine-tune for the Arabic sentiment foundation 3-class task.

Reads decisions from earlier gates:
  - max_length from reports/generated/arabic_foundation/token_length_audit.json (Gate 11)
  - loss variant from reports/generated/arabic_foundation/loss_imbalance_pilot.json (Gate 12)

Config per brief: UBC-NLP/MARBERT, end-to-end fine-tune (not frozen), lr=2e-5, AdamW,
weight_decay=0.01, warmup_ratio~=0.06 computed as warmup_steps (transformers 5.15.0 removed the
warmup_ratio TrainingArguments kwarg -- confirmed by direct inspect.signature check, same finding
as a prior session's Amazon pipeline hit), deterministic seed, DataCollatorWithPadding with
pad_to_multiple_of=8, bf16 (confirmed supported in Gate 1 hardware audit), TF32, fast tokenizer,
cached tokenization, best-model-by-validation-macro-F1, epoch-boundary eval/save,
save_total_limit=1, NO test-set evaluation during training. Max 3 epochs; continues past epoch 1
only while BOTH macro-F1 and Neutral/Mixed-F1 improve by >=0.2pp (defined here, before checking,
per Gate 15 instruction) -- enforced by a custom TrainerCallback, not by eyeballing after the run.

Run (smoke test first, then full):
  .venv/Scripts/python.exe scripts/arabic_foundation_train_marbert.py --smoke_test
  .venv/Scripts/python.exe scripts/arabic_foundation_train_marbert.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from nlp.arabic_foundation import transformer as af_tf  # noqa: E402
from nlp.arabic_foundation.normalization import LABEL_NAMES_3CLASS  # noqa: E402

CHECKPOINT = "UBC-NLP/MARBERT"
SEED = af_tf.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "primary_model"
CACHE_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "tokenized_cache"
MEANINGFUL_DELTA = 0.002  # 0.2 percentage points, defined BEFORE checking, per Gate 15 instruction


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_max_length() -> int:
    p = REPORTS_DIR / "token_length_audit.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return int(d["decision"])


def get_loss_variant():
    p = REPORTS_DIR / "loss_imbalance_pilot.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    weights = d["class_weights_inverse_frequency"] if d["decision"] == "B_class_weighted_ce" else None
    return d["decision"], weights


def dry_run_batch_size(model_ctor, tokenizer, max_length: int, candidates=(128, 96, 64, 48, 32, 16)) -> int:
    """Find the largest stable batch size via a real forward+backward pass at max_length,
    per the 'dry-run for largest stable batch size' instruction."""
    if not torch.cuda.is_available():
        return 16
    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = model_ctor().to("cuda")
            model.train()
            input_ids = torch.randint(0, tokenizer.vocab_size, (bs, max_length), device="cuda")
            attn = torch.ones((bs, max_length), dtype=torch.long, device="cuda")
            labels = torch.randint(0, 3, (bs,), device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
            loss.backward()
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            del model, out, loss, input_ids, attn, labels
            torch.cuda.empty_cache()
            log(f"batch_size={bs} OK, peak_vram={peak_mb:.0f}MB")
            return bs
        except torch.cuda.OutOfMemoryError:
            log(f"batch_size={bs} OOM, trying smaller")
            torch.cuda.empty_cache()
            continue
    return 8


class MeaningfulImprovementCallback:
    """Stops training if the most recent epoch's eval didn't improve BOTH macro-F1 and
    Neutral/Mixed-F1 by >= MEANINGFUL_DELTA over the previous epoch. Rule defined above the
    training loop, not adjusted after seeing results."""

    def __init__(self):
        self.history = []

    def __call__(self, macro_f1: float, neutral_mixed_f1: float) -> bool:
        """Returns True if training should CONTINUE."""
        if not self.history:
            self.history.append((macro_f1, neutral_mixed_f1))
            return True
        prev_macro, prev_neutral = self.history[-1]
        improved = (macro_f1 - prev_macro >= MEANINGFUL_DELTA) and (neutral_mixed_f1 - prev_neutral >= MEANINGFUL_DELTA)
        self.history.append((macro_f1, neutral_mixed_f1))
        return improved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
        Trainer, TrainingArguments, TrainerCallback,
    )

    max_length = get_max_length()
    loss_variant, class_weights = get_loss_variant()
    log(f"max_length={max_length}, loss_variant={loss_variant}, class_weights={class_weights}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    train = af_tf.load_split("train")
    val = af_tf.load_split("val_natural")

    if args.smoke_test:
        log("SMOKE TEST MODE: using small subsets, 1 epoch")
        train = train.sample(n=min(500, len(train)), random_state=SEED).reset_index(drop=True)
        val = val.sample(n=min(200, len(val)), random_state=SEED).reset_index(drop=True)

    log(f"train n={len(train)}, val n={len(val)}")
    log("Spot-checking label mapping on 5 random rows:")
    for _, row in train.sample(5, random_state=SEED).iterrows():
        log(f"  rating={row['rating']} -> label={row['label']} ({row['label_name']})")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_tag = "smoke" if args.smoke_test else "full"
    train_cache = CACHE_DIR / f"train_{cache_tag}_maxlen{max_length}"
    val_cache = CACHE_DIR / f"val_{cache_tag}_maxlen{max_length}"

    if train_cache.exists():
        from datasets import load_from_disk
        log(f"Loading cached tokenization from {train_cache}")
        ds_train = load_from_disk(str(train_cache))
        ds_val = load_from_disk(str(val_cache))
    else:
        log("Tokenizing (will cache to disk)...")
        ds_train = af_tf.tokenize_dataframe(train, tokenizer, max_length)
        ds_val = af_tf.tokenize_dataframe(val, tokenizer, max_length)
        ds_train.save_to_disk(str(train_cache))
        ds_val.save_to_disk(str(val_cache))

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    def model_ctor():
        return AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=3)

    log("Dry-running for largest stable batch size...")
    batch_size = dry_run_batch_size(model_ctor, tokenizer, max_length)
    if args.smoke_test:
        batch_size = min(batch_size, 8)
    log(f"Selected batch_size={batch_size}")

    model = model_ctor()

    num_epochs = 1 if args.smoke_test else 3
    steps_per_epoch = max(1, len(ds_train) // batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.06 * total_steps)
    log(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / ("smoke_run" if args.smoke_test else "run")),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        report_to=[],
        bf16=bf16_ok,
        fp16=not bf16_ok and torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        dataloader_num_workers=0,
        gradient_checkpointing=False,  # not needed -- no OOM at selected batch size
    )
    log(f"TrainingArguments: bf16={training_args.bf16}, fp16={training_args.fp16}, tf32={training_args.tf32}")

    if class_weights is not None:
        w = torch.tensor(class_weights)

        class WeightedCETrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = F.cross_entropy(logits, labels, weight=w.to(logits.device, logits.dtype))
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedCETrainer
    else:
        trainer_cls = Trainer

    improvement_gate = MeaningfulImprovementCallback()

    class EpochGateCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return control
            macro_f1 = metrics.get("eval_macro_f1", 0.0)
            neutral_f1 = metrics.get("eval_neutral_mixed_f1", 0.0)
            should_continue = improvement_gate(macro_f1, neutral_f1)
            log(f"epoch={state.epoch} eval_macro_f1={macro_f1:.4f} eval_neutral_mixed_f1={neutral_f1:.4f} "
                f"should_continue(meaningful improvement rule)={should_continue}")
            if not should_continue and state.epoch and state.epoch >= 2:
                log("Stopping early: neither macro-F1 nor Neutral/Mixed-F1 improved by >=0.2pp this epoch.")
                control.should_training_stop = True
            return control

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        compute_metrics=af_tf.compute_hf_metrics,
        callbacks=[EpochGateCallback()],
    )

    log("Checking gradients are finite / loss decreases on a tiny warm-up batch (Gate 14 smoke check component)...")
    model.train()
    model_input_keys = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    rows = [{k: v for k, v in ds_train[i].items() if k in model_input_keys} for i in range(min(4, len(ds_train)))]
    batch = collator(rows)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out1 = model(**batch)
    loss1 = out1.loss.item()
    assert np.isfinite(loss1), f"non-finite loss at init: {loss1}"
    log(f"initial loss on tiny batch: {loss1:.4f} (finite: OK)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    log(f"Training complete in {train_seconds:.1f}s, peak_vram={peak_vram_mb}")

    final_metrics = trainer.evaluate()
    log(f"Final (best-checkpoint) eval metrics: {final_metrics}")

    if args.smoke_test:
        smoke_report = {
            "cuda_active": torch.cuda.is_available(),
            "initial_loss_finite": bool(np.isfinite(loss1)),
            "train_seconds": train_seconds,
            "peak_vram_mb": peak_vram_mb,
            "final_eval_metrics": final_metrics,
            "batch_size_used": batch_size,
            "max_length": max_length,
            "loss_variant": loss_variant,
            "runtime_estimate_full_run_seconds_per_epoch_x3": (
                train_seconds / num_epochs if num_epochs else None
            ),
        }
        (REPORTS_DIR / "smoke_test_report.json").write_text(json.dumps(smoke_report, indent=2, default=str), encoding="utf-8")
        log("Wrote smoke_test_report.json")
        return

    # ---- Gate 15/26: save frozen artifacts ----
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    final_dir = ARTIFACT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    label_mapping = {"id2label": LABEL_NAMES_3CLASS, "label2id": {v: k for k, v in LABEL_NAMES_3CLASS.items()}}
    (final_dir / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")

    training_config = {
        "checkpoint": CHECKPOINT,
        "max_length": max_length,
        "batch_size": batch_size,
        "num_epochs_configured": num_epochs,
        "num_epochs_actually_run": trainer.state.epoch,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": warmup_steps,
        "seed": SEED,
        "loss_variant": loss_variant,
        "class_weights": class_weights,
        "bf16": training_args.bf16,
        "tf32": training_args.tf32,
        "epoch_gate_rule": f"continue past epoch 1 only if macro-F1 AND Neutral/Mixed-F1 both improved by >={MEANINGFUL_DELTA*100:.1f}pp",
        "epoch_history": improvement_gate.history,
        "train_seconds": train_seconds,
        "peak_vram_mb": peak_vram_mb,
        "final_eval_metrics": final_metrics,
        "log_history": trainer.state.log_history,
    }
    (final_dir / "training_config.json").write_text(json.dumps(training_config, indent=2, default=str), encoding="utf-8")
    (REPORTS_DIR / "marbert_training_manifest.json").write_text(json.dumps(training_config, indent=2, default=str), encoding="utf-8")

    # save raw logits on val_natural for calibration (Gate 16), do NOT touch test here
    log("Generating raw logits on val_natural for calibration...")
    val_preds = trainer.predict(ds_val)
    np.save(final_dir / "val_natural_logits.npy", val_preds.predictions)
    np.save(final_dir / "val_natural_labels.npy", val_preds.label_ids)
    val_uids = ds_val["review_uid"] if "review_uid" in ds_val.column_names else val["review_uid"].tolist()
    (final_dir / "val_natural_review_uids.json").write_text(json.dumps(list(val_uids)), encoding="utf-8")

    log(f"Saved final model to {final_dir}")
    print("\nDONE. training_config summary:")
    print(json.dumps({k: v for k, v in training_config.items() if k != "log_history"}, indent=2, default=str))


if __name__ == "__main__":
    main()
