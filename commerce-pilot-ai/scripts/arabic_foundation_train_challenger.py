"""Gate 17: controlled CAMeLBERT-Mix challenger run.

Verified official model card before running (see reports/generated/arabic_foundation/
challenger_model_card_verification.json): CAMeL-Lab/bert-base-arabic-camelbert-mix, Apache-2.0,
pretrained on a mix of MSA (Gigaword/Abu El-Khair/OSIAN/Wikipedia/OSCAR) + Dialectal Arabic +
Classical Arabic (OpenITI), 167GB / 17.3B words. The model card makes NO claim about
Arabic-English code-switching support -- this script does not repeat that unsupported claim
anywhere. It answers a genuinely distinct question from MARBERT (which is pretrained on ~1B
dialectal-heavy Arabic tweets): does an MSA+CA+DA-mixed pretraining corpus transfer better or
worse to LABR's book-review domain than a dialectal-tweet-pretrained model?

Same split/label-contract/max-length/loss-variant/eval-protocol/epoch-budget as MARBERT (Gates
11/12 decisions reused verbatim, NOT re-tuned per-model, per Gate 17 instruction).

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_train_challenger.py
"""
from __future__ import annotations

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

CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
SEED = af_tf.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "challenger"
CACHE_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "tokenized_cache"
MEANINGFUL_DELTA = 0.002


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
        Trainer, TrainingArguments, TrainerCallback,
    )

    # reuse MARBERT's decisions verbatim -- SAME protocol, not re-tuned per model (Gate 17)
    marbert_config = json.loads((REPO_ROOT / "artifacts/experiments/arabic_foundation/primary_model/final/training_config.json").read_text(encoding="utf-8"))
    max_length = marbert_config["max_length"]
    batch_size = marbert_config["batch_size"]
    loss_variant = marbert_config["loss_variant"]
    class_weights = marbert_config["class_weights"]
    num_epochs = marbert_config["num_epochs_configured"]
    log(f"Reusing MARBERT protocol: max_length={max_length} batch_size={batch_size} loss_variant={loss_variant} num_epochs={num_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    train = af_tf.load_split("train")
    val = af_tf.load_split("val_natural")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_cache = CACHE_DIR / f"train_challenger_maxlen{max_length}"
    val_cache = CACHE_DIR / f"val_challenger_maxlen{max_length}"
    if train_cache.exists():
        from datasets import load_from_disk
        ds_train = load_from_disk(str(train_cache))
        ds_val = load_from_disk(str(val_cache))
    else:
        ds_train = af_tf.tokenize_dataframe(train, tokenizer, max_length)
        ds_val = af_tf.tokenize_dataframe(val, tokenizer, max_length)
        ds_train.save_to_disk(str(train_cache))
        ds_val.save_to_disk(str(val_cache))

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=3)

    steps_per_epoch = max(1, len(ds_train) // batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.06 * total_steps)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / "run"),
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
    )

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

    history = []

    class EpochGateCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return control
            macro_f1 = metrics.get("eval_macro_f1", 0.0)
            neutral_f1 = metrics.get("eval_neutral_mixed_f1", 0.0)
            if history:
                prev_macro, prev_neutral = history[-1]
                improved = (macro_f1 - prev_macro >= MEANINGFUL_DELTA) and (neutral_f1 - prev_neutral >= MEANINGFUL_DELTA)
            else:
                improved = True
            history.append((macro_f1, neutral_f1))
            log(f"epoch={state.epoch} eval_macro_f1={macro_f1:.4f} eval_neutral_mixed_f1={neutral_f1:.4f} improved={improved}")
            if not improved and state.epoch and state.epoch >= 2:
                control.should_training_stop = True
            return control

    trainer = trainer_cls(
        model=model, args=training_args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=collator, compute_metrics=af_tf.compute_hf_metrics, callbacks=[EpochGateCallback()],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    final_metrics = trainer.evaluate()
    log(f"Challenger training complete in {train_seconds:.1f}s. Final eval: {final_metrics}")

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
        "num_epochs_actually_run": trainer.state.epoch,
        "loss_variant": loss_variant,
        "class_weights": class_weights,
        "train_seconds": train_seconds,
        "peak_vram_mb": peak_vram_mb,
        "final_eval_metrics": final_metrics,
        "epoch_history": history,
        "protocol_note": "Reused MARBERT's max_length/batch_size/loss_variant/epoch-budget verbatim (Gate 17: same protocol, not re-tuned per model).",
    }
    (final_dir / "training_config.json").write_text(json.dumps(training_config, indent=2, default=str), encoding="utf-8")
    (REPORTS_DIR / "challenger_training_manifest.json").write_text(json.dumps(training_config, indent=2, default=str), encoding="utf-8")

    # predictions on all eval splits for later comparison/significance testing
    pred_dir = ARTIFACT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)
    import pandas as pd
    for split_name in ["val_natural", "val_balanced", "test_natural", "item_holdout_stress"]:
        df = af_tf.load_split(split_name)
        ds = af_tf.tokenize_dataframe(df, tokenizer, max_length)
        preds_out = trainer.predict(ds)
        probs = af_tf.softmax(preds_out.predictions)
        pred_labels = probs.argmax(axis=1)
        out = pd.DataFrame({
            "review_uid": df["review_uid"].values,
            "true_label": df["label"].values,
            "pred_label": pred_labels,
            "proba_negative": probs[:, 0], "proba_neutral_mixed": probs[:, 1], "proba_positive": probs[:, 2],
        })
        out.to_parquet(pred_dir / f"{split_name}_predictions.parquet", index=False)
        log(f"{split_name}: challenger macro_f1={af_tf.compute_hf_metrics((preds_out.predictions, df['label'].values))['macro_f1']:.4f}")

    print("\nDONE. Challenger training_config:")
    print(json.dumps({k: v for k, v in training_config.items() if k != "log_history"}, indent=2, default=str))


if __name__ == "__main__":
    main()
