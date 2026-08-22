"""Gates 5 & 6: smoke test, then the one controlled 50k-row transformer training run, plus the
gated 100k-expansion decision.

Run once from repo root (after scripts/amazon_transformer_gate4_token_length.py has written
reports/generated/amazon/transformer_token_length_audit.json, which supplies max_length):
  .venv/Scripts/python.exe scripts/amazon_transformer_train.py
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
    TrainerCallback,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.amazon import transformer as amz_tf  # noqa: E402

CHECKPOINT_PRIMARY = "distilroberta-base"
CHECKPOINT_FALLBACK = "distilbert-base-uncased"
SEED = amz_tf.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "amazon"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "experiments" / "amazon" / "transformer"
CACHE_DIR = ARTIFACTS_DIR / "tokenized_cache"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_checkpoint():
    """Load the primary checkpoint; fall back only if it genuinely fails to load."""
    try:
        tok = AutoTokenizer.from_pretrained(CHECKPOINT_PRIMARY)
        model_fn = lambda: AutoModelForSequenceClassification.from_pretrained(  # noqa: E731
            CHECKPOINT_PRIMARY, num_labels=2
        )
        model_fn()  # verify it actually loads before committing
        return CHECKPOINT_PRIMARY, tok, model_fn, False
    except Exception as exc:  # noqa: BLE001
        log(f"PRIMARY checkpoint {CHECKPOINT_PRIMARY} failed to load ({exc}); falling back to {CHECKPOINT_FALLBACK}")
        tok = AutoTokenizer.from_pretrained(CHECKPOINT_FALLBACK)
        model_fn = lambda: AutoModelForSequenceClassification.from_pretrained(  # noqa: E731
            CHECKPOINT_FALLBACK, num_labels=2
        )
        return CHECKPOINT_FALLBACK, tok, model_fn, True


class NaturalValReportingCallback(TrainerCallback):
    """Runs a REPORTING-ONLY prediction pass on val_natural after each epoch's evaluation. Never
    writes into `metrics`/`control`, so it cannot influence load_best_model_at_end, checkpoint
    selection, or early stopping -- those are driven exclusively by the balanced `val` set passed
    as Trainer's eval_dataset. Satisfies Gate 6's "evaluate on both validation sets" reporting
    requirement while honoring Gate 5's "natural val reserved for Gate 7 calibration only, not
    used to pick training config" rule.
    """

    def __init__(self, trainer_ref_holder: dict, val_natural_ds, records: list):
        self.trainer_ref_holder = trainer_ref_holder
        self.val_natural_ds = val_natural_ds
        self.records = records

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref_holder["trainer"]
        pred = trainer.predict(self.val_natural_ds, metric_key_prefix="val_natural_reporting_only")
        metrics = {k: v for k, v in pred.metrics.items() if isinstance(v, (int, float))}
        metrics["epoch"] = state.epoch
        self.records.append(metrics)
        log(f"  [reporting-only, NOT used for any training decision] val_natural epoch={state.epoch}: {metrics}")


def dry_run_batch_size(model_fn, tokenizer, collator, max_length, candidate_sizes, sample_texts):
    """Try candidate batch sizes largest-first; return the largest that completes a forward+
    backward pass without OOM. This is a genuine dry run on real GPU memory, not a guess.
    """
    for bs in candidate_sizes:
        try:
            model = model_fn().to("cuda")
            model.train()
            batch_texts = (sample_texts * ((bs // len(sample_texts)) + 1))[:bs]
            enc = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            enc = {k: v.to("cuda") for k, v in enc.items()}
            labels = torch.zeros(bs, dtype=torch.long, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**enc, labels=labels)
            out.loss.backward()
            torch.cuda.synchronize()
            del model, out, enc, labels
            torch.cuda.empty_cache()
            log(f"  batch_size={bs}: OK")
            return bs
        except torch.OutOfMemoryError:
            log(f"  batch_size={bs}: OOM, trying smaller")
            torch.cuda.empty_cache()
            continue
    raise RuntimeError("Even the smallest candidate batch size OOM'd -- cannot proceed")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    token_audit = json.loads((REPORTS_DIR / "transformer_token_length_audit.json").read_text())
    max_length = token_audit["decision"]["selected_max_length"]
    log(f"max_length selected in Gate 4: {max_length}")

    log("Loading checkpoint (primary distilroberta-base, fallback distilbert-base-uncased)...")
    checkpoint_name, tokenizer, model_fn, used_fallback = load_checkpoint()
    log(f"Using checkpoint: {checkpoint_name} (fallback used: {used_fallback})")

    log("Reproducing full pool with text and rejoining splits...")
    full_pool = amz_tf.reproduce_full_pool_with_text(seed=SEED)
    train_50k = amz_tf.load_split_with_text("learning_curve_50000", full_pool)
    val = amz_tf.load_split_with_text("val", full_pool)
    val_natural = amz_tf.load_split_with_text("val_natural", full_pool)
    log(f"train_50k n={len(train_50k)} (label counts: {train_50k['label'].value_counts().to_dict()})")
    log(f"val n={len(val)}, val_natural n={len(val_natural)}")

    # ---- sub-rating mix / spread report for the reused 50k training subset (Gate 3 requirement)
    sub_rating = amz_data_sub_rating = __import__("src.nlp.amazon.data", fromlist=["sub_rating_breakdown"]).sub_rating_breakdown(train_50k)
    log(f"train_50k sub-rating breakdown: {sub_rating}")
    time_span = (train_50k["review_datetime_utc"].min(), train_50k["review_datetime_utc"].max())
    log(f"train_50k timestamp span: {time_span}")
    n_products = train_50k["parent_asin"].nunique()
    log(f"train_50k unique products: {n_products}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # ---- tokenize + cache -------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_cache_path = CACHE_DIR / f"train_50k_maxlen{max_length}_{checkpoint_name.replace('/', '_')}"
    val_cache_path = CACHE_DIR / f"val_maxlen{max_length}_{checkpoint_name.replace('/', '_')}"
    val_natural_cache_path = CACHE_DIR / f"val_natural_maxlen{max_length}_{checkpoint_name.replace('/', '_')}"

    from datasets import load_from_disk

    def tok_or_load(df, cache_path):
        if cache_path.exists():
            log(f"  loading cached tokenized dataset: {cache_path}")
            return load_from_disk(str(cache_path))
        log(f"  tokenizing and caching: {cache_path}")
        ds = amz_tf.tokenize_dataframe(df, tokenizer, max_length)
        ds.save_to_disk(str(cache_path))
        return ds

    log("Tokenizing (or loading cached tokenization)...")
    train_ds = tok_or_load(train_50k, train_cache_path)
    val_ds = tok_or_load(val, val_cache_path)
    val_natural_ds = tok_or_load(val_natural, val_natural_cache_path)

    # ---- dry run: find largest stable batch size --------------------------------------------
    log("Dry run: finding largest stable batch size...")
    sample_texts = train_50k[amz_tf.TEXT_COL].tolist()[:8]
    batch_size = dry_run_batch_size(
        model_fn, tokenizer, collator, max_length, candidate_sizes=[128, 96, 64, 48, 32], sample_texts=sample_texts
    )
    log(f"Selected batch_size={batch_size}")

    # ---- Gate 5 smoke test: 300 steps on a subset --------------------------------------------
    log("=== SMOKE TEST (300 steps) ===")
    smoke_out_dir = ARTIFACTS_DIR / "_smoke_test"
    smoke_model = model_fn().to("cuda")
    smoke_warmup_steps = max(1, round(0.06 * 300))
    smoke_args = TrainingArguments(
        output_dir=str(smoke_out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        max_steps=300,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=smoke_warmup_steps,
        logging_steps=25,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=300,
        save_total_limit=1,
        bf16=True,
        tf32=True,
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )
    smoke_trainer = Trainer(
        model=smoke_model,
        args=smoke_args,
        train_dataset=train_ds.remove_columns(["review_uid"]),
        eval_dataset=val_ds.remove_columns(["review_uid"]),
        data_collator=collator,
        compute_metrics=amz_tf.compute_hf_metrics,
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    smoke_result = smoke_trainer.train()
    smoke_seconds = time.time() - t0
    smoke_peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    log_history = smoke_trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    loss_decreased = len(train_losses) >= 2 and train_losses[-1] < train_losses[0]
    grads_finite = all(
        torch.isfinite(p.grad).all().item() for p in smoke_model.parameters() if p.grad is not None
    )
    smoke_eval = smoke_trainer.evaluate()

    # label-inversion spot check: pull 6 rows straight from the tokenized dataset and confirm
    # their `labels` field matches the source rating (1-2 -> 0, 4-5 -> 1)
    spot_check_idx = [0, 1, len(train_50k) // 2, len(train_50k) // 2 + 1, -2, -1]
    spot_checks = []
    for i in spot_check_idx:
        row = train_50k.iloc[i]
        expected_label = 0 if row["rating"] in (1.0, 2.0) else 1
        ds_label = train_ds[i % len(train_ds)]["labels"]
        spot_checks.append(
            {"rating": float(row["rating"]), "expected_label": expected_label, "dataset_label": int(ds_label), "match": expected_label == int(ds_label)}
        )
    label_inversion_ok = all(c["match"] for c in spot_checks)

    # eval-metrics-compute-correctly check: perfect predictions must give macro_f1 == 1.0
    synthetic_logits = np.array([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0]])
    synthetic_labels = np.array([0, 1, 0, 1])
    synthetic_metrics = amz_tf.compute_hf_metrics((synthetic_logits, synthetic_labels))
    eval_metrics_correct = abs(synthetic_metrics["macro_f1"] - 1.0) < 1e-9

    # checkpoint save/load smoke check
    ckpt_dirs = sorted(smoke_out_dir.glob("checkpoint-*"))
    checkpoint_save_load_ok = False
    if ckpt_dirs:
        try:
            reloaded = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dirs[-1]))
            checkpoint_save_load_ok = True
            del reloaded
        except Exception as exc:  # noqa: BLE001
            log(f"  checkpoint reload FAILED: {exc}")

    steps_per_sec = smoke_trainer.state.global_step / smoke_seconds
    examples_per_sec = steps_per_sec * batch_size
    full_run_steps = (len(train_50k) // batch_size) * 2  # 2 epochs
    est_full_seconds = full_run_steps / steps_per_sec

    smoke_report = {
        "checkpoint": checkpoint_name,
        "used_fallback_checkpoint": used_fallback,
        "batch_size": batch_size,
        "max_length": max_length,
        "steps_run": smoke_trainer.state.global_step,
        "seconds": smoke_seconds,
        "steps_per_sec": steps_per_sec,
        "examples_per_sec": examples_per_sec,
        "peak_vram_allocated_mb": smoke_peak_vram_mb,
        "train_loss_sequence": train_losses,
        "loss_decreased": loss_decreased,
        "gradients_finite": grads_finite,
        "label_inversion_spot_check": spot_checks,
        "label_inversion_check_passed": label_inversion_ok,
        "eval_metrics_compute_correctly": eval_metrics_correct,
        "checkpoint_save_load_ok": checkpoint_save_load_ok,
        "smoke_eval_on_val": {k: v for k, v in smoke_eval.items() if isinstance(v, (int, float))},
        "estimated_full_50k_2epoch_seconds": est_full_seconds,
        "estimated_full_50k_2epoch_minutes": est_full_seconds / 60,
    }
    log(f"SMOKE TEST REPORT: {json.dumps(smoke_report, indent=2, default=str)}")

    all_checks_passed = (
        loss_decreased and grads_finite and label_inversion_ok and eval_metrics_correct and checkpoint_save_load_ok
    )
    if not all_checks_passed:
        (REPORTS_DIR / "transformer_smoke_test.json").write_text(json.dumps(smoke_report, indent=2, default=str))
        raise RuntimeError(f"SMOKE TEST FAILED -- see reports/generated/amazon/transformer_smoke_test.json. Aborting before full run. {smoke_report}")

    (REPORTS_DIR / "transformer_smoke_test.json").write_text(json.dumps(smoke_report, indent=2, default=str))
    log("Smoke test PASSED. Proceeding to full 50k training run.")

    del smoke_model, smoke_trainer
    torch.cuda.empty_cache()
    import shutil

    shutil.rmtree(smoke_out_dir, ignore_errors=True)

    # ---- Gate 6: full 50k training run, up to 2 epochs ---------------------------------------
    log("=== FULL 50k TRAINING RUN (up to 2 epochs) ===")
    full_out_dir = ARTIFACTS_DIR / "checkpoints_50k"
    model = model_fn().to("cuda")
    full_total_steps = (len(train_50k) // batch_size) * 2
    full_warmup_steps = max(1, round(0.06 * full_total_steps))
    full_args = TrainingArguments(
        output_dir=str(full_out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=full_warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        bf16=True,
        tf32=True,
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )
    trainer_holder: dict = {}
    natural_val_records: list = []
    trainer = Trainer(
        model=model,
        args=full_args,
        train_dataset=train_ds.remove_columns(["review_uid"]),
        eval_dataset=val_ds.remove_columns(["review_uid"]),
        data_collator=collator,
        compute_metrics=amz_tf.compute_hf_metrics,
        callbacks=[NaturalValReportingCallback(trainer_holder, val_natural_ds.remove_columns(["review_uid"]), natural_val_records)],
    )
    trainer_holder["trainer"] = trainer

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_output = trainer.train()
    full_train_seconds = time.time() - t0
    full_peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    log(f"Full training done in {full_train_seconds:.1f}s ({full_train_seconds/60:.1f} min), peak VRAM {full_peak_vram_mb:.0f} MB")

    epoch_eval_records = [e for e in trainer.state.log_history if "eval_macro_f1" in e]
    epoch_train_loss_records = [e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    log(f"Per-epoch val (balanced) eval records: {json.dumps(epoch_eval_records, indent=2, default=str)}")
    log(f"Per-epoch val_natural (reporting only) records: {json.dumps(natural_val_records, indent=2, default=str)}")

    # ---- plateau / overfitting assessment (documented decision, see script docstring) --------
    if len(epoch_eval_records) >= 2:
        f1_e1, f1_e2 = epoch_eval_records[0]["eval_macro_f1"], epoch_eval_records[1]["eval_macro_f1"]
        loss_e1, loss_e2 = epoch_eval_records[0]["eval_loss"], epoch_eval_records[1]["eval_loss"]
        neg_f1_e1, neg_f1_e2 = epoch_eval_records[0]["eval_negative_f1"], epoch_eval_records[1]["eval_negative_f1"]
        plateaued = (f1_e2 - f1_e1) < 0.002
        overfitting = (loss_e2 > loss_e1) and (f1_e2 <= f1_e1)
        epoch_decision = {
            "epoch1_macro_f1": f1_e1,
            "epoch2_macro_f1": f1_e2,
            "epoch1_negative_f1": neg_f1_e1,
            "epoch2_negative_f1": neg_f1_e2,
            "epoch1_eval_loss": loss_e1,
            "epoch2_eval_loss": loss_e2,
            "plateaued": plateaued,
            "overfitting_signal": overfitting,
            "reasoning": (
                "Both epochs were run (max allowed by config is 2, so the cost of running epoch 2 "
                "to observe the actual trend -- rather than guessing after epoch 1 alone -- is "
                "just the epoch-2 wall-clock time, no extra hyperparameter search). "
                f"macro-F1 change epoch1->epoch2: {f1_e2-f1_e1:+.4f}. "
                + ("Plateaued/regressed -- epoch 1 model is likely as good or better." if plateaued or overfitting
                   else "Meaningful improvement -- epoch 2 model is the better checkpoint.")
            ),
            "best_model_loaded_by_trainer": "epoch with highest eval_macro_f1 (load_best_model_at_end=True)",
        }
    else:
        epoch_decision = {"note": "fewer than 2 epoch eval records logged", "records": epoch_eval_records}
    log(f"EPOCH DECISION: {epoch_decision}")

    # ---- 100k expansion decision --------------------------------------------------------------
    best_macro_f1 = max(e["eval_macro_f1"] for e in epoch_eval_records) if epoch_eval_records else None
    tfidf_baseline = json.loads((REPORTS_DIR / "metrics.json").read_text())
    tfidf_val_macro_f1_at_100k = tfidf_baseline["learning_curve_validation_results"]["tfidf_wordchar_linearsvc"]["100000"]["val_macro_f1"]
    expansion_decision = {
        "fifty_k_run_completed_cleanly": True,
        "fifty_k_best_val_macro_f1": best_macro_f1,
        "tfidf_baseline_val_macro_f1_at_100k_for_context": tfidf_val_macro_f1_at_100k,
        "epoch1_to_epoch2_gain": (epoch_decision.get("epoch2_macro_f1", 0) - epoch_decision.get("epoch1_macro_f1", 0)) if len(epoch_eval_records) >= 2 else None,
        "time_for_50k_2epoch_run_minutes": full_train_seconds / 60,
    }
    # Decision rule: expand to 100k only if the 50k run is already GPU/time-cheap (it is -- see
    # time_for_50k_2epoch_run_minutes) AND the in-run learning signal (epoch1->epoch2 gain) shows
    # the model was still improving substantially with more optimization steps late in training,
    # suggesting more DATA (not just more steps) plausibly has remaining gain to offer, AND the
    # 50k result has not already reached/exceeded the strong classical baseline (in which case
    # further scaling the transformer has little to prove). This mirrors Gate 6's literal
    # criteria without re-running a redundant 25k probe just to redraw the same curve TF-IDF
    # already drew for a different model family.
    still_improving = expansion_decision["epoch1_to_epoch2_gain"] is not None and expansion_decision["epoch1_to_epoch2_gain"] > 0.005
    already_strong = best_macro_f1 is not None and best_macro_f1 >= tfidf_val_macro_f1_at_100k - 0.005
    expand_to_100k = still_improving and not already_strong
    expansion_decision["still_improving_epoch_to_epoch"] = still_improving
    expansion_decision["already_matches_or_beats_tfidf_context_baseline"] = already_strong
    expansion_decision["DECISION_expand_to_100k"] = expand_to_100k
    expansion_decision["reasoning"] = (
        ("Expanding to 100k: epoch-to-epoch gain suggests the model has not saturated on 50k rows "
         "and more data is likely to help further.")
        if expand_to_100k else
        ("Stopping at 50k: " + (
            "the 50k result already matches/exceeds the TF-IDF 100k-row validation macro-F1, so "
            "there is no evidence more transformer training data would change the final Gate 8 "
            "conclusion enough to justify the extra runtime. "
            if already_strong else "") +
         ("epoch-to-epoch gain was small (<0.5pp), indicating the model is close to its ceiling on "
          "this amount of data already." if not still_improving else ""))
    )
    log(f"100K EXPANSION DECISION: {expansion_decision}")

    # ---- save the frozen 50k checkpoint as the selected model ---------------------------------
    final_dir = ARTIFACTS_DIR / "model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log(f"Saved final selected checkpoint + tokenizer to {final_dir}")

    result = {
        "generated_at": "2026-08-17",
        "checkpoint": checkpoint_name,
        "used_fallback_checkpoint": used_fallback,
        "max_length": max_length,
        "batch_size": batch_size,
        "training_size": len(train_50k),
        "train_config": {
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "num_train_epochs_max": 2,
            "warmup_ratio_target": 0.06,
            "warmup_steps_computed": full_warmup_steps,
            "warmup_note": "this transformers version (5.15.0) removed TrainingArguments.warmup_ratio; "
            "warmup_steps = round(0.06 * total_train_steps) is used instead to reproduce the same "
            "6% linear-warmup schedule.",
            "optimizer": "adamw_torch",
            "seed": SEED,
            "bf16": True,
            "tf32": True,
            "pad_to_multiple_of": 8,
            "metric_for_best_model": "eval_macro_f1",
            "load_best_model_at_end": True,
            "save_total_limit": 1,
        },
        "hardware": {
            "peak_vram_allocated_mb_full_run": full_peak_vram_mb,
            "full_train_seconds": full_train_seconds,
            "full_train_minutes": full_train_seconds / 60,
        },
        "train_50k_composition": {
            "sub_rating_breakdown": sub_rating,
            "timestamp_span": [str(time_span[0]), str(time_span[1])],
            "n_unique_products": int(n_products),
        },
        "per_epoch_eval_on_balanced_val": epoch_eval_records,
        "per_epoch_eval_on_val_natural_REPORTING_ONLY_not_used_for_any_decision": natural_val_records,
        "epoch_decision": epoch_decision,
        "hundred_k_expansion_decision": expansion_decision,
        "final_selected_checkpoint_dir": str(final_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    (REPORTS_DIR / "transformer_training_run.json").write_text(json.dumps(result, indent=2, default=str))
    log("Wrote reports/generated/amazon/transformer_training_run.json")


if __name__ == "__main__":
    main()
