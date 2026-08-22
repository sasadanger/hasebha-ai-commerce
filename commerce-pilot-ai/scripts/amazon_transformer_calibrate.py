"""Gate 7: natural-distribution calibration.

Loads the frozen, Gate-6-selected transformer checkpoint, computes RAW logits on val_natural
(the naturally-distributed validation set built in Gate 3 -- NEVER a test set), fits a single-
scalar temperature via NLL minimization, and picks an operational decision threshold using
val_natural only. Saves both raw and calibrated predictions, reports Brier score and ECE for
both. No test set is touched anywhere in this script.

Run once from repo root (after scripts/amazon_transformer_train.py has produced
artifacts/experiments/amazon/transformer/model/):
  .venv/Scripts/python.exe scripts/amazon_transformer_calibrate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.amazon import transformer as amz_tf  # noqa: E402

SEED = amz_tf.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "amazon"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "experiments" / "amazon" / "transformer"
MODEL_DIR = ARTIFACTS_DIR / "model"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_run = json.loads((REPORTS_DIR / "transformer_training_run.json").read_text())
    max_length = train_run["max_length"]
    batch_size = train_run["batch_size"]
    log(f"Loading frozen checkpoint from {MODEL_DIR} (max_length={max_length})")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to("cuda")
    model.eval()

    full_pool = amz_tf.reproduce_full_pool_with_text(seed=SEED)
    val_natural = amz_tf.load_split_with_text("val_natural", full_pool)
    log(f"val_natural n={len(val_natural)}, label counts={val_natural['label'].value_counts().to_dict()}")

    val_natural_ds = amz_tf.tokenize_dataframe(val_natural, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    infer_args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR / "_calibration_infer_tmp"),
        per_device_eval_batch_size=batch_size * 2,
        bf16=True,
        tf32=True,
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=infer_args, data_collator=collator)

    t0 = time.time()
    pred_output = trainer.predict(val_natural_ds.remove_columns(["review_uid"]))
    infer_seconds = time.time() - t0
    logits = np.asarray(pred_output.predictions)
    labels = np.asarray(pred_output.label_ids)
    log(f"Inference on val_natural done in {infer_seconds:.2f}s ({len(val_natural)/infer_seconds:.1f} examples/sec)")

    raw_probs = amz_tf.softmax(logits)
    raw_positive_prob = raw_probs[:, 1]

    # ---- fit temperature on val_natural only -------------------------------------------------
    temperature = amz_tf.fit_temperature(logits, labels)
    calibrated_probs = amz_tf.temperature_scale(logits, temperature)
    calibrated_positive_prob = calibrated_probs[:, 1]
    log(f"Fitted temperature: {temperature:.4f}")

    # ---- default (argmax / 0.5) decision rule metrics ----------------------------------------
    default_preds = (raw_positive_prob >= 0.5).astype(int)
    default_acc = float((default_preds == labels).mean())

    raw_brier = amz_tf.brier_score(labels, raw_positive_prob)
    raw_ece = amz_tf.expected_calibration_error(labels, raw_positive_prob)
    calibrated_brier = amz_tf.brier_score(labels, calibrated_positive_prob)
    calibrated_ece = amz_tf.expected_calibration_error(labels, calibrated_positive_prob)
    log(f"RAW: brier={raw_brier:.4f} ece={raw_ece:.4f} | CALIBRATED: brier={calibrated_brier:.4f} ece={calibrated_ece:.4f}")

    # ---- operational threshold selection on val_natural only (maximize macro-F1) -------------
    from sklearn.metrics import f1_score

    candidate_thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold, best_macro_f1 = 0.5, -1.0
    threshold_curve = []
    for thr in candidate_thresholds:
        preds = (calibrated_positive_prob >= thr).astype(int)
        mf1 = f1_score(labels, preds, average="macro", zero_division=0)
        threshold_curve.append({"threshold": float(thr), "macro_f1": float(mf1)})
        if mf1 > best_macro_f1:
            best_macro_f1, best_threshold = mf1, float(thr)
    log(f"Selected operational threshold (val_natural only, calibrated probs, max macro-F1): {best_threshold:.3f} (macro_f1={best_macro_f1:.4f})")

    # ---- save raw + calibrated predictions on val_natural -------------------------------------
    pred_dir = REPORTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    val_natural_pred_df = pd.DataFrame(
        {
            "id": val_natural["review_uid"].to_numpy(),
            "true_label": labels,
            "raw_positive_prob": raw_positive_prob,
            "calibrated_positive_prob": calibrated_positive_prob,
            "predicted_label_default_threshold": default_preds,
            "predicted_label_selected_threshold": (calibrated_positive_prob >= best_threshold).astype(int),
        }
    )
    val_natural_pred_df.to_parquet(pred_dir / "val_natural_transformer_predictions.parquet", index=False)
    log(f"Saved {pred_dir / 'val_natural_transformer_predictions.parquet'}")

    calibration_config = {
        "generated_at": "2026-08-17",
        "calibration_method": "single-scalar temperature scaling (Guo et al. 2017), fit by NLL "
        "minimization via numpy finite-difference gradient descent -- see fit_temperature() in "
        "src/nlp/amazon/transformer.py",
        "fit_on": "val_natural ONLY (reports/generated/amazon/split_ids/val_natural.parquet) -- "
        "never any test set",
        "n_val_natural": len(val_natural),
        "val_natural_label_counts": {str(k): int(v) for k, v in val_natural["label"].value_counts().items()},
        "temperature": temperature,
        "default_threshold": {
            "threshold": 0.5,
            "accuracy": default_acc,
        },
        "selected_operational_threshold": {
            "threshold": best_threshold,
            "selection_method": "argmax of macro-F1 over calibrated positive-class probability, "
            "swept threshold 0.05-0.95 step 0.005, on val_natural only",
            "val_natural_macro_f1_at_threshold": best_macro_f1,
        },
        "threshold_sweep_curve": threshold_curve,
        "raw": {"brier_score": raw_brier, "ece": raw_ece},
        "calibrated": {"brier_score": calibrated_brier, "ece": calibrated_ece},
        "inference_examples_per_sec": len(val_natural) / infer_seconds,
        "predictions_saved_to": "reports/generated/amazon/predictions/val_natural_transformer_predictions.parquet",
    }
    (REPORTS_DIR / "transformer_calibration.json").write_text(json.dumps(calibration_config, indent=2, default=str))
    log("Wrote reports/generated/amazon/transformer_calibration.json")

    import shutil

    shutil.rmtree(ARTIFACTS_DIR / "_calibration_infer_tmp", ignore_errors=True)


if __name__ == "__main__":
    main()
