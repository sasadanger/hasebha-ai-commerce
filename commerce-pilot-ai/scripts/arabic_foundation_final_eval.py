"""Gate 21: final test evaluation of the FROZEN MARBERT primary model, once, on every fixed
LABR split (val_natural/val_balanced/test_natural/item_holdout_stress). Only run this after
preprocessing/label-contract/max-length/loss/model/calibration/challenger-decision are all
frozen. Compares directly against the frozen baseline on identical review_uid rows.

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_final_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nlp.arabic_foundation import transformer as af_tf  # noqa: E402

FINAL_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "primary_model" / "final"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
PRED_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "primary_model" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def full_metrics(y_true, y_pred, y_prob) -> dict:
    from sklearn.metrics import (
        f1_score, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score,
        matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score,
    )

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    onehot = np.eye(3)[y_true]
    try:
        roc_auc_ovr = roc_auc_score(onehot, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc_ovr = None
    pr_auc = {}
    for c in [0, 1, 2]:
        try:
            pr_auc[str(c)] = float(average_precision_score(onehot[:, c], y_prob[:, c]))
        except ValueError:
            pr_auc[str(c)] = None

    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            "negative": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "neutral_mixed": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
            "positive": {"precision": float(precision[2]), "recall": float(recall[2]), "f1": float(f1[2]), "support": int(support[2])},
        },
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "roc_auc_macro_ovr": roc_auc_ovr,
        "pr_auc_per_class": pr_auc,
        "brier_multiclass": af_tf.multiclass_brier_score(y_true, y_prob),
        "ece": af_tf.multiclass_ece(y_true, y_prob),
        "prediction_distribution": {str(c): int((y_pred == c).sum()) for c in [0, 1, 2]},
        "true_distribution": {str(c): int((np.asarray(y_true) == c).sum()) for c in [0, 1, 2]},
    }


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    log(f"Loading frozen model/tokenizer from {FINAL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    training_config = json.loads((FINAL_DIR / "training_config.json").read_text(encoding="utf-8"))
    max_length = training_config["max_length"]

    calibration_path = FINAL_DIR / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else None
    use_calibration = calibration is not None and calibration["decision"] == "USE_CALIBRATED"
    temperature = calibration["temperature"] if calibration else 1.0
    log(f"Calibration: use_calibration={use_calibration}, temperature={temperature}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    all_metrics = {}
    all_metrics["calibration_used"] = use_calibration
    all_metrics["temperature"] = temperature

    latency_results = {}

    for split_name in ["val_natural", "val_balanced", "test_natural", "item_holdout_stress"]:
        df = af_tf.load_split(split_name)
        ds = af_tf.tokenize_dataframe(df, tokenizer, max_length)

        # batch inference throughput
        model_input_keys = {"input_ids", "attention_mask", "token_type_ids"}
        t0 = time.time()
        all_logits = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(ds), batch_size):
                rows = [
                    {k: v for k, v in ds[j].items() if k in model_input_keys}
                    for j in range(i, min(i + batch_size, len(ds)))
                ]
                batch = collator(rows)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                all_logits.append(out.logits.cpu().numpy())
        logits = np.concatenate(all_logits, axis=0)
        batch_seconds = time.time() - t0

        # single-text latency (n=50 samples, one at a time)
        n_lat = min(50, len(df))
        single_texts = df[af_tf.TEXT_COL].iloc[:n_lat].tolist()
        single_times = []
        with torch.no_grad():
            for t in single_texts:
                t0s = time.time()
                enc = tokenizer(t, truncation=True, max_length=max_length, return_tensors="pt").to(device)
                _ = model(**enc)
                if device == "cuda":
                    torch.cuda.synchronize()
                single_times.append(time.time() - t0s)

        latency_results[split_name] = {
            "batch_inference_seconds_total": batch_seconds,
            "batch_inference_rows_per_second": len(df) / batch_seconds,
            "single_text_latency_ms_mean": float(np.mean(single_times) * 1000),
            "single_text_latency_ms_p95": float(np.percentile(single_times, 95) * 1000),
        }

        probs_raw = af_tf.softmax(logits)
        probs_used = af_tf.temperature_scale(logits, temperature) if use_calibration else probs_raw
        preds = probs_used.argmax(axis=1)
        y_true = df["label"].values

        m = full_metrics(y_true, preds, probs_used)
        m["latency"] = latency_results[split_name]
        all_metrics[split_name] = m
        log(f"{split_name}: macro_f1={m['macro_f1']:.4f} neutral_mixed_f1={m['per_class']['neutral_mixed']['f1']:.4f} n={m['n']}")

        out = pd.DataFrame({
            "review_uid": df["review_uid"].values,
            "true_label": y_true,
            "pred_label": preds,
            "proba_negative": probs_used[:, 0],
            "proba_neutral_mixed": probs_used[:, 1],
            "proba_positive": probs_used[:, 2],
        })
        out.to_parquet(PRED_DIR / f"{split_name}_predictions.parquet", index=False)

    (REPORTS_DIR / "final_test_evaluation.json").write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf-8")
    log("Wrote final_test_evaluation.json")

    # ---- direct comparison against frozen baseline on identical rows ----
    baseline_manifest = json.loads((REPORTS_DIR / "baseline_manifest.json").read_text(encoding="utf-8"))
    comparison = {}
    for split_name in ["val_natural", "val_balanced", "test_natural", "item_holdout_stress"]:
        base_preds = pd.read_parquet(REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "baseline" / "predictions" / f"{split_name}_predictions.parquet")
        marbert_preds = pd.read_parquet(PRED_DIR / f"{split_name}_predictions.parquet")
        merged = base_preds.merge(marbert_preds, on="review_uid", suffixes=("_baseline", "_marbert"), validate="one_to_one")
        assert (merged["true_label_baseline"] == merged["true_label_marbert"]).all(), "row mismatch between baseline and MARBERT predictions"
        comparison[split_name] = {
            "n_rows_compared": len(merged),
            "baseline_macro_f1": baseline_manifest["eval_summary"][split_name]["macro_f1"],
            "marbert_macro_f1": all_metrics[split_name]["macro_f1"],
            "delta": all_metrics[split_name]["macro_f1"] - baseline_manifest["eval_summary"][split_name]["macro_f1"],
        }
    (REPORTS_DIR / "baseline_vs_marbert_comparison.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    main()
