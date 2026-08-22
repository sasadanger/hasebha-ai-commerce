"""Gate 8: final, one-time evaluation of the frozen transformer on the four fixed eval sets, with
an honest, same-row-ID comparison against the frozen TF-IDF+LinearSVC baseline.

This is run EXACTLY ONCE per eval set, after every model/calibration/threshold decision from
Gates 5-7 is frozen. Test sets are never touched before this point.

Run once from repo root (after amazon_transformer_train.py and amazon_transformer_calibrate.py):
  .venv/Scripts/python.exe scripts/amazon_transformer_final_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.amazon import transformer as amz_tf  # noqa: E402

SEED = amz_tf.SEED
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "amazon"
PRED_DIR = REPORTS_DIR / "predictions"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "experiments" / "amazon" / "transformer"
MODEL_DIR = ARTIFACTS_DIR / "model"

EVAL_SETS = ["test_balanced", "test_representative", "chronological_stress", "product_holdout_stress"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def full_metric_set(y_true, y_pred, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {
        "n": int(len(y_true)),
        "prevalence": {"negative": float((y_true == 0).mean()), "positive": float((y_true == 1).mean())},
        "confusion_matrix": {"labels": ["negative", "positive"], "matrix": cm.tolist(), "reading": "rows=true, cols=predicted"},
        "per_class": {
            "negative": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "positive": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        },
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else float("nan"),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            out["average_precision"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = None
            out["average_precision"] = None
    return out


def slice_report(df: pd.DataFrame, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Slice results by rating, review length tertile, verified status, time period, product freq."""
    work = df.copy()
    work["_pred"] = y_pred
    work["_true"] = y_true

    def macro_f1_of(sub: pd.DataFrame) -> float | None:
        if sub["_true"].nunique() < 1 or len(sub) == 0:
            return None
        return float(f1_score(sub["_true"], sub["_pred"], average="macro", zero_division=0))

    slices: dict = {}

    slices["by_rating"] = {
        str(r): {"n": int((work["rating"] == r).sum()), "macro_f1": macro_f1_of(work[work["rating"] == r]), "accuracy": float((work.loc[work["rating"] == r, "_pred"] == work.loc[work["rating"] == r, "_true"]).mean()) if (work["rating"] == r).sum() else None}
        for r in [1.0, 2.0, 4.0, 5.0]
    }

    work["text_len"] = work[amz_tf.TEXT_COL].str.len()
    try:
        tertiles = pd.qcut(work["text_len"], 3, labels=["short", "medium", "long"], duplicates="drop")
        work["length_band"] = tertiles
        slices["by_length_tertile"] = {
            str(band): {"n": int((work["length_band"] == band).sum()), "macro_f1": macro_f1_of(work[work["length_band"] == band])}
            for band in work["length_band"].dropna().unique()
        }
    except ValueError as exc:
        slices["by_length_tertile"] = {"error": str(exc)}

    if "verified_purchase" in work.columns:
        slices["by_verified_status"] = {
            str(v): {"n": int((work["verified_purchase"] == v).sum()), "macro_f1": macro_f1_of(work[work["verified_purchase"] == v])}
            for v in work["verified_purchase"].unique()
        }

    if "review_datetime_utc" in work.columns:
        work["year"] = pd.to_datetime(work["review_datetime_utc"]).dt.year
        slices["by_year"] = {
            str(y): {"n": int((work["year"] == y).sum()), "macro_f1": macro_f1_of(work[work["year"] == y])}
            for y in sorted(work["year"].dropna().unique())
        }

    if "parent_asin" in work.columns:
        freq = work.groupby("parent_asin").size()
        work["product_freq"] = work["parent_asin"].map(freq)
        bands = pd.cut(work["product_freq"], bins=[0, 1, 5, np.inf], labels=["unseen_or_singleton", "rare_2_5", "frequent_6_plus"])
        work["product_freq_band"] = bands
        slices["by_product_frequency_band"] = {
            str(b): {"n": int((work["product_freq_band"] == b).sum()), "macro_f1": macro_f1_of(work[work["product_freq_band"] == b])}
            for b in work["product_freq_band"].dropna().unique()
        }

    return slices


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_run = json.loads((REPORTS_DIR / "transformer_training_run.json").read_text())
    calibration = json.loads((REPORTS_DIR / "transformer_calibration.json").read_text())
    max_length = train_run["max_length"]
    batch_size = train_run["batch_size"]
    temperature = calibration["temperature"]
    selected_threshold = calibration["selected_operational_threshold"]["threshold"]
    log(f"max_length={max_length}, batch_size={batch_size}, temperature={temperature:.4f}, selected_threshold={selected_threshold:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to("cuda")
    model.eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    infer_args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR / "_final_eval_infer_tmp"),
        per_device_eval_batch_size=batch_size * 2,
        bf16=True,
        tf32=True,
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=infer_args, data_collator=collator)

    full_pool = amz_tf.reproduce_full_pool_with_text(seed=SEED)

    # ---- load frozen TF-IDF baseline metrics + predictions for same-row-ID comparison ---------
    tfidf_metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())

    results = {}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for name in EVAL_SETS:
        log(f"=== Evaluating on {name} ===")
        df = amz_tf.load_split_with_text(name, full_pool)
        ds = amz_tf.tokenize_dataframe(df, tokenizer, max_length)

        t0 = time.time()
        pred_out = trainer.predict(ds.remove_columns(["review_uid"]))
        infer_seconds = time.time() - t0
        logits = np.asarray(pred_out.predictions)
        y_true = np.asarray(pred_out.label_ids)
        raw_probs = amz_tf.softmax(logits)
        raw_positive_prob = raw_probs[:, 1]
        calibrated_probs = amz_tf.temperature_scale(logits, temperature)
        calibrated_positive_prob = calibrated_probs[:, 1]

        y_pred_default = (raw_positive_prob >= 0.5).astype(int)
        y_pred_selected_threshold = (calibrated_positive_prob >= selected_threshold).astype(int)

        metrics_default = full_metric_set(y_true, y_pred_default, raw_positive_prob)
        metrics_selected = full_metric_set(y_true, y_pred_selected_threshold, calibrated_positive_prob)

        raw_brier = amz_tf.brier_score(y_true, raw_positive_prob)
        raw_ece = amz_tf.expected_calibration_error(y_true, raw_positive_prob)
        calibrated_brier = amz_tf.brier_score(y_true, calibrated_positive_prob)
        calibrated_ece = amz_tf.expected_calibration_error(y_true, calibrated_positive_prob)

        latency_ms_per_1000 = (infer_seconds / len(df)) * 1000.0 * 1000.0
        throughput_eps = len(df) / infer_seconds

        slices = slice_report(df, y_pred_default, y_true)

        # ---- save predictions (same id/true_label/predicted_label/score pattern as TF-IDF) ----
        pred_df = pd.DataFrame(
            {
                "id": df["review_uid"].to_numpy(),
                "true_label": y_true,
                "predicted_label": y_pred_default,
                "score": raw_positive_prob,
                "calibrated_score": calibrated_positive_prob,
                "predicted_label_selected_threshold": y_pred_selected_threshold,
            }
        )
        pred_df.to_parquet(PRED_DIR / f"{name}_transformer_predictions.parquet", index=False)

        # ---- same-row-ID comparison against frozen TF-IDF predictions -------------------------
        tfidf_pred_df = pd.read_parquet(PRED_DIR / f"{name}_predictions.parquet")
        joined = pred_df.merge(tfidf_pred_df, on="id", suffixes=("_transformer", "_tfidf"), how="inner")
        assert len(joined) == len(pred_df) == len(tfidf_pred_df), (
            f"{name}: row-ID join mismatch -- transformer n={len(pred_df)}, tfidf n={len(tfidf_pred_df)}, "
            f"joined n={len(joined)}. Comparison sets must be identical row IDs."
        )
        transformer_correct = joined["predicted_label_transformer"] == joined["true_label_transformer"]
        tfidf_correct = joined["predicted_label_tfidf"] == joined["true_label_tfidf"]
        agreement = float((joined["predicted_label_transformer"] == joined["predicted_label_tfidf"]).mean())
        both_correct = int((transformer_correct & tfidf_correct).sum())
        only_transformer_correct = int((transformer_correct & ~tfidf_correct).sum())
        only_tfidf_correct = int((~transformer_correct & tfidf_correct).sum())
        neither_correct = int((~transformer_correct & ~tfidf_correct).sum())

        tfidf_set_metrics = tfidf_metrics["results"][name]["final_model"]
        same_row_comparison = {
            "n_joined_rows": int(len(joined)),
            "prediction_agreement_rate": agreement,
            "both_correct": both_correct,
            "only_transformer_correct": only_transformer_correct,
            "only_tfidf_correct": only_tfidf_correct,
            "neither_correct": neither_correct,
            "transformer_macro_f1_default_threshold": metrics_default["macro_f1"],
            "tfidf_macro_f1_frozen": tfidf_set_metrics["macro_f1"],
            "macro_f1_delta_transformer_minus_tfidf": metrics_default["macro_f1"] - tfidf_set_metrics["macro_f1"],
            "transformer_wins": metrics_default["macro_f1"] > tfidf_set_metrics["macro_f1"],
        }
        log(f"  [{name}] transformer macro_f1={metrics_default['macro_f1']:.4f} vs tfidf macro_f1={tfidf_set_metrics['macro_f1']:.4f} (delta={same_row_comparison['macro_f1_delta_transformer_minus_tfidf']:+.4f})")

        results[name] = {
            "at_default_threshold_0_5_raw_scores": metrics_default,
            "at_selected_threshold_calibrated_scores": {"threshold": selected_threshold, **metrics_selected},
            "calibration": {
                "raw": {"brier_score": raw_brier, "ece": raw_ece},
                "calibrated": {"brier_score": calibrated_brier, "ece": calibrated_ece},
            },
            "inference": {"latency_ms_per_1000_reviews": latency_ms_per_1000, "throughput_examples_per_sec": throughput_eps},
            "slices": slices,
            "comparison_to_frozen_tfidf_baseline_same_row_ids": same_row_comparison,
        }

    # ---- overall honest verdict ----------------------------------------------------------------
    wins = sum(1 for r in results.values() if r["comparison_to_frozen_tfidf_baseline_same_row_ids"]["transformer_wins"])
    deltas = {name: r["comparison_to_frozen_tfidf_baseline_same_row_ids"]["macro_f1_delta_transformer_minus_tfidf"] for name, r in results.items()}
    verdict = {
        "eval_sets_transformer_wins_macro_f1": wins,
        "eval_sets_total": len(EVAL_SETS),
        "macro_f1_deltas_by_set": deltas,
        "mean_delta": float(np.mean(list(deltas.values()))),
    }
    log(f"OVERALL VERDICT: {verdict}")

    out = {
        "generated_at": "2026-08-17",
        "checkpoint": train_run["checkpoint"],
        "max_length": max_length,
        "temperature": temperature,
        "selected_threshold": selected_threshold,
        "results": results,
        "overall_verdict": verdict,
    }
    (REPORTS_DIR / "transformer_final_eval.json").write_text(json.dumps(out, indent=2, default=str))
    log("Wrote reports/generated/amazon/transformer_final_eval.json")

    import shutil

    shutil.rmtree(ARTIFACTS_DIR / "_final_eval_infer_tmp", ignore_errors=True)


if __name__ == "__main__":
    main()
