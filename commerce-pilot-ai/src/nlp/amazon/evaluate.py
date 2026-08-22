"""Metrics computation for Amazon Appliances sentiment models.

Every model is scored uniformly: `score_model` calls `.predict` (and `.predict_proba` if
available) and times inference; `compute_metrics` turns true/predicted labels (+ optional
positive-class scores) into the full metric set the project brief requires -- macro/weighted F1,
per-class precision/recall/F1, confusion matrix, ROC-AUC, a calibration table, and accuracy
(reported alongside the others, never alone).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def score_model(model, texts) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Run a fitted model over `texts`. Returns (y_pred, y_score_or_None, ms_per_1000_reviews).

    y_score is the positive-class probability/score (column 1 of predict_proba) if the model
    supports predict_proba, else None. Timing covers `.predict` only (the deployment-relevant
    inference cost), measured once over the whole batch, converted to ms per 1000 reviews.
    """
    n = len(texts)
    start = time.perf_counter()
    y_pred = np.asarray(model.predict(texts))
    elapsed_seconds = time.perf_counter() - start
    ms_per_1000 = (elapsed_seconds / n) * 1000.0 * 1000.0 if n else float("nan")
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = np.asarray(model.predict_proba(texts))[:, 1]
    return y_pred, y_score, ms_per_1000


def calibration_table(y_true, y_score, n_bins: int = 10) -> list[dict]:
    """Reliability table: for `n_bins` equal-width bins of predicted positive-probability, report
    the mean predicted probability vs. the observed positive rate in that bin. Empty bins are
    skipped (documented, not padded with fabricated values).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & (y_score <= hi if i == n_bins - 1 else y_score < hi)
        if mask.sum() == 0:
            continue
        table.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": int(mask.sum()),
                "mean_predicted_positive_prob": float(y_score[mask].mean()),
                "observed_positive_rate": float(y_true[mask].mean()),
            }
        )
    return table


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Full metric set for a binary Negative(0)/Positive(1) classification result.

    Inputs: true labels, predicted labels (arrays of 0/1), optional positive-class scores.
    Output: a JSON-serializable dict (macro_f1 is the primary headline metric per the brief).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            "negative": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "positive": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": {
            "labels": ["negative", "positive"],
            "matrix": cm.tolist(),
            "reading": "rows=true label, columns=predicted label",
        },
        "roc_auc": None,
        "calibration_bins": None,
    }
    if y_score is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            result["roc_auc"] = None
        result["calibration_bins"] = calibration_table(y_true, y_score)
    return result


def compare_to_baseline(model_metrics: dict, baseline_metrics: dict) -> dict:
    """Macro/weighted-F1 and accuracy deltas of a model over the dummy baseline on the same set."""
    return {
        "macro_f1_delta": model_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "weighted_f1_delta": model_metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
        "accuracy_delta": model_metrics["accuracy"] - baseline_metrics["accuracy"],
    }
