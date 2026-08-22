"""Shared transformer fine-tuning utilities for the Arabic sentiment foundation 3-class task.

Companion to normalization.py (label contract + text normalization) and features.py (classical
baseline). Mirrors the structure of src/nlp/amazon/transformer.py (same repo, prior session, same
house style) adapted for: 3 classes instead of 2, MARBERT/CAMeLBERT tokenizers instead of
distilroberta, and splits loaded directly from parquet (this task's splits already store text, so
no separate id-to-text rejoin step is needed -- see arabic_foundation_build_splits.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

TEXT_COL = "text_norm"
LABEL_COL = "label"
SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "splits"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}.parquet")


def tokenize_dataframe(df: pd.DataFrame, tokenizer, max_length: int):
    """Batch-tokenize `df[TEXT_COL]` with a fast tokenizer. Returns a HF `datasets.Dataset` with
    input_ids/attention_mask/labels, keeping review_uid alongside for later join-back."""
    from datasets import Dataset

    ds = Dataset.from_pandas(
        df[["review_uid", TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "labels"}),
        preserve_index=False,
    )

    def _tok(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)

    ds = ds.map(_tok, batched=True, remove_columns=[TEXT_COL])
    return ds


def compute_hf_metrics(eval_pred) -> dict:
    """`compute_metrics` callback for transformers.Trainer -- macro-F1 is the model-selection
    metric (`metric_for_best_model="eval_macro_f1"`), matching the classical baseline's headline
    metric so the two are directly comparable."""
    logits, labels = eval_pred
    logits = np.asarray(logits)
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "negative_f1": f1[0],
        "neutral_mixed_f1": f1[1],
        "positive_f1": f1[2],
        "negative_precision": precision[0],
        "negative_recall": recall[0],
        "neutral_mixed_precision": precision[1],
        "neutral_mixed_recall": recall[1],
        "positive_precision": precision[2],
        "positive_recall": recall[2],
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a single learned scalar temperature to logits before softmax (Guo et al. 2017)."""
    return softmax(logits / temperature)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 200) -> float:
    """Fit a single scalar temperature by minimizing NLL on (logits, labels) via numpy gradient
    descent (1-parameter convex optimization, no extra dependency needed)."""
    labels = np.asarray(labels).astype(int)
    t = 1.0
    n = len(labels)
    for _ in range(max_iter):
        eps = 1e-3
        probs_plus = softmax(logits / (t + eps))
        probs_minus = softmax(logits / (t - eps))
        nll_plus = -np.mean(np.log(np.clip(probs_plus[np.arange(n), labels], 1e-12, 1.0)))
        nll_minus = -np.mean(np.log(np.clip(probs_minus[np.arange(n), labels], 1e-12, 1.0)))
        grad = (nll_plus - nll_minus) / (2 * eps)
        t -= lr * grad
        t = max(t, 0.05)
    return float(t)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    """Multiclass Brier score: mean squared distance between the one-hot label and the predicted
    probability vector (generalizes the binary Brier score used in the Amazon pipeline)."""
    y_true = np.asarray(y_true).astype(int)
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def multiclass_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Top-label expected calibration error: bins by the model's own predicted-class confidence
    (max prob), compares bin-average confidence to bin accuracy."""
    y_true = np.asarray(y_true).astype(int)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)
