"""Shared transformer fine-tuning utilities for Amazon Appliances binary sentiment classification.

Companion to train.py (which holds the classical TF-IDF/sklearn models) -- this module holds
everything specific to fine-tuning a real Hugging Face sequence-classification transformer
end-to-end. Model input is ALWAYS `transformer_text` (see `build_transformer_text` /
`normalize_text_for_transformer` in data.py) -- rating/verified_purchase/helpful_vote are never
fed to the model, same no-leakage rule as the TF-IDF pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.nlp.amazon import data as amz_data

TEXT_COL = "transformer_text"
LABEL_COL = "label"
SEED = amz_data.SEED

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_IDS_DIR = REPO_ROOT / "reports" / "generated" / "amazon" / "split_ids"

ALL_SPLIT_NAMES = (
    "val",
    "val_natural",
    "test_balanced",
    "test_representative",
    "product_holdout_stress",
    "chronological_stress",
    "train_full_pool",
    "learning_curve_25000",
    "learning_curve_50000",
    "learning_curve_100000",
    "learning_curve_200000",
)


def reproduce_full_pool_with_text(seed: int = SEED) -> pd.DataFrame:
    """Deterministically reproduce the full pre-split pool (load -> dedup -> carve-outs), WITH
    the original text/title columns, so any split's review_uid set can be rejoined to real text.

    split_ids/*.parquet files intentionally store only review_uid/parent_asin/label/rating (an
    id manifest, not a text dump) -- this function reconstructs the text side deterministically
    from the same seeded pipeline used to build every split (see
    scripts/amazon_build_val_natural_split.py, which verifies this reproduction is byte-for-byte
    identical to the frozen on-disk splits before trusting it).
    """
    pool = amz_data.load_modeling_pool(seed=seed, positive_sample_size=amz_data.POSITIVE_SAMPLE_SIZE)
    pool, _dup_report = amz_data.remove_duplicate_text(pool)
    pool = pool.reset_index(drop=True)
    pool, product_holdout = amz_data.carve_out_product_holdout_stress(pool, seed=seed, target_rows=5_000)
    pool, chronological = amz_data.carve_out_chronological_stress(pool, target_rows=5_000)
    # product_holdout/chronological rows were carved OUT of `pool` above -- re-attach them so
    # every split name (including the two stress sets) can be rejoined from one frame.
    full = pd.concat([pool, product_holdout, chronological], ignore_index=True)
    full[TEXT_COL] = amz_data.build_transformer_text(full)
    return full


def load_split_with_text(name: str, full_pool: pd.DataFrame) -> pd.DataFrame:
    """Rejoin a saved split_ids/<name>.parquet (review_uid/parent_asin/label/rating only) to full
    text/title/rating/timestamp/etc columns via `full_pool` (see `reproduce_full_pool_with_text`).
    """
    ids = pd.read_parquet(SPLIT_IDS_DIR / f"{name}.parquet")
    merged = ids[["review_uid"]].merge(full_pool, on="review_uid", how="left", validate="one_to_one")
    if merged[TEXT_COL].isna().any():
        missing = int(merged[TEXT_COL].isna().sum())
        raise RuntimeError(
            f"load_split_with_text('{name}'): {missing} review_uid(s) from the saved split could "
            "not be rejoined to the reproduced pool -- the pipeline reproduction is not matching "
            "the frozen split artifacts, do not proceed."
        )
    return merged


def tokenize_dataframe(df: pd.DataFrame, tokenizer, max_length: int):
    """Batch-tokenize `df[TEXT_COL]` with a fast tokenizer. Returns a HF `datasets.Dataset` with
    input_ids/attention_mask/labels, keeping review_uid alongside for later join-back.
    """
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
    metric (`metric_for_best_model="eval_macro_f1"`), matching the TF-IDF pipeline's headline
    metric so the two are comparable.
    """
    logits, labels = eval_pred
    logits = np.asarray(logits)
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "negative_f1": f1[0],
        "positive_f1": f1[1],
        "negative_precision": precision[0],
        "negative_recall": recall[0],
        "positive_precision": precision[1],
        "positive_recall": recall[1],
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a single learned scalar temperature to logits before softmax (Guo et al. 2017)."""
    return softmax(logits / temperature)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 200) -> float:
    """Fit a single scalar temperature by minimizing NLL on (logits, labels) via gradient descent
    in numpy (no extra dependency needed -- this is a 1-parameter convex optimization).
    """
    labels = np.asarray(labels).astype(int)
    t = 1.0
    n = len(labels)
    for _ in range(max_iter):
        probs = softmax(logits / t)
        # d(NLL)/dT for a single temperature scalar
        p_true = probs[np.arange(n), labels]
        # gradient via finite difference is simplest and robust for a 1-D convex problem
        eps = 1e-3
        probs_plus = softmax(logits / (t + eps))
        probs_minus = softmax(logits / (t - eps))
        nll_plus = -np.mean(np.log(np.clip(probs_plus[np.arange(n), labels], 1e-12, 1.0)))
        nll_minus = -np.mean(np.log(np.clip(probs_minus[np.arange(n), labels], 1e-12, 1.0)))
        grad = (nll_plus - nll_minus) / (2 * eps)
        t -= lr * grad
        t = max(t, 0.05)
    return float(t)


def brier_score(y_true: np.ndarray, y_prob_positive: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    return float(np.mean((y_prob_positive - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob_positive: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob_positive = np.asarray(y_prob_positive)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob_positive >= lo) & (y_prob_positive <= hi if i == n_bins - 1 else y_prob_positive < hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_prob_positive[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)
