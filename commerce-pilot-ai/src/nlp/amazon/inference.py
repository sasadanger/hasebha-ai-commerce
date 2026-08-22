"""Single-review and batch inference for the Amazon Appliances sentiment models.

Two model families are exposed here, selected via a `model` parameter (never a separate module,
so callers have one place to switch): the original TF-IDF+LinearSVC pipeline (`predict()`,
unchanged from the first version of this file) and the fine-tuned transformer added later
(`predict_with_model()` / `predict_batch()`, `model="tfidf"` or `model="transformer"`).

TF-IDF path: loads the pipeline named in `artifacts/experiments/amazon/models/
best_model_manifest.json`, hash-verified against the sha256 recorded at save time.
Transformer path: loads the checkpoint saved under `artifacts/experiments/amazon/transformer/
model/`, hash-verified against `artifacts/experiments/amazon/transformer/model_manifest.json`,
and applies the temperature scaling + operational threshold selected in Gate 7 (see
`reports/generated/amazon/transformer_calibration.json`) -- fit ONLY on the natural-distribution
validation set, never on any test set. Both paths use the same integrity-check pattern as
src/ai_service/services/fulfillment_risk.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.nlp.amazon.data import build_model_text, build_transformer_text

MODEL_DIR = Path("artifacts/experiments/amazon/models")
BEST_MODEL_MANIFEST_PATH = MODEL_DIR / "best_model_manifest.json"

TRANSFORMER_MODEL_DIR = Path("artifacts/experiments/amazon/transformer/model")
TRANSFORMER_MANIFEST_PATH = Path("artifacts/experiments/amazon/transformer/model_manifest.json")


class ModelIntegrityError(RuntimeError):
    """Raised when the on-disk Amazon sentiment model artifact does not match its recorded hash."""


_cache: dict = {}


def _load_best_model():
    """Load (and cache) the saved best pipeline, hash-verified against its manifest entry."""
    if "model" in _cache:
        return _cache["model"], _cache["manifest"]
    if not BEST_MODEL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Amazon best-model manifest not found: {BEST_MODEL_MANIFEST_PATH}. "
            "Run the training pipeline before calling predict()."
        )
    manifest = json.loads(BEST_MODEL_MANIFEST_PATH.read_text())
    model_path = MODEL_DIR / manifest["file_name"]
    if not model_path.exists():
        raise FileNotFoundError(f"Amazon model artifact not found: {model_path}")
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != manifest["sha256"]:
        raise ModelIntegrityError(
            f"Amazon sentiment model hash mismatch: expected {manifest['sha256']}, got {actual_hash}"
        )
    model = joblib.load(model_path)
    _cache["model"] = model
    _cache["manifest"] = manifest
    return model, manifest


def predict(text: str, title: str | None = None) -> dict:
    """Score one raw review string with the saved best pipeline.

    Inputs: `text` (required, non-empty review body), `title` (optional review title -- combined
    exactly as at training time via `build_model_text`, i.e. `title + '. ' + text` when present).
    Output: dict with `label` (0/1), `label_name` ("negative"/"positive"), `positive_score`
    (P(positive) in [0,1] if the model supports predict_proba, else None), and model metadata
    (`model_name`, `model_sha256`) so callers can trace which artifact produced the score.
    Raises ValueError if text is empty/not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("predict() requires a non-empty review text string")
    model, manifest = _load_best_model()
    row = pd.DataFrame({"title": [title], "text": [text]})
    model_text = build_model_text(row)
    label = int(model.predict(model_text)[0])
    positive_score = None
    if hasattr(model, "predict_proba"):
        positive_score = float(model.predict_proba(model_text)[0][1])
    return {
        "label": label,
        "label_name": "positive" if label == 1 else "negative",
        "positive_score": positive_score,
        "model_name": manifest["model_name"],
        "model_sha256": manifest["sha256"],
    }


def _load_transformer_model():
    """Load (and cache) the fine-tuned transformer + tokenizer, hash-verified against
    `TRANSFORMER_MANIFEST_PATH`, plus the temperature/threshold selected in Gate 7 calibration.
    """
    if "model" in _transformer_cache:
        return _transformer_cache
    if not TRANSFORMER_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Amazon transformer manifest not found: {TRANSFORMER_MANIFEST_PATH}. "
            "Run scripts/amazon_transformer_train.py and scripts/amazon_transformer_calibrate.py first."
        )
    manifest = json.loads(TRANSFORMER_MANIFEST_PATH.read_text())
    weights_path = TRANSFORMER_MODEL_DIR / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Amazon transformer weights not found: {weights_path}")
    actual_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if actual_hash != manifest["model_weights_sha256"]:
        raise ModelIntegrityError(
            f"Amazon transformer weights hash mismatch: expected {manifest['model_weights_sha256']}, "
            f"got {actual_hash}"
        )
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_MODEL_DIR))
    model.eval()
    _transformer_cache.update(
        tokenizer=tokenizer,
        model=model,
        manifest=manifest,
        temperature=manifest["temperature"],
        threshold=manifest["selected_threshold"],
        max_length=manifest["max_length"],
        torch=torch,
    )
    return _transformer_cache


_transformer_cache: dict = {}


def _transformer_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def _predict_transformer_batch(texts: list[str]) -> list[dict]:
    cache = _load_transformer_model()
    torch = cache["torch"]
    tokenizer, model = cache["tokenizer"], cache["model"]
    enc = tokenizer(texts, truncation=True, max_length=cache["max_length"], padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.numpy()
    calibrated_probs = _transformer_softmax(logits / cache["temperature"])
    positive_prob = calibrated_probs[:, 1]
    threshold = cache["threshold"]
    manifest = cache["manifest"]
    out = []
    for prob in positive_prob:
        label = int(prob >= threshold)
        out.append(
            {
                "label": label,
                "label_name": "positive" if label == 1 else "negative",
                "positive_score": float(prob),
                "model_name": manifest["model_name"],
                "model_sha256": manifest["model_weights_sha256"],
                "threshold_used": threshold,
            }
        )
    return out


def _predict_tfidf_batch(texts: list[str], titles: list[str | None]) -> list[dict]:
    model, manifest = _load_best_model()
    row = pd.DataFrame({"title": titles, "text": texts})
    model_text = build_model_text(row)
    labels = model.predict(model_text)
    scores = model.predict_proba(model_text)[:, 1] if hasattr(model, "predict_proba") else [None] * len(texts)
    return [
        {
            "label": int(label),
            "label_name": "positive" if int(label) == 1 else "negative",
            "positive_score": float(score) if score is not None else None,
            "model_name": manifest["model_name"],
            "model_sha256": manifest["sha256"],
        }
        for label, score in zip(labels, scores)
    ]


def predict_batch(texts: list[str], titles: list[str | None] | None = None, model: str = "tfidf") -> list[dict]:
    """Score a batch of raw review strings with either model family.

    Inputs: `texts` (non-empty list of non-empty review bodies), `titles` (optional, same length
    as `texts`, elementwise paired), `model` ("tfidf" -- default, the original pipeline -- or
    "transformer" -- the Gate 6-selected, Gate 7-calibrated fine-tuned checkpoint).
    Output: a list of per-review dicts, same shape as `predict()`'s single-review output, in
    input order. For `model="transformer"`, `positive_score` is the TEMPERATURE-CALIBRATED
    positive-class probability and `label` uses the Gate-7 validation-selected operational
    threshold (not a hardcoded 0.5) -- see reports/generated/amazon/transformer_calibration.json.
    Raises ValueError for empty input, mismatched lengths, or an unknown `model` value.
    """
    if not texts:
        raise ValueError("predict_batch() requires a non-empty list of texts")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("predict_batch() requires every text to be a non-empty string")
    if titles is None:
        titles = [None] * len(texts)
    if len(titles) != len(texts):
        raise ValueError("titles must be the same length as texts")
    if model == "tfidf":
        return _predict_tfidf_batch(texts, titles)
    if model == "transformer":
        row = pd.DataFrame({"title": titles, "text": texts})
        combined = build_transformer_text(row).tolist()
        return _predict_transformer_batch(combined)
    raise ValueError(f"Unknown model '{model}' -- expected 'tfidf' or 'transformer'")


def predict_with_model(text: str, title: str | None = None, model: str = "tfidf") -> dict:
    """Score one raw review string with either model family (see `predict_batch` for details).

    `model="tfidf"` reproduces `predict()`'s exact output. `model="transformer"` uses the fine-
    tuned, Gate-7-calibrated checkpoint instead. This is the single entry point new callers
    should use when they need to choose the model; `predict()` is kept unchanged for existing
    callers that only ever wanted the TF-IDF pipeline.
    """
    return predict_batch([text], [title], model=model)[0]
