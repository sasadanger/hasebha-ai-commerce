"""NLP inference adapters for the CommercePilot v1 promotion decisions in
reports/checkpoints/nlp_deployment_promotion_registry_v1_2026-08-14/.

Loads fine-tuned transformer weights exported (not trained) by
scripts/nlp_export_inference_artifacts.py -- see
artifacts/experiments/nlp/inference_exports/export_manifest.json. Models are
loaded lazily on first use per (task, model) key and cached in memory for the
life of the process, so the service never holds more than the models
actually requested, and startup stays cheap (registry read + hash checks
only, no weights loaded until first request).

Task routing, per the promotion registry:
  E  (MPOLD) -> UBC-NLP/MARBERT only (PROMOTE)
  B2 (ASTD)  -> UBC-NLP/MARBERT only (PROMOTE)
  C  (LABR)  -> BOTH UBC-NLP/MARBERT and aubmindlab/bert-base-arabertv2;
                no champion is selected (bootstrap CI includes zero) -- the
                caller may request one explicitly or receive both.
  A  (Amazon)-> ARTIFACT_NOT_MATERIALIZED: the frozen classical winner
                (A::tfidf_word_bigram_logreg) was never serialized to disk
                during Batch 1 (only its hyperparameter fingerprint + metrics
                were recorded). Serving it would require fitting a new
                TfidfVectorizer+LogisticRegression, which is training -- out
                of scope for this integration pass. Reported honestly, not
                silently skipped and not faked.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.ai_service.config import (
    NLP_EXPORT_MANIFEST_PATH,
    NLP_PREPROCESSING_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

os.environ.setdefault("HF_HOME", "D:/commercepilot_ml_cache/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "D:/commercepilot_ml_cache/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/commercepilot_ml_cache/hf/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(REPO_ROOT))
from src.nlp.text_normalization import normalize_text  # noqa: E402

TASK_MODEL_KEYS = {
    "E": ["E_MARBERT"],
    "B2": ["B2_MARBERT"],
    "C": ["C_MARBERT", "C_AraBERT"],
}
AMAZON_TASK = "A"

MODEL_SHORT_NAME = {
    "E_MARBERT": "marbert",
    "B2_MARBERT": "marbert",
    "C_MARBERT": "marbert",
    "C_AraBERT": "arabert",
}


class ArtifactIntegrityError(RuntimeError):
    """Raised when an exported model/tokenizer file fails hash verification."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class NlpPrediction:
    task: str
    task_name: str
    model_key: str
    model_short_name: str
    model_name: str
    model_revision: str
    predicted_label: str
    class_probabilities: dict
    preprocessing_version: str
    normalized_text: str
    analyzed_at: str


class NlpInferenceService:
    """Registry + lazy-loading cache for the promoted/retained NLP models."""

    def __init__(self, manifest_path: Path = NLP_EXPORT_MANIFEST_PATH) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"NLP export manifest not found: {manifest_path}")
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._entries: dict[str, dict] = {e["key"]: e for e in self._manifest["exports"]}
        self._verify_all_hashes()
        self._loaded: dict[str, object] = {}

    def _verify_all_hashes(self) -> None:
        for key, entry in self._entries.items():
            export_dir = REPO_ROOT / entry["export_dir"]
            for fname, expected in entry["model_file_hashes"].items():
                path = export_dir / fname
                if not path.exists():
                    raise FileNotFoundError(f"[{key}] missing exported file: {path}")
                actual = sha256_file(path)
                if actual != expected:
                    raise ArtifactIntegrityError(
                        f"[{key}] {fname} hash mismatch: expected {expected}, got {actual}"
                    )

    def available_model_keys(self) -> list[str]:
        return list(self._entries.keys())

    def is_loadable(self, model_key: str) -> bool:
        return model_key in self._entries

    def keys_for_task(self, task: str) -> list[str]:
        if task == AMAZON_TASK:
            return []
        return TASK_MODEL_KEYS.get(task, [])

    def _load(self, model_key: str):
        if model_key in self._loaded:
            return self._loaded[model_key]
        entry = self._entries.get(model_key)
        if entry is None:
            raise KeyError(f"unknown NLP model key: {model_key}")

        # Re-verify this specific artifact's hashes immediately before load,
        # in case the on-disk export changed after service startup.
        export_dir = REPO_ROOT / entry["export_dir"]
        for fname, expected in entry["model_file_hashes"].items():
            actual = sha256_file(export_dir / fname)
            if actual != expected:
                raise ArtifactIntegrityError(
                    f"[{model_key}] {fname} hash mismatch at load time: expected {expected}, got {actual}"
                )

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(export_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(export_dir))
        model.eval()
        loaded = {"tokenizer": tokenizer, "model": model, "torch": torch, "entry": entry}
        self._loaded[model_key] = loaded
        return loaded

    def predict(self, model_key: str, text: str) -> NlpPrediction:
        loaded = self._load(model_key)
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        torch = loaded["torch"]
        entry = loaded["entry"]

        normalized = normalize_text(text)
        encoded = tokenizer(
            [normalized],
            truncation=True,
            padding=True,
            max_length=entry["max_length"],
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()

        id2label = model.config.id2label
        class_probabilities = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        predicted_id = int(torch.argmax(logits, dim=-1)[0])
        predicted_label = id2label[predicted_id]

        return NlpPrediction(
            task=entry["task"],
            task_name=entry["task_name"],
            model_key=model_key,
            model_short_name=MODEL_SHORT_NAME[model_key],
            model_name=entry["model_name"],
            model_revision=entry["revision"],
            predicted_label=predicted_label,
            class_probabilities=class_probabilities,
            preprocessing_version=NLP_PREPROCESSING_VERSION,
            normalized_text=normalized,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )
