"""Gate 28: single-text and batch inference functions for the frozen Arabic sentiment foundation
model. Library/artifact only -- NOT wired into any production service (per hard scope boundary).

Usage:
    from nlp.arabic_foundation.inference import ArabicSentimentFoundationModel
    model = ArabicSentimentFoundationModel.load()  # loads the frozen artifact by default
    result = model.predict_one("الكتاب رائع جدا")
    results = model.predict_batch(["نص اول", "نص ثاني"])
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nlp.arabic_foundation.normalization import normalize_text, LABEL_NAMES_3CLASS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "primary_model" / "final"


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass
class SentimentPrediction:
    text: str
    predicted_label_id: int
    predicted_label_name: str
    raw_logits: list
    raw_probabilities: list
    calibrated_probabilities: "list | None" = None


@dataclass
class ArabicSentimentFoundationModel:
    """Loads the frozen MARBERT primary model + tokenizer + calibration for inference.

    Explicitly an Arabic sentiment FOUNDATION model (LABR book-review domain), not an Egyptian
    e-commerce production model -- see reports/generated/arabic_foundation/ARABIC_FOUNDATION_MODEL_CARD.md.
    """

    model: object
    tokenizer: object
    max_length: int
    label_names: dict = field(default_factory=lambda: dict(LABEL_NAMES_3CLASS))
    temperature: float = 1.0
    use_calibration: bool = False
    device: str = "cpu"

    @classmethod
    def load(cls, model_dir: "str | Path | None" = None) -> "ArabicSentimentFoundationModel":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Frozen model dir not found: {model_dir}. Run scripts/arabic_foundation_train_marbert.py "
                "to produce it before calling ArabicSentimentFoundationModel.load()."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        training_config_path = model_dir / "training_config.json"
        max_length = 192
        if training_config_path.exists():
            max_length = json.loads(training_config_path.read_text(encoding="utf-8")).get("max_length", 192)

        label_mapping_path = model_dir / "label_mapping.json"
        label_names = dict(LABEL_NAMES_3CLASS)
        if label_mapping_path.exists():
            lm = json.loads(label_mapping_path.read_text(encoding="utf-8"))
            label_names = {int(k): v for k, v in lm.get("id2label", label_names).items()}

        temperature = 1.0
        use_calibration = False
        calib_path = model_dir / "calibration.json"
        if calib_path.exists():
            calib = json.loads(calib_path.read_text(encoding="utf-8"))
            if calib.get("decision") == "USE_CALIBRATED":
                temperature = calib["temperature"]
                use_calibration = True

        return cls(
            model=model, tokenizer=tokenizer, max_length=max_length, label_names=label_names,
            temperature=temperature, use_calibration=use_calibration, device=device,
        )

    def _predict_logits(self, texts: list) -> np.ndarray:
        import torch

        norm_texts = [normalize_text(t) for t in texts]
        enc = self.tokenizer(
            norm_texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        return out.logits.detach().cpu().numpy()

    def predict_batch(self, texts: list) -> list:
        if len(texts) == 0:
            return []
        logits = self._predict_logits(list(texts))
        raw_probs = _softmax(logits)
        cal_probs = _softmax(logits / self.temperature) if self.use_calibration else None
        preds = raw_probs.argmax(axis=1)
        results = []
        for i, t in enumerate(texts):
            results.append(
                SentimentPrediction(
                    text=t,
                    predicted_label_id=int(preds[i]),
                    predicted_label_name=self.label_names[int(preds[i])],
                    raw_logits=logits[i].tolist(),
                    raw_probabilities=raw_probs[i].tolist(),
                    calibrated_probabilities=cal_probs[i].tolist() if cal_probs is not None else None,
                )
            )
        return results

    def predict_one(self, text: str) -> SentimentPrediction:
        return self.predict_batch([text])[0]
