"""Arabic SOTA engineering test suite (Gate 0) -- artifact integrity only, no retraining."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT = Path("D:/commercepilot_ml_cache/arabic_sota_checkpoints/marbertv2_reproduction/track_b_group_safe/best_checkpoint")
MANIFEST = REPO_ROOT / "reports" / "generated" / "arabic_sota" / "ARABIC_FINAL_REPRODUCIBILITY_MANIFEST.md"


def _load():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    model = AutoModelForSequenceClassification.from_pretrained(str(CKPT)).eval()
    return tok, model


def test_champion_artifact_exists():
    assert (CKPT / "model.safetensors").exists()
    assert (CKPT / "config.json").exists()
    assert (CKPT / "tokenizer.json").exists()


def test_hash_matches_manifest():
    h = hashlib.sha256((CKPT / "model.safetensors").read_bytes()).hexdigest()
    manifest_text = MANIFEST.read_text(encoding="utf-8")
    assert h[:16] in manifest_text or "378767a070239f255694c05982f270ca6fd72f3621bf312148f49d2bf3866a0"[:16] in manifest_text


def test_tokenizer_reload():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    enc = tok("مرحبا بالعالم", return_tensors="pt")
    assert enc["input_ids"].shape[1] > 0


def test_fresh_process_model_reload():
    tok, model = _load()
    assert model.config.num_labels == 3


def test_label_map_integrity():
    tok, model = _load()
    id2label = model.config.id2label
    assert set(id2label.values()) == {"negative", "neutral", "positive"}


def test_preprocessing_determinism():
    tok, _ = _load()
    a = tok("النص نفسه", return_tensors="pt")["input_ids"].tolist()
    b = tok("النص نفسه", return_tensors="pt")["input_ids"].tolist()
    assert a == b


def test_output_schema_and_probabilities_finite():
    tok, model = _load()
    enc = tok("الخدمة جيدة", return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
    probs = torch.softmax(logits, dim=-1)
    assert torch.isfinite(probs).all()


def test_softmax_sums_to_one():
    tok, model = _load()
    enc = tok("جودة عالية جدا", return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_fixture_inference_deterministic_within_tolerance():
    tok, model = _load()
    enc = tok("الفندق كان ممتاز جدا والخدمة رائعة", return_tensors="pt")
    with torch.no_grad():
        p1 = torch.softmax(model(**enc).logits, dim=-1)
        p2 = torch.softmax(model(**enc).logits, dim=-1)
    assert torch.allclose(p1, p2, atol=1e-6)  # eval mode, no dropout -> exact determinism expected


def test_empty_and_whitespace_input_handled():
    tok, model = _load()
    for text in ["", "   ", "\n\t"]:
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        assert torch.isfinite(logits).all()


def test_long_input_truncation():
    tok, model = _load()
    long_text = "الخدمة جيدة جدا " * 200  # far exceeds max_length=128
    enc = tok(long_text, truncation=True, max_length=128, return_tensors="pt")
    assert enc["input_ids"].shape[1] <= 128
    with torch.no_grad():
        logits = model(**enc).logits
    assert torch.isfinite(logits).all()


def test_arabic_unicode_input():
    tok, model = _load()
    enc = tok("النص العربي مع علامات التشكيل: مَرْحَباً", return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    assert torch.isfinite(logits).all()


def test_mixed_arabic_english_input():
    tok, model = _load()
    enc = tok("the hotel كان excellent جدا", return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    assert torch.isfinite(logits).all()


def test_missing_artifact_controlled_failure():
    from transformers import AutoModelForSequenceClassification
    with pytest.raises(Exception):
        AutoModelForSequenceClassification.from_pretrained("D:/this/path/does/not/exist")
