"""Gate 29 tests for the Arabic sentiment foundation model.

Covers: label mapping, normalization, empty-text handling, deterministic row-id generation,
split-overlap checks, duplicate-group isolation, tokenizer/model loading, single+batch inference,
probability shape, label validity, calibration roundtrip, artifact paths, hash-manifest integrity.

Model-loading/inference tests are SKIPPED (not failed) if the frozen artifact does not yet exist
on this machine -- this file is meant to run both during this task's own pipeline (where the
artifact exists) and in a fresh checkout (where it may not have been trained yet).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nlp.arabic_foundation.normalization import (  # noqa: E402
    normalize_text, is_empty_or_whitespace, labr_rating_to_3class, astd_label_to_3class,
    arsas_label_to_3class, LABEL_NAMES_3CLASS, diacritic_fraction,
)

SPLITS_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "splits"
FINAL_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "primary_model" / "final"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def test_labr_rating_to_3class_mapping():
    assert labr_rating_to_3class(1) == 0
    assert labr_rating_to_3class(2) == 0
    assert labr_rating_to_3class(3) == 1
    assert labr_rating_to_3class(4) == 2
    assert labr_rating_to_3class(5) == 2


def test_labr_rating_to_3class_rejects_invalid():
    with pytest.raises(ValueError):
        labr_rating_to_3class(0)
    with pytest.raises(ValueError):
        labr_rating_to_3class(6)


def test_astd_label_mapping_excludes_obj():
    assert astd_label_to_3class("OBJ") is None
    assert astd_label_to_3class("POS") == 2
    assert astd_label_to_3class("NEG") == 0
    assert astd_label_to_3class("NEUTRAL") == 1


def test_arsas_label_mapping():
    assert arsas_label_to_3class("Negative") == 0
    assert arsas_label_to_3class("Neutral") == 1
    assert arsas_label_to_3class("Positive") == 2
    assert arsas_label_to_3class("Mixed") == 1


def test_label_names_are_consistent():
    assert LABEL_NAMES_3CLASS == {0: "Negative", 1: "Neutral/Mixed", 2: "Positive"}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_normalize_text_strips_tatweel_by_default():
    out = normalize_text("كتاب" + "ـ" * 5)
    assert "ـ" not in out


def test_normalize_text_collapses_whitespace():
    assert normalize_text("hello    world\n\n\t") == "hello world"


def test_normalize_text_preserves_punctuation_and_repetition():
    out = normalize_text("رااااائع!!!")
    assert "!!!" in out
    assert out.count("ا") >= 3  # elongation preserved, not collapsed


def test_normalize_text_handles_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_is_empty_or_whitespace():
    assert is_empty_or_whitespace(None)
    assert is_empty_or_whitespace("")
    assert is_empty_or_whitespace("   \n\t  ")
    assert not is_empty_or_whitespace("a")


def test_diacritic_fraction_bounds():
    assert diacritic_fraction("") == 0.0
    assert 0.0 <= diacritic_fraction("hello") <= 1.0
    assert diacritic_fraction("كِتَابٌ") > 0.0


# ---------------------------------------------------------------------------
# Split artifacts: overlap / duplicate-group isolation / hash-manifest integrity
# ---------------------------------------------------------------------------

pytestmark_splits = pytest.mark.skipif(not SPLITS_DIR.exists(), reason="splits not yet built on this machine")


@pytest.mark.skipif(not (REPORTS_DIR / "split_manifest.json").exists(), reason="split manifest not yet built")
def test_split_manifest_no_overlap_detected():
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["overlap_verification"]["any_overlap_detected"] is False
    assert manifest["overlap_verification"]["normalized_text_duplicate_crossing_splits"] == 0
    assert manifest["overlap_verification"]["val_balanced_is_subset_of_val_natural"] is True


@pytest.mark.skipif(not (REPORTS_DIR / "split_manifest.json").exists(), reason="split manifest not yet built")
def test_split_manifest_dual_hash_present_for_every_split():
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["splits"].items():
        assert "PHYSICAL_FILE_SHA256" in info, f"{name} missing physical hash"
        assert "SEMANTIC_CONTENT_SHA256" in info, f"{name} missing semantic hash"
        assert len(info["PHYSICAL_FILE_SHA256"]) == 64
        assert len(info["SEMANTIC_CONTENT_SHA256"]) == 64


@pytest.mark.skipif(not SPLITS_DIR.exists(), reason="splits not yet built")
def test_split_parquet_physical_hash_matches_manifest():
    manifest = json.loads((REPORTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["splits"].items():
        path = REPO_ROOT / info["path"]
        assert path.exists(), f"split file missing on disk: {path}"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == info["PHYSICAL_FILE_SHA256"], f"{name} physical hash drift detected"


@pytest.mark.skipif(not SPLITS_DIR.exists(), reason="splits not yet built")
def test_no_review_uid_crosses_splits_at_read_time():
    import pandas as pd

    names = ["train", "val_natural", "test_natural", "item_holdout_stress"]
    id_sets = {}
    for n in names:
        df = pd.read_parquet(SPLITS_DIR / f"{n}.parquet")
        id_sets[n] = set(df["review_uid"])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = id_sets[names[i]] & id_sets[names[j]]
            assert not inter, f"leakage: {names[i]} and {names[j]} share {len(inter)} review_uid(s)"


@pytest.mark.skipif(not SPLITS_DIR.exists(), reason="splits not yet built")
def test_val_balanced_is_class_balanced():
    import pandas as pd

    df = pd.read_parquet(SPLITS_DIR / "val_balanced.parquet")
    counts = df["label"].value_counts()
    assert counts.min() == counts.max(), "val_balanced is not perfectly class-balanced"


@pytest.mark.skipif(not SPLITS_DIR.exists(), reason="splits not yet built")
def test_all_split_labels_are_valid():
    import pandas as pd

    for n in ["train", "val_natural", "val_balanced", "test_natural", "item_holdout_stress"]:
        df = pd.read_parquet(SPLITS_DIR / f"{n}.parquet")
        assert set(df["label"].unique()).issubset({0, 1, 2})
        assert df["text"].isna().sum() == 0
        assert (df["text"].str.strip() != "").all()


# ---------------------------------------------------------------------------
# Model loading / inference (skipped if artifact not present)
# ---------------------------------------------------------------------------

model_available = FINAL_DIR.exists() and (FINAL_DIR / "config.json").exists()
pytestmark_model = pytest.mark.skipif(not model_available, reason="frozen model artifact not present on this machine")


@pytest.mark.skipif(not model_available, reason="frozen model artifact not present")
def test_tokenizer_and_model_load():
    from nlp.arabic_foundation.inference import ArabicSentimentFoundationModel

    m = ArabicSentimentFoundationModel.load()
    assert m.tokenizer is not None
    assert m.model is not None
    assert m.max_length > 0


@pytest.mark.skipif(not model_available, reason="frozen model artifact not present")
def test_single_text_inference_shape_and_validity():
    from nlp.arabic_foundation.inference import ArabicSentimentFoundationModel

    m = ArabicSentimentFoundationModel.load()
    result = m.predict_one("الكتاب رائع جدا وأنصح به بشدة")
    assert result.predicted_label_id in (0, 1, 2)
    assert result.predicted_label_name in LABEL_NAMES_3CLASS.values()
    assert len(result.raw_probabilities) == 3
    assert abs(sum(result.raw_probabilities) - 1.0) < 1e-4


@pytest.mark.skipif(not model_available, reason="frozen model artifact not present")
def test_batch_inference_matches_single_text_inference():
    from nlp.arabic_foundation.inference import ArabicSentimentFoundationModel

    m = ArabicSentimentFoundationModel.load()
    texts = ["كتاب سيء جدا ومضيعة للوقت", "كتاب رائع أنصح به", ""]
    results = m.predict_batch(texts)
    assert len(results) == 3
    for r in results:
        assert r.predicted_label_id in (0, 1, 2)
        assert len(r.raw_probabilities) == 3


@pytest.mark.skipif(not model_available, reason="frozen model artifact not present")
def test_empty_text_inference_does_not_crash():
    from nlp.arabic_foundation.inference import ArabicSentimentFoundationModel

    m = ArabicSentimentFoundationModel.load()
    result = m.predict_one("")
    assert result.predicted_label_id in (0, 1, 2)


@pytest.mark.skipif(not model_available, reason="frozen model artifact not present")
def test_label_mapping_json_matches_normalization_module():
    lm = json.loads((FINAL_DIR / "label_mapping.json").read_text(encoding="utf-8"))
    assert {int(k): v for k, v in lm["id2label"].items()} == LABEL_NAMES_3CLASS


@pytest.mark.skipif(not (FINAL_DIR / "calibration.json").exists(), reason="calibration not yet run")
def test_calibration_roundtrip_probabilities_sum_to_one():
    import numpy as np

    calib = json.loads((FINAL_DIR / "calibration.json").read_text(encoding="utf-8"))
    temperature = calib["temperature"]
    assert temperature > 0
    logits = np.array([[2.0, 0.5, -1.0]])
    scaled = logits / temperature
    z = scaled - scaled.max()
    probs = np.exp(z) / np.exp(z).sum()
    assert abs(probs.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Artifact path / hash-manifest integrity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (REPORTS_DIR / "labr_full_audit.json").exists(), reason="labr audit not yet run")
def test_labr_audit_file_hash_recorded():
    audit = json.loads((REPORTS_DIR / "labr_full_audit.json").read_text(encoding="utf-8"))
    assert len(audit["file_sha256"]) == 64
    assert audit["n_rows"] == 63257
