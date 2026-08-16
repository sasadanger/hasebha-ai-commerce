"""Phase 8 -- Egyptian error / domain-shift analysis. Loads the already
fine-tuned seed-303 (representative) checkpoint for INFERENCE ONLY (no
further training) on Jumia validation, joins predictions with heuristic
slice labels, and compares LABR-native vs Jumia-direct-transfer vs
Jumia-adapted performance. All heuristic slices are explicitly labeled as
heuristic, not ground truth.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nlp.text_normalization import normalize_text  # noqa: E402

CSV_PATH = ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
SPLIT_PATH = ROOT / "artifacts" / "experiments" / "jumia" / "phase1_split" / "jumia_split_assignments.parquet"
CHECKPOINT_DIR = Path("D:/commercepilot_ml_cache/checkpoints/jumia_UBC-NLP__MARBERT_seed303_confirm/checkpoint-501")
TOKENIZER_SOURCE = ROOT / "artifacts" / "experiments" / "nlp" / "inference_exports" / "C_MARBERT"
OUT_DIR = ROOT / "reports" / "generated" / "jumia" / "domain_shift"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS = ["1", "2", "3", "4", "5"]

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
LATIN_RE = re.compile(r"[A-Za-z]")
ELONGATION_RE = re.compile(r"(.)\1{2,}")
# Heuristic Egyptian/MSA negation markers (word-boundary-ish substring match)
NEGATION_MARKERS = ["مش ", "مِش", "لا ", "ماكانش", "مكنش", "ملهاش", "مفيش", "بدون", "غير كده", "للأسف"]


def script_profile(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return "empty"
    has_ar = bool(ARABIC_RE.search(s))
    has_la = bool(LATIN_RE.search(s))
    if has_ar and has_la:
        return "mixed_arabic_english"
    if has_ar:
        return "arabic"
    if has_la:
        return "english"
    return "other_no_letters"


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["customer_rating"] = df["customer_rating"].astype(str)
    split_df = pd.read_parquet(SPLIT_PATH)
    val_idx = split_df.loc[split_df["split"] == "validation", "row_index"].tolist()
    val = df.iloc[val_idx].copy()

    val["_script"] = val["review"].fillna("").map(script_profile)
    val["_word_count"] = val["review"].fillna("").map(lambda s: len(str(s).split()))
    val["_is_short"] = val["_word_count"] <= 3
    val["_is_long"] = val["_word_count"] >= 10
    val["_has_elongation"] = val["review"].fillna("").map(lambda s: bool(ELONGATION_RE.search(str(s))))
    val["_has_negation"] = val["review"].fillna("").map(
        lambda s: any(marker in str(s) for marker in NEGATION_MARKERS)
    )
    val["_is_minority_class"] = val["customer_rating"].isin(["2", "3"])

    texts_normalized = val["review"].map(normalize_text).tolist()

    print(f"loading seed-303 checkpoint (inference only) from {CHECKPOINT_DIR}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_SOURCE))
    model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT_DIR))
    model.eval()
    id2label = model.config.id2label

    predictions = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(texts_normalized), batch_size):
            batch = texts_normalized[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            predictions.extend(id2label[i] for i in pred_ids)
    val["_pred_adapted"] = predictions

    def slice_metrics(mask_name: str, mask: pd.Series) -> dict:
        sub = val[mask]
        if len(sub) == 0:
            return {"n": 0, "macro_f1": None, "note": "no rows in this slice"}
        return {
            "n": int(len(sub)),
            "macro_f1": float(f1_score(sub["customer_rating"], sub["_pred_adapted"], labels=LABELS, average="macro", zero_division=0)),
        }

    slices = {
        "arabic": val["_script"] == "arabic",
        "english": val["_script"] == "english",
        "mixed_arabic_english": val["_script"] == "mixed_arabic_english",
        "short_le3_words": val["_is_short"],
        "long_ge10_words": val["_is_long"],
        "noisy_elongation": val["_has_elongation"],
        "negation_heuristic": val["_has_negation"],
        "minority_classes_2_3": val["_is_minority_class"],
    }
    slice_results = {name: slice_metrics(name, mask) for name, mask in slices.items()}

    overall_adapted_macro_f1 = float(f1_score(val["customer_rating"], val["_pred_adapted"], labels=LABELS, average="macro", zero_division=0))

    # Cross-reference: LABR-native (from existing project evidence, not
    # recomputed here) vs Jumia direct transfer (Phase 5) vs Jumia adapted
    # (this script, overall).
    direct_transfer = json.loads(
        (ROOT / "reports" / "generated" / "jumia" / "direct_transfer" / "direct_transfer_metrics_and_tiebreak.json").read_text()
    )
    comparison = {
        "LABR_native_test_domain_MARBERT_macro_f1": 0.4872,  # from configs/nlp_champion_registry.yaml (frozen, pre-existing evidence, not recomputed)
        "jumia_direct_transfer_MARBERT_macro_f1": direct_transfer["MARBERT"]["macro_f1"],
        "jumia_adapted_MARBERT_macro_f1_seed303": overall_adapted_macro_f1,
        "jumia_adapted_MARBERT_mean_3seed": 0.3962557754783857,
    }

    out = {
        "schema_version": "jumia-domain-shift-analysis-v1",
        "generated_at": "2026-08-15",
        "representative_seed": 303,
        "n_validation_rows": len(val),
        "heuristic_disclaimer": "All slice labels below (script, length, elongation, negation) are heuristic pattern-matches, not human-annotated ground truth.",
        "slice_results_heuristic": slice_results,
        "cross_domain_comparison": comparison,
        "internal_test_accessed": False,
        "checkpoint_used_for_inference_only": str(CHECKPOINT_DIR),
        "no_additional_training_performed": True,
    }
    out_path = OUT_DIR / "jumia_domain_shift_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str), file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
