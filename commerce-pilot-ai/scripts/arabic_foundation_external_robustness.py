"""Gate 22: external robustness evaluation on ASTD and ArSAS -- reported as separately-labeled
CROSS-DOMAIN STRESS EVIDENCE only, never as Egyptian e-commerce performance. Label-semantic
differences are reported prominently. No retuning happens after this (post-hoc reporting only).

ASTD: Positive->Positive, Negative->Negative, Mixed->Neutral/Mixed; OBJ rows EXCLUDED from this
3-class eval (documented in Gate 6 / astd_audit.json -- OBJ != Neutral/Mixed sentiment).
ArSAS: Negative->Negative, Neutral->Neutral/Mixed, Positive->Positive, Mixed->Neutral/Mixed.

Run (after the frozen primary model exists):
  .venv/Scripts/python.exe scripts/arabic_foundation_external_robustness.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nlp.arabic_foundation import transformer as af_tf  # noqa: E402
from nlp.arabic_foundation.normalization import (  # noqa: E402
    normalize_text, astd_label_to_3class, arsas_label_to_3class,
)

FINAL_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "primary_model" / "final"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
DATA_ROOT = REPO_ROOT / "data" / "quarantine" / "nlp"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def full_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            "negative": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "neutral_mixed": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
            "positive": {"precision": float(precision[2]), "recall": float(recall[2]), "f1": float(f1[2]), "support": int(support[2])},
        },
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
    }


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    training_config = json.loads((FINAL_DIR / "training_config.json").read_text(encoding="utf-8"))
    max_length = training_config["max_length"]

    def run_inference(texts: list) -> np.ndarray:
        all_logits = []
        bs = 64
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch_texts = [normalize_text(t) for t in texts[i : i + bs]]
                enc = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
                out = model(**enc)
                all_logits.append(out.logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    results = {}

    # ---- ASTD ----
    log("Evaluating on ASTD (OBJ excluded)...")
    astd_path = DATA_ROOT / "astd" / "data_Tweets.txt"
    astd = pd.read_csv(astd_path, sep="\t", header=None, names=["text", "label"], na_filter=False, engine="python", quoting=3)
    astd["label_3class"] = astd["label"].map(astd_label_to_3class)
    n_obj_excluded = int(astd["label_3class"].isna().sum())
    astd_eval = astd.dropna(subset=["label_3class"]).copy()
    astd_eval["label_3class"] = astd_eval["label_3class"].astype(int)
    logits = run_inference(astd_eval["text"].tolist())
    preds = af_tf.softmax(logits).argmax(axis=1)
    m = full_metrics(astd_eval["label_3class"].values, preds)
    m["n_obj_excluded"] = n_obj_excluded
    m["label_mapping"] = "POS->Positive, NEG->Negative, NEUTRAL->Neutral/Mixed; OBJ EXCLUDED (not sentiment-equivalent)"
    results["astd"] = m
    log(f"ASTD: macro_f1={m['macro_f1']:.4f} n={m['n']} (n_obj_excluded={n_obj_excluded})")

    # ---- ArSAS ----
    log("Evaluating on ArSAS...")
    arsas_path = DATA_ROOT / "arsas" / "extracted" / "ArSAS..txt"
    arsas = pd.read_csv(arsas_path, sep="\t", encoding="utf-8")
    arsas["label_3class"] = arsas["Sentiment_label"].map(arsas_label_to_3class)
    logits = run_inference(arsas["Tweet_text"].tolist())
    preds = af_tf.softmax(logits).argmax(axis=1)
    m = full_metrics(arsas["label_3class"].values, preds)
    m["label_mapping"] = "Negative->Negative, Positive->Positive, Neutral->Neutral/Mixed, Mixed->Neutral/Mixed"
    results["arsas"] = m
    log(f"ArSAS: macro_f1={m['macro_f1']:.4f} n={m['n']}")

    results["IMPORTANT_CAVEAT"] = (
        "These are CROSS-DOMAIN ROBUSTNESS STRESS results on Twitter-sourced Arabic sentiment "
        "datasets with DIFFERENT label semantics and DIFFERENT domain (social media, not book "
        "reviews, not e-commerce) than the primary LABR-trained model. They are NOT a measure of "
        "Egyptian e-commerce production readiness and must never be reported as such. Label "
        "mapping differences (ASTD's OBJ exclusion, ArSAS's Mixed->Neutral/Mixed fold) mean these "
        "numbers are not directly comparable to the primary LABR test metrics either -- they "
        "measure a genuinely different question (does the LABR-trained model transfer at all to "
        "dialectal/social-media text) and are reported separately for that reason. Model selection "
        "(Gate 25) was already finalized before this evaluation ran; these results do not retune "
        "anything (post-hoc reporting only, per Gate 22 instruction)."
    )

    (REPORTS_DIR / "external_robustness_report.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "IMPORTANT_CAVEAT"}, indent=2, default=str))
    print("\nWrote external_robustness_report.json")


if __name__ == "__main__":
    main()
