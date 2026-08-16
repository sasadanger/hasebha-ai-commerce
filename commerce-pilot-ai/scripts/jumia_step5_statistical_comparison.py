"""Step 5 -- paired bootstrap comparisons (best transformer vs strongest
classical; remediation vs current base-MARBERT) + Step 6 ordinal
diagnostics. Inference-only against already-trained checkpoints -- no
further training. Protected test never accessed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nlp.text_normalization import normalize_text  # noqa: E402

CSV_PATH = ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
SPLIT_PATH = ROOT / "artifacts" / "experiments" / "jumia" / "phase1_split" / "jumia_split_assignments.parquet"
LABELS = ["1", "2", "3", "4", "5"]
SEED = 42
RNG_SEED = 20260815
N_BOOTSTRAP = 2000

LABR_INIT_CHECKPOINT = Path("D:/commercepilot_ml_cache/checkpoints/jumia_from_labr_MARBERT_seed202/checkpoint-167")
BASE_MARBERT_CHECKPOINT = Path("D:/commercepilot_ml_cache/checkpoints/jumia_UBC-NLP__MARBERT_seed303_confirm/checkpoint-501")
TOKENIZER_SOURCE = ROOT / "artifacts" / "experiments" / "nlp" / "inference_exports" / "C_MARBERT"


def transformer_predict(checkpoint_dir: Path, texts: list[str]) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_SOURCE))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.eval()
    id2label = model.config.id2label
    preds = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            preds.extend(id2label[i] for i in pred_ids)
    del model
    return preds


def paired_bootstrap(true, pred_a, pred_b, label_a, label_b, seed=RNG_SEED, n=N_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    true_arr = np.array(true)
    a_arr = np.array(pred_a)
    b_arr = np.array(pred_b)
    n_items = len(true)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_items, n_items)
        f1_a = f1_score(true_arr[idx], a_arr[idx], labels=LABELS, average="macro", zero_division=0)
        f1_b = f1_score(true_arr[idx], b_arr[idx], labels=LABELS, average="macro", zero_division=0)
        diffs.append(f1_a - f1_b)
    diffs = np.array(diffs)
    point = f1_score(true_arr, a_arr, labels=LABELS, average="macro") - f1_score(true_arr, b_arr, labels=LABELS, average="macro")
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        "comparison": f"{label_a} - {label_b} (macro_f1)",
        "point_estimate_diff": float(point),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "n_bootstrap": n,
    }


def ordinal_diagnostics(true, pred) -> dict:
    true_int = np.array([int(t) for t in true])
    pred_int = np.array([int(p) for p in pred])
    abs_err = np.abs(true_int - pred_int)
    return {
        "mae": float(abs_err.mean()),
        "exact_accuracy": float((abs_err == 0).mean()),
        "within_1_star_accuracy": float((abs_err <= 1).mean()),
        "severe_error_rate_ge2": float((abs_err >= 2).mean()),
    }


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["customer_rating"] = df["customer_rating"].astype(str)
    df["_norm"] = df["review"].map(normalize_text)
    split_df = pd.read_parquet(SPLIT_PATH)
    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].tolist()
    val_idx = split_df.loc[split_df["split"] == "validation", "row_index"].tolist()
    train, val = df.iloc[train_idx], df.iloc[val_idx]
    true = val["customer_rating"].tolist()
    val_texts_norm = val["_norm"].tolist()

    print("re-deriving classical char_wb baseline predictions (fast, deterministic)...", file=sys.stderr)
    char_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ]).fit(train["_norm"], train["customer_rating"])
    pred_classical = char_pipeline.predict(val_texts_norm).tolist()

    print("loading LABR-init MARBERT seed202 (inference only)...", file=sys.stderr)
    pred_labr_init = transformer_predict(LABR_INIT_CHECKPOINT, val_texts_norm)

    print("loading base-MARBERT seed303 (inference only)...", file=sys.stderr)
    pred_base = transformer_predict(BASE_MARBERT_CHECKPOINT, val_texts_norm)

    bootstrap_vs_classical = paired_bootstrap(true, pred_labr_init, pred_classical, "LABR_init_MARBERT", "classical_char_wb")
    bootstrap_vs_base = paired_bootstrap(true, pred_labr_init, pred_base, "LABR_init_MARBERT", "base_MARBERT")

    out = {
        "schema_version": "jumia-step5-statistical-comparison-v1",
        "generated_at": "2026-08-15",
        "n_validation": len(val),
        "internal_test_accessed": False,
        "bootstrap_labr_init_vs_classical": bootstrap_vs_classical,
        "bootstrap_labr_init_vs_base_marbert": bootstrap_vs_base,
        "ordinal_diagnostics": {
            "classical_char_wb": ordinal_diagnostics(true, pred_classical),
            "base_MARBERT": ordinal_diagnostics(true, pred_base),
            "LABR_init_MARBERT": ordinal_diagnostics(true, pred_labr_init),
        },
    }
    out_path = ROOT / "reports" / "generated" / "jumia" / "step5_statistical_comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
