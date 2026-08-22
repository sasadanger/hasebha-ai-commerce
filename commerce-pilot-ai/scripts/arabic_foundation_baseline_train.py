"""Gate 10: mandatory classical baseline for the Arabic sentiment foundation task.

Word+char TF-IDF + LinearSVC (calibrated for predict_proba), matching this project's established
Amazon-pipeline convention (src/nlp/amazon/features.py / train.py), adapted for 3-class Arabic.
ONE baseline only, no hyperparameter search, per Gate 10 instruction. Trained on the exact same
leakage-safe splits/row-ids/label-contract that the transformer will use (Gate 5 split_manifest).

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_baseline_train.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nlp.arabic_foundation.features import build_word_char_union  # noqa: E402

SEED = 42
SPLITS_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "splits"
MODEL_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation" / "baseline"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}.parquet")


def main() -> None:
    train = load_split("train")
    val_natural = load_split("val_natural")
    val_balanced = load_split("val_balanced")
    test_natural = load_split("test_natural")
    item_holdout = load_split("item_holdout_stress")

    print(f"train={len(train)} val_natural={len(val_natural)} val_balanced={len(val_balanced)} "
          f"test_natural={len(test_natural)} item_holdout={len(item_holdout)}")

    pipe = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
        [
            ("tfidf", build_word_char_union()),
            ("clf", CalibratedClassifierCV(LinearSVC(random_state=SEED, dual="auto"), cv=3, method="sigmoid")),
        ]
    )

    t0 = time.time()
    pipe.fit(train["text_norm"].tolist(), train["label"].tolist())
    train_seconds = time.time() - t0
    print(f"fit done in {train_seconds:.1f}s")

    model_path = MODEL_DIR / "tfidf_wordchar_linearsvc.joblib"
    joblib.dump(pipe, model_path)
    model_hash = sha256_file(model_path)

    preds_dir = MODEL_DIR / "predictions"
    preds_dir.mkdir(exist_ok=True)

    eval_summary = {}
    for split_name, df in [
        ("val_natural", val_natural), ("val_balanced", val_balanced),
        ("test_natural", test_natural), ("item_holdout_stress", item_holdout),
    ]:
        texts = df["text_norm"].tolist()
        t0 = time.time()
        proba = pipe.predict_proba(texts)
        infer_seconds = time.time() - t0
        pred = proba.argmax(axis=1)
        out = pd.DataFrame({
            "review_uid": df["review_uid"].values,
            "true_label": df["label"].values,
            "pred_label": pred,
            "proba_negative": proba[:, 0],
            "proba_neutral_mixed": proba[:, 1],
            "proba_positive": proba[:, 2],
        })
        out_path = preds_dir / f"{split_name}_predictions.parquet"
        out.to_parquet(out_path, index=False)

        from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
        macro_f1 = f1_score(df["label"], pred, average="macro")
        neutral_f1 = f1_score(df["label"], pred, average=None, labels=[0, 1, 2])[1]
        acc = accuracy_score(df["label"], pred)
        bal_acc = balanced_accuracy_score(df["label"], pred)
        eval_summary[split_name] = {
            "n": len(df),
            "macro_f1": float(macro_f1),
            "neutral_mixed_f1": float(neutral_f1),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "inference_seconds_total": infer_seconds,
            "inference_rows_per_second": len(df) / infer_seconds if infer_seconds > 0 else None,
            "predictions_path": str(out_path.relative_to(REPO_ROOT)),
        }
        print(split_name, eval_summary[split_name])

    manifest = {
        "model": "tfidf_word_char_union + LinearSVC (sigmoid-calibrated via CalibratedClassifierCV, cv=3)",
        "seed": SEED,
        "train_rows": len(train),
        "train_seconds": train_seconds,
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "model_sha256": model_hash,
        "eval_summary": eval_summary,
        "note": "One fixed baseline config, no hyperparameter search, per Gate 10 instruction. "
                "Same splits/row-ids/label-contract as the MARBERT transformer for direct comparability.",
    }
    (REPORT_DIR / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("\nWrote baseline_manifest.json")


if __name__ == "__main__":
    main()
