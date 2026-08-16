"""Phase 6 -- compact classical Jumia baselines. Train-only fit,
validation-only evaluation. No protected-test access. No hyperparameter
sweep, no experiment sprawl -- exactly majority, stratified random,
TF-IDF word/bigram linear, and char_wb TF-IDF linear (justified below).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.nlp.text_normalization import normalize_text  # noqa: E402

CSV_PATH = REPO_ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
SPLIT_PATH = REPO_ROOT / "artifacts" / "experiments" / "jumia" / "phase1_split" / "jumia_split_assignments.parquet"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "jumia" / "classical_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS = ["1", "2", "3", "4", "5"]
SEED = 42


def metrics_block(true, pred):
    report = classification_report(true, pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(true, pred, labels=LABELS)
    return {
        "macro_f1": f1_score(true, pred, labels=LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "accuracy": accuracy_score(true, pred),
        "per_class": {lbl: report[lbl] for lbl in LABELS},
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["customer_rating"] = df["customer_rating"].astype(str)
    df["_norm"] = df["review"].map(normalize_text)
    split_df = pd.read_parquet(SPLIT_PATH)

    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].tolist()
    val_idx = split_df.loc[split_df["split"] == "validation", "row_index"].tolist()

    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    X_train, y_train = train["_norm"], train["customer_rating"]
    X_val, y_val = val["_norm"], val["customer_rating"]

    results = {}

    # 1. majority
    dummy_majority = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(X_train, y_train)
    results["majority"] = metrics_block(y_val, dummy_majority.predict(X_val))

    # 2. stratified random
    dummy_stratified = DummyClassifier(strategy="stratified", random_state=SEED).fit(X_train, y_train)
    results["stratified_random"] = metrics_block(y_val, dummy_stratified.predict(X_val))

    # 3. TF-IDF word unigram+bigram + LogisticRegression
    word_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ]).fit(X_train, y_train)
    results["tfidf_word_bigram_logreg"] = metrics_block(y_val, word_pipeline.predict(X_val))

    # 4. char_wb TF-IDF -- justified by EDA: 5.1% of reviews contain
    # character-elongation noise (repeated-letter emphasis, e.g. "جمييييل"),
    # a real (if modest) script-noise signal per the EDA report
    # (reports/generated/jumia/jumia_eda_report.json#reviews_with_char_elongation).
    char_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ]).fit(X_train, y_train)
    results["char_wb_tfidf_logreg"] = metrics_block(y_val, char_pipeline.predict(X_val))

    best_key = max(results, key=lambda k: results[k]["macro_f1"])
    out = {
        "schema_version": "jumia-classical-baselines-v1",
        "generated_at": "2026-08-15",
        "n_train": len(train),
        "n_validation": len(val),
        "char_wb_justification": (
            f"{int((df['review'].fillna('').str.contains('(.)\\1{{2,}}', regex=True)).sum())} rows "
            "with character-elongation noise in full dataset EDA (5.1% share) -- modest but real "
            "script-noise signal, per Phase 6 instructions ('ONLY if script/noise/Arabizi evidence "
            "justifies it')."
        ),
        "internal_test_accessed": False,
        "results": results,
        "best_classical_candidate": best_key,
        "best_classical_macro_f1": results[best_key]["macro_f1"],
    }
    out_path = OUT_DIR / "jumia_classical_baselines.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in results.items():
        print(f"{k}: macro_f1={v['macro_f1']:.4f} balanced_acc={v['balanced_accuracy']:.4f} acc={v['accuracy']:.4f}", file=sys.stderr)
    print(f"Best: {best_key} ({results[best_key]['macro_f1']:.4f})", file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
