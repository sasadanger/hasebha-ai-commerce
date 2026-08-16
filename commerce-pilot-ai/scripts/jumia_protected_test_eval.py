"""Phase 9 -- the single, one-time PROTECTED_TEST read for the frozen Jumia
champion. Must only run after PRE_TEST_RELEASE_RECORD.md exists. No tuning,
seed change, or candidate switch follows this run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nlp.text_normalization import normalize_text  # noqa: E402

CSV_PATH = ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
SPLIT_PATH = ROOT / "artifacts" / "experiments" / "jumia" / "phase1_split" / "jumia_split_assignments.parquet"
CHAMPION_DIR = ROOT / "artifacts" / "experiments" / "jumia" / "phase2_champion" / "JUMIA_MARBERT_LABR_INIT_seed202"
FREEZE_RECORD = ROOT / "reports" / "checkpoints" / "jumia_phase9_freeze_and_protected_test_2026-08-15" / "PRE_TEST_RELEASE_RECORD.md"
LABELS = ["1", "2", "3", "4", "5"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not FREEZE_RECORD.exists():
        print("PRE_TEST_RELEASE_RECORD.md missing -- refusing to access PROTECTED_TEST.", file=sys.stderr)
        sys.exit(1)
    freeze_hash = sha256_file(FREEZE_RECORD)
    print(f"Freeze record present, sha256={freeze_hash}", file=sys.stderr)

    manifest = json.loads((CHAMPION_DIR / "export_manifest.json").read_text())
    for fname, expected in manifest["file_hashes"].items():
        actual = sha256_file(CHAMPION_DIR / fname)
        if actual != expected:
            raise RuntimeError(f"{fname} hash mismatch: expected {expected}, got {actual}")
    print("Champion artifact hashes verified.", file=sys.stderr)

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["customer_rating"] = df["customer_rating"].astype(str)
    df["_norm"] = df["review"].map(normalize_text)
    split_df = pd.read_parquet(SPLIT_PATH)

    print(">>> Opening PROTECTED_TEST (one-time access) <<<", file=sys.stderr)
    test_idx = split_df.loc[split_df["split"] == "internal_test", "row_index"].tolist()
    test = df.iloc[test_idx]
    true = test["customer_rating"].tolist()
    texts = test["_norm"].tolist()
    print(f"n_protected_test_rows={len(test)}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(str(CHAMPION_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(CHAMPION_DIR))
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

    macro_f1 = f1_score(true, preds, labels=LABELS, average="macro")
    bal_acc = balanced_accuracy_score(true, preds)
    acc = accuracy_score(true, preds)
    report = classification_report(true, preds, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(true, preds, labels=LABELS)

    true_int = np.array([int(t) for t in true])
    pred_int = np.array([int(p) for p in preds])
    abs_err = np.abs(true_int - pred_int)

    # bootstrap CI on the final protected-test macro-F1 (uncertainty, not a
    # comparison -- no re-tuning follows this)
    rng = np.random.default_rng(20260815)
    n_items = len(true)
    true_arr = np.array(true)
    pred_arr = np.array(preds)
    boot_f1s = []
    for _ in range(2000):
        idx = rng.integers(0, n_items, n_items)
        boot_f1s.append(f1_score(true_arr[idx], pred_arr[idx], labels=LABELS, average="macro", zero_division=0))
    ci_low, ci_high = np.percentile(boot_f1s, [2.5, 97.5])

    result = {
        "schema_version": "jumia-protected-test-final-v1",
        "generated_at": "2026-08-15",
        "freeze_record": "reports/checkpoints/jumia_phase9_freeze_and_protected_test_2026-08-15/PRE_TEST_RELEASE_RECORD.md",
        "freeze_record_sha256": freeze_hash,
        "champion_export": str(CHAMPION_DIR.relative_to(ROOT)).replace("\\", "/"),
        "n_protected_test_rows": len(test),
        "macro_f1": macro_f1,
        "macro_f1_bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABELS,
        "class2_f1": report["2"]["f1-score"],
        "class2_recall": report["2"]["recall"],
        "rating_diagnostics": {
            "mae": float(abs_err.mean()),
            "exact_accuracy": float((abs_err == 0).mean()),
            "within_1_star_accuracy": float((abs_err <= 1).mean()),
            "severe_error_rate_ge2": float((abs_err >= 2).mean()),
        },
        "one_time_access_note": "This is the single PROTECTED_TEST read for this freeze. No tuning follows this run.",
    }

    out_path = ROOT / "reports" / "generated" / "jumia" / "protected_test_final_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "confusion_matrix"}, indent=2), file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
