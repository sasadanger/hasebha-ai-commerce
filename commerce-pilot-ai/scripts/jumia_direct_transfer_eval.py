"""Phase 5 -- direct Egyptian-domain transfer evaluation of the two frozen
LABR finalists on Jumia VALIDATION only. No fine-tuning, no gradient step,
no protected-test access. Uses the exact already-frozen model revision,
tokenizer, preprocessing, and label mapping exported for the FastAPI
service (artifacts/experiments/nlp/inference_exports/C_MARBERT|C_AraBERT).

Run once per model (separate process invocations) so RAM is fully released
between the two -- one heavy workload at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nlp.text_normalization import normalize_text  # noqa: E402

EXPORT_ROOT = REPO_ROOT / "artifacts" / "experiments" / "nlp" / "inference_exports"
SPLIT_PATH = REPO_ROOT / "artifacts" / "experiments" / "jumia" / "phase1_split" / "jumia_split_assignments.parquet"
CSV_PATH = REPO_ROOT / "data" / "raw" / "jumia" / "extracted" / "jumia_reviews.csv"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "jumia" / "direct_transfer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEYS = {"marbert": "C_MARBERT", "arabert": "C_AraBERT"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["marbert", "arabert"])
    parser.add_argument("--split", default="validation", choices=["validation"])  # protected_test intentionally not selectable here
    args = parser.parse_args()

    model_key = MODEL_KEYS[args.model]
    export_dir = EXPORT_ROOT / model_key
    manifest = json.loads((EXPORT_ROOT / "export_manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["exports"] if e["key"] == model_key)

    for fname, expected_hash in entry["model_file_hashes"].items():
        actual = sha256_file(export_dir / fname)
        if actual != expected_hash:
            raise RuntimeError(f"{fname} hash mismatch: expected {expected_hash}, got {actual}")

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["customer_rating"] = df["customer_rating"].astype(str)
    split_df = pd.read_parquet(SPLIT_PATH)
    val_indices = split_df.loc[split_df["split"] == args.split, "row_index"].tolist()
    eval_rows = df.iloc[val_indices].copy()
    print(f"[{model_key}] evaluating on Jumia {args.split}: n={len(eval_rows)}", file=sys.stderr)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(export_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(export_dir))
    model.eval()

    id2label = model.config.id2label
    predictions = []
    all_probs = []
    texts_normalized = eval_rows["review"].map(normalize_text).tolist()
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(texts_normalized), batch_size):
            batch_texts = texts_normalized[start : start + batch_size]
            encoded = tokenizer(
                batch_texts, truncation=True, padding=True, max_length=entry["max_length"], return_tensors="pt"
            )
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            for i, pid in enumerate(pred_ids):
                predictions.append(id2label[pid])
                all_probs.append(probs[i].tolist())
            if start % (batch_size * 5) == 0:
                print(f"[{model_key}] {start + len(batch_texts)}/{len(texts_normalized)}", file=sys.stderr)

    out = {
        "schema_version": "jumia-direct-transfer-v1",
        "model_key": model_key,
        "model_name": entry["model_name"],
        "revision": entry["revision"],
        "max_length": entry["max_length"],
        "split_evaluated": args.split,
        "n_rows": len(eval_rows),
        "row_indices": val_indices,
        "true_labels": eval_rows["customer_rating"].tolist(),
        "predicted_labels": predictions,
        "class_labels_order": [id2label[i] for i in range(len(id2label))],
        "probabilities": all_probs,
        "internal_test_accessed": False,
        "fine_tuned_this_session": False,
    }
    out_path = OUT_DIR / f"predictions_{model_key}_{args.split}.json"
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"[{model_key}] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
