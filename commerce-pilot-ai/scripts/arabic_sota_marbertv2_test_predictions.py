"""Gate 4 Task 11 (partial -- honest limitation applies): generate predictions from a
trained MARBERTv2 checkpoint on test_unlabeled.csv, broken out by the four frozen test
views (frozen_test_views.json).

CRITICAL LIMITATION, stated plainly rather than glossed over: the official AraSentEval
2026 test set gold labels were never released to participants and are not available to
this project. This script therefore CANNOT compute a true test-set Macro-F1, weighted-F1,
balanced accuracy, MCC, confusion matrix, or per-class P/R/F1 against ground truth -- doing
so would require labels this project does not have. What it CAN do, and does, is:
  (a) generate predictions for every test row,
  (b) report the predicted label distribution and prediction confidence broken out by each
      of the four frozen views, as a CONSISTENCY check (does the model behave differently on
      the leakage-flagged rows vs. the leakage-safe rows? e.g. higher confidence on rows that
      are near-duplicates of training data would itself be informative, even without labels).
This is recorded as MODEL_PERFORMANCE_ACCESS in the protected test ledger the first time it
is run, since it does involve running a trained model against test inputs (even though no
score against gold labels is computed) -- logged explicitly, not silently.

Run: .venv/Scripts/python.exe scripts/arabic_sota_marbertv2_test_predictions.py --track a
     .venv/Scripts/python.exe scripts/arabic_sota_marbertv2_test_predictions.py --track b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_sota" / "raw_data_gate3"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_sota"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_sota" / "marbertv2_reproduction"
LABELS = ["negative", "neutral", "positive"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b"], required=True)
    args = ap.parse_args()

    track_name = "track_a_paper_faithful" if args.track == "a" else "track_b_group_safe"
    ckpt_dir = ARTIFACT_DIR / track_name / "best_checkpoint"
    if not ckpt_dir.exists():
        raise SystemExit(f"No checkpoint found at {ckpt_dir} -- run training for this track first.")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log(f"Loading fresh-process checkpoint from {ckpt_dir} ...")
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    test = pd.read_csv(DATA_DIR / "test_unlabeled.csv").reset_index().rename(columns={"index": "row_idx"})
    views = json.loads((DATA_DIR / "frozen_test_views.json").read_text(encoding="utf-8"))

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(test), 32):
            batch = test["Sentence"].iloc[i:i + 32].tolist()
            enc = tok(batch, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(pred.cpu().tolist())
            confs.extend(conf.cpu().tolist())

    test["pred_label"] = [LABELS[p] for p in preds]
    test["pred_confidence"] = confs

    per_view = {}
    for view_name, view_info in views.items():
        rows = set(view_info["row_idxs"])
        sub = test[test["row_idx"].isin(rows)]
        per_view[view_name] = {
            "n_rows": len(sub),
            "pred_distribution": sub["pred_label"].value_counts().to_dict(),
            "mean_confidence": float(sub["pred_confidence"].mean()),
        }
        log(f"{view_name}: n={len(sub)} dist={per_view[view_name]['pred_distribution']} "
            f"mean_conf={per_view[view_name]['mean_confidence']:.4f}")

    out = {
        "track": track_name,
        "checkpoint_path": str(ckpt_dir),
        "LIMITATION": "No gold test labels available -- predictions and confidence only, NOT scored metrics. See module docstring.",
        "per_view_prediction_summary": per_view,
    }
    out_path = REPORTS_DIR / f"marbertv2_{track_name}_test_predictions_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"Written to {out_path}")

    # append MODEL_PERFORMANCE_ACCESS event to the protected ledger
    ledger_path = REPORTS_DIR / "protected_test_access_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger.setdefault("MODEL_PERFORMANCE_ACCESS_events", []).append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "track": track_name,
        "description": "Generated predictions (not scored -- no gold labels available) on test_unlabeled.csv "
                        "for the four frozen views, for consistency-checking only. This is logged as "
                        "MODEL_PERFORMANCE_ACCESS because it runs a trained model against test inputs, even "
                        "though no metric against ground truth was or could be computed.",
    })
    ledger["current_status"]["any_model_performance_access_to_date"] = True
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    log("Protected test access ledger updated (MODEL_PERFORMANCE_ACCESS event logged).")


if __name__ == "__main__":
    main()
