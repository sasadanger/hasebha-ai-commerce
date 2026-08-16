"""Phase 5 -- compute metrics + paired bootstrap comparison for the direct
Egyptian-domain transfer evaluation of the two frozen LABR finalists on
Jumia validation. No training. No protected-test access.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "reports" / "generated" / "jumia" / "direct_transfer"
LABELS = ["1", "2", "3", "4", "5"]

RNG_SEED = 20260815
N_BOOTSTRAP = 2000


def load(model_key: str) -> dict:
    return json.loads((IN_DIR / f"predictions_{model_key}_validation.json").read_text(encoding="utf-8"))


def compute_metrics(true, pred) -> dict:
    macro_f1 = f1_score(true, pred, labels=LABELS, average="macro")
    bal_acc = balanced_accuracy_score(true, pred)
    acc = accuracy_score(true, pred)
    report = classification_report(true, pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(true, pred, labels=LABELS)
    true_int = [int(t) for t in true]
    pred_int = [int(p) for p in pred]
    abs_errors = [abs(t - p) for t, p in zip(true_int, pred_int)]
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "per_class": {lbl: report[lbl] for lbl in LABELS},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABELS,
        "support": {lbl: int(report[lbl]["support"]) for lbl in LABELS},
        "rating_diagnostics": {
            "mean_absolute_error": float(np.mean(abs_errors)),
            "adjacent_class_error_rate": float(np.mean([e == 1 for e in abs_errors])),
            "severe_error_rate_ge2": float(np.mean([e >= 2 for e in abs_errors])),
        },
    }


def paired_bootstrap_macro_f1_diff(true, pred_a, pred_b, seed=RNG_SEED, n=N_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    n_items = len(true)
    true_arr = np.array(true)
    pred_a_arr = np.array(pred_a)
    pred_b_arr = np.array(pred_b)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_items, n_items)
        f1_a = f1_score(true_arr[idx], pred_a_arr[idx], labels=LABELS, average="macro", zero_division=0)
        f1_b = f1_score(true_arr[idx], pred_b_arr[idx], labels=LABELS, average="macro", zero_division=0)
        diffs.append(f1_a - f1_b)
    diffs = np.array(diffs)
    point_estimate = f1_score(true_arr, pred_a_arr, labels=LABELS, average="macro") - f1_score(
        true_arr, pred_b_arr, labels=LABELS, average="macro"
    )
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        "comparison": "MARBERT - AraBERT (Jumia direct transfer, macro_f1)",
        "point_estimate_diff": float(point_estimate),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "n_bootstrap": n,
    }


def main() -> None:
    marbert = load("C_MARBERT")
    arabert = load("C_AraBERT")

    assert marbert["row_indices"] == arabert["row_indices"], "row alignment mismatch between the two models"
    assert marbert["true_labels"] == arabert["true_labels"], "label alignment mismatch"

    true = marbert["true_labels"]
    pred_marbert = marbert["predicted_labels"]
    pred_arabert = arabert["predicted_labels"]

    marbert_metrics = compute_metrics(true, pred_marbert)
    arabert_metrics = compute_metrics(true, pred_arabert)
    bootstrap = paired_bootstrap_macro_f1_diff(true, pred_marbert, pred_arabert)

    decision = "STILL_INCONCLUSIVE"
    if bootstrap["ci_excludes_zero"]:
        decision = "MARBERT" if bootstrap["point_estimate_diff"] > 0 else "ARABERT"

    out = {
        "schema_version": "jumia-direct-transfer-metrics-v1",
        "generated_at": "2026-08-15",
        "n_validation_rows": marbert["n_rows"],
        "internal_test_accessed": False,
        "MARBERT": {
            "model_name": marbert["model_name"],
            "revision": marbert["revision"],
            **marbert_metrics,
        },
        "AraBERT": {
            "model_name": arabert["model_name"],
            "revision": arabert["revision"],
            **arabert_metrics,
        },
        "paired_bootstrap": bootstrap,
        "egyptian_domain_tie_break_decision": decision,
    }

    out_path = IN_DIR / "direct_transfer_metrics_and_tiebreak.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("MARBERT", "AraBERT")}, indent=2), file=sys.stderr)
    print("MARBERT macro_f1:", marbert_metrics["macro_f1"], "balanced_accuracy:", marbert_metrics["balanced_accuracy"], file=sys.stderr)
    print("AraBERT macro_f1:", arabert_metrics["macro_f1"], "balanced_accuracy:", arabert_metrics["balanced_accuracy"], file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
