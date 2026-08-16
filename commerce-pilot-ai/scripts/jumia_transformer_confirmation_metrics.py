"""Aggregate the 3-seed Jumia MARBERT confirmation results and compute a
paired bootstrap comparison against the frozen classical baseline
(char_wb_tfidf_logreg) on the same validation rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_DIR = REPO_ROOT / "reports" / "generated" / "jumia" / "transformer_adaptation"
SEEDS = [101, 202, 303]


def main() -> None:
    runs = []
    for seed in SEEDS:
        path = TRANSFORMER_DIR / f"jumia_UBC-NLP__MARBERT_seed{seed}_confirm.json"
        runs.append(json.loads(path.read_text(encoding="utf-8")))

    macro_f1s = [r["macro_f1"] for r in runs]
    bal_accs = [r["balanced_accuracy"] for r in runs]
    worst_class_f1s = [r["worst_class_f1"] for r in runs]

    classical_path = REPO_ROOT / "reports" / "generated" / "jumia" / "classical_baselines" / "jumia_classical_baselines.json"
    classical = json.loads(classical_path.read_text(encoding="utf-8"))
    classical_macro_f1 = classical["best_classical_macro_f1"]
    classical_key = classical["best_classical_candidate"]

    summary = {
        "schema_version": "jumia-transformer-confirmation-summary-v1",
        "generated_at": "2026-08-15",
        "model_name": "UBC-NLP/MARBERT",
        "revision": "88e1fa192dd723cf0b3563500aec46209762eb22",
        "seeds": SEEDS,
        "macro_f1_per_seed": macro_f1s,
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "std_macro_f1": float(np.std(macro_f1s, ddof=1)),
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "mean_worst_class_f1": float(np.mean(worst_class_f1s)),
        "min_worst_class_f1": float(np.min(worst_class_f1s)),
        "classical_baseline_key": classical_key,
        "classical_baseline_macro_f1": classical_macro_f1,
        "mean_diff_vs_classical": float(np.mean(macro_f1s) - classical_macro_f1),
        "materiality_threshold": 0.02,
        "materially_better_by_mean": bool(np.mean(macro_f1s) - classical_macro_f1 >= 0.02),
        "internal_test_accessed": False,
    }

    out_path = TRANSFORMER_DIR / "jumia_marbert_confirmation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
