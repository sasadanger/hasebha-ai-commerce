"""Gate 17/23/25: MARBERT vs. CAMeLBERT-Mix challenger comparison -- VALIDATION ONLY.

Per the binding constraint recorded in reports/generated/arabic_foundation/
protected_test_access_ledger.json (test_natural/item_holdout_stress were exposed early, during a
gate-ordering violation later corrected), this comparison deliberately uses ONLY val_natural and
val_balanced predictions. It does NOT read final_test_evaluation.json,
baseline_vs_marbert_comparison.json, or statistical_significance.json (the already-exposed
test-set files) -- this is enforced by simply never importing/opening those paths below.

Decision hierarchy (Gate 25, applied here for the challenger sub-decision): validation macro-F1
-> Neutral/Mixed F1 -> robustness (not assessed head-to-head here, external robustness Gate 22 was
MARBERT-only) -> calibration (not required for this decision) -> runtime/complexity as tie-breaker.

Run (after the challenger has finished training and produced val_natural/val_balanced predictions):
  .venv/Scripts/python.exe scripts/arabic_foundation_challenger_validation_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
MARBERT_PRED_DIR = REPO_ROOT / "artifacts/experiments/arabic_foundation/primary_model/predictions"
CHALLENGER_PRED_DIR = REPO_ROOT / "artifacts/experiments/arabic_foundation/challenger/predictions"
N_RESAMPLES = 1000
SEED = 42


def paired_bootstrap_macro_f1(y_true, pred_a, pred_b, n_resamples=N_RESAMPLES, seed=SEED) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        f1_a = f1_score(y_true[idx], pred_a[idx], average="macro")
        f1_b = f1_score(y_true[idx], pred_b[idx], average="macro")
        deltas[i] = f1_b - f1_a
    point_delta = f1_score(y_true, pred_b, average="macro") - f1_score(y_true, pred_a, average="macro")
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_resamples": n_resamples, "n_rows": int(n),
        "point_delta_challenger_minus_marbert": float(point_delta),
        "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi),
        "ci_excludes_zero": bool(not (ci_lo <= 0 <= ci_hi)),
    }


def main() -> None:
    if not CHALLENGER_PRED_DIR.exists():
        print("Challenger predictions not found -- Gate 17 has not completed. Nothing to compare.")
        return

    comparison = {"protocol_note": "VALIDATION ONLY per binding constraint -- test_natural/item_holdout_stress NOT read by this script."}

    marbert_training_cfg = json.loads((REPO_ROOT / "artifacts/experiments/arabic_foundation/primary_model/final/training_config.json").read_text(encoding="utf-8"))
    challenger_training_cfg = json.loads((REPO_ROOT / "artifacts/experiments/arabic_foundation/challenger/final/training_config.json").read_text(encoding="utf-8"))

    for split_name in ["val_natural", "val_balanced"]:
        m = pd.read_parquet(MARBERT_PRED_DIR / f"{split_name}_predictions.parquet")
        c = pd.read_parquet(CHALLENGER_PRED_DIR / f"{split_name}_predictions.parquet")
        merged = m.merge(c, on="review_uid", suffixes=("_marbert", "_challenger"), validate="one_to_one")
        assert (merged["true_label_marbert"] == merged["true_label_challenger"]).all()

        y_true = merged["true_label_marbert"].values
        pred_m = merged["pred_label_marbert"].values
        pred_c = merged["pred_label_challenger"].values

        macro_f1_m = f1_score(y_true, pred_m, average="macro")
        macro_f1_c = f1_score(y_true, pred_c, average="macro")
        neutral_f1_m = f1_score(y_true, pred_m, average=None, labels=[0, 1, 2])[1]
        neutral_f1_c = f1_score(y_true, pred_c, average=None, labels=[0, 1, 2])[1]

        boot = paired_bootstrap_macro_f1(y_true, pred_m, pred_c)

        comparison[split_name] = {
            "n": int(len(merged)),
            "marbert_macro_f1": float(macro_f1_m),
            "challenger_macro_f1": float(macro_f1_c),
            "marbert_neutral_mixed_f1": float(neutral_f1_m),
            "challenger_neutral_mixed_f1": float(neutral_f1_c),
            "bootstrap": boot,
        }
        print(f"{split_name}: MARBERT macro_f1={macro_f1_m:.4f} neutral_f1={neutral_f1_m:.4f} | "
              f"Challenger macro_f1={macro_f1_c:.4f} neutral_f1={neutral_f1_c:.4f} | "
              f"delta={boot['point_delta_challenger_minus_marbert']:+.4f} CI=[{boot['ci_95_lo']:+.4f},{boot['ci_95_hi']:+.4f}]")

    # ---- Gate 25 decision hierarchy, applied here, VALIDATION ONLY ----
    vn = comparison["val_natural"]
    vb = comparison["val_balanced"]
    macro_meaningfully_better = (
        vn["bootstrap"]["ci_excludes_zero"] and vn["bootstrap"]["point_delta_challenger_minus_marbert"] > 0
        and vb["bootstrap"]["ci_excludes_zero"] and vb["bootstrap"]["point_delta_challenger_minus_marbert"] > 0
    )
    neutral_not_regressed = (vn["challenger_neutral_mixed_f1"] >= vn["marbert_neutral_mixed_f1"] - 0.01) and \
                             (vb["challenger_neutral_mixed_f1"] >= vb["marbert_neutral_mixed_f1"] - 0.01)

    if macro_meaningfully_better and neutral_not_regressed:
        decision = "CHALLENGER_REPLACES_MARBERT"
        reasoning = (
            "CAMeLBERT-Mix shows a statistically-supported macro-F1 improvement over MARBERT on "
            "BOTH val_natural and val_balanced (95% CI excludes zero and favors the challenger on "
            "both), without regressing Neutral/Mixed-F1 by more than 1pp on either. Per the Gate 25 "
            "decision hierarchy, the challenger becomes primary."
        )
    else:
        decision = "MARBERT_STAYS_PRIMARY"
        reasoning = (
            "CAMeLBERT-Mix does NOT show a statistically-supported, non-regressing improvement over "
            "MARBERT on validation evidence (val_natural and val_balanced, macro-F1 and Neutral/Mixed-F1, "
            "with a 95% bootstrap CI). Per the Gate 17 instruction, MARBERT stays primary -- this is a "
            "legitimate, expected outcome, not a failure. No test-set numbers were consulted in reaching "
            "this decision (see protected_test_access_ledger.json for why that matters here)."
        )

    comparison["training_cost_comparison"] = {
        "marbert_train_seconds": marbert_training_cfg.get("train_seconds"),
        "challenger_train_seconds": challenger_training_cfg.get("train_seconds"),
        "marbert_peak_vram_mb": marbert_training_cfg.get("peak_vram_mb"),
        "challenger_peak_vram_mb": challenger_training_cfg.get("peak_vram_mb"),
    }
    comparison["DECISION"] = decision
    comparison["DECISION_REASONING"] = reasoning
    comparison["binding_constraint_honored"] = "This script reads ONLY val_natural/val_balanced prediction parquets and training_config.json files. It does not open final_test_evaluation.json, baseline_vs_marbert_comparison.json, or statistical_significance.json (the exposed test-set files)."

    (REPORTS_DIR / "challenger_validation_only_comparison.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print("\nDECISION:", decision)
    print(reasoning)
    print("\nWrote challenger_validation_only_comparison.json")


if __name__ == "__main__":
    main()
