"""Gate 23: paired bootstrap significance testing for MARBERT vs. baseline (and, if produced,
MARBERT vs. CAMeLBERT-Mix challenger). >=1000 resamples, reports point delta + 95% CI +
interpretation. Never declares a winner on a point estimate alone.

Run (after both baseline predictions and final MARBERT predictions exist for test_natural):
  .venv/Scripts/python.exe scripts/arabic_foundation_significance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
N_RESAMPLES = 1000
SEED = 42


def paired_bootstrap_macro_f1(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_resamples=N_RESAMPLES, seed=SEED) -> dict:
    """Bootstrap the macro-F1 DELTA (b - a) over paired predictions on the same rows."""
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
    significant = not (ci_lo <= 0 <= ci_hi)
    return {
        "n_resamples": n_resamples,
        "n_rows": int(n),
        "point_delta_b_minus_a": float(point_delta),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "ci_excludes_zero": bool(significant),
        "interpretation": (
            f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}] {'excludes' if significant else 'includes'} zero -> "
            f"the delta is {'statistically distinguishable from zero' if significant else 'NOT statistically distinguishable from zero (inconclusive at 95%)'}."
        ),
    }


def main() -> None:
    comparisons = {}

    baseline_pred = pd.read_parquet(REPO_ROOT / "artifacts/experiments/arabic_foundation/baseline/predictions/test_natural_predictions.parquet")
    marbert_pred = pd.read_parquet(REPO_ROOT / "artifacts/experiments/arabic_foundation/primary_model/predictions/test_natural_predictions.parquet")
    merged = baseline_pred.merge(marbert_pred, on="review_uid", suffixes=("_baseline", "_marbert"), validate="one_to_one")
    assert (merged["true_label_baseline"] == merged["true_label_marbert"]).all()

    y_true = merged["true_label_baseline"].values
    pred_baseline = merged["pred_label_baseline"].values
    pred_marbert = merged["pred_label_marbert"].values

    result = paired_bootstrap_macro_f1(y_true, pred_baseline, pred_marbert)
    comparisons["marbert_vs_baseline_test_natural"] = result
    print("MARBERT vs baseline (test_natural):", json.dumps(result, indent=2))

    challenger_pred_path = REPO_ROOT / "artifacts/experiments/arabic_foundation/challenger/predictions/test_natural_predictions.parquet"
    if challenger_pred_path.exists():
        challenger_pred = pd.read_parquet(challenger_pred_path)
        merged2 = challenger_pred.merge(marbert_pred, on="review_uid", suffixes=("_challenger", "_marbert"), validate="one_to_one")
        assert (merged2["true_label_challenger"] == merged2["true_label_marbert"]).all()
        y_true2 = merged2["true_label_challenger"].values
        pred_challenger = merged2["pred_label_challenger"].values
        pred_marbert2 = merged2["pred_label_marbert"].values
        result2 = paired_bootstrap_macro_f1(y_true2, pred_marbert2, pred_challenger)
        comparisons["challenger_vs_marbert_test_natural"] = result2
        print("Challenger vs MARBERT (test_natural):", json.dumps(result2, indent=2))
    else:
        comparisons["challenger_vs_marbert_test_natural"] = {"skipped": True, "reason": "challenger predictions not present (Gate 17 not run or challenger declined)"}

    (REPORTS_DIR / "statistical_significance.json").write_text(json.dumps(comparisons, indent=2, default=str), encoding="utf-8")
    print("\nWrote statistical_significance.json")


if __name__ == "__main__":
    main()
