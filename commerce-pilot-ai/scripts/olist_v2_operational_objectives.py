"""Gate 7: separated operational objectives (catch late orders vs protect exposure),
using the strategy selected via historical periods only, evaluated as a POST_SELECTION
diagnostic on the already-exposed LATEST_TEMPORAL_STRESS_BLOCK."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "olist_v2"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "olist_v2"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from olist_v2_build_pipeline import FEATURE_COLS  # noqa: E402
from olist_v2_temporal_forensics_and_remedies import load, prequential_periods, fit_predict_lgbm  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if __name__ == "__main__":
    df = load()
    periods, dev, stress = prequential_periods(df, n_periods=6)

    # recency_weighted was selected -- reproduce its stress-block predictions
    Xtr, ytr = dev[FEATURE_COLS], dev["late_binary"].values
    half_life = max(1, len(Xtr) // 4)
    age = np.arange(len(Xtr))[::-1]
    weights = 0.5 ** (age / half_life)
    Xstress = stress[FEATURE_COLS]
    proba_stress = fit_predict_lgbm(Xtr, ytr, Xstress, sample_weight=weights)

    y = stress["late_binary"].values
    value = (stress["total_price"] + stress["total_freight"]).values
    n = len(stress)

    def topk(scores, k):
        order = np.argsort(-scores)[:k]
        return order

    objectives = {}
    rng_boot = np.random.RandomState(123)
    for budget in [5, 10, 20]:
        k = max(1, int(n * budget / 100))
        rand_scores = np.random.RandomState(42).rand(n)

        obj_a = {}  # CATCH MOST LATE ORDERS
        obj_b = {}  # PROTECT LATE EXPOSURE
        for name, scores in [("RANDOM", rand_scores), ("VALUE_ONLY", value), ("RISK_ONLY", proba_stress),
                              ("RISK_X_VALUE", proba_stress * (value / value.max()))]:
            order = topk(scores, k)
            obj_a[name] = {"late_recall": float(y[order].sum() / max(1, y.sum())),
                            "precision": float(y[order].sum() / k)}
            late_mask_order = y[order] == 1
            obj_b[name] = {"late_exposure_captured": float(value[order][late_mask_order].sum()),
                            "late_exposure_captured_frac": float(value[order][late_mask_order].sum() / max(1e-6, value[y == 1].sum()))}

        # bootstrap CI: RISK_ONLY vs VALUE_ONLY, both objectives
        boot_a, boot_b = [], []
        idx_all = np.arange(n)
        for _ in range(1000):
            bs = rng_boot.choice(idx_all, size=n, replace=True)
            y_bs, val_bs, risk_bs = y[bs], value[bs], proba_stress[bs]
            o_val = topk(val_bs, k)
            o_risk = topk(risk_bs, k)
            recall_val = y_bs[o_val].sum() / max(1, y_bs.sum())
            recall_risk = y_bs[o_risk].sum() / max(1, y_bs.sum())
            boot_a.append(recall_risk - recall_val)
            late_mask_val = y_bs[o_val] == 1
            late_mask_risk = y_bs[o_risk] == 1
            exp_val = val_bs[o_val][late_mask_val].sum() / max(1e-6, val_bs[y_bs == 1].sum())
            exp_risk = val_bs[o_risk][late_mask_risk].sum() / max(1e-6, val_bs[y_bs == 1].sum())
            boot_b.append(exp_risk - exp_val)
        boot_a, boot_b = np.array(boot_a), np.array(boot_b)

        objectives[f"budget_{budget}pct"] = {
            "OBJECTIVE_A_catch_late_orders": obj_a,
            "OBJECTIVE_A_bootstrap_CI_risk_vs_value": {"mean": float(boot_a.mean()), "ci_2.5": float(np.percentile(boot_a, 2.5)), "ci_97.5": float(np.percentile(boot_a, 97.5))},
            "OBJECTIVE_B_protect_exposure": obj_b,
            "OBJECTIVE_B_bootstrap_CI_risk_vs_value": {"mean": float(boot_b.mean()), "ci_2.5": float(np.percentile(boot_b, 2.5)), "ci_97.5": float(np.percentile(boot_b, 97.5))},
        }
        log(f"budget {budget}%: A(late_recall) RISK={obj_a['RISK_ONLY']['late_recall']:.3f} VALUE={obj_a['VALUE_ONLY']['late_recall']:.3f} "
            f"CI=[{objectives[f'budget_{budget}pct']['OBJECTIVE_A_bootstrap_CI_risk_vs_value']['ci_2.5']:.4f},{objectives[f'budget_{budget}pct']['OBJECTIVE_A_bootstrap_CI_risk_vs_value']['ci_97.5']:.4f}] | "
            f"B(exposure) RISK={obj_b['RISK_ONLY']['late_exposure_captured_frac']:.3f} VALUE={obj_b['VALUE_ONLY']['late_exposure_captured_frac']:.3f} "
            f"CI=[{objectives[f'budget_{budget}pct']['OBJECTIVE_B_bootstrap_CI_risk_vs_value']['ci_2.5']:.4f},{objectives[f'budget_{budget}pct']['OBJECTIVE_B_bootstrap_CI_risk_vs_value']['ci_97.5']:.4f}]")

    (REPORTS_DIR / "operational_objective_matrix.json").write_text(json.dumps({
        "note": "Evaluated using the recency_weighted strategy (selected via historical periods only) applied to the LATEST_TEMPORAL_STRESS_BLOCK as a POST_SELECTION_STRESS_DIAGNOSTIC, not a blind test.",
        "objectives": objectives,
    }, indent=2, default=str), encoding="utf-8")
    log("Written to operational_objective_matrix.json")
