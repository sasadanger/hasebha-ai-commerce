"""Olist V2: temporal split, leakage tests, baseline+GBDT ladder, calibration,
operational ranking vs VALUE_ONLY. Loads the canonical dataset built by
olist_v2_build_pipeline.py."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "olist_v2"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "olist_v2"

from olist_v2_build_pipeline import FEATURE_COLS  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_temporal_splits(df):
    """3 rolling-origin development folds + 1 protected final temporal block."""
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(df)
    # Protect final 15% as the temporal test (latest period, untouched until freeze)
    protected_start_idx = int(n * 0.85)
    protected = df.iloc[protected_start_idx:].copy()
    dev = df.iloc[:protected_start_idx].copy()

    # 3 rolling-origin folds within dev: expanding train, next-period val
    n_dev = len(dev)
    fold_bounds = [int(n_dev * f) for f in (0.5, 0.65, 0.8, 1.0)]
    folds = []
    for i in range(3):
        tr = dev.iloc[:fold_bounds[i]]
        va = dev.iloc[fold_bounds[i]:fold_bounds[i + 1]]
        folds.append((tr, va))

    manifest = {
        "n_total": n, "n_dev": n_dev, "n_protected_test": len(protected),
        "protected_test_date_range": [str(protected["order_purchase_timestamp"].min()),
                                        str(protected["order_purchase_timestamp"].max())],
        "protected_test_late_prevalence": float(protected["late_binary"].mean()),
        "folds": [{"train_n": len(tr), "val_n": len(va),
                    "train_date_range": [str(tr["order_purchase_timestamp"].min()), str(tr["order_purchase_timestamp"].max())],
                    "val_date_range": [str(va["order_purchase_timestamp"].min()), str(va["order_purchase_timestamp"].max())],
                    "val_late_prevalence": float(va["late_binary"].mean())}
                   for tr, va in folds],
    }
    (REPORTS_DIR / "temporal_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(json.dumps(manifest, indent=2, default=str))
    return folds, dev, protected


def run_leakage_tests(dev):
    """GATE 8: recompute seller history for a sample of rows from preceding rows only, verify match."""
    dev_sorted = dev.sort_values("order_purchase_timestamp").reset_index(drop=True)
    sample_idx = np.random.RandomState(42).choice(len(dev_sorted), size=min(30, len(dev_sorted)), replace=False)
    failures = []
    for idx in sample_idx:
        row = dev_sorted.iloc[idx]
        seller = row["seller_id"]
        prior = dev_sorted.iloc[:idx]
        prior_seller_rows = prior[prior["seller_id"] == seller]
        expected_count = len(prior_seller_rows)
        actual_count = row["seller_prior_order_count"]
        if expected_count != actual_count:
            failures.append({"idx": int(idx), "expected_count": int(expected_count), "actual_count": int(actual_count)})
            continue
        if expected_count > 0:
            expected_late_rate = prior_seller_rows["late_binary"].mean()
            actual_late_rate = row["seller_prior_late_rate"]
            if not (pd.isna(expected_late_rate) and pd.isna(actual_late_rate)) and abs(expected_late_rate - actual_late_rate) > 1e-9:
                failures.append({"idx": int(idx), "expected_late_rate": float(expected_late_rate), "actual_late_rate": float(actual_late_rate)})
    result = {"n_sampled": len(sample_idx), "n_failures": len(failures), "failures": failures[:5],
              "LEAKAGE_TESTS_PASS": len(failures) == 0}
    (REPORTS_DIR / "leakage_unit_tests.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log(f"leakage tests: {len(failures)} failures of {len(sample_idx)} sampled -- PASS={len(failures)==0}")
    if failures:
        raise SystemExit("LEAKAGE TEST FAILED -- blocking modeling per Gate 8.")
    return result


def prep_xy(df, feature_cols):
    X = df[feature_cols].copy()
    cat_cols = []  # all numeric in FEATURE_COLS as constructed
    y = df["late_binary"].values
    return X, y


def train_and_eval(folds, protected):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, brier_score_loss

    results = {"folds": [], "logistic": [], "lightgbm": []}

    for i, (tr, va) in enumerate(folds):
        Xtr, ytr = prep_xy(tr, FEATURE_COLS)
        Xva, yva = prep_xy(va, FEATURE_COLS)
        Xtr = Xtr.fillna(Xtr.median())
        Xva = Xva.fillna(Xtr.median())  # impute val with TRAIN median only (no future info)

        # MODEL 0: majority baseline
        majority_pred = np.full(len(yva), ytr.mean())

        # MODEL 1: Logistic Regression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
        lr_proba = lr.predict_proba(scaler.transform(Xva))[:, 1]

        # MODEL 2: LightGBM (chosen: already installed, fast, handles NaN/imbalance natively)
        gbm = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   class_weight="balanced", verbosity=-1, random_state=42)
        gbm.fit(Xtr, ytr)
        gbm_proba = gbm.predict_proba(Xva)[:, 1]

        fold_result = {
            "fold": i, "train_n": len(tr), "val_n": len(va), "val_late_prevalence": float(yva.mean()),
            "logistic": {"roc_auc": float(roc_auc_score(yva, lr_proba)), "pr_auc": float(average_precision_score(yva, lr_proba)),
                          "brier": float(brier_score_loss(yva, lr_proba))},
            "lightgbm": {"roc_auc": float(roc_auc_score(yva, gbm_proba)), "pr_auc": float(average_precision_score(yva, gbm_proba)),
                         "brier": float(brier_score_loss(yva, gbm_proba))},
        }
        results["folds"].append(fold_result)
        log(f"fold {i}: LR roc_auc={fold_result['logistic']['roc_auc']:.4f} "
            f"LGBM roc_auc={fold_result['lightgbm']['roc_auc']:.4f}")

    # Train final model on ALL dev data, evaluate ONCE on protected test
    dev_all = pd.concat([tr for tr, va in folds] + [folds[-1][1]])
    Xdev, ydev = prep_xy(dev_all, FEATURE_COLS)
    Xdev = Xdev.fillna(Xdev.median())
    Xprot, yprot = prep_xy(protected, FEATURE_COLS)
    Xprot = Xprot.fillna(Xdev.median())

    gbm_final = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     class_weight="balanced", verbosity=-1, random_state=42)
    gbm_final.fit(Xdev, ydev)
    prot_proba_raw = gbm_final.predict_proba(Xprot)[:, 1]

    # GATE 12: calibration on the LAST dev fold's validation set only (never protected test)
    from sklearn.isotonic import IsotonicRegression
    last_tr, last_va = folds[-1]
    Xlast_va, ylast_va = prep_xy(last_va, FEATURE_COLS)
    Xlast_va = Xlast_va.fillna(Xdev.median())
    gbm_for_calib = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                         class_weight="balanced", verbosity=-1, random_state=42)
    Xlast_tr, ylast_tr = prep_xy(last_tr, FEATURE_COLS)
    Xlast_tr = Xlast_tr.fillna(Xlast_tr.median())
    gbm_for_calib.fit(Xlast_tr, ylast_tr)
    calib_val_proba = gbm_for_calib.predict_proba(Xlast_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(calib_val_proba, ylast_va)
    raw_brier = float(brier_score_loss(ylast_va, calib_val_proba))
    calib_brier = float(brier_score_loss(ylast_va, iso.predict(calib_val_proba)))
    calibration_result = {"raw_brier_on_calib_val": raw_brier, "isotonic_calibrated_brier": calib_brier,
                           "chosen": "isotonic" if calib_brier < raw_brier else "raw"}
    log(f"calibration: raw_brier={raw_brier:.4f} isotonic_brier={calib_brier:.4f}")

    prot_proba_calibrated = iso.predict(prot_proba_raw) if calibration_result["chosen"] == "isotonic" else prot_proba_raw

    protected_result = {
        "n": len(protected), "late_prevalence": float(yprot.mean()),
        "roc_auc_raw": float(roc_auc_score(yprot, prot_proba_raw)),
        "pr_auc_raw": float(average_precision_score(yprot, prot_proba_raw)),
        "brier_raw": float(brier_score_loss(yprot, prot_proba_raw)),
        "brier_calibrated": float(brier_score_loss(yprot, prot_proba_calibrated)),
    }
    log(f"PROTECTED TEST (one-time): roc_auc={protected_result['roc_auc_raw']:.4f} pr_auc={protected_result['pr_auc_raw']:.4f}")

    # GATE 13: operational ranking vs VALUE_ONLY at 5/10/20% budgets
    value = protected["total_price"].values + protected["total_freight"].values
    op_results = {}
    for budget_pct in [5, 10, 20]:
        k = max(1, int(len(protected) * budget_pct / 100))

        def topk_metrics(scores):
            order = np.argsort(-scores)[:k]
            captured_late = int(yprot[order].sum())
            captured_value = float(value[order].sum())
            return {
                "late_captured": captured_late,
                "late_recall_at_budget": captured_late / max(1, yprot.sum()),
                "precision_at_budget": captured_late / k,
                "value_captured": captured_value,
                "value_captured_fraction": captured_value / value.sum(),
            }

        rng = np.random.RandomState(42)
        random_scores = rng.rand(len(protected))
        risk_scores = prot_proba_calibrated
        combined_scores = prot_proba_calibrated * (value / value.max())

        op_results[f"budget_{budget_pct}pct"] = {
            "k": k,
            "RANDOM": topk_metrics(random_scores),
            "VALUE_ONLY": topk_metrics(value),
            "RISK_ONLY": topk_metrics(risk_scores),
            "CALIBRATED_RISK_X_VALUE": topk_metrics(combined_scores),
        }
        vo = op_results[f"budget_{budget_pct}pct"]["VALUE_ONLY"]
        rx = op_results[f"budget_{budget_pct}pct"]["CALIBRATED_RISK_X_VALUE"]
        ro = op_results[f"budget_{budget_pct}pct"]["RISK_ONLY"]
        op_results[f"budget_{budget_pct}pct"]["ML_LATE_RECALL_DELTA_VS_VALUE_ONLY"] = ro["late_recall_at_budget"] - vo["late_recall_at_budget"]
        log(f"budget {budget_pct}%: VALUE_ONLY late_recall={vo['late_recall_at_budget']:.3f} "
            f"RISK_ONLY late_recall={ro['late_recall_at_budget']:.3f} "
            f"CALIBRATED_RISK_X_VALUE late_recall={rx['late_recall_at_budget']:.3f}")

    # Paired bootstrap CI for the 10% budget RISK_ONLY vs VALUE_ONLY late-recall delta
    k10 = max(1, int(len(protected) * 0.10))
    boot_deltas = []
    rng = np.random.RandomState(123)
    idx_all = np.arange(len(protected))
    for _ in range(1000):
        bs = rng.choice(idx_all, size=len(idx_all), replace=True)
        y_bs, val_bs, risk_bs = yprot[bs], value[bs], prot_proba_calibrated[bs]
        order_val = np.argsort(-val_bs)[:k10]
        order_risk = np.argsort(-risk_bs)[:k10]
        recall_val = y_bs[order_val].sum() / max(1, y_bs.sum())
        recall_risk = y_bs[order_risk].sum() / max(1, y_bs.sum())
        boot_deltas.append(recall_risk - recall_val)
    boot_deltas = np.array(boot_deltas)
    bootstrap_ci = {"mean_delta": float(boot_deltas.mean()), "ci_2.5": float(np.percentile(boot_deltas, 2.5)),
                     "ci_97.5": float(np.percentile(boot_deltas, 97.5)), "n_resamples": 1000}
    log(f"bootstrap CI (RISK_ONLY - VALUE_ONLY late-recall @10%): mean={bootstrap_ci['mean_delta']:.4f} "
        f"CI=[{bootstrap_ci['ci_2.5']:.4f}, {bootstrap_ci['ci_97.5']:.4f}]")

    ml_operationally_useful = bootstrap_ci["ci_2.5"] > 0
    final = {
        "protected_test_result": protected_result,
        "calibration": calibration_result,
        "operational_ranking": op_results,
        "bootstrap_ci_risk_vs_value_at_10pct": bootstrap_ci,
        "ML_OPERATIONALLY_USEFUL": bool(ml_operationally_useful),
        "note": "ML_OPERATIONALLY_USEFUL=True requires the bootstrap CI lower bound for RISK_ONLY vs VALUE_ONLY late-recall delta to exceed 0 -- a real uncertainty-supported advantage, not just a positive point estimate.",
    }
    (REPORTS_DIR / "operational_ranking_results.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    (REPORTS_DIR / "temporal_fold_model_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results, final


if __name__ == "__main__":
    df = pd.read_parquet(OUT_DIR / "canonical_dataset.parquet")
    folds, dev, protected = build_temporal_splits(df)
    run_leakage_tests(dev)
    fold_results, final = train_and_eval(folds, protected)
    log("DONE. See reports/generated/olist_v2/ for all outputs.")
