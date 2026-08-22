"""Part A: Seller-SLA calibration (Platt vs Isotonic vs raw), using temporal OOF dev predictions only."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ART = Path("artifacts/experiments/olist_v3_multistage")
OUT = Path("reports/generated/olist_v3_multistage")

df = pd.read_parquet(ART / "seller_sla_canonical.parquet").sort_values("order_purchase_timestamp").reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ("order_id", "order_purchase_timestamp", "seller_id", "SELLER_HANDOFF_SLA_BREACH")]
y = df["SELLER_HANDOFF_SLA_BREACH"].values
X = df[feature_cols]
df["_month"] = df["order_purchase_timestamp"].dt.to_period("M")
month_counts = df["_month"].value_counts()
valid_months = sorted([m for m in df["_month"].unique() if month_counts[m] >= 200])
periods = [(str(p[0]), str(p[-1])) for p in np.array_split(valid_months, 7) if len(p) > 0]

def month_mask(lo, hi):
    return (df["_month"] >= pd.Period(lo)) & (df["_month"] <= pd.Period(hi))

LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=42, verbose=-1)

# collect temporal OOF dev predictions (historical periods 1..4, excluding period 0 [no history] and period 5 [exposed stress block])
oof_p, oof_y = [], []
per_period_raw = []
for i in range(1, len(periods) - 1):
    train_mask = df["_month"] <= pd.Period(periods[i - 1][1])
    test_lo, test_hi = periods[i]
    test_mask = month_mask(test_lo, test_hi)
    if train_mask.sum() < 500 or test_mask.sum() < 100:
        continue
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X[train_mask], y[train_mask])
    p = m.predict_proba(X[test_mask])[:, 1]
    yte = y[test_mask]
    oof_p.append(p); oof_y.append(yte)
    per_period_raw.append({"period": f"{test_lo}..{test_hi}", "auc": float(roc_auc_score(yte, p)), "brier": float(brier_score_loss(yte, p))})

oof_p = np.concatenate(oof_p); oof_y = np.concatenate(oof_y)

# split OOF pool chronologically-agnostic 50/50 for calibrator-fit vs calibrator-eval (both still historical dev periods, never the stress block)
rng = np.random.default_rng(42)
idx = np.arange(len(oof_p)); rng.shuffle(idx)
half = len(idx) // 2
fit_idx, eval_idx = idx[:half], idx[half:]

raw_eval = oof_p[eval_idx]
y_eval = oof_y[eval_idx]

platt = LogisticRegression()
platt.fit(oof_p[fit_idx].reshape(-1, 1), oof_y[fit_idx])
platt_eval = platt.predict_proba(raw_eval.reshape(-1, 1))[:, 1]

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_p[fit_idx], oof_y[fit_idx])
iso_eval = iso.predict(raw_eval)

def metrics(p, y):
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p_clip)),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ece": float(np.mean([
            abs(np.mean(y[(p >= b) & (p < b + 0.1)]) - np.mean(p[(p >= b) & (p < b + 0.1)]))
            * (((p >= b) & (p < b + 0.1)).sum() / len(p))
            for b in np.arange(0, 1, 0.1) if ((p >= b) & (p < b + 0.1)).sum() > 0
        ])),
    }

m_raw = metrics(raw_eval, y_eval)
m_platt = metrics(platt_eval, y_eval)
m_iso = metrics(iso_eval, y_eval)

candidates = {"RAW": m_raw, "PLATT": m_platt, "ISOTONIC": m_iso}
# selection rule: lowest Brier among candidates whose AUC does not drop >0.005 vs raw (ranking must not be materially damaged)
selected = "RAW"
best_brier = m_raw["brier"]
for name, m in candidates.items():
    if name == "RAW":
        continue
    if m["auc"] >= m_raw["auc"] - 0.005 and m["brier"] < best_brier:
        selected = name
        best_brier = m["brier"]

report = {
    "gate": "A2_A3_SELLER_SLA_CALIBRATION",
    "method_pool": ["RAW", "PLATT", "ISOTONIC"],
    "candidates": candidates,
    "SELECTED_METHOD": selected,
    "selection_rule": "lowest Brier among methods with AUC >= raw_AUC - 0.005 (ranking must not be materially damaged), evaluated on a held-out half of temporal OOF dev predictions never including the exposed stress block",
    "RAW_BRIER": m_raw["brier"], "CALIBRATED_BRIER": candidates[selected]["brier"],
    "RAW_ECE": m_raw["ece"], "CALIBRATED_ECE": candidates[selected]["ece"],
    "RAW_AUC": m_raw["auc"], "CALIBRATED_AUC": candidates[selected]["auc"],
    "per_period_raw_temporal": per_period_raw,
    "n_oof_dev_predictions_used": len(oof_p),
    "stress_block_used_for_selection": False,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "SELLER_SLA_CALIBRATION_REPORT.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ["SELECTED_METHOD","RAW_BRIER","CALIBRATED_BRIER","RAW_ECE","CALIBRATED_ECE","RAW_AUC","CALIBRATED_AUC"]}, indent=2))

# ---- fit FINAL calibrator on ALL historical OOF dev predictions (not just eval half), freeze it ----
if selected == "PLATT":
    final_cal = LogisticRegression().fit(oof_p.reshape(-1, 1), oof_y)
    import pickle
    with open(ART / "seller_sla_calibrator.pkl", "wb") as f:
        pickle.dump({"method": "PLATT", "model": final_cal}, f)
elif selected == "ISOTONIC":
    final_cal = IsotonicRegression(out_of_bounds="clip").fit(oof_p, oof_y)
    import pickle
    with open(ART / "seller_sla_calibrator.pkl", "wb") as f:
        pickle.dump({"method": "ISOTONIC", "model": final_cal}, f)
else:
    import pickle
    with open(ART / "seller_sla_calibrator.pkl", "wb") as f:
        pickle.dump({"method": "RAW", "model": None}, f)
print("calibrator saved, method=", selected)
