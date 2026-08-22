"""Gate 2: production-availability stress simulation on the FROZEN 22-feature Seller-SLA model.

Re-scores the SAME historical temporal folds, but with every feature that Gate 1's re-audit
found NOT_AVAILABLE in HASEBHA forced to the EXACT production sentinel the live service would
use (never a fabricated 'best guess' value). Model itself is NOT retrained.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

ART = Path("artifacts/experiments/olist_v3_multistage")
OUT = Path("reports/generated/olist_v3_multistage")

df = pd.read_parquet(ART / "seller_sla_canonical.parquet").sort_values("order_purchase_timestamp").reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ("order_id", "order_purchase_timestamp", "seller_id", "SELLER_HANDOFF_SLA_BREACH")]
y = df["SELLER_HANDOFF_SLA_BREACH"].values

# NOT_AVAILABLE in HASEBHA per Gate 1 re-audit -- forced to production sentinel for EVERY row
# (not just cold-start orders, since these features are structurally never available):
ALWAYS_UNAVAILABLE_SENTINEL = {
    "days_to_shipping_deadline": -1.0,   # no SLA config exists (Gate 5)
    "n_installments": -1.0,               # no core Medusa field found
    "seller_past_order_count": 0,
    "seller_past_breach_rate_expanding": -1.0,
    "seller_past_handling_median_expanding": -1.0,
    "seller_past_handling_std_expanding": -1.0,
    "seller_breach_rate_30d": -1.0,
    "seller_breach_rate_90d": -1.0,
    "seller_handling_mean_30d": -1.0,
    "seller_recent_load_7d": 0.0,
}
# total_freight_over_price depends on total_freight/total_price which remain real -- unaffected.

df_sim = df.copy()
for col, val in ALWAYS_UNAVAILABLE_SENTINEL.items():
    df_sim[col] = val

df["_month"] = df["order_purchase_timestamp"].dt.to_period("M")
month_counts = df["_month"].value_counts()
valid_months = sorted([m for m in df["_month"].unique() if month_counts[m] >= 200])
periods = [(str(p[0]), str(p[-1])) for p in np.array_split(valid_months, 7) if len(p) > 0]

def month_mask(lo, hi):
    return (df["_month"] >= pd.Period(lo)) & (df["_month"] <= pd.Period(hi))

LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=42, verbose=-1)

results = []
for i in range(1, len(periods) - 1):
    train_mask = (df["_month"] <= pd.Period(periods[i - 1][1])).values
    test_lo, test_hi = periods[i]
    test_mask = month_mask(test_lo, test_hi).values
    if train_mask.sum() < 500 or test_mask.sum() < 100:
        continue
    ytr, yte = y[train_mask], y[test_mask]
    if ytr.sum() < 5 or yte.sum() < 5:
        continue
    # model trained on REAL historical features (this is the frozen model's own training regime,
    # unchanged) -- only the SIMULATED-PRODUCTION test rows get the sentinel substitution.
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(df.loc[train_mask, feature_cols], ytr)

    p_full = model.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
    p_sim = model.predict_proba(df_sim.loc[test_mask, feature_cols])[:, 1]

    def metrics(p, yt):
        order = np.argsort(-p)
        k = max(1, int(0.10 * len(p)))
        return {
            "auc": float(roc_auc_score(yt, p)), "pr_auc": float(average_precision_score(yt, p)),
            "brier": float(brier_score_loss(yt, p)), "recall_at_10": float(yt[order[:k]].sum() / max(1, yt.sum())),
        }

    m_full = metrics(p_full, yte)
    m_sim = metrics(p_sim, yte)
    results.append({
        "test_period": f"{test_lo}..{test_hi}", "n_test": int(test_mask.sum()), "prevalence": float(yte.mean()),
        "FULL_22_FEATURE": m_full, "PRODUCTION_SIM": m_sim, "auc_delta": float(m_sim["auc"] - m_full["auc"]),
    })

mean_full_auc = float(np.mean([r["FULL_22_FEATURE"]["auc"] for r in results]))
mean_sim_auc = float(np.mean([r["PRODUCTION_SIM"]["auc"] for r in results]))
worst_sim_auc = float(np.min([r["PRODUCTION_SIM"]["auc"] for r in results]))

out = {
    "gate": "GATE_2_PRODUCTION_AVAILABILITY_SIMULATION",
    "sentinel_substitution": ALWAYS_UNAVAILABLE_SENTINEL,
    "per_period": results,
    "FULL_22_FEATURE_AUC_MEAN": mean_full_auc,
    "PRODUCTION_SIM_AUC_MEAN": mean_sim_auc,
    "PRODUCTION_SIM_AUC_WORST": worst_sim_auc,
    "AUC_DELTA_MEAN": mean_sim_auc - mean_full_auc,
    "DECISION_GATE_VERDICT": None,
}
delta = mean_sim_auc - mean_full_auc
if delta > -0.03:
    out["DECISION_GATE_VERDICT"] = "MATERIAL_COLLAPSE_NOT_OBSERVED -- but see note"
else:
    out["DECISION_GATE_VERDICT"] = "MATERIAL_COLLAPSE -- frozen 22-feature model's 0.7702 status is RESEARCH_FULL_FEATURE_MODEL, NOT product performance"
out["note"] = (
    "This simulation zeroes/sentinels 10 features but LEAVES same_state, order/product/payment "
    "features intact from the real historical data -- it measures pure feature-loss impact, "
    "isolated from the separate question of whether HASEBHA's actual population/regime "
    "resembles Olist's at all (a distinct, unaddressed generalization risk)."
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "OLIST_SELLER_SLA_PRODUCTION_SIMULATION.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: out[k] for k in ["FULL_22_FEATURE_AUC_MEAN","PRODUCTION_SIM_AUC_MEAN","PRODUCTION_SIM_AUC_WORST","AUC_DELTA_MEAN","DECISION_GATE_VERDICT"]}, indent=2))
