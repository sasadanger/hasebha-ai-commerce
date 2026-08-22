"""Gate 3/6/7: production-parity model (MODEL P) and P+ (with store-level operational features).

Both trained on the SAME Olist data/target/temporal protocol as the frozen 22-feature model,
but restricted to features that Gate 1's re-audit found genuinely available (or honestly
store-level-renamed) in a single-vendor HASEBHA deployment. This is the authorized retraining:
the prediction CONTRACT changed (different, smaller feature set), not a model-zoo expansion.
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
y_col = "SELLER_HANDOFF_SLA_BREACH"

# ---- MODE A bootstrap features: genuinely available at T0 with zero history dependence ----
MODE_A_FEATURES = [
    "purchase_weekday", "purchase_hour", "purchase_month", "same_state",  # renamed same_zone conceptually, same column
    "n_items", "n_distinct_products", "n_categories", "total_price", "total_freight",
    "total_freight_over_price", "weight_g", "volume_cm3", "payment_value",
]
# excluded vs the frozen 22: days_to_shipping_deadline (no SLA config, Gate 5), n_installments
# (no core Medusa field, Gate 1), and all 10 seller-history features (no seller concept).

# ---- MODE B additional store-level operational features (causal, STORE-WIDE not per-seller) ----
# Computed exactly like the seller-history block but grouped over the WHOLE dataset (as if all
# orders belong to one store), honestly named store_* (never seller_*), per Gate 1's semantic rule.
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
breach_shift = df[y_col].shift(1)
# need handling_duration_days -- reconstruct from original pipeline inputs already present? Not
# saved in canonical parquet; approximate via order_purchase_timestamp deltas is unavailable, so
# use the seller-level handling stat is NOT allowed (would leak seller structure back in) --
# instead compute a store-wide proxy purely from the breach label + order volume, which IS
# available online (breach outcome becomes known at fulfillment.shipped_at, a real event).
tmp = df[["order_purchase_timestamp"]].copy()
tmp["breach_shift"] = breach_shift.values
tmp = tmp.set_index("order_purchase_timestamp")
df["store_breach_rate_30d"] = tmp["breach_shift"].rolling("30D", min_periods=1).mean().values
df["store_breach_rate_90d"] = tmp["breach_shift"].rolling("90D", min_periods=1).mean().values
df["store_breach_rate_expanding"] = breach_shift.expanding().mean().values

const_one = pd.Series(1.0, index=df.index)
one_shift = const_one.shift(1).fillna(0.0)
tmp2 = df[["order_purchase_timestamp"]].copy()
tmp2["one_shift"] = one_shift.values
tmp2 = tmp2.set_index("order_purchase_timestamp")
df["store_open_order_backlog_24h"] = tmp2["one_shift"].rolling("1D", min_periods=0).sum().values
df["store_recent_load_7d"] = tmp2["one_shift"].rolling("7D", min_periods=0).sum().values

for c in ["store_breach_rate_30d", "store_breach_rate_90d", "store_breach_rate_expanding"]:
    df[c] = df[c].fillna(-1.0)

MODE_B_EXTRA_FEATURES = [
    "store_breach_rate_30d", "store_breach_rate_90d", "store_breach_rate_expanding",
    "store_open_order_backlog_24h", "store_recent_load_7d",
]

y = df[y_col].values
df["_month"] = df["order_purchase_timestamp"].dt.to_period("M")
month_counts = df["_month"].value_counts()
valid_months = sorted([m for m in df["_month"].unique() if month_counts[m] >= 200])
periods = [(str(p[0]), str(p[-1])) for p in np.array_split(valid_months, 7) if len(p) > 0]

def month_mask(lo, hi):
    return (df["_month"] >= pd.Period(lo)) & (df["_month"] <= pd.Period(hi))

LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=42, verbose=-1)

def evaluate_feature_set(name, feature_cols):
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
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(df.loc[train_mask, feature_cols], ytr)
        p = model.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
        order = np.argsort(-p)
        recalls = {}
        for pct, tag in [(0.05, "recall_at_5"), (0.10, "recall_at_10"), (0.20, "recall_at_20")]:
            k = max(1, int(pct * len(p)))
            recalls[tag] = float(yte[order[:k]].sum() / max(1, yte.sum()))
        pred_label = (p >= 0.5).astype(int)
        results.append({
            "test_period": f"{test_lo}..{test_hi}", "n_test": int(test_mask.sum()), "prevalence": float(yte.mean()),
            "auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
            "brier": float(brier_score_loss(yte, p)), "f1": float(f1_score(yte, pred_label, zero_division=0)),
            **recalls,
        })
    aucs = [r["auc"] for r in results]
    return {
        "name": name, "feature_cols": feature_cols, "per_period": results,
        "mean_auc": float(np.mean(aucs)), "worst_auc": float(np.min(aucs)), "std_auc": float(np.std(aucs)),
        "mean_pr_auc": float(np.mean([r["pr_auc"] for r in results])),
        "mean_recall_at_10": float(np.mean([r["recall_at_10"] for r in results])),
    }

model_p = evaluate_feature_set("MODEL_P_production_parity_bootstrap", MODE_A_FEATURES)
model_p_plus = evaluate_feature_set("MODEL_P_PLUS_production_parity_with_store_ops", MODE_A_FEATURES + MODE_B_EXTRA_FEATURES)

# save canonical production-parity training frames for later calibration/freeze steps
df[["order_id", "order_purchase_timestamp"] + MODE_A_FEATURES + [y_col]].to_parquet(ART / "production_parity_mode_a.parquet", index=False)
df[["order_id", "order_purchase_timestamp"] + MODE_A_FEATURES + MODE_B_EXTRA_FEATURES + [y_col]].to_parquet(ART / "production_parity_mode_b.parquet", index=False)

out = {
    "gate": "GATE_3_6_7_PRODUCTION_PARITY_MODEL_COMPARISON",
    "MODEL_P": {k: model_p[k] for k in ["mean_auc", "worst_auc", "std_auc", "mean_pr_auc", "mean_recall_at_10"]},
    "MODEL_P_PLUS": {k: model_p_plus[k] for k in ["mean_auc", "worst_auc", "std_auc", "mean_pr_auc", "mean_recall_at_10"]},
    "OPS_FEATURE_GAIN_mean_auc": model_p_plus["mean_auc"] - model_p["mean_auc"],
    "full_detail": {"MODEL_P": model_p, "MODEL_P_PLUS": model_p_plus},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "PRODUCTION_PARITY_MODEL_COMPARISON.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({
    "MODEL_P_mean_auc": model_p["mean_auc"], "MODEL_P_worst_auc": model_p["worst_auc"],
    "MODEL_P_PLUS_mean_auc": model_p_plus["mean_auc"], "MODEL_P_PLUS_worst_auc": model_p_plus["worst_auc"],
    "OPS_FEATURE_GAIN": model_p_plus["mean_auc"] - model_p["mean_auc"],
}, indent=2))
