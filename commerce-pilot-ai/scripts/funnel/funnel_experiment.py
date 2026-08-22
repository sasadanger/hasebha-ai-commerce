"""Phase 3: controlled experiment -- FULL-23 baseline vs FULL-23 + funnel block.

Same rolling prequential protocol and LightGBM params as the existing forensic reproduction
(scripts/forensics/production_model_forensics.py) for direct comparability. New code only.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ART_V3 = ROOT / "artifacts/experiments/olist_v3_multistage"
ART_FUNNEL = ROOT / "artifacts/experiments/olist_funnel"
OUT = ROOT / "reports/generated/olist_funnel"
OUT.mkdir(parents=True, exist_ok=True)

y_col = "SELLER_HANDOFF_SLA_BREACH"
LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=42, verbose=-1)

FULL_23 = [
    "purchase_weekday", "purchase_hour", "purchase_month", "same_state",
    "n_items", "n_distinct_products", "n_categories", "total_price", "total_freight",
    "total_freight_over_price", "weight_g", "volume_cm3", "payment_value",
    "days_to_shipping_deadline", "n_installments",
    "seller_past_order_count", "seller_past_breach_rate_expanding",
    "seller_past_handling_median_expanding", "seller_past_handling_std_expanding",
    "seller_breach_rate_30d", "seller_breach_rate_90d", "seller_handling_mean_30d",
    "seller_recent_load_7d",
]
FUNNEL_FEATURES = ["seller_history_available_funnel", "seller_tenure_days", "time_to_close_days", "origin", "lead_type", "business_segment"]

canon = pd.read_parquet(ART_V3 / "seller_sla_canonical.parquet").sort_values("order_purchase_timestamp").reset_index(drop=True)
funnel = pd.read_parquet(ART_FUNNEL / "seller_sla_with_funnel.parquet")
df = canon.merge(funnel[["order_id"] + FUNNEL_FEATURES], on="order_id", how="left")
y = df[y_col].values

df["_month"] = df["order_purchase_timestamp"].dt.to_period("M")
month_counts = df["_month"].value_counts()
valid_months = sorted([m for m in df["_month"].unique() if month_counts[m] >= 200])
periods = [(str(p[0]), str(p[-1])) for p in np.array_split(valid_months, 7) if len(p) > 0]

def month_mask(lo, hi):
    return (df["_month"] >= pd.Period(lo)) & (df["_month"] <= pd.Period(hi))

def evaluate(name, feature_cols):
    per_period = []
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
        per_period.append({"test_period": f"{test_lo}..{test_hi}", "n_test": int(test_mask.sum()),
                            "prevalence": float(yte.mean()), "auc": float(roc_auc_score(yte, p))})
    aucs = [r["auc"] for r in per_period]
    return {"name": name, "feature_cols": feature_cols, "per_period": per_period,
            "mean_auc": float(np.mean(aucs)), "worst_auc": float(np.min(aucs)), "std_auc": float(np.std(aucs))}

baseline = evaluate("FULL_23_BASELINE_REPRO", FULL_23)
treatment = evaluate("FULL_23_PLUS_FUNNEL", FULL_23 + FUNNEL_FEATURES)

paired_deltas = []
for b, t in zip(baseline["per_period"], treatment["per_period"]):
    assert b["test_period"] == t["test_period"]
    paired_deltas.append({"test_period": b["test_period"], "baseline_auc": b["auc"], "treatment_auc": t["auc"], "delta": t["auc"] - b["auc"]})

# funnel-only univariate AUCs (pooled, matching the forensic study's own methodology)
univariate = {}
for c in FUNNEL_FEATURES:
    x = df[c].fillna(-1).values.astype(float)
    if np.nanstd(x) == 0:
        univariate[c] = None
        continue
    auc = roc_auc_score(y, x)
    univariate[c] = {"auc": float(auc), "auc_flipped": float(max(auc, 1 - auc))}

out = {
    "gate": "PHASE_3_FUNNEL_CONTROLLED_EXPERIMENT",
    "predeclared_hypothesis": "small gain, +0.01 to +0.03 mean AUC vs FULL-23 baseline; zero or negative is a legitimate, honestly-reported result",
    "baseline": baseline,
    "treatment": treatment,
    "mean_auc_delta": treatment["mean_auc"] - baseline["mean_auc"],
    "worst_auc_delta": treatment["worst_auc"] - baseline["worst_auc"],
    "paired_per_period_deltas": paired_deltas,
    "funnel_only_univariate_auc": univariate,
    "coverage_caveat": "Only 4.5% of canonical rows have a real funnel record (see FUNNEL_DATA_QUALITY_AUDIT.json) -- this bounds any possible effect size regardless of the result below.",
}
(OUT / "FUNNEL_EXPERIMENT_RESULTS.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({"baseline_mean_auc": baseline["mean_auc"], "treatment_mean_auc": treatment["mean_auc"],
                   "mean_auc_delta": out["mean_auc_delta"], "worst_auc_delta": out["worst_auc_delta"]}, indent=2))
print("univariate:", json.dumps(univariate, indent=2))
