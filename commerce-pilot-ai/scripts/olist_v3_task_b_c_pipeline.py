"""Gate O7 (Task B: stacked customer-late T0) and Gate O8 (Task C: dynamic customer-late T1).

Reuses the seller_sla_canonical cohort (single-seller, clean shipping_limit_date) and the
rolling-origin seller-SLA model already trained in olist_v3_seller_sla_train_eval.py, applied
strictly out-of-time (a period's rows only ever get predictions from a model trained on
strictly earlier periods) to produce PREDICTED_SELLER_SLA_RISK as a legitimate T0 feature.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ART = Path("artifacts/experiments/olist_v3_multistage")
OUT = Path("reports/generated/olist_v3_multistage")
RAW = Path("data/raw/olist/extracted")

df = pd.read_parquet(ART / "seller_sla_canonical.parquet")
orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date",
    "order_delivered_carrier_date"])
df = df.merge(orders[["order_id", "order_delivered_customer_date", "order_estimated_delivery_date",
                       "order_delivered_carrier_date"]], on="order_id", how="left")
df = df[df["order_delivered_customer_date"].notna() & df["order_estimated_delivery_date"].notna()].copy()
df["CUSTOMER_LATE"] = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(int)
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
df["_month"] = df["order_purchase_timestamp"].dt.to_period("M")

seller_sla_feature_cols = [c for c in df.columns if c not in (
    "order_id", "order_purchase_timestamp", "seller_id", "SELLER_HANDOFF_SLA_BREACH",
    "order_delivered_customer_date", "order_estimated_delivery_date", "order_delivered_carrier_date",
    "CUSTOMER_LATE", "_month")]

month_counts = df["_month"].value_counts()
valid_months = sorted([m for m in df["_month"].unique() if month_counts[m] >= 200])
n_periods = 7
period_bounds = np.array_split(valid_months, n_periods)
periods = [(str(p[0]), str(p[-1])) for p in period_bounds if len(p) > 0]

def month_mask(lo, hi):
    return (df["_month"] >= pd.Period(lo)) & (df["_month"] <= pd.Period(hi))

LGB_SLA_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                       min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                       random_state=42, verbose=-1)

# ---- produce out-of-time PREDICTED_SELLER_SLA_RISK for every row (period i gets predictions from model trained on periods < i only) ----
pred_seller_risk = np.full(len(df), np.nan)
for i in range(1, len(periods)):
    train_mask = df["_month"] <= pd.Period(periods[i - 1][1])
    test_lo, test_hi = periods[i]
    test_mask = month_mask(test_lo, test_hi).values
    if train_mask.sum() < 500 or test_mask.sum() == 0:
        continue
    m = lgb.LGBMClassifier(**LGB_SLA_PARAMS)
    m.fit(df.loc[train_mask, seller_sla_feature_cols], df.loc[train_mask, "SELLER_HANDOFF_SLA_BREACH"])
    pred_seller_risk[test_mask] = m.predict_proba(df.loc[test_mask, seller_sla_feature_cols])[:, 1]
df["PREDICTED_SELLER_SLA_RISK"] = pred_seller_risk

# rows in period[0] (no prior training data) get no OOF seller-risk prediction -- excluded from Task B comparison
task_b_eligible = df["PREDICTED_SELLER_SLA_RISK"].notna()

old_feature_cols = [c for c in seller_sla_feature_cols if c not in ("seller_recent_load_7d",)]  # T0 order/seller-history features, no realized carrier info either way
new_feature_cols = old_feature_cols + ["PREDICTED_SELLER_SLA_RISK"]

LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=42, verbose=-1)

b_results = []
for i in range(2, len(periods) - 1):  # need >=1 prior period for OOF seller-risk to exist in training data too
    train_mask = (df["_month"] <= pd.Period(periods[i - 1][1])) & task_b_eligible
    test_lo, test_hi = periods[i]
    test_mask = month_mask(test_lo, test_hi).values & task_b_eligible.values
    if train_mask.sum() < 500 or test_mask.sum() < 100:
        continue
    ytr, yte = df.loc[train_mask, "CUSTOMER_LATE"], df.loc[test_mask, "CUSTOMER_LATE"]
    if ytr.sum() < 5 or yte.sum() < 5:
        continue
    m_old = lgb.LGBMClassifier(**LGB_PARAMS); m_old.fit(df.loc[train_mask, old_feature_cols], ytr)
    m_new = lgb.LGBMClassifier(**LGB_PARAMS); m_new.fit(df.loc[train_mask, new_feature_cols], ytr)
    p_old = m_old.predict_proba(df.loc[test_mask, old_feature_cols])[:, 1]
    p_new = m_new.predict_proba(df.loc[test_mask, new_feature_cols])[:, 1]
    auc_old, auc_new = roc_auc_score(yte, p_old), roc_auc_score(yte, p_new)
    b_results.append({
        "test_period": f"{test_lo}..{test_hi}", "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "prevalence_test": float(yte.mean()),
        "OLD_CUSTOMER_T0_AUC": float(auc_old), "NEW_STACKED_CUSTOMER_T0_AUC": float(auc_new),
        "delta": float(auc_new - auc_old),
        "OLD_PR_AUC": float(average_precision_score(yte, p_old)), "NEW_PR_AUC": float(average_precision_score(yte, p_new)),
    })

# ---- Task C: T1 dynamic model, only orders that reached carrier handoff ----
t1_df = df[df["order_delivered_carrier_date"].notna()].copy()
t1_df["handling_duration_days"] = (t1_df["order_delivered_carrier_date"] - t1_df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
t1_df["remaining_promise_slack_days"] = (t1_df["order_estimated_delivery_date"] - t1_df["order_delivered_carrier_date"]).dt.total_seconds() / 86400.0
t1_df["handoff_weekday"] = t1_df["order_delivered_carrier_date"].dt.weekday
t1_df["handoff_hour"] = t1_df["order_delivered_carrier_date"].dt.hour

t1_feature_cols = [
    "handling_duration_days", "remaining_promise_slack_days", "handoff_weekday", "handoff_hour",
    "same_state", "total_price", "total_freight", "weight_g", "volume_cm3", "n_items",
    "seller_past_breach_rate_expanding", "seller_breach_rate_30d", "seller_handling_mean_30d",
]
t1_df = t1_df.sort_values("order_purchase_timestamp").reset_index(drop=True)
t1_df["_month"] = t1_df["order_purchase_timestamp"].dt.to_period("M")
t1_month_counts = t1_df["_month"].value_counts()
t1_valid_months = sorted([m for m in t1_df["_month"].unique() if t1_month_counts[m] >= 200])
t1_period_bounds = np.array_split(t1_valid_months, n_periods)
t1_periods = [(str(p[0]), str(p[-1])) for p in t1_period_bounds if len(p) > 0]

def t1_month_mask(lo, hi):
    return (t1_df["_month"] >= pd.Period(lo)) & (t1_df["_month"] <= pd.Period(hi))

c_results = []
for i in range(1, len(t1_periods) - 1):
    train_mask = t1_df["_month"] <= pd.Period(t1_periods[i - 1][1])
    test_lo, test_hi = t1_periods[i]
    test_mask = t1_month_mask(test_lo, test_hi)
    if train_mask.sum() < 500 or test_mask.sum() < 100:
        continue
    ytr, yte = t1_df.loc[train_mask, "CUSTOMER_LATE"], t1_df.loc[test_mask, "CUSTOMER_LATE"]
    if ytr.sum() < 5 or yte.sum() < 5:
        continue
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(t1_df.loc[train_mask, t1_feature_cols], ytr)
    p = m.predict_proba(t1_df.loc[test_mask, t1_feature_cols])[:, 1]
    order = np.argsort(-p)
    k = max(1, int(0.10 * len(p)))
    c_results.append({
        "test_period": f"{test_lo}..{test_hi}", "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "prevalence_test": float(yte.mean()),
        "roc_auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
        "f1": float(f1_score(yte, (p >= 0.5).astype(int), zero_division=0)),
        "recall_at_10": float(yte.values[order[:k]].sum() / max(1, yte.sum())),
    })

out = {
    "gate": "O7_O8_TASK_B_C",
    "TASK_B_stacked_customer_T0": {
        "note": "OLD and NEW trained on IDENTICAL rows/periods (olist_v3 cohort), OLD = order/seller-history features only, NEW = OLD + causal out-of-time PREDICTED_SELLER_SLA_RISK. This is a controlled ablation on this session's cohort, not a re-run of the separate olist_v2-track model (different cohort/features) -- olist_v2's recency_weighted result (mean AUC 0.7126) is cited for context only, not as the literal OLD model object.",
        "results": b_results,
        "mean_OLD_AUC": float(np.mean([r["OLD_CUSTOMER_T0_AUC"] for r in b_results])) if b_results else None,
        "mean_NEW_AUC": float(np.mean([r["NEW_STACKED_CUSTOMER_T0_AUC"] for r in b_results])) if b_results else None,
        "mean_delta": float(np.mean([r["delta"] for r in b_results])) if b_results else None,
    },
    "TASK_C_dynamic_customer_T1": {
        "note": "T1 = carrier handoff has occurred; features = handling duration, remaining promise slack, handoff time, lane/geo, seller history. No realized customer delivery info used.",
        "results": c_results,
        "mean_auc": float(np.mean([r["roc_auc"] for r in c_results])) if c_results else None,
        "worst_auc": float(np.min([r["roc_auc"] for r in c_results])) if c_results else None,
        "mean_pr_auc": float(np.mean([r["pr_auc"] for r in c_results])) if c_results else None,
        "mean_recall_at_10": float(np.mean([r["recall_at_10"] for r in c_results])) if c_results else None,
    },
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "TASK_B_C_RESULTS.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
