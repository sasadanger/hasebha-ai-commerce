"""Gate O5-O6: train primary GBDT for Seller SLA breach, rolling/prequential temporal eval."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, brier_score_loss

ART = Path("artifacts/experiments/olist_v3_multistage")
OUT = Path("reports/generated/olist_v3_multistage")

df = pd.read_parquet(ART / "seller_sla_canonical.parquet")
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in ("order_id", "order_purchase_timestamp", "seller_id", "SELLER_HANDOFF_SLA_BREACH")]
y = df["SELLER_HANDOFF_SLA_BREACH"].values
X = df[feature_cols]

# 6 rolling prequential periods (historical only) + 1 latest stress block, mirroring olist_v2 convention.
df["_month"] = df["order_purchase_timestamp"].dt.to_period("M")
months = sorted(df["_month"].unique())
# drop first/last partial months with too few rows
month_counts = df["_month"].value_counts()
valid_months = [m for m in months if month_counts[m] >= 200]
n_periods = 7
period_bounds = np.array_split(valid_months, n_periods)
periods = [(str(p[0]), str(p[-1])) for p in period_bounds if len(p) > 0]

def month_mask(lo, hi):
    return (df["_month"] >= pd.Period(lo)) & (df["_month"] <= pd.Period(hi))

results = []
LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                   reg_lambda=1.0, random_state=42, verbose=-1)

for i in range(1, len(periods) - 1):  # rolling-origin: train on periods[0..i-1], test on periods[i]; last period reserved as stress block
    train_mask = df["_month"] <= pd.Period(periods[i - 1][1])
    test_lo, test_hi = periods[i]
    test_mask = month_mask(test_lo, test_hi)
    if train_mask.sum() < 500 or test_mask.sum() < 100:
        continue
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    if ytr.sum() < 5 or yte.sum() < 5:
        continue
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    pr_auc = average_precision_score(yte, p)
    pred_label = (p >= 0.5).astype(int)
    results.append({
        "period_index": i, "train_through": periods[i - 1][1], "test_period": f"{test_lo}..{test_hi}",
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()), "prevalence_test": float(yte.mean()),
        "roc_auc": float(auc), "pr_auc": float(pr_auc),
        "f1": float(f1_score(yte, pred_label, zero_division=0)),
        "precision": float(precision_score(yte, pred_label, zero_division=0)),
        "recall": float(recall_score(yte, pred_label, zero_division=0)),
        "brier": float(brier_score_loss(yte, p)),
    })

# final block = last period, used only as post-selection stress diagnostic (EXPOSED, not blind), not for any tuning
final_lo, final_hi = periods[-1]
train_mask = df["_month"] < pd.Period(final_lo)
test_mask = month_mask(final_lo, final_hi)
stress = None
if train_mask.sum() > 500 and test_mask.sum() > 50 and y[train_mask].sum() > 5 and y[test_mask].sum() > 5:
    model_final = lgb.LGBMClassifier(**LGB_PARAMS)
    model_final.fit(X[train_mask], y[train_mask])
    p = model_final.predict_proba(X[test_mask])[:, 1]
    yte = y[test_mask]
    pred_label = (p >= 0.5).astype(int)
    stress = {
        "period": f"{final_lo}..{final_hi}", "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "prevalence_test": float(yte.mean()), "roc_auc": float(roc_auc_score(yte, p)),
        "pr_auc": float(average_precision_score(yte, p)),
        "f1": float(f1_score(yte, pred_label, zero_division=0)),
        "recall_at_10pct": None,
    }
    order = np.argsort(-p)
    k = max(1, int(0.10 * len(p)))
    top_idx = order[:k]
    stress["recall_at_10pct"] = float(yte[top_idx].sum() / max(1, yte.sum()))
    stress["LABEL"] = "LATEST_TEMPORAL_STRESS_BLOCK -- EXPOSED, post-selection diagnostic only, NOT used for model/feature selection"

# baseline comparisons on the same test periods: global prevalence baseline + seller historical breach-rate-only baseline
baseline_rows = []
for r in results:
    test_lo, test_hi = r["test_period"].split("..")
    test_mask = month_mask(test_lo, test_hi)
    yte = y[test_mask]
    seller_hist = df.loc[test_mask, "seller_past_breach_rate_expanding"].values
    seller_hist_valid = seller_hist.copy()
    seller_hist_valid[seller_hist_valid < 0] = df.loc[df["_month"] <= pd.Period(r["train_through"]), "SELLER_HANDOFF_SLA_BREACH"].mean()
    try:
        auc_seller_hist = roc_auc_score(yte, seller_hist_valid)
    except ValueError:
        auc_seller_hist = None
    baseline_rows.append({"test_period": r["test_period"], "seller_history_only_auc": auc_seller_hist,
                           "global_prevalence_baseline_auc": 0.5})

mean_auc = float(np.mean([r["roc_auc"] for r in results])) if results else None
worst_auc = float(np.min([r["roc_auc"] for r in results])) if results else None
recall_at_10_periods = []
for r_idx, r in enumerate(results):
    test_lo, test_hi = r["test_period"].split("..")
    test_mask = month_mask(test_lo, test_hi)
    Xte, yte = X[test_mask], y[test_mask]
    train_mask = df["_month"] <= pd.Period(r["train_through"])
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X[train_mask], y[train_mask])
    p = model.predict_proba(Xte)[:, 1]
    order = np.argsort(-p)
    for pct, name in [(0.05, "recall_at_5"), (0.10, "recall_at_10"), (0.20, "recall_at_20")]:
        k = max(1, int(pct * len(p)))
        recall_at_10_periods.append({"period": r["test_period"], "budget": name, "value": float(yte[order[:k]].sum() / max(1, yte.sum()))})

scorecard = {
    "gate": "O3-O6_SELLER_SLA",
    "T0": "order_purchase_timestamp",
    "n_rows_total": len(df),
    "n_periods_evaluated": len(results),
    "rolling_prequential_results": results,
    "mean_temporal_auc": mean_auc,
    "worst_temporal_auc": worst_auc,
    "std_temporal_auc": float(np.std([r["roc_auc"] for r in results])) if results else None,
    "baseline_comparisons": baseline_rows,
    "stress_block_diagnostic": stress,
    "recall_at_budgets": recall_at_10_periods,
    "model": "LightGBM (mostly numeric/rolling causal features -> LightGBM per Gate O5 rule)",
    "params": LGB_PARAMS,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "SELLER_SLA_TEMPORAL_EVAL.json").write_text(json.dumps(scorecard, indent=2, default=str))
print("mean_auc", mean_auc, "worst_auc", worst_auc, "n_periods", len(results))
print(json.dumps(results, indent=2, default=str))
if stress:
    print("STRESS:", json.dumps(stress, indent=2, default=str))
