"""FORENSIC INVESTIGATION: why is the production-parity model weak (AUC 0.5551)?

Read-only with respect to frozen tracks. Uses the SAME canonical dataset, target
(SELLER_HANDOFF_SLA_BREACH) and rolling prequential protocol as
scripts/olist_v3_production_parity_model.py.

Experiments:
  PHASE 7  - ablation of the frozen 23-feature research model (where did 0.77 -> 0.555 come from)
  PHASE 4  - univariate AUC of every candidate feature + permutation importance
  PHASE 5  - new HASEBHA-derivable feature groups (geography, customer history,
             product category, basket price structure)
  PHASE 6  - model-family comparison on the best legitimate feature set

All new features are point-in-time correct at T0 = order_purchase_timestamp.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/experiments/olist_v3_multistage"
RAW = ROOT / "data/raw/olist/extracted"
OUT = ROOT / "reports/generated/olist_v3_multistage/forensics"
OUT.mkdir(parents=True, exist_ok=True)

y_col = "SELLER_HANDOFF_SLA_BREACH"
LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                  min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=42, verbose=-1)

# ---------------------------------------------------------------- load + extend
df = pd.read_parquet(ART / "seller_sla_canonical.parquet").sort_values(
    "order_purchase_timestamp").reset_index(drop=True)

orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
products = pd.read_csv(RAW / "olist_products_dataset.csv", usecols=["product_id", "product_category_name"])
geo = pd.read_csv(RAW / "olist_geolocation_dataset.csv")

zip_geo = geo.groupby("geolocation_zip_code_prefix").agg(
    lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean")).reset_index()

cust = customers.merge(zip_geo, left_on="customer_zip_code_prefix",
                       right_on="geolocation_zip_code_prefix", how="left")
sell = sellers.merge(zip_geo, left_on="seller_zip_code_prefix",
                     right_on="geolocation_zip_code_prefix", how="left")
sell = sell.rename(columns={"lat": "lat_s", "lng": "lng_s"})
df = df.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
df = df.merge(cust[["customer_id", "customer_unique_id", "customer_state",
                    "customer_zip_code_prefix", "lat", "lng"]], on="customer_id", how="left")
df = df.merge(sell[["seller_id", "lat_s", "lng_s"]], on="seller_id", how="left")

# --- geography: haversine distance customer->seller (zip-centroid proxy) -----
def haversine(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

df["geo_dist_km"] = haversine(df["lat"], df["lng"], df["lat_s"], df["lng_s"])
df["geo_lat_diff"] = (df["lat_s"] - df["lat"]).abs()
df["geo_lng_diff"] = (df["lng_s"] - df["lng"]).abs()
GEO_FEATURES = ["geo_dist_km", "geo_lat_diff", "geo_lng_diff"]

# --- customer history: strictly point-in-time at T0 --------------------------
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
df["cust_prior_orders"] = df.groupby("customer_unique_id").cumcount()
df["cust_is_repeat"] = (df["cust_prior_orders"] > 0).astype(int)
first_seen = df.groupby("customer_unique_id")["order_purchase_timestamp"].transform("first")
df["cust_tenure_days"] = (df["order_purchase_timestamp"] - first_seen).dt.total_seconds() / 86400.0
# prior late-delivery rate: only counts prior orders whose OUTCOME was already
# observable (delivered) strictly before the current order's purchase timestamp
o = orders[["order_id", "customer_id", "order_purchase_timestamp",
            "order_delivered_customer_date", "order_estimated_delivery_date"]].copy()
o["late"] = (o["order_delivered_customer_date"] > o["order_estimated_delivery_date"]).astype(float)
o = o.dropna(subset=["order_delivered_customer_date"])
cust_orders = customers[["customer_id", "customer_unique_id"]].merge(
    o[["customer_id", "order_purchase_timestamp", "order_delivered_customer_date", "late"]],
    on="customer_id")
# per-customer sorted outcome arrays; "observable" = delivered strictly before T0
cust_orders = cust_orders.sort_values("order_delivered_customer_date").reset_index(drop=True)
by_cust = {}
for cuid, grp in cust_orders.groupby("customer_unique_id"):
    by_cust[cuid] = (grp["order_delivered_customer_date"].values.astype("datetime64[ns]"),
                     grp["late"].values)
rates, known = [], []
for t0, cuid in zip(df["order_purchase_timestamp"].values, df["customer_unique_id"].values):
    if cuid in by_cust:
        dl, late = by_cust[cuid]
        k = np.searchsorted(dl, t0, side="left")
    else:
        k, late = 0, np.array([])
    known.append(k)
    rates.append(float(late[:k].mean()) if k > 0 else -1.0)
df["cust_prior_late_rate"] = rates
df["cust_prior_observed_orders"] = np.array(known, dtype=float)
CUST_FEATURES = ["cust_prior_orders", "cust_is_repeat", "cust_tenure_days",
                 "cust_prior_late_rate", "cust_prior_observed_orders"]

# --- basket price structure + product category ------------------------------
agg = items.groupby("order_id").agg(
    item_price_mean=("price", "mean"), item_price_std=("price", "std"),
    item_price_max=("price", "max"), freight_per_item=("freight_value", "mean"),
    first_product_id=("product_id", "first")).reset_index()
df = df.merge(agg, on="order_id", how="left")
df = df.merge(products, left_on="first_product_id", right_on="product_id", how="left")
df["product_category"] = df["product_category_name"].fillna("__missing__")
df["product_category_code"] = df["product_category"].astype("category").cat.codes
PRICE_FEATURES = ["item_price_mean", "item_price_max", "freight_per_item"]
CAT_FEATURES = ["product_category_code"]

P_FEATURES = ["purchase_weekday", "purchase_hour", "purchase_month", "same_state",
              "n_items", "n_distinct_products", "n_categories", "total_price",
              "total_freight", "total_freight_over_price", "weight_g", "volume_cm3",
              "payment_value"]
SELLER_HIST = ["seller_past_order_count", "seller_past_breach_rate_expanding",
               "seller_past_handling_median_expanding", "seller_past_handling_std_expanding",
               "seller_breach_rate_30d", "seller_breach_rate_90d",
               "seller_handling_mean_30d", "seller_recent_load_7d"]
DEADLINE = ["days_to_shipping_deadline"]
INSTALL = ["n_installments"]
NUM_FILL = P_FEATURES + GEO_FEATURES + CUST_FEATURES + PRICE_FEATURES + DEADLINE + INSTALL
for c in NUM_FILL:
    df[c] = df[c].fillna(-1.0)

# ---------------------------------------------------------------- protocol
y = df[y_col].values
df["_month"] = df["order_purchase_timestamp"].dt.to_period("M")
month_counts = df["_month"].value_counts()
valid_months = sorted([m for m in df["_month"].unique() if month_counts[m] >= 200])
periods = [(str(p[0]), str(p[-1])) for p in np.array_split(valid_months, 7) if len(p) > 0]

def month_mask(lo, hi):
    return (df["_month"] >= pd.Period(lo)) & (df["_month"] <= pd.Period(hi))

def evaluate(name, feature_cols, model="lgbm"):
    per_period, pooled_p, pooled_y = [], [], []
    for i in range(1, len(periods) - 1):
        train_mask = (df["_month"] <= pd.Period(periods[i - 1][1])).values
        test_lo, test_hi = periods[i]
        test_mask = month_mask(test_lo, test_hi).values
        if train_mask.sum() < 500 or test_mask.sum() < 100:
            continue
        ytr, yte = y[train_mask], y[test_mask]
        Xtr, Xte = df.loc[train_mask, feature_cols], df.loc[test_mask, feature_cols]
        if model == "lgbm":
            m = lgb.LGBMClassifier(**LGB_PARAMS)
        elif model == "logreg":
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        elif model == "rf":
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                       random_state=42, n_jobs=-1)
        elif model == "histgb":
            m = HistGradientBoostingClassifier(random_state=42)
        elif model == "xgb":
            m = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              random_state=42, eval_metric="logloss", n_jobs=-1)
        elif model == "catboost":
            m = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                   random_seed=42, verbose=False)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pooled_p.append(p); pooled_y.append(yte)
        per_period.append({
            "test_period": f"{test_lo}..{test_hi}", "n_test": int(test_mask.sum()),
            "prevalence": float(yte.mean()),
            "auc": float(roc_auc_score(yte, p)),
            "pr_auc": float(average_precision_score(yte, p))})
    aucs = [r["auc"] for r in per_period]
    pooled_y = np.concatenate(pooled_y); pooled_p = np.concatenate(pooled_p)
    return {"name": name, "model": model, "feature_cols": list(feature_cols),
            "per_period": per_period,
            "mean_auc": float(np.mean(aucs)), "worst_auc": float(np.min(aucs)),
            "std_auc": float(np.std(aucs)),
            "pooled_auc": float(roc_auc_score(pooled_y, pooled_p)),
            "pooled_pr_auc": float(average_precision_score(pooled_y, pooled_p))}

results = {}

# ---------------------------------------------------------------- PHASE 7 ablation
results["AB_FULL_23"] = evaluate("AB_FULL_23_frozen_research", P_FEATURES + DEADLINE + INSTALL + SELLER_HIST)
results["AB_NO_SELLER_HIST"] = evaluate("AB_NO_SELLER_HIST", P_FEATURES + DEADLINE + INSTALL)
results["AB_NO_SELLER_HIST_NO_DEADLINE"] = evaluate("AB_NO_SELLER_HIST_NO_DEADLINE", P_FEATURES + INSTALL)
results["AB_NO_SELLER_HIST_NO_INSTALL"] = evaluate("AB_NO_SELLER_HIST_NO_INSTALL", P_FEATURES + DEADLINE)
results["AB_SELLER_HIST_ONLY"] = evaluate("AB_SELLER_HIST_ONLY", SELLER_HIST)
results["AB_DEADLINE_ONLY"] = evaluate("AB_DEADLINE_ONLY", DEADLINE)
results["P_BASELINE_REPRO"] = evaluate("P_BASELINE_REPRO", P_FEATURES)

# ---------------------------------------------------------------- PHASE 4 univariate
univariate = {}
for c in P_FEATURES + DEADLINE + SELLER_HIST + GEO_FEATURES + CUST_FEATURES + PRICE_FEATURES + INSTALL:
    x = df[c].values
    if np.nanstd(x) == 0:
        univariate[c] = None
        continue
    auc = roc_auc_score(y, x)
    univariate[c] = {"auc": float(auc), "auc_flipped": float(max(auc, 1 - auc))}
results["_univariate_auc"] = univariate

# ---------------------------------------------------------------- PHASE 5 new groups
results["P_GEO"] = evaluate("P_GEO", P_FEATURES + GEO_FEATURES)
results["P_CUST"] = evaluate("P_CUST", P_FEATURES + CUST_FEATURES)
results["P_CAT"] = evaluate("P_CAT", P_FEATURES + CAT_FEATURES)
results["P_PRICE"] = evaluate("P_PRICE", P_FEATURES + PRICE_FEATURES)
results["P_GEO_CUST"] = evaluate("P_GEO_CUST", P_FEATURES + GEO_FEATURES + CUST_FEATURES)
results["P_ALL"] = evaluate("P_ALL", P_FEATURES + GEO_FEATURES + CUST_FEATURES + PRICE_FEATURES + CAT_FEATURES)

# paired per-period deltas vs P_BASELINE
def paired_delta(a, b):
    ia = {r["test_period"]: r["auc"] for r in results[a]["per_period"]}
    ib = {r["test_period"]: r["auc"] for r in results[b]["per_period"]}
    common = sorted(set(ia) & set(ib))
    return {k: round(ia[k] - ib[k], 4) for k in common}
results["_paired_deltas_vs_P"] = {k: paired_delta(k, "P_BASELINE_REPRO")
                                  for k in ["P_GEO", "P_CUST", "P_CAT", "P_PRICE",
                                            "P_GEO_CUST", "P_ALL"]}

# ---------------------------------------------------------------- PHASE 6 model family
family = {}
for fam in ["logreg", "histgb", "xgb", "catboost", "rf"]:
    family[fam] = evaluate(f"P_ALL_{fam}", P_FEATURES + GEO_FEATURES + CUST_FEATURES + PRICE_FEATURES, model=fam)
results["_model_family_P_ALLNUM"] = family

# ---------------------------------------------------------------- permutation importance
from sklearn.inspection import permutation_importance
last_i = len(periods) - 2
train_mask = (df["_month"] <= pd.Period(periods[last_i - 1][1])).values
test_lo, test_hi = periods[last_i]
test_mask = month_mask(test_lo, test_hi).values
imp_cols = P_FEATURES + GEO_FEATURES + CUST_FEATURES + PRICE_FEATURES
m = lgb.LGBMClassifier(**LGB_PARAMS).fit(df.loc[train_mask, imp_cols], y[train_mask])
pi = permutation_importance(m, df.loc[test_mask, imp_cols], y[test_mask],
                            scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
results["_permutation_importance_final_period"] = {
    c: float(v) for c, v in zip(imp_cols, pi.importances_mean)}

summary = {k: (v if not isinstance(v, dict) or "mean_auc" not in v else
               {kk: v[kk] for kk in ["mean_auc", "worst_auc", "std_auc", "pooled_auc", "pooled_pr_auc"]})
           for k, v in results.items()}
(OUT / "FORENSIC_EXPERIMENT_RESULTS.json").write_text(json.dumps(results, indent=2, default=str))
(OUT / "FORENSIC_EXPERIMENT_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
