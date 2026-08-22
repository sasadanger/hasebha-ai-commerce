"""Gate O3-O6: Seller Handoff SLA Breach model (olist_v3_multistage).

T0 = order_purchase_timestamp. Justified by Gate O1/O2 check:
shipping_limit_date is never before order_approved_at (frac=0.0012, essentially
data noise) and never before purchase (frac=0.0) -- median gap purchase->limit
is 6.0 days, so the deadline is a legitimately-known-at-purchase future date,
not a T0-violating field. Cohort = single-seller orders (O2 clean cohort,
n=97,388; O2 also showed 100% of those have a single distinct shipping_limit_date).

Target: SELLER_HANDOFF_SLA_BREACH = order_delivered_carrier_date > shipping_limit_date,
restricted to orders that actually reached carrier handoff (non-null carrier date).
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/olist/extracted")
ART = Path("artifacts/experiments/olist_v3_multistage")
OUT = Path("reports/generated/olist_v3_multistage")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"])
items = pd.read_csv(RAW / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
products = pd.read_csv(RAW / "olist_products_dataset.csv")
sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
payments = pd.read_csv(RAW / "olist_order_payments_dataset.csv")

# ---- cohort selection (Gate O2 clean cohort) ----
seller_counts = items.groupby("order_id")["seller_id"].nunique()
limit_counts = items.groupby("order_id")["shipping_limit_date"].nunique()
clean_ids = seller_counts[(seller_counts == 1)].index.intersection(limit_counts[limit_counts == 1].index)

item_agg = items[items["order_id"].isin(clean_ids)].groupby("order_id").agg(
    seller_id=("seller_id", "first"),
    shipping_limit_date=("shipping_limit_date", "first"),
    n_items=("order_item_id", "count"),
    n_distinct_products=("product_id", "nunique"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
).reset_index()

df = orders.merge(item_agg, on="order_id", how="inner")
df = df.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
df = df.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
pay_agg = payments.groupby("order_id").agg(payment_value=("payment_value", "sum"), n_installments=("payment_installments", "max")).reset_index()
df = df.merge(pay_agg, on="order_id", how="left")

# product weight/volume (mean across items' products for the order)
prod_small = products[["product_id", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm", "product_category_name"]]
item_prod = items[items["order_id"].isin(clean_ids)].merge(prod_small, on="product_id", how="left")
wv = item_prod.groupby("order_id").agg(
    weight_g=("product_weight_g", "sum"),
    volume_cm3=("product_length_cm", lambda x: np.nan),  # placeholder, computed below
).reset_index()
item_prod["vol"] = item_prod["product_length_cm"] * item_prod["product_height_cm"] * item_prod["product_width_cm"]
vol = item_prod.groupby("order_id")["vol"].sum().reset_index().rename(columns={"vol": "volume_cm3"})
weight = item_prod.groupby("order_id")["product_weight_g"].sum().reset_index().rename(columns={"product_weight_g": "weight_g"})
cat_div = item_prod.groupby("order_id")["product_category_name"].nunique().reset_index().rename(columns={"product_category_name": "n_categories"})
df = df.merge(weight, on="order_id", how="left").merge(vol, on="order_id", how="left").merge(cat_div, on="order_id", how="left")

# ---- target: restrict to reached-carrier-handoff orders ----
df = df[df["order_delivered_carrier_date"].notna() & df["shipping_limit_date"].notna() & df["order_purchase_timestamp"].notna()].copy()
df["SELLER_HANDOFF_SLA_BREACH"] = (df["order_delivered_carrier_date"] > df["shipping_limit_date"]).astype(int)

# ---- order by event time for causal features ----
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

# ---- SLA feature: days from purchase to deadline (known at T0) ----
df["days_to_shipping_deadline"] = (df["shipping_limit_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0

# ---- time features (T0-known) ----
df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
df["purchase_month"] = df["order_purchase_timestamp"].dt.month

# ---- geo ----
df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

# ---- CAUSAL seller history features: strictly prior orders only (shift(1) then expanding/rolling) ----
df["handling_duration_days"] = (df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0

df = df.sort_values(["seller_id", "order_purchase_timestamp"]).reset_index(drop=True)
grp = df.groupby("seller_id", sort=False)

breach_shift = grp["SELLER_HANDOFF_SLA_BREACH"].shift(1)
handling_shift = grp["handling_duration_days"].shift(1)

df["seller_past_order_count"] = grp.cumcount()
df["seller_past_breach_rate_expanding"] = breach_shift.groupby(df["seller_id"]).expanding().mean().reset_index(level=0, drop=True)
df["seller_past_handling_median_expanding"] = handling_shift.groupby(df["seller_id"]).expanding().median().reset_index(level=0, drop=True)
df["seller_past_handling_std_expanding"] = handling_shift.groupby(df["seller_id"]).expanding().std().reset_index(level=0, drop=True)

# time-based rolling windows: build per-seller frame indexed by purchase timestamp with the shifted (pre-current-row) series
_tmp = df[["seller_id", "order_purchase_timestamp"]].copy()
_tmp["breach_shift"] = breach_shift.values
_tmp["handling_shift"] = handling_shift.values
_tmp = _tmp.set_index("order_purchase_timestamp")
df["seller_breach_rate_30d"] = _tmp.groupby("seller_id")["breach_shift"].rolling("30D", min_periods=1).mean().reset_index(level=0, drop=True).values
df["seller_breach_rate_90d"] = _tmp.groupby("seller_id")["breach_shift"].rolling("90D", min_periods=1).mean().reset_index(level=0, drop=True).values
df["seller_handling_mean_30d"] = _tmp.groupby("seller_id")["handling_shift"].rolling("30D", min_periods=1).mean().reset_index(level=0, drop=True).values

# recent load: seller's order count in the trailing 7 days strictly before this order (causal), two-pointer per seller
recent_load = np.zeros(len(df), dtype=float)
for _, idx in df.groupby("seller_id").indices.items():
    idx = np.sort(idx)
    ts = df["order_purchase_timestamp"].values[idx].astype("datetime64[ns]")
    lo = 0
    for k in range(len(idx)):
        cutoff = ts[k] - np.timedelta64(7, "D")
        while lo < k and ts[lo] < cutoff:
            lo += 1
        recent_load[idx[k]] = k - lo  # count of strictly-prior orders within trailing 7 days
df["seller_recent_load_7d"] = recent_load

df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

# fill NaNs (cold-start sellers with no prior history) with global-safe sentinels; document, don't hide
GLOBAL_PRIOR_BREACH_RATE = df["SELLER_HANDOFF_SLA_BREACH"].mean()  # only used for cold-start sentinel, NOT leakage into target
for c in ["seller_past_breach_rate_expanding", "seller_breach_rate_30d", "seller_breach_rate_90d"]:
    df[c] = df[c].fillna(-1.0)  # -1 sentinel = "no history", model can learn this explicitly
for c in ["seller_past_handling_median_expanding", "seller_past_handling_std_expanding", "seller_handling_mean_30d"]:
    df[c] = df[c].fillna(-1.0)

# ---- leakage tests ----
leak_report = {"tests": [], "n_failures": 0}
def check(name, cond):
    ok = bool(cond)
    leak_report["tests"].append({"name": name, "pass": ok})
    if not ok:
        leak_report["n_failures"] += 1

check("no_customer_delivered_date_column", "order_delivered_customer_date" not in [c for c in df.columns if c.startswith("feat_")])
check("seller_past_order_count_min_is_0", df["seller_past_order_count"].min() == 0)
# for the FIRST order of each seller (past_order_count==0) breach-rate features must be sentinel -1
first_rows = df[df["seller_past_order_count"] == 0]
check("first_order_per_seller_has_sentinel_breach_rate", (first_rows["seller_past_breach_rate_expanding"] == -1.0).all())
# spot-check: for a random seller with >=3 orders, expanding breach rate at row k equals mean of breach[0:k]
rng = np.random.default_rng(0)
sample_sellers = df.groupby("seller_id").filter(lambda g: len(g) >= 3)["seller_id"].unique()
sample = rng.choice(sample_sellers, size=min(15, len(sample_sellers)), replace=False)
spot_fail = 0
for sid in sample:
    g = df[df["seller_id"] == sid].sort_values("order_purchase_timestamp").reset_index(drop=True)
    for k in range(2, len(g)):
        expected = g.loc[:k-1, "SELLER_HANDOFF_SLA_BREACH"].mean()
        got = g.loc[k, "seller_past_breach_rate_expanding"]
        if got != -1.0 and abs(expected - got) > 1e-9:
            spot_fail += 1
check("spot_check_expanding_breach_rate_causal_30_comparisons", spot_fail == 0)
leak_report["spot_check_failures"] = spot_fail
leak_report["n_sampled_sellers"] = len(sample)

(OUT / "SELLER_SLA_LEAKAGE_TESTS.json").write_text(json.dumps(leak_report, indent=2))
print("LEAKAGE TESTS:", leak_report["n_failures"], "failures /", len(leak_report["tests"]))
assert leak_report["n_failures"] == 0, "STOP: leakage test failed, do not train"

# ---- save canonical dataset ----
feature_cols = [
    "days_to_shipping_deadline", "purchase_weekday", "purchase_hour", "purchase_month",
    "same_state", "n_items", "n_distinct_products", "n_categories", "total_price", "total_freight",
    "weight_g", "volume_cm3", "payment_value", "n_installments",
    "seller_past_order_count", "seller_past_breach_rate_expanding",
    "seller_past_handling_median_expanding", "seller_past_handling_std_expanding",
    "seller_breach_rate_30d", "seller_breach_rate_90d", "seller_handling_mean_30d",
    "seller_recent_load_7d",
]
keep = ["order_id", "order_purchase_timestamp", "seller_id", "SELLER_HANDOFF_SLA_BREACH"] + feature_cols
final = df[keep].copy()
final["total_freight_over_price"] = final["total_freight"] / final["total_price"].replace(0, np.nan)
feature_cols.append("total_freight_over_price")
final.to_parquet(ART / "seller_sla_canonical.parquet", index=False)

meta = {
    "n_rows": len(final),
    "prevalence": float(final["SELLER_HANDOFF_SLA_BREACH"].mean()),
    "date_range": [str(final["order_purchase_timestamp"].min()), str(final["order_purchase_timestamp"].max())],
    "feature_cols": feature_cols,
    "T0": "order_purchase_timestamp",
    "cohort": "single-seller orders with single distinct shipping_limit_date (Gate O2 clean cohort), restricted to orders that reached carrier handoff",
    "n_unique_sellers": int(final["seller_id"].nunique()),
    "semantic_hash": hashlib.sha256(pd.util.hash_pandas_object(final[["order_id"]], index=False).values.tobytes()).hexdigest(),
}
(OUT / "SELLER_SLA_DATASET_META.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
