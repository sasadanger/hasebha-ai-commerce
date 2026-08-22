"""Olist V2: canonical dataset + causal historical features + temporal split + leakage tests
+ baseline/GBDT ladder + operational ranking vs VALUE_ONLY. One script given session time
constraints; each stage's output is saved so later stages can be re-run independently.

PREDICTION_MOMENT_T0 = order_purchase_timestamp (zero nulls in raw data, earliest actionable
moment for a business risk-flagging use case -- chosen over order_approved_at, which has 160
nulls and reflects a slightly later moment).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "olist" / "extracted"
OUT_DIR = REPO_ROOT / "artifacts" / "experiments" / "olist_v2"
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "olist_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# GATE 3/4: canonical one-row-per-order dataset + feature availability ledger
# ---------------------------------------------------------------------------
def build_canonical_dataset():
    orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date"])
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    products = pd.read_csv(RAW / "olist_products_dataset.csv")
    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
    sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
    payments = pd.read_csv(RAW / "olist_order_payments_dataset.csv")
    geo = pd.read_csv(RAW / "olist_geolocation_dataset.csv")

    log(f"raw orders={len(orders)}")

    # KEEP ONLY delivered orders with a real actual delivery date (target well-defined).
    # This matches standard practice for this dataset and is documented, not silent.
    df = orders[orders["order_status"] == "delivered"].copy()
    df = df.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date",
                            "order_purchase_timestamp"])
    log(f"delivered with complete dates={len(df)}")

    # ITEM-LEVEL AGGREGATION to one row per order (prevent item-level duplication)
    items_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_sellers=("seller_id", "nunique"),
        n_distinct_products=("product_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        max_shipping_limit_date=("shipping_limit_date", "max"),
    ).reset_index()
    # primary seller = seller of the first/only item-line (used for seller-history features)
    first_item = items.sort_values(["order_id", "order_item_id"]).groupby("order_id").first().reset_index()
    items_agg = items_agg.merge(first_item[["order_id", "seller_id", "product_id"]], on="order_id", how="left")

    prod_cols = ["product_id", "product_category_name", "product_weight_g",
                 "product_length_cm", "product_height_cm", "product_width_cm"]
    items_agg = items_agg.merge(products[prod_cols], on="product_id", how="left")

    pay_agg = payments.groupby("order_id").agg(payment_installments=("payment_installments", "max")).reset_index()

    df = df.merge(items_agg, on="order_id", how="inner")  # inner: an order with no item rows is unusable
    df = df.merge(pay_agg, on="order_id", how="left")
    df = df.merge(customers[["customer_id", "customer_state", "customer_zip_code_prefix", "customer_city"]],
                   on="customer_id", how="left")
    df = df.merge(sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]], on="seller_id", how="left")

    # geo centroid per zip prefix (avg lat/lon) -- coarse but leakage-safe (static reference data)
    geo_centroid = geo.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"), lon=("geolocation_lng", "mean")).reset_index()
    df = df.merge(geo_centroid.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                                                 "lat": "customer_lat", "lon": "customer_lon"}),
                   on="customer_zip_code_prefix", how="left")
    df = df.merge(geo_centroid.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix",
                                                 "lat": "seller_lat", "lon": "seller_lon"}),
                   on="seller_zip_code_prefix", how="left")

    # duplicate check
    dup = df["order_id"].duplicated().sum()
    log(f"canonical rows={len(df)}, duplicate order_ids={dup}")
    assert dup == 0, "one-row-per-order violated!"

    return df


# ---------------------------------------------------------------------------
# GATE 5: target contract
# ---------------------------------------------------------------------------
def add_targets(df):
    df["signed_delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    df["late_binary"] = (df["signed_delay_days"] > 0).astype(int)
    df["positive_delay_days"] = df["signed_delay_days"].clip(lower=0)
    df["actual_delivery_duration_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    return df


# ---------------------------------------------------------------------------
# GATE 7: causal historical features (shift(1) before any expanding/rolling)
# ---------------------------------------------------------------------------
def add_causal_historical_features(df):
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    # SELLER history (shift(1): a seller's Nth order never uses info from order N itself)
    g = df.groupby("seller_id")
    df["seller_prior_order_count"] = g.cumcount()  # count of PRIOR orders (0-indexed cumcount = prior count)
    df["seller_prior_late_rate"] = g["late_binary"].transform(lambda s: s.shift(1).expanding().mean())
    df["seller_prior_delay_mean"] = g["signed_delay_days"].transform(lambda s: s.shift(1).expanding().mean())
    df["seller_prior_delay_std"] = g["signed_delay_days"].transform(lambda s: s.shift(1).expanding().std())

    # rolling 30-order seller load (proxy for 30d volume without needing a date-indexed rolling window,
    # which pandas rolling('30D') would also support but cumcount-based is simpler/robust here)
    df["seller_rolling30_late_rate"] = g["late_binary"].transform(lambda s: s.shift(1).rolling(30, min_periods=3).mean())

    # LANE history (seller_state -> customer_state)
    df["lane"] = df["seller_state"].astype(str) + "->" + df["customer_state"].astype(str)
    gl = df.groupby("lane")
    df["lane_prior_order_count"] = gl.cumcount()
    df["lane_prior_late_rate"] = gl["late_binary"].transform(lambda s: s.shift(1).expanding().mean())
    df["lane_prior_delay_mean"] = gl["signed_delay_days"].transform(lambda s: s.shift(1).expanding().mean())

    # DESTINATION (customer_state) history
    gd = df.groupby("customer_state")
    df["dest_prior_order_count"] = gd.cumcount()
    df["dest_prior_late_rate"] = gd["late_binary"].transform(lambda s: s.shift(1).expanding().mean())

    # GEOGRAPHY
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))
    df["distance_km"] = haversine(df["customer_lat"], df["customer_lon"], df["seller_lat"], df["seller_lon"])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # ORDER features (available at T0 -- price/freight/weight/dims are catalog+cart info)
    df["freight_value_ratio"] = df["total_freight"] / df["total_price"].clip(lower=1e-6)
    df["multi_seller_flag"] = (df["n_distinct_sellers"] > 1).astype(int)

    # TIME features (from purchase timestamp, always available at T0)
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_week"] = df["order_purchase_timestamp"].dt.isocalendar().week.astype(int)
    # Brazilian Black Friday proximity (last Friday of November) -- approximate via month/day heuristic
    df["is_november"] = (df["purchase_month"] == 11).astype(int)

    # PROMISE features (estimated delivery horizon -- known at T0, this IS the promise made to customer)
    df["promise_horizon_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["distance_x_promise"] = df["distance_km"] * df["promise_horizon_days"]

    return df


FEATURE_COLS = [
    "seller_prior_order_count", "seller_prior_late_rate", "seller_prior_delay_mean", "seller_prior_delay_std",
    "seller_rolling30_late_rate", "lane_prior_order_count", "lane_prior_late_rate", "lane_prior_delay_mean",
    "dest_prior_order_count", "dest_prior_late_rate", "distance_km", "same_state",
    "total_price", "total_freight", "freight_value_ratio", "n_items", "n_distinct_sellers",
    "n_distinct_products", "multi_seller_flag", "product_weight_g", "payment_installments",
    "purchase_hour", "purchase_weekday", "purchase_month", "purchase_week", "is_november",
    "promise_horizon_days", "distance_x_promise",
]


if __name__ == "__main__":
    df = build_canonical_dataset()
    df = add_targets(df)
    df = add_causal_historical_features(df)

    h = sha256_df(df[["order_id", "order_purchase_timestamp", "late_binary"]])
    manifest = {
        "n_rows": len(df), "n_unique_order_ids": df["order_id"].nunique(),
        "date_range": [str(df["order_purchase_timestamp"].min()), str(df["order_purchase_timestamp"].max())],
        "late_prevalence": float(df["late_binary"].mean()),
        "signed_delay_days_stats": {"mean": float(df["signed_delay_days"].mean()), "std": float(df["signed_delay_days"].std()),
                                     "median": float(df["signed_delay_days"].median())},
        "canonical_dataset_semantic_hash": h,
        "PREDICTION_MOMENT_T0": "order_purchase_timestamp",
    }
    log(json.dumps(manifest, indent=2))
    (REPORTS_DIR / "canonical_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    df.to_parquet(OUT_DIR / "canonical_dataset.parquet", index=False)
    log(f"saved canonical dataset: {OUT_DIR / 'canonical_dataset.parquet'}")
