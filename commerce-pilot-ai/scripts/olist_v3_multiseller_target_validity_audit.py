"""Gate O2: multi-seller / target validity audit for olist_v3_multistage."""
import json
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/olist/extracted")
OUT = Path("reports/generated/olist_v3_multistage/MULTI_SELLER_TARGET_VALIDITY_AUDIT.json")

orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
items = pd.read_csv(RAW / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])

# only orders that reached carrier handoff / delivered are relevant for this audit's "physical events happened" framing
n_orders_total = orders["order_id"].nunique()

per_order = items.groupby("order_id").agg(
    n_items=("order_item_id", "count"),
    n_distinct_sellers=("seller_id", "nunique"),
    n_distinct_shipping_limits=("shipping_limit_date", "nunique"),
).reset_index()

single_item = (per_order["n_items"] == 1)
multi_item = ~single_item
single_seller = (per_order["n_distinct_sellers"] == 1)
multi_seller = ~single_seller
single_seller_consistent_limit = single_seller & (per_order["n_distinct_shipping_limits"] == 1)
single_seller_inconsistent_limit = single_seller & (per_order["n_distinct_shipping_limits"] > 1)

n = len(per_order)

# revenue proxy: price+freight sum, to report "fraction of ecommerce volume"
items["line_value"] = items["price"] + items["freight_value"]
order_value = items.groupby("order_id")["line_value"].sum()
total_value = order_value.sum()
clean_cohort_ids = per_order.loc[single_seller_consistent_limit, "order_id"]
clean_cohort_value = order_value.reindex(clean_cohort_ids).sum()

audit = {
    "gate": "O2_MULTI_SELLER_TARGET_VALIDITY",
    "n_orders_in_items_table": int(n),
    "n_orders_in_orders_table": int(n_orders_total),
    "single_item_orders": int(single_item.sum()),
    "multi_item_orders": int(multi_item.sum()),
    "single_seller_orders": int(single_seller.sum()),
    "multi_seller_orders": int(multi_seller.sum()),
    "single_seller_fraction": float(single_seller.mean()),
    "single_seller_consistent_shipping_limit_orders": int(single_seller_consistent_limit.sum()),
    "single_seller_inconsistent_shipping_limit_orders": int(single_seller_inconsistent_limit.sum()),
    "determination": {
        "question": "Can order_delivered_carrier_date represent seller handoff for (A) all single-seller orders or only (B) single-seller + consistent-shipping-limit orders?",
        "answer": "B -- restrict to single-seller AND single distinct shipping_limit_date. A small fraction of single-seller multi-item orders carry >1 distinct shipping_limit_date (different lead times per SKU from the same seller), so even single-seller orders are not automatically safe; the per-order deadline is only unambiguous when all items share one shipping_limit_date.",
        "PREFERRED_CLEAN_COHORT": "single_seller_orders AND single_distinct_shipping_limit_date",
    },
    "clean_cohort": {
        "n_orders": int(single_seller_consistent_limit.sum()),
        "fraction_of_all_item_orders": float(single_seller_consistent_limit.sum() / n),
        "fraction_of_total_gmv_proxy(price+freight)": float(clean_cohort_value / total_value) if total_value else None,
    },
    "note": "Fractions computed over the order_items-joinable population (orders with at least one item; excludes a small number of orders absent from order_items, e.g. some canceled/unavailable orders).",
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(audit, indent=2, default=str))
print(json.dumps({k: audit[k] for k in ["single_seller_orders","multi_seller_orders","single_seller_fraction","clean_cohort"]}, indent=2, default=str))
