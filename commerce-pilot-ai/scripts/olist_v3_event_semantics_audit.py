"""Gate O1: raw Olist event-timestamp semantics audit (olist_v3_multistage track).
Reads ORIGINAL raw CSVs only (not the engineered V2 dataset) per mission instruction.
"""
import json
import hashlib
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/olist/extracted")
OUT = Path("reports/generated/olist_v3_multistage/EVENT_SEMANTICS_AUDIT.json")

orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")

def sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()

ts_cols_orders = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for c in ts_cols_orders:
    orders[c] = pd.to_datetime(orders[c], errors="coerce")
items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")

audit = {
    "gate": "O1_RAW_EVENT_SEMANTICS",
    "source_files": {
        "olist_orders_dataset.csv": {"sha256": sha256(RAW / "olist_orders_dataset.csv"), "rows": len(orders)},
        "olist_order_items_dataset.csv": {"sha256": sha256(RAW / "olist_order_items_dataset.csv"), "rows": len(items)},
    },
    "timestamps": {
        "order_purchase_timestamp": {
            "table": "orders", "granularity": "order-level, 1 row per order_id",
            "meaning": "customer places the order (checkout completion)",
            "known_at_purchase": True, "known_at_approval": True, "known_at_carrier_handoff": True,
            "post_outcome": False, "target_only": False,
            "null_count": int(orders["order_purchase_timestamp"].isna().sum()),
        },
        "order_approved_at": {
            "table": "orders", "granularity": "order-level",
            "meaning": "payment approval by Olist (not seller/carrier action)",
            "known_at_purchase": False, "known_at_approval": True, "known_at_carrier_handoff": True,
            "post_outcome": False, "target_only": False,
            "null_count": int(orders["order_approved_at"].isna().sum()),
        },
        "shipping_limit_date": {
            "table": "order_items", "granularity": "ITEM-LEVEL, one row per order_item (NOT order-level)",
            "meaning": "seller's deadline to hand the item to the logistics carrier; set at order-item creation, effectively known very close to purchase/approval time",
            "known_at_purchase": "approximately (present in items table populated near purchase; treated as known at/just after purchase, not truly T0 pre-checkout)",
            "known_at_approval": True, "known_at_carrier_handoff": True,
            "post_outcome": False, "target_only": False,
            "null_count": int(items["shipping_limit_date"].isna().sum()),
            "CRITICAL_NOTE": "Item-level, not order-level. Multiple items in one order can carry DIFFERENT shipping_limit_date values (different sellers or different lead times). Do NOT collapse to order-level without checking per-order distinct-value count (see Gate O2).",
        },
        "order_delivered_carrier_date": {
            "table": "orders", "granularity": "order-level in the orders table, but represents the carrier pickup/handoff event which is physically a per-item/per-seller event; Olist collapses it to one order-level timestamp",
            "meaning": "date the order (or its last constituent shipment) was handed to the logistics carrier",
            "known_at_purchase": False, "known_at_approval": False, "known_at_carrier_handoff": True,
            "post_outcome": True, "target_only": False,
            "null_count": int(orders["order_delivered_carrier_date"].isna().sum()),
            "CRITICAL_NOTE": "For multi-seller orders this single order-level field cannot unambiguously represent EACH seller's individual handoff -- see Gate O2 multi-seller audit before using as a seller-specific label source.",
        },
        "order_delivered_customer_date": {
            "table": "orders", "granularity": "order-level",
            "meaning": "date the customer actually received the order",
            "known_at_purchase": False, "known_at_approval": False, "known_at_carrier_handoff": False,
            "post_outcome": True, "target_only": True,
            "null_count": int(orders["order_delivered_customer_date"].isna().sum()),
        },
        "order_estimated_delivery_date": {
            "table": "orders", "granularity": "order-level",
            "meaning": "promised delivery date shown to customer at purchase time",
            "known_at_purchase": True, "known_at_approval": True, "known_at_carrier_handoff": True,
            "post_outcome": False, "target_only": False,
            "null_count": int(orders["order_estimated_delivery_date"].isna().sum()),
        },
    },
    "order_status_distribution": orders["order_status"].value_counts().to_dict(),
    "method": "Read directly from raw CSVs (data/raw/olist/extracted/), not from engineered olist_v2 parquet, per mission Gate O1 instruction.",
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(audit, indent=2, default=str))
print("WROTE", OUT, "rows_orders", len(orders), "rows_items", len(items))
