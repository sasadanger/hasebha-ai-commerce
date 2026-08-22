"""Gate D4: DataCo raw column leakage audit."""
import json
import pandas as pd
from pathlib import Path

RAW = Path("D:/commercepilot_ml_cache/data/dataco/raw/DataCoSupplyChainDataset.csv")
OUT = Path("reports/generated/dataco")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, encoding="latin1")

# Direct/derived target leakage per paper + our own scrutiny
DIRECT_LEAKAGE = {
    "Delivery Status": "Directly encodes on-time/late/shipping-canceled outcome -- IS the label (or a superset of it) for the delay task. TARGET_DERIVED=YES, POST_OUTCOME=YES.",
    "Days for shipping (real)": "Actual realized shipping duration -- direct outcome. TARGET_DERIVED=YES, POST_OUTCOME=YES.",
    "Late_delivery_risk": "Binary flag algebraically derived from (Days for shipping (real) > Days for shipment (scheduled)); paper reports r>0.99 with label. TARGET_DERIVED=YES, POST_OUTCOME=YES.",
}
SCHEDULED_OK = {"Days for shipment (scheduled)": "Scheduled (promised) transit time, known at order time -- NOT an outcome."}
PII_EXCLUDE = ["Customer Email", "Customer Fname", "Customer Lname", "Customer Password", "Customer Street", "Product Description", "Product Image"]
POST_OUTCOME_FINANCIAL = ["Benefit per order", "Order Profit Per Order", "Order Item Profit Ratio"]  # realized profit, not known at order placement, exclude from strict-online at minimum

rows = []
for c in df.columns:
    known_at_order_time = True
    target_derived = False
    post_outcome = False
    strict_allowed = True
    reason = "Order/product/customer attribute known at order placement."
    if c in DIRECT_LEAKAGE:
        known_at_order_time = False
        target_derived = True
        post_outcome = True
        strict_allowed = False
        reason = DIRECT_LEAKAGE[c]
    elif c == "shipping date (DateOrders)":
        known_at_order_time = False
        post_outcome = True
        strict_allowed = False
        reason = "Actual shipment/delivery date -- realized outcome timestamp, used only to construct labels/edge history, never as an input feature."
    elif c in POST_OUTCOME_FINANCIAL:
        known_at_order_time = False
        post_outcome = True
        strict_allowed = False
        reason = "Realized profit/margin, not knowable at order-placement time (depends on final settled costs); excluded from both tracks' feature sets to be safe, paper-faithful track may use as edge covariate only if computed train-history-only."
    elif c in PII_EXCLUDE:
        strict_allowed = False
        reason = "PII / free text, not a modeling feature in either track."
    elif c == "Days for shipment (scheduled)":
        reason = SCHEDULED_OK[c]

    rows.append({
        "FEATURE": c,
        "DESCRIPTION": reason,
        "PAPER_USES": c not in PII_EXCLUDE and c not in DIRECT_LEAKAGE and c != "shipping date (DateOrders)",
        "KNOWN_AT_ORDER_TIME": known_at_order_time,
        "KNOWN_AT_WINDOW_CUTOFF": known_at_order_time,
        "TARGET_DERIVED": target_derived,
        "POST_OUTCOME": post_outcome,
        "STRICT_TRACK_ALLOWED": strict_allowed,
        "REASON": reason,
    })

audit = {
    "gate": "D4_DATACO_RAW_FEATURE_AUDIT",
    "n_rows": len(df), "n_columns": len(df.columns),
    "direct_leakage_blacklist": list(DIRECT_LEAKAGE.keys()),
    "columns": rows,
}
(OUT / "DATACO_RAW_FEATURE_AUDIT.json").write_text(json.dumps(audit, indent=2))
print("blacklisted:", list(DIRECT_LEAKAGE.keys()))
print("n_columns audited:", len(rows))
