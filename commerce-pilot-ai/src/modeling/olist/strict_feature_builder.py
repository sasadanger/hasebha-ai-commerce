"""Leakage-safe Olist strict-core feature construction."""
from __future__ import annotations
from pathlib import Path
import duckdb, numpy as np, pandas as pd

FEATURES = ["purchase_year","purchase_month","purchase_day_of_week","purchase_hour",
            "approval_year","approval_month","approval_day_of_week","approval_hour",
            "purchase_to_approval_seconds"]
RANGES = {"purchase_month":(1,12),"approval_month":(1,12),
          "purchase_day_of_week":(0,6),"approval_day_of_week":(0,6),
          "purchase_hour":(0,23),"approval_hour":(0,23),
          "purchase_to_approval_seconds":(0,float("inf"))}

def build_split(orders_path: Path, start: str, end_exclusive: str) -> pd.DataFrame:
    """Build one deterministic row per eligible order in one explicit time window."""
    sql = f"""
    WITH source AS (
      SELECT *, count(*) OVER(PARTITION BY order_id) AS order_id_count
      FROM read_parquet(?)
    ), eligible AS (
      SELECT * FROM source WHERE order_id_count=1 AND order_status='delivered'
       AND order_approved_at IS NOT NULL AND order_delivered_customer_date IS NOT NULL
       AND order_estimated_delivery_date IS NOT NULL
       AND order_approved_at>=order_purchase_timestamp
       AND order_delivered_customer_date>=order_purchase_timestamp
       AND order_estimated_delivery_date>=order_approved_at
       AND (order_delivered_carrier_date IS NULL OR order_delivered_carrier_date>=order_approved_at)
       AND (order_delivered_carrier_date IS NULL OR order_delivered_customer_date>=order_delivered_carrier_date)
       AND order_approved_at>=TIMESTAMP '{start}' AND order_approved_at<TIMESTAMP '{end_exclusive}'
    )
    SELECT order_id,
      CAST(order_delivered_customer_date>order_estimated_delivery_date AS INTEGER) AS late_delivery,
      year(order_purchase_timestamp) purchase_year, month(order_purchase_timestamp) purchase_month,
      dayofweek(order_purchase_timestamp) purchase_day_of_week, hour(order_purchase_timestamp) purchase_hour,
      year(order_approved_at) approval_year, month(order_approved_at) approval_month,
      dayofweek(order_approved_at) approval_day_of_week, hour(order_approved_at) approval_hour,
      date_diff('second',order_purchase_timestamp,order_approved_at) purchase_to_approval_seconds
    FROM eligible ORDER BY order_approved_at,order_id"""
    with duckdb.connect() as con:
        df = con.execute(sql, [orders_path.as_posix()]).fetchdf()
    if list(df.columns) != ["order_id","late_delivery",*FEATURES]:
        raise ValueError("Unexpected column entered strict-core frame")
    if not df.order_id.is_unique or df.isna().any().any():
        raise ValueError("Duplicate or null strict-core values")
    values=df[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all(): raise ValueError("Non-finite strict-core value")
    for col,(lo,hi) in RANGES.items():
        if not df[col].between(lo,hi).all(): raise ValueError(f"Impossible {col}")
    return df

def matrix(frame: pd.DataFrame):
    if list(frame.columns) != ["order_id","late_delivery",*FEATURES]:
        raise ValueError("Forbidden or unexpected predictor detected")
    return frame[FEATURES], frame["late_delivery"].astype(int)
