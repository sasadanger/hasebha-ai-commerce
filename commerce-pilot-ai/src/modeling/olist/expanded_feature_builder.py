"""Development-only Olist Contract A and retrospective Contract B construction."""
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from .strict_feature_builder import FEATURES as STRICT
EXPANDED=["payment_record_count","max_payment_installments","total_payment_value","item_count","unique_product_count","unique_seller_count","total_item_price","total_freight_value","mean_freight_value","max_freight_value","product_category_diversity","mean_product_weight_g","mean_product_length_cm","mean_product_height_cm","mean_product_width_cm"]
META=["order_id","approved_at","late_delivery"]
def build_development(processed:Path,start="2016-09-01",end="2018-05-01"):
    if end>"2018-05-01": raise ValueError("Phase 2B Test boundary violation")
    q=f"""WITH s AS (SELECT *,count(*) over(partition by order_id) n FROM read_parquet(?) WHERE order_approved_at<TIMESTAMP '{end}'),
    e AS (SELECT * FROM s WHERE n=1 AND order_status='delivered' AND order_approved_at IS NOT NULL AND order_delivered_customer_date IS NOT NULL AND order_estimated_delivery_date IS NOT NULL AND order_approved_at>=order_purchase_timestamp AND order_delivered_customer_date>=order_purchase_timestamp AND order_estimated_delivery_date>=order_approved_at AND (order_delivered_carrier_date IS NULL OR order_delivered_carrier_date>=order_approved_at) AND (order_delivered_carrier_date IS NULL OR order_delivered_customer_date>=order_delivered_carrier_date) AND order_approved_at>=TIMESTAMP '{start}'),
    pay AS (SELECT order_id,count(*) payment_record_count,max(payment_installments) max_payment_installments,sum(payment_value) total_payment_value FROM read_parquet(?) GROUP BY 1),
    ia AS (SELECT i.order_id,count(*) item_count,count(distinct i.product_id) unique_product_count,count(distinct i.seller_id) unique_seller_count,sum(i.price) total_item_price,sum(i.freight_value) total_freight_value,avg(i.freight_value) mean_freight_value,max(i.freight_value) max_freight_value,count(distinct p.product_category_name) product_category_diversity,avg(p.product_weight_g) mean_product_weight_g,avg(p.product_length_cm) mean_product_length_cm,avg(p.product_height_cm) mean_product_height_cm,avg(p.product_width_cm) mean_product_width_cm FROM read_parquet(?) i LEFT JOIN read_parquet(?) p USING(product_id) GROUP BY 1)
    SELECT e.order_id,e.order_approved_at approved_at,cast(e.order_delivered_customer_date>e.order_estimated_delivery_date as integer) late_delivery,
    year(order_purchase_timestamp) purchase_year,month(order_purchase_timestamp) purchase_month,dayofweek(order_purchase_timestamp) purchase_day_of_week,hour(order_purchase_timestamp) purchase_hour,year(order_approved_at) approval_year,month(order_approved_at) approval_month,dayofweek(order_approved_at) approval_day_of_week,hour(order_approved_at) approval_hour,date_diff('second',order_purchase_timestamp,order_approved_at) purchase_to_approval_seconds,pay.* EXCLUDE(order_id),ia.* EXCLUDE(order_id)
    FROM e LEFT JOIN pay USING(order_id) LEFT JOIN ia USING(order_id) ORDER BY approved_at,order_id"""
    files=[processed/"olist_orders_dataset.parquet",processed/"olist_order_payments_dataset.parquet",processed/"olist_order_items_dataset.parquet",processed/"olist_products_dataset.parquet"]
    with duckdb.connect() as con: d=con.execute(q,[x.as_posix() for x in files]).fetchdf()
    expected=META+STRICT+EXPANDED
    if list(d)!=expected or not d.order_id.is_unique: raise ValueError("Feature schema/cardinality violation")
    if d.approved_at.max()>=pd.Timestamp(end): raise ValueError("Test row materialized")
    return d
def predictors(d,contract):
    if contract not in ("strict","expanded"): raise ValueError("Unknown feature contract")
    cols=STRICT if contract=="strict" else STRICT+EXPANDED
    unexpected=set(cols)-set(STRICT+EXPANDED)
    if unexpected: raise ValueError(f"Unapproved features: {unexpected}")
    return d[cols],d.late_delivery.astype(int)
