"""Generate focused factual Phase 1B observations for one processed dataset."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import duckdb
from .download import PROJECT_ROOT, SUPPORTED_DATASETS

def rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[list[Any]]:
    return [[value.isoformat() if hasattr(value,"isoformat") else value for value in row] for row in con.execute(sql).fetchall()]

def observe(dataset: str) -> dict[str, Any]:
    con=duckdb.connect(); base=(PROJECT_ROOT/"data"/"processed"/dataset).as_posix(); result={"dataset":dataset,"scope":"One processed dataset only."}
    try:
        if dataset=="olist":
            orders=f"read_parquet('{base}/olist_orders_dataset.parquet')"; items=f"read_parquet('{base}/olist_order_items_dataset.parquet')"; reviews=f"read_parquet('{base}/olist_order_reviews_dataset.parquet')"; geo=f"read_parquet('{base}/olist_geolocation_dataset.parquet')"
            result.update({"order_status_distribution":rows(con,f"select order_status,count(*) from {orders} group by 1 order by 2 desc"),"purchase_timestamp_coverage":rows(con,f"select min(order_purchase_timestamp),max(order_purchase_timestamp) from {orders}")[0],"timestamp_sequence_checks":dict(zip(["approval_before_purchase","carrier_before_approval","customer_before_carrier","delivered_after_estimate"],rows(con,f"select sum(order_approved_at < order_purchase_timestamp),sum(order_delivered_carrier_date < order_approved_at),sum(order_delivered_customer_date < order_delivered_carrier_date),sum(order_delivered_customer_date > order_estimated_delivery_date) from {orders}")[0])),"item_value_ranges":dict(zip(["min_price","max_price","nonpositive_price","min_freight","max_freight","negative_freight"],rows(con,f"select min(price),max(price),sum(price<=0),min(freight_value),max(freight_value),sum(freight_value<0) from {items}")[0])),"review_score_distribution":rows(con,f"select review_score,count(*) from {reviews} group by 1 order by 1"),"invalid_coordinate_counts":dict(zip(["latitude","longitude"],rows(con,f"select sum(geolocation_lat not between -90 and 90),sum(geolocation_lng not between -180 and 180) from {geo}")[0]))})
        elif dataset=="instacart":
            orders=f"read_parquet('{base}/orders.parquet')"; prior=f"read_parquet('{base}/order_products__prior.parquet')"; products=f"read_parquet('{base}/products.parquet')"; aisles=f"read_parquet('{base}/aisles.parquet')"; departments=f"read_parquet('{base}/departments.parquet')"
            result.update({"eval_set_distribution":rows(con,f"select eval_set,count(*) from {orders} group by 1 order by 1"),"order_ranges":dict(zip(["min_order_number","max_order_number","min_dow","max_dow","min_hour","max_hour","unparseable_hour","min_days_since_prior","max_days_since_prior"],rows(con,f"select min(order_number),max(order_number),min(order_dow),max(order_dow),min(try_cast(order_hour_of_day as integer)),max(try_cast(order_hour_of_day as integer)),sum(try_cast(order_hour_of_day as integer) is null),min(days_since_prior_order),max(days_since_prior_order) from {orders}")[0])),"reordered_distribution_prior":rows(con,f"select reordered,count(*) from {prior} group by 1 order by 1"),"taxonomy_orphans":dict(zip(["products_without_aisle","products_without_department"],rows(con,f"select (select count(*) from {products} x anti join {aisles} a using(aisle_id)),(select count(*) from {products} x anti join {departments} d using(department_id))")[0]))})
        else:
            reviews=f"read_parquet('{base}/reviews_text_ready.parquet')"
            result.update({"review_timestamp_coverage_utc":rows(con,f"select cast(min(review_datetime_utc) as varchar),cast(max(review_datetime_utc) as varchar) from {reviews}")[0],"rating_distribution":rows(con,f"select rating,count(*) from {reviews} group by 1 order by 1"),"quality_flags":dict(zip(["negative_helpful_votes","min_helpful_vote","max_helpful_vote","usable_text","empty_or_whitespace_text"],rows(con,f"select sum(helpful_vote<0),min(helpful_vote),max(helpful_vote),sum(has_usable_text),sum(not has_usable_text) from {reviews}")[0])),"verified_purchase_distribution":rows(con,f"select verified_purchase,count(*) from {reviews} group by 1 order by 1")})
    finally: con.close()
    return result

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("dataset",choices=SUPPORTED_DATASETS); args=parser.parse_args(); report=observe(args.dataset); output=PROJECT_ROOT/"data"/"profiles"/f"phase1b_{args.dataset}.json"; output.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
