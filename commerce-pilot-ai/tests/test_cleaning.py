from __future__ import annotations

import gzip
import json
from pathlib import Path

import duckdb
import pytest

from src.data_pipeline.clean_amazon_appliances import clean as clean_amazon
from src.data_pipeline.clean_instacart import FILES as INSTACART_FILES, clean as clean_instacart
from src.data_pipeline.clean_olist import FILES as OLIST_FILES, clean as clean_olist
from src.data_pipeline.data_quality import dataset_paths, prepare_output


HEADERS = {
    "olist_customers_dataset.csv": "customer_id,customer_unique_id,customer_zip_code_prefix,customer_city,customer_state\n1,u,1,c,s\n",
    "olist_geolocation_dataset.csv": "geolocation_zip_code_prefix,geolocation_lat,geolocation_lng,geolocation_city,geolocation_state\n1,1.0,2.0,c,s\n",
    "olist_order_items_dataset.csv": "order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value\n1,1,p,s,2018-01-01,2,1\n",
    "olist_order_payments_dataset.csv": "order_id,payment_sequential,payment_type,payment_installments,payment_value\n1,1,card,1,3\n",
    "olist_order_reviews_dataset.csv": "review_id,order_id,review_score,review_comment_title,review_comment_message,review_creation_date,review_answer_timestamp\nr,1,5,,,2018-01-02,2018-01-03\n",
    "olist_orders_dataset.csv": "order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date\n1,1,delivered,2018-01-01,2018-01-01,2018-01-02,2018-01-03,2018-01-04\n",
    "olist_products_dataset.csv": "product_id,product_category_name,product_name_lenght,product_description_lenght,product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm\np,c,1,1,1,1,1,1,1\n",
    "olist_sellers_dataset.csv": "seller_id,seller_zip_code_prefix,seller_city,seller_state\ns,1,c,s\n",
    "product_category_name_translation.csv": "product_category_name,product_category_name_english\nc,c\n",
    "aisles.csv": "aisle_id,aisle\n1,a\n", "departments.csv": "department_id,department\n1,d\n",
    "order_products__prior.csv": "order_id,product_id,add_to_cart_order,reordered\n1,1,1,0\n",
    "order_products__train.csv": "order_id,product_id,add_to_cart_order,reordered\n2,1,1,1\n",
    "orders.csv": "order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order\n1,1,prior,1,0,1,\n",
    "products.csv": "product_id,product_name,aisle_id,department_id\n1,p,1,1\n",
}


def config(tmp_path: Path) -> Path:
    path = tmp_path / "datasets.yaml"
    lines = ["datasets:"]
    for name in ("olist", "instacart", "amazon_reviews_appliances"):
        lines += [f"  {name}:", "    source_url: https://example.invalid/data", f"    local_raw_data_path: '{(tmp_path/'raw'/name).as_posix()}'", f"    local_processed_data_path: '{(tmp_path/'processed'/name).as_posix()}'", "    expected_file_format: test"]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def populate_csvs(root: Path, names: list[str]) -> None:
    extracted = root / "extracted"; extracted.mkdir(parents=True)
    for name in names: (extracted / name).write_text(HEADERS[name], encoding="utf-8")


def test_dataset_paths_are_isolated(tmp_path: Path) -> None:
    cfg=config(tmp_path); paths=[dataset_paths(cfg,name) for name in ("olist","instacart","amazon_reviews_appliances")]
    assert len({raw for raw,_ in paths}) == 3 and len({processed for _,processed in paths}) == 3


def test_missing_input_and_dry_run(tmp_path: Path) -> None:
    cfg=config(tmp_path)
    with pytest.raises(FileNotFoundError): clean_olist(cfg,dry_run=True)
    populate_csvs(tmp_path/"raw"/"olist",OLIST_FILES)
    target=clean_olist(cfg,dry_run=True)
    assert not target.exists()


def test_refuses_processed_overwrite(tmp_path: Path) -> None:
    output=tmp_path/"processed"; output.mkdir(); (output/"kept.txt").write_text("keep",encoding="utf-8")
    with pytest.raises(FileExistsError): prepare_output(output,force=False,dry_run=False)
    assert (output/"kept.txt").read_text(encoding="utf-8") == "keep"


def test_olist_safe_cleaning_retains_row(tmp_path: Path) -> None:
    cfg=config(tmp_path); populate_csvs(tmp_path/"raw"/"olist",OLIST_FILES); output=clean_olist(cfg)
    assert duckdb.sql(f"select count(*) from read_parquet('{(output/'olist_orders_dataset.parquet').as_posix()}')").fetchone()[0] == 1


def test_instacart_safe_cleaning_preserves_missing_value(tmp_path: Path) -> None:
    cfg=config(tmp_path); populate_csvs(tmp_path/"raw"/"instacart",INSTACART_FILES); output=clean_instacart(cfg)
    assert duckdb.sql(f"select days_since_prior_order is null from read_parquet('{(output/'orders.parquet').as_posix()}')").fetchone()[0] is True


def test_amazon_safe_cleaning_flags_empty_text_without_dropping(tmp_path: Path) -> None:
    cfg=config(tmp_path); raw=tmp_path/"raw"/"amazon_reviews_appliances"; raw.mkdir(parents=True)
    record={"rating":3.0,"title":"t","text":" ","images":[],"asin":"a","parent_asin":"p","user_id":"u","timestamp":1000,"helpful_vote":0,"verified_purchase":True}
    with gzip.open(raw/"Appliances.jsonl.gz","wt",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
    output=clean_amazon(cfg); row=duckdb.sql(f"select count(*), bool_and(not has_usable_text) from read_parquet('{(output/'reviews_text_ready.parquet').as_posix()}')").fetchone()
    assert row == (1, True)
    columns={x[0] for x in duckdb.sql(f"describe select * from read_parquet('{(output/'reviews_text_ready.parquet').as_posix()}')").fetchall()}
    assert {"overall","review_title","review_text"} <= columns
    assert not {"rating","title","text"} & columns
