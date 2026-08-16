from pathlib import Path
import duckdb
import pytest
from src.modeling.olist import readiness_audit as audit

TABLES={
"olist_orders_dataset.parquet":"SELECT 'o1'::VARCHAR order_id,'c1'::VARCHAR customer_id,'delivered'::VARCHAR order_status,TIMESTAMP '2017-01-01' order_purchase_timestamp,TIMESTAMP '2017-01-01 01:00:00' order_approved_at,TIMESTAMP '2017-01-02' order_delivered_carrier_date,TIMESTAMP '2017-01-04' order_delivered_customer_date,TIMESTAMP '2017-01-03' order_estimated_delivery_date",
"olist_order_items_dataset.parquet":"SELECT 'o1'::VARCHAR order_id,1::BIGINT order_item_id,'p1'::VARCHAR product_id,'s1'::VARCHAR seller_id,TIMESTAMP '2017-01-02' shipping_limit_date,10.0::DOUBLE price,2.0::DOUBLE freight_value",
"olist_order_payments_dataset.parquet":"SELECT 'o1'::VARCHAR order_id,1::BIGINT payment_sequential,'credit_card'::VARCHAR payment_type,1::BIGINT payment_installments,12.0::DOUBLE payment_value",
"olist_customers_dataset.parquet":"SELECT 'c1'::VARCHAR customer_id,'u1'::VARCHAR customer_unique_id,'1'::VARCHAR customer_zip_code_prefix,'x'::VARCHAR customer_city,'SP'::VARCHAR customer_state",
"olist_products_dataset.parquet":"SELECT 'p1'::VARCHAR product_id,'cat'::VARCHAR product_category_name,1::BIGINT product_name_lenght,1::BIGINT product_description_lenght,1::BIGINT product_photos_qty,1::BIGINT product_weight_g,1::BIGINT product_length_cm,1::BIGINT product_height_cm,1::BIGINT product_width_cm",
"olist_sellers_dataset.parquet":"SELECT 's1'::VARCHAR seller_id,'1'::VARCHAR seller_zip_code_prefix,'x'::VARCHAR seller_city,'SP'::VARCHAR seller_state",
"olist_geolocation_dataset.parquet":"SELECT '1'::VARCHAR geolocation_zip_code_prefix,0.0::DOUBLE geolocation_lat,0.0::DOUBLE geolocation_lng,'x'::VARCHAR geolocation_city,'SP'::VARCHAR geolocation_state",
"olist_order_reviews_dataset.parquet":"SELECT 'r1'::VARCHAR review_id,'o1'::VARCHAR order_id,5::BIGINT review_score,NULL::VARCHAR review_comment_title,NULL::VARCHAR review_comment_message,TIMESTAMP '2017-01-04' review_creation_date,TIMESTAMP '2017-01-05' review_answer_timestamp",
"product_category_name_translation.parquet":"SELECT 'cat'::VARCHAR product_category_name,'category'::VARCHAR product_category_name_english"}

@pytest.fixture
def processed(tmp_path:Path,monkeypatch:pytest.MonkeyPatch)->Path:
    root=tmp_path/"olist"; root.mkdir(); con=duckdb.connect()
    for name,query in TABLES.items(): con.execute(f"COPY ({query}) TO '{(root/name).as_posix()}' (FORMAT PARQUET)")
    con.close(); monkeypatch.setattr(audit,"PROCESSED",root); return root

def test_schema_and_target_waterfall(processed:Path)->None:
    con=duckdb.connect(); audit.verify_schemas(con); audit.build_audit_view(con); flow,eligible=audit.waterfall(con); stats=audit._stats(con,eligible); con.close()
    assert stats == {"orders":1,"positive":1,"negative":0,"unlabeled":0,"positive_prevalence":1.0}
    assert sum(step["excluded_orders"] for step in flow)==0

def test_order_level_joins_are_unique_and_deterministic(processed:Path)->None:
    con=duckdb.connect(); audit.build_audit_view(con); _,eligible=audit.waterfall(con); first=audit.join_checks(con,eligible); second=audit.join_checks(con,eligible); con.close()
    assert first==second
    assert first["eligible_unique_orders"]==first["item_aggregate_join_rows"]==first["payment_aggregate_join_rows"]==1

def test_forbidden_columns_cannot_enter_candidates()->None:
    assert audit.CANDIDATE_FEATURE_COLUMNS.isdisjoint(audit.FORBIDDEN_COLUMNS)

def test_schema_mismatch_fails(processed:Path)->None:
    (processed/"olist_orders_dataset.parquet").unlink(); duckdb.sql(f"COPY (SELECT 'x' AS wrong) TO '{(processed/'olist_orders_dataset.parquet').as_posix()}' (FORMAT PARQUET)")
    with pytest.raises(ValueError,match="Schema mismatch"): audit.verify_schemas(duckdb.connect())

def feature_contract()->dict:
    return audit.load_yaml(audit.DEFAULT_FEATURE_CONTRACT)

def resolution()->dict:
    return audit.load_yaml(audit.DEFAULT_RESOLUTION)

def modeling_config()->dict:
    return audit.load_config(audit.DEFAULT_CONFIG)

def test_target_equality_is_negative_and_missing_is_excluded(processed:Path)->None:
    path=processed/"olist_orders_dataset.parquet"; path.unlink(); con=duckdb.connect()
    con.execute(f"COPY (SELECT * FROM ({TABLES['olist_orders_dataset.parquet']}) UNION ALL SELECT 'o2','c1','delivered',TIMESTAMP '2017-01-01',TIMESTAMP '2017-01-01 01:00:00',TIMESTAMP '2017-01-02',TIMESTAMP '2017-01-03',TIMESTAMP '2017-01-03' UNION ALL SELECT 'o3','c1','delivered',TIMESTAMP '2017-01-01',TIMESTAMP '2017-01-01 01:00:00',TIMESTAMP '2017-01-02',NULL,TIMESTAMP '2017-01-03') TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close(); report=audit.phase1d_reassessment(modeling_config(),resolution(),feature_contract())
    assert report["target_checks"]["orders"]==2
    assert report["target_checks"]["positive"]==1 and report["target_checks"]["negative"]==1
    assert report["target_checks"]["equal_timestamp_negative_count"]==1
    assert report["target_checks"]["missing_target_components"]==0

def test_strict_features_exclude_target_reviews_status_and_shipping()->None:
    contract=feature_contract(); checks=audit.validate_feature_contract(contract); strict=set(contract["features"]["STRICT_CORE_ALLOWED"]); forbidden=set(contract["features"]["FORBIDDEN"]); target=set(contract["features"]["TARGET_CONSTRUCTION_ONLY"])
    assert checks["mutually_exclusive"] and not strict&(forbidden|target)
    assert {"order_delivered_customer_date","order_estimated_delivery_date"}==target
    assert {"review_score","review_comment_message","order_status","order_delivered_carrier_date","shipping_limit_date"} <= forbidden

def test_identifiers_are_separate_from_predictive_features()->None:
    features=feature_contract()["features"]
    assert set(features["IDENTIFIER_ONLY"]).isdisjoint(features["STRICT_CORE_ALLOWED"])
    assert set(features["IDENTIFIER_ONLY"]).isdisjoint(features["CONDITIONALLY_ALLOWED"])

def test_frozen_splits_are_chronological_and_exact()->None:
    split=resolution()["temporal_split"]
    assert split["train"]=={"start":"2016-09-01","end_exclusive":"2018-01-01"}
    assert split["validation"]=={"start":"2018-01-01","end_exclusive":"2018-05-01"}
    assert split["test"]=={"start":"2018-05-01","end_exclusive":"2018-09-01"}
    assert split["train"]["end_exclusive"]==split["validation"]["start"]
    assert split["validation"]["end_exclusive"]==split["test"]["start"]

def test_feature_classification_is_exhaustive_and_mutually_exclusive()->None:
    contract=feature_contract(); checks=audit.validate_feature_contract(contract)
    assert checks["classification_count"]==sum(len(contract["features"][tier]) for tier in audit.FEATURE_TIERS)

def test_explicit_criteria_generate_conditional_go(processed:Path)->None:
    report=audit.phase1d_reassessment(modeling_config(),resolution(),feature_contract())
    assert report["technical_offline_verdict"]=="CONDITIONAL GO"

def test_reassessment_does_not_modify_processed_fixture(processed:Path)->None:
    before={p.name:audit._hash(p) for p in processed.glob("*.parquet")}; audit.phase1d_reassessment(modeling_config(),resolution(),feature_contract()); after={p.name:audit._hash(p) for p in processed.glob("*.parquet")}
    assert before==after
