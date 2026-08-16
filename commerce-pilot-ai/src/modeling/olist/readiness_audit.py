"""Audit Olist target, cohort, joins, anomalies, and temporal splits without training."""

from __future__ import annotations
import argparse, hashlib, json, platform, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import duckdb, yaml

PROJECT_ROOT=Path(__file__).resolve().parents[3]
DEFAULT_CONFIG=PROJECT_ROOT/"configs"/"olist_modeling.example.yaml"
DEFAULT_RESOLUTION=PROJECT_ROOT/"configs"/"olist_readiness_resolution.yaml"
DEFAULT_FEATURE_CONTRACT=PROJECT_ROOT/"configs"/"olist_feature_contract_v1.yaml"
PROCESSED=PROJECT_ROOT/"data"/"processed"/"olist"
REQUIRED_SCHEMAS={
"olist_orders_dataset.parquet":["order_id","customer_id","order_status","order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"],
"olist_order_items_dataset.parquet":["order_id","order_item_id","product_id","seller_id","shipping_limit_date","price","freight_value"],
"olist_order_payments_dataset.parquet":["order_id","payment_sequential","payment_type","payment_installments","payment_value"],
"olist_customers_dataset.parquet":["customer_id","customer_unique_id","customer_zip_code_prefix","customer_city","customer_state"],
"olist_products_dataset.parquet":["product_id","product_category_name","product_name_lenght","product_description_lenght","product_photos_qty","product_weight_g","product_length_cm","product_height_cm","product_width_cm"],
"olist_sellers_dataset.parquet":["seller_id","seller_zip_code_prefix","seller_city","seller_state"],
"olist_geolocation_dataset.parquet":["geolocation_zip_code_prefix","geolocation_lat","geolocation_lng","geolocation_city","geolocation_state"],
"olist_order_reviews_dataset.parquet":["review_id","order_id","review_score","review_comment_title","review_comment_message","review_creation_date","review_answer_timestamp"],
"product_category_name_translation.parquet":["product_category_name","product_category_name_english"]}
FORBIDDEN_COLUMNS={"order_status","order_delivered_carrier_date","order_delivered_customer_date","review_id","review_score","review_comment_title","review_comment_message","review_creation_date","review_answer_timestamp"}
CANDIDATE_FEATURE_COLUMNS={"order_purchase_timestamp","order_approved_at","payment_type","payment_installments","payment_value","price","freight_value","product_category_name","product_weight_g","product_length_cm","product_height_cm","product_width_cm","customer_zip_code_prefix","customer_city","customer_state","seller_zip_code_prefix","seller_city","seller_state"}
FEATURE_TIERS=("STRICT_CORE_ALLOWED","CONDITIONALLY_ALLOWED","FORBIDDEN","TARGET_CONSTRUCTION_ONLY","IDENTIFIER_ONLY","UNRESOLVED")

def _path(name:str)->str: return (PROCESSED/name).as_posix()
def _hash(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def load_config(path:Path)->dict[str,Any]:
    if not path.is_file(): raise FileNotFoundError(f"Configuration not found: {path}")
    value=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict) or "experiment" not in value: raise ValueError("Invalid Olist modeling configuration.")
    return value
def load_yaml(path:Path)->dict[str,Any]:
    if not path.is_file(): raise FileNotFoundError(f"Required contract not found: {path}")
    value=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict): raise ValueError(f"Invalid YAML contract: {path}")
    return value
def validate_feature_contract(contract:dict[str,Any])->dict[str,Any]:
    features=contract.get("features")
    if not isinstance(features,dict) or set(features)!=set(FEATURE_TIERS): raise ValueError("Feature contract must define every required tier exactly once.")
    seen:dict[str,str]={}; duplicates=[]
    for tier in FEATURE_TIERS:
        values=features[tier]
        if not isinstance(values,list): raise ValueError(f"Feature tier {tier} must be a list.")
        for value in values:
            if value in seen: duplicates.append(value)
            seen[value]=tier
    if duplicates: raise ValueError(f"Feature classifications overlap: {sorted(set(duplicates))}")
    strict=set(features["STRICT_CORE_ALLOWED"]); forbidden=set(features["FORBIDDEN"]); target=set(features["TARGET_CONSTRUCTION_ONLY"]); identifiers=set(features["IDENTIFIER_ONLY"])
    if not strict: raise ValueError("Strict-core feature set is empty.")
    if strict & (forbidden|target|identifiers): raise ValueError("Strict-core feature contamination detected.")
    required_forbidden={"order_status","shipping_limit_date","order_delivered_carrier_date","review_score","review_comment_title","review_comment_message","review_creation_date","review_answer_timestamp"}
    if not required_forbidden <= forbidden: raise ValueError("Required post-approval/review columns are not all forbidden.")
    if {"order_delivered_customer_date","order_estimated_delivery_date"} != target: raise ValueError("Target-only columns are not exact.")
    return {"classification_count":len(seen),"mutually_exclusive":True,"strict_core_count":len(strict),"strict_forbidden_overlap":sorted(strict&forbidden),"strict_target_overlap":sorted(strict&target),"strict_identifier_overlap":sorted(strict&identifiers),"required_forbidden_enforced":True}
def verify_schemas(con:duckdb.DuckDBPyConnection)->dict[str,list[str]]:
    observed={}
    for name,expected in REQUIRED_SCHEMAS.items():
        path=PROCESSED/name
        if not path.is_file(): raise FileNotFoundError(f"Required processed file not found: {path}")
        columns=[row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_path(name)}')").fetchall()]
        if columns!=expected: raise ValueError(f"Schema mismatch for {name}: {columns}")
        observed[name]=columns
    return observed
def _stats(con:duckdb.DuckDBPyConnection,where:str)->dict[str,Any]:
    row=con.execute(f"SELECT count(*),count(*) FILTER (WHERE late_delivery),count(*) FILTER (WHERE not late_delivery),count(*) FILTER (WHERE late_delivery IS NULL) FROM audit_orders WHERE {where}").fetchone(); labeled=row[1]+row[2]
    return {"orders":row[0],"positive":row[1],"negative":row[2],"unlabeled":row[3],"positive_prevalence":round(row[1]/labeled,8) if labeled else None}
def build_audit_view(con:duckdb.DuckDBPyConnection)->None:
    con.execute(f"""CREATE TEMP VIEW audit_orders AS SELECT *, CASE WHEN order_delivered_customer_date IS NOT NULL AND order_estimated_delivery_date IS NOT NULL THEN order_delivered_customer_date > order_estimated_delivery_date END AS late_delivery, count(*) OVER(PARTITION BY order_id) AS order_id_count FROM read_parquet('{_path('olist_orders_dataset.parquet')}')""")
def waterfall(con:duckdb.DuckDBPyConnection)->tuple[list[dict[str,Any]],str]:
    rules=[("duplicated_order_id","order_id_count > 1"),("non_delivered_status","order_status <> 'delivered'"),("missing_payment_approval","order_approved_at IS NULL"),("missing_actual_customer_delivery","order_delivered_customer_date IS NULL"),("missing_estimated_delivery","order_estimated_delivery_date IS NULL"),("approval_before_purchase","order_approved_at < order_purchase_timestamp"),("delivery_before_purchase","order_delivered_customer_date < order_purchase_timestamp"),("estimate_before_approval","order_estimated_delivery_date < order_approved_at"),("carrier_before_approval","order_delivered_carrier_date IS NOT NULL AND order_delivered_carrier_date < order_approved_at"),("delivery_before_carrier","order_delivered_carrier_date IS NOT NULL AND order_delivered_customer_date < order_delivered_carrier_date")]
    original=con.execute("SELECT count(*) FROM audit_orders").fetchone()[0]; keep="TRUE"; output=[]
    for name,condition in rules:
        excluded=con.execute(f"SELECT count(*) FROM audit_orders WHERE {keep} AND ({condition})").fetchone()[0]; keep=f"({keep}) AND NOT ({condition})"; remaining=_stats(con,keep); output.append({"rule":name,"excluded_orders":excluded,"excluded_pct_original":round(100*excluded/original,6),"remaining":remaining})
    return output,keep
def anomaly_counts(con:duckdb.DuckDBPyConnection)->dict[str,int]:
    checks={"missing_approval":"order_approved_at IS NULL","carrier_before_approval":"order_delivered_carrier_date < order_approved_at","delivery_before_carrier":"order_delivered_customer_date < order_delivered_carrier_date","delivery_before_purchase":"order_delivered_customer_date < order_purchase_timestamp","estimate_before_approval":"order_estimated_delivery_date < order_approved_at","approval_before_purchase":"order_approved_at < order_purchase_timestamp","carrier_before_purchase":"order_delivered_carrier_date < order_purchase_timestamp","estimate_before_purchase":"order_estimated_delivery_date < order_purchase_timestamp","unparsable_timestamps":"FALSE"}
    return {name:con.execute(f"SELECT count(*) FROM audit_orders WHERE {condition}").fetchone()[0] for name,condition in checks.items()}
def split_stats(con:duckdb.DuckDBPyConnection,eligible:str,config:dict[str,Any])->dict[str,Any]:
    split=config["temporal_split"]; result={}; train_start=split["train"]["start"]; train_end=split["train"]["end_exclusive"]
    for name in ("train","validation","test"):
        spec=split[name]; where=f"{eligible} AND order_approved_at >= TIMESTAMP '{spec['start']}' AND order_approved_at < TIMESTAMP '{spec['end_exclusive']}'"; values=_stats(con,where); bounds=con.execute(f"SELECT min(order_approved_at),max(order_approved_at) FROM audit_orders WHERE {where}").fetchone(); values.update({"configured_start":spec["start"],"configured_end_exclusive":spec["end_exclusive"],"observed_min":str(bounds[0]),"observed_max":str(bounds[1])});
        if name!="train":
            entity_sql=f"""WITH train_orders AS (SELECT order_id,customer_id FROM audit_orders WHERE {eligible} AND order_approved_at >= TIMESTAMP '{train_start}' AND order_approved_at < TIMESTAMP '{train_end}'), current_orders AS (SELECT order_id,customer_id FROM audit_orders WHERE {where}), train_items AS (SELECT DISTINCT i.product_id,i.seller_id,p.product_category_name FROM read_parquet('{_path('olist_order_items_dataset.parquet')}') i LEFT JOIN read_parquet('{_path('olist_products_dataset.parquet')}') p USING(product_id) JOIN train_orders USING(order_id)), current_items AS (SELECT DISTINCT i.product_id,i.seller_id,p.product_category_name FROM read_parquet('{_path('olist_order_items_dataset.parquet')}') i LEFT JOIN read_parquet('{_path('olist_products_dataset.parquet')}') p USING(product_id) JOIN current_orders USING(order_id)), train_customers AS (SELECT DISTINCT c.customer_unique_id FROM read_parquet('{_path('olist_customers_dataset.parquet')}') c JOIN train_orders USING(customer_id)), current_customers AS (SELECT DISTINCT c.customer_unique_id FROM read_parquet('{_path('olist_customers_dataset.parquet')}') c JOIN current_orders USING(customer_id)) SELECT (SELECT count(*) FROM current_customers x ANTI JOIN train_customers t USING(customer_unique_id)),(SELECT count(*) FROM (SELECT DISTINCT seller_id FROM current_items) x ANTI JOIN (SELECT DISTINCT seller_id FROM train_items) t USING(seller_id)),(SELECT count(*) FROM (SELECT DISTINCT product_id FROM current_items) x ANTI JOIN (SELECT DISTINCT product_id FROM train_items) t USING(product_id)),(SELECT count(*) FROM (SELECT DISTINCT product_category_name FROM current_items WHERE product_category_name IS NOT NULL) x ANTI JOIN (SELECT DISTINCT product_category_name FROM train_items WHERE product_category_name IS NOT NULL) t USING(product_category_name))"""; unseen=con.execute(entity_sql).fetchone(); values["unseen_vs_training"]={"customers":unseen[0],"sellers":unseen[1],"products":unseen[2],"categories":unseen[3]}
        result[name]=values
    result["outside_frozen_windows"]=_stats(con,f"{eligible} AND NOT (order_approved_at >= TIMESTAMP '{split['train']['start']}' AND order_approved_at < TIMESTAMP '{split['test']['end_exclusive']}')")
    return result
def join_checks(con:duckdb.DuckDBPyConnection,eligible:str)->dict[str,Any]:
    eligible_count=con.execute(f"SELECT count(*) FROM audit_orders WHERE {eligible}").fetchone()[0]
    item_rows=con.execute(f"WITH e AS (SELECT order_id FROM audit_orders WHERE {eligible}), a AS (SELECT order_id,count(*) item_count,count(distinct product_id) product_count,count(distinct seller_id) seller_count,sum(price) total_price,sum(freight_value) total_freight,max(freight_value) max_freight,avg(freight_value) avg_freight FROM read_parquet('{_path('olist_order_items_dataset.parquet')}') GROUP BY order_id) SELECT count(*),count(distinct e.order_id) FROM e LEFT JOIN a USING(order_id)").fetchone()
    payment_rows=con.execute(f"WITH e AS (SELECT order_id FROM audit_orders WHERE {eligible}), a AS (SELECT order_id,sum(payment_value) total_payment,max(payment_installments) max_installments FROM read_parquet('{_path('olist_order_payments_dataset.parquet')}') GROUP BY order_id) SELECT count(*),count(distinct e.order_id) FROM e LEFT JOIN a USING(order_id)").fetchone()
    multi=con.execute(f"""WITH e AS (SELECT order_id FROM audit_orders WHERE {eligible}), ia AS (SELECT order_id,count(*) n_items,count(distinct seller_id) n_sellers FROM read_parquet('{_path('olist_order_items_dataset.parquet')}') GROUP BY 1), pa AS (SELECT order_id,count(*) n_payments FROM read_parquet('{_path('olist_order_payments_dataset.parquet')}') GROUP BY 1) SELECT count(*) FILTER(WHERE n_items>1),count(*) FILTER(WHERE n_sellers>1),count(*) FILTER(WHERE n_payments>1),count(*) FILTER(WHERE n_items IS NULL),count(*) FILTER(WHERE n_payments IS NULL) FROM e LEFT JOIN ia USING(order_id) LEFT JOIN pa USING(order_id)""").fetchone()
    geo=con.execute(f"SELECT count(*),count(distinct geolocation_zip_code_prefix) FROM read_parquet('{_path('olist_geolocation_dataset.parquet')}')").fetchone()
    return {"eligible_unique_orders":eligible_count,"item_aggregate_join_rows":item_rows[0],"item_aggregate_distinct_orders":item_rows[1],"payment_aggregate_join_rows":payment_rows[0],"payment_aggregate_distinct_orders":payment_rows[1],"orders_with_multiple_items":multi[0],"orders_with_multiple_sellers":multi[1],"orders_with_multiple_payments":multi[2],"orders_without_items":multi[3],"orders_without_payments":multi[4],"geolocation_rows":geo[0],"geolocation_distinct_zip_prefixes":geo[1],"geolocation_duplicate_zip_rows_beyond_distinct":geo[0]-geo[1],"candidate_forbidden_overlap":sorted(CANDIDATE_FEATURE_COLUMNS & FORBIDDEN_COLUMNS)}
def audit(config:dict[str,Any])->dict[str,Any]:
    con=duckdb.connect()
    try:
        schemas=verify_schemas(con); build_audit_view(con); flow,eligible=waterfall(con); monthly=[{"month":str(r[0]),"orders":r[1],"positive":r[2],"prevalence":round(r[2]/r[1],8)} for r in con.execute(f"SELECT date_trunc('month',order_approved_at),count(*),count(*) FILTER(WHERE late_delivery) FROM audit_orders WHERE {eligible} GROUP BY 1 ORDER BY 1").fetchall()]
        return {"experiment":config["experiment"],"target":config["target"],"schemas":schemas,"original_orders":con.execute("SELECT count(*) FROM audit_orders").fetchone()[0],"waterfall":flow,"eligible":_stats(con,eligible),"timestamp_anomalies":anomaly_counts(con),"monthly_eligible_distribution":monthly,"splits":split_stats(con,eligible,config),"join_aggregation_checks":join_checks(con,eligible),"processed_hashes":{name:_hash(PROCESSED/name) for name in REQUIRED_SCHEMAS},"environment":{"python":platform.python_version(),"duckdb":duckdb.__version__}}
    finally: con.close()
def phase1d_reassessment(modeling:dict[str,Any],resolution:dict[str,Any],feature_contract:dict[str,Any])->dict[str,Any]:
    base=audit(modeling); checks=validate_feature_contract(feature_contract); con=duckdb.connect()
    try:
        build_audit_view(con); _,eligible=waterfall(con)
        target_row=con.execute(f"""WITH eligible_orders AS (SELECT * EXCLUDE(late_delivery) FROM audit_orders WHERE {eligible}), labeled AS (SELECT *,order_delivered_customer_date > order_estimated_delivery_date AS late_delivery FROM eligible_orders) SELECT count(*),count(*) FILTER(WHERE late_delivery),count(*) FILTER(WHERE NOT late_delivery),count(*) FILTER(WHERE order_delivered_customer_date=order_estimated_delivery_date),count(*) FILTER(WHERE order_delivered_customer_date IS NULL OR order_estimated_delivery_date IS NULL) FROM labeled""").fetchone()
        target_checks={"generated_after_eligibility":True,"orders":target_row[0],"positive":target_row[1],"negative":target_row[2],"equal_timestamp_negative_count":target_row[3],"missing_target_components":target_row[4],"actual_delivery_in_strict_inputs":False,"estimated_delivery_in_strict_inputs":False}
    finally: con.close()
    criteria=resolution["readiness_gate_criteria"]
    technical_keys=("reproducible_target_and_cohort","frozen_chronological_split","nontrivial_strict_core","forbidden_features_programmatically_blocked","leakage_safe_preprocessing_protocol")
    if not all(criteria.get(key) is True for key in technical_keys): verdict="NO-GO"
    elif resolution["open_assumptions"]: verdict="CONDITIONAL GO"
    else: verdict="GO"
    declared=resolution["reassessment"]["technical_offline_verdict"]
    if verdict!=declared: raise ValueError(f"Derived verdict {verdict} disagrees with declared verdict {declared}.")
    return {"phase":"1D","previous_technical_verdict":resolution["reassessment"]["previous_technical_verdict"],"technical_offline_verdict":verdict,"legal_licensing_verdict":resolution["reassessment"]["legal_licensing_verdict"],"production_verdict":resolution["reassessment"]["production_verdict"],"egyptian_market_external_validity_verdict":resolution["reassessment"]["egyptian_market_external_validity_verdict"],"target_semantics_status":resolution["target"]["semantics_status"],"scientific_claim":resolution["prediction"]["scientific_claim"],"target_checks":target_checks,"feature_contract_checks":checks,"strict_core_features":feature_contract["features"]["STRICT_CORE_ALLOWED"],"conditionally_allowed_features":feature_contract["features"]["CONDITIONALLY_ALLOWED"],"forbidden_features":feature_contract["features"]["FORBIDDEN"],"target_only_columns":feature_contract["features"]["TARGET_CONSTRUCTION_ONLY"],"identifier_only_columns":feature_contract["features"]["IDENTIFIER_ONLY"],"unresolved_features":feature_contract["features"]["UNRESOLVED"],"cohort":base["eligible"],"splits":base["splits"],"processed_hashes":base["processed_hashes"],"evidence_status_summary":{"repository_cohort_and_schema":"VERIFIED_FROM_REPOSITORY","owner_dataset_identity_and_license":"VERIFIED_FROM_PRIMARY_SOURCE","stored_estimate_as_approval_time_immutable_promise":"UNVERIFIED","retrospective_stored_estimate_target_validity":"SUPPORTED_INFERENCE","strict_calendar_feature_availability":"SUPPORTED_INFERENCE","business_costs_and_recall_requirement":"UNVERIFIED"},"resolved_blockers":["Future outcome is separated from model input and may construct a retrospective label.","A non-trivial strict core uses only purchase/approval timing facts.","Missing business costs block operational threshold claims, not threshold-independent offline ranking evaluation.","Snapshot-dependent inputs are isolated to a sensitivity contract."],"open_assumptions":resolution["open_assumptions"],"readiness_gate_criteria":criteria,"environment":base["environment"]}
def phase1d_markdown(report:dict[str,Any])->str:
    c=report["cohort"]; lines=["# Generated Phase 1D Olist Readiness Reassessment","",f"Technical offline verdict: **{report['technical_offline_verdict']}**",f"Legal/licensing verdict: **{report['legal_licensing_verdict']}**",f"Production verdict: **{report['production_verdict']}**",f"Egyptian-market external validity: **{report['egyptian_market_external_validity_verdict']}**","",f"Target status: `{report['target_semantics_status']}`",f"Cohort: {c['orders']:,} orders; {c['positive']:,} positive; {c['negative']:,} negative; prevalence {c['positive_prevalence']:.8f}.","","Strict-core features:",""]
    lines += [f"- `{value}`" for value in report["strict_core_features"]]
    lines += ["","This report validates contracts only. It contains no model, feature matrix, fitted preprocessing, or predictions.",""]
    return "\n".join(lines)
def markdown(report:dict[str,Any])->str:
    lines=["# Generated Olist Readiness Audit","",f"Original orders: {report['original_orders']:,}. Eligible orders: {report['eligible']['orders']:,}.","","## Exclusion waterfall","","| Rule | Excluded | % original | Remaining | Positive | Negative | Prevalence |","|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["waterfall"]: r=row["remaining"]; lines.append(f"| {row['rule']} | {row['excluded_orders']:,} | {row['excluded_pct_original']:.6f}% | {r['orders']:,} | {r['positive']:,} | {r['negative']:,} | {r['positive_prevalence']} |")
    lines += ["","## Frozen splits","","| Split | Start | End exclusive | Orders | Positive | Negative | Prevalence |","|---|---|---|---:|---:|---:|---:|"]
    for name in ("train","validation","test"): s=report["splits"][name]; lines.append(f"| {name} | {s['configured_start']} | {s['configured_end_exclusive']} | {s['orders']:,} | {s['positive']:,} | {s['negative']:,} | {s['positive_prevalence']} |")
    return "\n".join(lines)+"\n"
def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--resolution",type=Path,default=DEFAULT_RESOLUTION); parser.add_argument("--feature-contract",type=Path,default=DEFAULT_FEATURE_CONTRACT); parser.add_argument("--output-dir",type=Path); parser.add_argument("--phase1d",action="store_true"); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--force",action="store_true"); args=parser.parse_args()
    try:
        config=load_config(args.config.resolve())
        if args.phase1d:
            resolution=load_yaml(args.resolution.resolve()); feature_contract=load_yaml(args.feature_contract.resolve()); report=phase1d_reassessment(config,resolution,feature_contract); json_output=(args.output_dir/"phase1d_readiness_reassessment.json" if args.output_dir else PROJECT_ROOT/resolution["output"]["json"]).resolve(); md_output=(args.output_dir/"phase1d_readiness_reassessment.md" if args.output_dir else PROJECT_ROOT/resolution["output"]["markdown"]).resolve()
            if args.dry_run: print(json.dumps(report,indent=2,default=str)); return 0
            if (json_output.exists() or md_output.exists()) and not args.force: raise FileExistsError(f"Refusing to overwrite Phase 1D report output. Use --force explicitly.")
            json_output.parent.mkdir(parents=True,exist_ok=True); report["execution_timestamp_utc"]=datetime.now(timezone.utc).isoformat(); json_output.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8"); md_output.write_text(phase1d_markdown(report),encoding="utf-8"); print(f"Wrote Phase 1D reassessment to {json_output.parent}"); return 0
        report=audit(config); output=(args.output_dir or PROJECT_ROOT/config["outputs"]["audit_directory"]).resolve()
        if args.dry_run: print(json.dumps(report,indent=2,default=str)); return 0
        if output.exists() and any(output.iterdir()) and not args.force: raise FileExistsError(f"Refusing to overwrite audit output: {output}. Use --force explicitly.")
        staging=output.with_name(output.name+".staging");
        if staging.exists(): shutil.rmtree(staging)
        staging.mkdir(parents=True); report["execution_timestamp_utc"]=datetime.now(timezone.utc).isoformat(); (staging/"readiness_audit.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8"); (staging/"readiness_audit.md").write_text(markdown(report),encoding="utf-8")
        if output.exists(): shutil.rmtree(output)
        staging.replace(output); print(f"Wrote Olist readiness audit to {output}"); return 0
    except (OSError,ValueError,duckdb.Error) as exc: print(f"ERROR: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
