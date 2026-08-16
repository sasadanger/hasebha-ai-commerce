"""Phase 2B development-only sensitivity CLI; deliberately has no Test route."""
from pathlib import Path
import argparse,hashlib,json,platform,time,os
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd,yaml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from .expanded_feature_builder import build_development,predictors,STRICT,EXPANDED
from .temporal_validation import fold_indices,fold_hash,basic,paired_bootstrap
from .evaluation import metrics,prediction_hash
ROOT=Path(__file__).resolve().parents[3]
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256();
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def chash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def write(p,v): Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,default=str),encoding="utf-8")
def load(p): return yaml.safe_load(Path(p).read_text(encoding="utf-8"))
def protected():
 paths=[ROOT/"reports/generated/olist/phase2a/selection_manifest.json",ROOT/"reports/generated/olist/phase2a/test_access_ledger.json"]
 paths += list((ROOT/"artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/predictions").glob("test_*"))
 return {str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in paths}
def gate(c):
 mpath=ROOT/"reports/generated/olist/phase2a/selection_manifest.json"; m=json.loads(mpath.read_text()); embedded=m.pop("manifest_sha256")
 if chash(m)!=embedded or embedded!=c["parent_manifest"]["canonical_sha256"] or sha(mpath)!=c["parent_manifest"]["physical_sha256"]:raise ValueError("Phase 2A manifest verification failed")
 ledger=json.loads((ROOT/"reports/generated/olist/phase2a/test_access_ledger.json").read_text())
 if ledger["status"]!="CONSUMED" or ledger["access_count"]!=1:raise ValueError("Phase 2A Test ledger invalid")
 return ledger,protected()
def audit(c,out):
 ledger,pro=gate(c); processed=ROOT/"data/processed/olist"; d=build_development(processed)
 candidates=[]
 approved=set(EXPANDED)
 source={"payment_record_count":"payments","max_payment_installments":"payments","total_payment_value":"payments","item_count":"items","unique_product_count":"items","unique_seller_count":"items","total_item_price":"items","total_freight_value":"items","mean_freight_value":"items","max_freight_value":"items","product_category_diversity":"items+products","mean_product_weight_g":"items+products","mean_product_length_cm":"items+products","mean_product_height_cm":"items+products","mean_product_width_cm":"items+products"}
 for name in yaml.safe_load((ROOT/"configs/olist_feature_contract_v1.yaml").read_text())["features"]["CONDITIONALLY_ALLOWED"]:
  concrete=[x for x in EXPANDED if (name==x or name=="payment_type_indicators" and False or name=="product_dimension_aggregates" and "product_" in x and "weight" not in x or name=="product_weight_aggregates" and "weight" in x)]
  status="CONDITIONALLY_ALLOWED_RETROSPECTIVE_SENSITIVITY" if concrete or name in approved else "UNVERIFIED_ASOF_AVAILABILITY"
  candidates.append({"contract_candidate":name,"concrete_features":concrete or ([name] if name in EXPANDED else []),"source_table":sorted({source.get(x,"unimplemented") for x in concrete or [name]}),"prediction_timestamp":"order_approved_at","availability_evidence":"final export presence only; immutable approval-time snapshot not verified","snapshot_immutability_verified":False,"classification":status,"primary_asof_allowed":False,"retrospective_allowed":bool(concrete or name in approved),"leakage_risk":"snapshot revision","target_proxy_risk":"none identified for admitted aggregates","post_approval_risk":"excluded when present","temporal_aggregate_risk":"historical families excluded","reason":"Repository Contract B authorizes retrospective upper-bound only" if status.startswith("CONDITIONALLY") else "Safe snapshot/event-time implementation not established"})
 a={"created_utc":now(),"classification":"Retrospective expanded-feature sensitivity with unverified as-of semantics","primary_asof_whitelist":[],"retrospective_expanded_whitelist":EXPANDED,"whitelist_sha256":chash(EXPANDED),"candidates":candidates,"missingness":{x:float(d[x].isna().mean()) for x in EXPANDED},"join_rule":"aggregate to unique order_id before left join","development_rows":len(d),"max_development_timestamp":str(d.approved_at.max()),"test_start":"2018-05-01","protected_hashes_before":pro,"ledger_before":ledger};write(out/"feature_availability_audit.json",a);return d,a
def model(family,p,expanded):
 if family=="dummy":return DummyClassifier(strategy="prior")
 prep=[("imputer",SimpleImputer(strategy="median"))]
 if family=="logistic_regression":prep += [("scale",StandardScaler()),("model",LogisticRegression(C=p["C"],solver=p["solver"],penalty=p["penalty"],max_iter=p["max_iter"],random_state=42))]
 elif family=="catboost":prep += [("model",CatBoostClassifier(loss_function="Logloss",random_seed=42,thread_count=1,verbose=False,allow_writing_files=False,**p))]
 else:prep += [("model",LGBMClassifier(objective="binary",random_state=42,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1,**p))]
 return Pipeline(prep)
def development(c,d,out,art):
 if (out/"development_metrics.json").exists():raise FileExistsError("Phase 2B outputs exist; overwrite prohibited")
 families=list(c["models"]); records=[]; preds=[]; saved={}; foldman=[]
 for fs in c["folds"]:
  tr,va=fold_indices(d,fs); foldman.append({**fs,"train_rows":len(tr),"validation_rows":len(va),"validation_positive":int(d.iloc[va].late_delivery.sum()),"validation_negative":int((d.iloc[va].late_delivery==0).sum()),"prevalence":float(d.iloc[va].late_delivery.mean()),"train_hash":fold_hash(d.iloc[tr].order_id.tolist()),"validation_hash":fold_hash(d.iloc[va].order_id.tolist()),"max_train_timestamp":str(d.iloc[tr].approved_at.max()),"min_validation_timestamp":str(d.iloc[va].approved_at.min())})
  for family in families:
   for contract in ("strict","expanded"):
    X,y=predictors(d,contract); mo=model(family,c["models"][family],contract=="expanded");t=time.perf_counter();mo.fit(X.iloc[tr],y.iloc[tr]);dur=time.perf_counter()-t;s=mo.predict_proba(X.iloc[va])[:,1]; met=metrics(y.iloc[va],s);records.append({"fold_id":fs["id"],"model_name":family,"feature_contract":contract,"metrics":met,"duration_seconds":dur,"configuration":c["models"][family]});thr=met["research_threshold"]["threshold"]
    preds.append(pd.DataFrame({"order_id":d.iloc[va].order_id,"fold_id":fs["id"],"model_name":family,"feature_contract":contract,"y_true":y.iloc[va].to_numpy(),"y_score":s,"prediction_at_0_5":(s>=.5).astype("int8"),"prediction_at_research_threshold":(s>=thr).astype("int8")}));saved[(family,contract)]=mo
 pred=pd.concat(preds,ignore_index=True); art.mkdir(parents=True,exist_ok=True); predpath=art/"development_predictions.parquet";pred.to_parquet(predpath,index=False)
 modeldir=art/"models";modeldir.mkdir(exist_ok=True)
 for (f,k),mo in saved.items():joblib.dump(mo,modeldir/f"{f}_{k}_last_fold.joblib")
 paired={}
 for f in families:
  a=pred[(pred.model_name==f)&(pred.feature_contract=="strict")].sort_values(["fold_id","order_id"]);b=pred[(pred.model_name==f)&(pred.feature_contract=="expanded")].sort_values(["fold_id","order_id"])
  pb=paired_bootstrap(a.y_true.to_numpy(),a.y_score.to_numpy(),b.y_score.to_numpy(),c["metrics"]["bootstrap_resamples"],c["metrics"]["bootstrap_seed"]);improved=sum(next(x for x in records if x["fold_id"]==fold and x["model_name"]==f and x["feature_contract"]=="expanded")["metrics"]["average_precision"]>next(x for x in records if x["fold_id"]==fold and x["model_name"]==f and x["feature_contract"]=="strict")["metrics"]["average_precision"] for fold in [z["id"] for z in c["folds"]]);ci=pb["delta_ap"];label="SUPPORTED_INCREMENTAL_SIGNAL" if ci["lower"]>0 and improved>=2 else "INCONCLUSIVE_INCREMENTAL_SIGNAL" if ci["point"]>0 else "DEGRADED_PERFORMANCE" if improved==0 else "NO_INCREMENTAL_SIGNAL";paired[f]={**pb,"folds_improved":improved,"evidence_label":label}
 write(out/"development_fold_manifest.json",foldman);write(out/"development_metrics.json",records);write(out/"paired_comparisons.json",paired);write(out/"prediction_manifest.json",{"path":str(predpath.relative_to(ROOT)).replace("\\","/"),"sha256":sha(predpath),"rows":len(pred)});print(json.dumps({"status":"DEVELOPMENT_COMPLETE","test_access":"NOT_PERFORMED","evidence":{f:v["evidence_label"] for f,v in paired.items()}},indent=2))
def reproduce(c,d,out):
 old=json.loads((out/"development_metrics.json").read_text());checks=[]
 for r in old:
  fs=next(x for x in c["folds"] if x["id"]==r["fold_id"]);tr,va=fold_indices(d,fs);X,y=predictors(d,r["feature_contract"]);mo=model(r["model_name"],r["configuration"],r["feature_contract"]=="expanded");mo.fit(X.iloc[tr],y.iloc[tr]);s=mo.predict_proba(X.iloc[va])[:,1];checks.append({"fold_id":r["fold_id"],"model_name":r["model_name"],"feature_contract":r["feature_contract"],"prediction_hash_match":prediction_hash(s)==r["metrics"]["prediction_hash"]})
 status="PASS" if all(x["prediction_hash_match"] for x in checks) else "FAIL";write(out/"reproducibility_report.json",{"status":status,"timestamp_utc":now(),"checks":checks,"environment":{"python":platform.python_version(),"platform":platform.platform(),"cpu":platform.processor(),"logical_cpu":os.cpu_count(),"gpu":"not used","threads":1,"seed":42}});print(status)
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--stage",required=True,choices=["audit","dry-run","development","reproduce"]);a=p.parse_args();c=load(a.config);out=ROOT/c["paths"]["report_dir"];art=ROOT/c["paths"]["artifact_dir"]
 if a.stage=="audit":
  if out.exists():
   ledger,_=gate(c); au=json.loads((out/"feature_availability_audit.json").read_text()); print(json.dumps({"status":"CORRECTION_AUDIT_PASS","ledger":ledger["status"],"access_count":ledger["access_count"],"whitelist_sha256":au["whitelist_sha256"],"test_opened":False},indent=2))
  else:
   out.mkdir(parents=True);d,au=audit(c,out);print(json.dumps({"status":"AUDIT_LOCKED","whitelist_sha256":au["whitelist_sha256"],"primary_asof_count":0,"retrospective_count":len(EXPANDED)},indent=2))
 elif a.stage=="dry-run":
  ledger,_=gate(c);d=build_development(ROOT/"data/processed/olist");folds=[{"id":x["id"],"train":len(fold_indices(d,x)[0]),"validation":len(fold_indices(d,x)[1])} for x in c["folds"]];print(json.dumps({"status":"DRY_RUN_PASS","rows":len(d),"positive":int(d.late_delivery.sum()),"max_timestamp":str(d.approved_at.max()),"test_opened":False,"ledger":ledger,"folds":folds},indent=2))
 elif a.stage=="development":d=build_development(ROOT/"data/processed/olist");development(c,d,out,art)
 else:d=build_development(ROOT/"data/processed/olist");reproduce(c,d,out)
if __name__=="__main__":main()
