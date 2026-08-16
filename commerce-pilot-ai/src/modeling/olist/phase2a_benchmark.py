"""Controlled two-stage Olist Phase 2A benchmark."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .strict_feature_builder import FEATURES, build_split, matrix
from .evaluation import metrics, bootstrap_ci

ROOT=Path(__file__).resolve().parents[3]; CONFIG=ROOT/"configs/olist_phase2a_benchmark.yaml"
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def canonical_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def write_json(path,value): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(value,indent=2,default=str),encoding="utf-8")
def load(): return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
def paths(c): return ROOT/c["paths"]["report_dir"],ROOT/c["paths"]["artifact_dir"]
def verify(c):
    order=ROOT/c["dataset"]["orders_path"]
    if sha(order)!=c["dataset"]["orders_sha256"]: raise ValueError("Processed orders SHA-256 changed")
    if yaml.safe_load((ROOT/"configs/olist_readiness_resolution.yaml").read_text())["reassessment"]["technical_offline_verdict"] not in ("CONDITIONAL GO","GO"): raise ValueError("Phase 1D gate is closed")
    fc=yaml.safe_load((ROOT/"configs/olist_feature_contract_v1.yaml").read_text())
    if c["features"]!=fc["features"]["STRICT_CORE_ALLOWED"] or FEATURES!=c["features"]: raise ValueError("Strict feature contract mismatch")
    return order
def split(c,name,order):
    s=c["splits"][name]; frame=build_split(order,s["start"],s["end_exclusive"])
    observed=(len(frame),int(frame.late_delivery.sum()),int((frame.late_delivery==0).sum()))
    expected=(s["rows"],s["positive"],s["negative"])
    if observed!=expected: raise ValueError(f"{name} count mismatch: {observed} != {expected}")
    return frame
def package_versions():
    import importlib.metadata as md
    return {x:md.version(x) for x in ["numpy","pandas","pyarrow","scikit-learn","catboost","lightgbm","matplotlib","duckdb"]}
def make_model(family,p,seed,iters=None):
    if family=="dummy": return DummyClassifier(strategy="prior")
    if family=="logistic_regression": return Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(C=p["C"],solver=p["solver"],penalty=p["penalty"],max_iter=p["max_iter"],random_state=seed))])
    if family=="catboost": return CatBoostClassifier(loss_function="Logloss",random_seed=seed,thread_count=1,verbose=False,allow_writing_files=False,depth=p["depth"],learning_rate=p["learning_rate"],iterations=iters or p["iterations"])
    q=dict(p); q["n_estimators"]=iters or q["n_estimators"]
    return LGBMClassifier(objective="binary",random_state=seed,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1,**q)
def fit_score(family,p,X,y,Xv,yv,seed,early=True):
    model=make_model(family,p,seed); start=time.perf_counter()
    if family=="catboost" and early: model.fit(X,y,eval_set=(Xv,yv),early_stopping_rounds=40,verbose=False)
    elif family=="lightgbm" and early: model.fit(X,y,eval_set=[(Xv,yv)],callbacks=[early_stopping(40,verbose=False),log_evaluation(0)])
    else: model.fit(X,y)
    duration=time.perf_counter()-start; scores=model.predict_proba(Xv)[:,1]
    best=(model.get_best_iteration()+1 if family=="catboost" else model.best_iteration_ if family=="lightgbm" else None)
    return model,scores,duration,best
def save_predictions(path,frame,scores):
    pd.DataFrame({"order_id":frame.order_id,"late_delivery":frame.late_delivery.astype("int8"),"score":scores}).to_parquet(path,index=False)
def source_hashes(): return {str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in [ROOT/"src/modeling/olist/strict_feature_builder.py",ROOT/"src/modeling/olist/evaluation.py",Path(__file__),CONFIG]}
def validation(c):
    report,art=paths(c)
    if (report/"selection_manifest.json").exists(): raise FileExistsError("Locked selection manifest already exists")
    order=verify(c); train=split(c,"train",order); val=split(c,"validation",order); X,y=matrix(train); Xv,yv=matrix(val)
    results={}; selected={}; pred_dir=art/"predictions"; pred_dir.mkdir(parents=True,exist_ok=True)
    for family,grid in c["models"].items():
        runs=[]
        for rank,p in enumerate(grid):
            try:
                model,s,d,b=fit_score(family,p,X,y,Xv,yv,c["experiment"]["random_seed"]); m=metrics(yv,s)
                runs.append({"rank":rank,"configuration":p,"metrics":m,"duration_seconds":d,"fitted_iterations":b,"status":"PASS"})
            except Exception as e: runs.append({"rank":rank,"configuration":p,"status":"FAIL","error":repr(e)})
        passing=[r for r in runs if r["status"]=="PASS"]
        if not passing: raise RuntimeError(f"All {family} candidates failed")
        best=sorted(passing,key=lambda r:(-r["metrics"]["average_precision"],r["metrics"]["brier_score"],r["rank"]))[0]
        _,s1,_,_=fit_score(family,best["configuration"],X,y,Xv,yv,c["experiment"]["random_seed"])
        _,s2,_,_=fit_score(family,best["configuration"],X,y,Xv,yv,c["experiment"]["random_seed"])
        if not np.allclose(s1,s2,rtol=0,atol=c["determinism"]["prediction_tolerance"]): raise RuntimeError(f"{family} reproducibility failed")
        save_predictions(pred_dir/f"validation_{family}.parquet",val,s1)
        results[family]=runs; selected[family]={**best,"reproduction_prediction_hash":metrics(yv,s2)["prediction_hash"]}
    champion=sorted(selected,key=lambda f:(-selected[f]["metrics"]["average_precision"],selected[f]["metrics"]["brier_score"],list(c["models"]).index(f)))[0]
    manifest={"experiment_id":c["experiment"]["id"],"created_utc":now(),"status":"LOCKED_BEFORE_TEST_ACCESS","test_metrics_available_during_selection":False,"champion":champion,"selected":selected,"candidate_results":results,"strict_features":FEATURES,"strict_feature_hash":canonical_hash(FEATURES),"configuration_hash":sha(CONFIG),"source_hashes":source_hashes(),"dataset_hashes":{"processed_orders":sha(order)},"random_seed":c["experiment"]["random_seed"],"package_versions":package_versions(),"preprocessing":c["preprocessing"]}
    manifest["manifest_sha256"]=canonical_hash(manifest); write_json(report/"selection_manifest.json",manifest)
    repro={"status":"PASS","checked_utc":now(),"tolerance":c["determinism"]["prediction_tolerance"],"families":{f:{"prediction_hash_match":selected[f]["metrics"]["prediction_hash"]==selected[f]["reproduction_prediction_hash"]} for f in selected},"environment":{"python":platform.python_version(),"os":platform.platform(),"cpu":platform.processor(),"logical_cpu":os.cpu_count(),"gpu":"not used; CPU policy","threads":1}}
    write_json(report/"reproducibility_report.json",repro); write_json(report/"validation_metrics.json",results)
    print(json.dumps({"status":"VALIDATION_LOCKED","champion":champion,"manifest_sha256":manifest["manifest_sha256"]},indent=2))
def final_test(c,manifest_hash):
    report,art=paths(c); mp=report/"selection_manifest.json"; ledger=report/"test_access_ledger.json"
    if ledger.exists(): raise FileExistsError("Final Test access ledger already exists; re-evaluation prohibited")
    manifest=json.loads(mp.read_text(encoding="utf-8")); stored=manifest.pop("manifest_sha256")
    if stored!=canonical_hash(manifest) or stored!=manifest_hash: raise ValueError("Locked manifest hash verification failed")
    access=now(); write_json(ledger,{"status":"EVALUATION_IN_PROGRESS","test_access_began_utc":access,"manifest_sha256":stored})
    order=verify(c); train=split(c,"train",order); val=split(c,"validation",order); test=split(c,"test",order)
    dev=pd.concat([train,val],ignore_index=True); X,y=matrix(dev); Xt,yt=matrix(test); outputs={}; pred_dir=art/"predictions"; model_dir=art/"models"; plot_dir=art/"plots"; model_dir.mkdir(parents=True,exist_ok=True); plot_dir.mkdir(parents=True,exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for family,sel in manifest["selected"].items():
        p=sel["configuration"]; it=sel["fitted_iterations"]; model=make_model(family,p,c["experiment"]["random_seed"],it); start=time.perf_counter(); model.fit(X,y); duration=time.perf_counter()-start; s=model.predict_proba(Xt)[:,1]; threshold=sel["metrics"]["research_threshold"]["threshold"]; m=metrics(yt,s,threshold); m["confidence_intervals_95"]=bootstrap_ci(yt,s,c["experiment"]["random_seed"],c["evaluation"]["bootstrap_replicates"]); m["training_duration_seconds"]=duration
        pp=pred_dir/f"test_{family}.parquet"; save_predictions(pp,test,s)
        if family=="catboost": model_path=model_dir/f"{family}.cbm"; model.save_model(model_path)
        elif family=="lightgbm": model_path=model_dir/f"{family}.txt"; model.booster_.save_model(model_path)
        else: model_path=model_dir/f"{family}.joblib"; joblib.dump(model,model_path)
        m["prediction_path"]=str(pp.relative_to(ROOT)).replace("\\","/"); m["prediction_file_sha256"]=sha(pp); m["model_path"]=str(model_path.relative_to(ROOT)).replace("\\","/"); m["model_sha256"]=sha(model_path); outputs[family]=m
        fig,ax=plt.subplots(); ax.hist(s,bins=30); ax.set(title=f"{family} Test score distribution",xlabel="score",ylabel="orders"); fig.tight_layout(); fig.savefig(plot_dir/f"{family}_score_distribution.png"); plt.close(fig)
    write_json(report/"test_metrics.json",outputs)
    artifacts={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in art.rglob("*") if p.is_file()}; write_json(report/"artifact_manifest.json",artifacts)
    write_json(ledger,{"status":"CONSUMED","test_access_began_utc":access,"test_access_completed_utc":now(),"manifest_sha256":stored,"access_count":1,"champion_unchanged":manifest["champion"]})
    print(json.dumps({"status":"COMPLETE","test_access_began_utc":access,"champion":manifest["champion"]},indent=2))
def dry(c):
    order=verify(c); result={n:{"rows":len(f),"positive":int(f.late_delivery.sum()),"negative":int((f.late_delivery==0).sum())} for n in ("train","validation") for f in [split(c,n,order)]}; print(json.dumps({"status":"DRY_RUN_PASS","test_opened":False,"splits":result},indent=2))
def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["dry-run","validate-and-lock","final-test"]); p.add_argument("--manifest-sha256"); a=p.parse_args(); c=load()
    if a.command=="dry-run": dry(c)
    elif a.command=="validate-and-lock": validation(c)
    elif not a.manifest_sha256: p.error("final-test requires --manifest-sha256")
    else: final_test(c,a.manifest_sha256)
if __name__=="__main__": main()
