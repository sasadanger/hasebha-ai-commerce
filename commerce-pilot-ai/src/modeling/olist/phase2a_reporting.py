"""Create Phase 2A reports from saved predictions and fitted artifacts only."""
from pathlib import Path
import hashlib,json,platform,os
import joblib,numpy as np,pandas as pd,yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.calibration import calibration_curve
from .strict_feature_builder import FEATURES
ROOT=Path(__file__).resolve().parents[3]; R=ROOT/"reports/generated/olist/phase2a"; A=ROOT/"artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=json.loads((R/"selection_manifest.json").read_text()); test=json.loads((R/"test_metrics.json").read_text()); models=list(manifest["selected"]); (R/"final_test_metrics.json").write_text(json.dumps(test,indent=2),encoding="utf-8")
    env={"python":platform.python_version(),"platform":platform.platform(),"processor":platform.processor(),"logical_cpu":os.cpu_count(),"gpu":"not used","thread_count":1,"random_seed":42,"packages":manifest["package_versions"]}; (R/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    for split_name in ("validation","test"):
        fig,ax=plt.subplots()
        for family in models:
            p=A/"predictions"/f"{split_name}_{family}.parquet"; d=pd.read_parquet(p); y=d.late_delivery.to_numpy(); s=d.score.to_numpy(); locked=manifest["selected"][family]["metrics"]["research_threshold"]["threshold"]
            enriched=pd.DataFrame({"order_id":d.order_id,"split":split_name,"model_name":family,"y_true":y,"y_score":s,"prediction_at_0_5":(s>=.5).astype("int8"),"prediction_at_locked_research_threshold":(s>=locked).astype("int8")}); enriched.to_parquet(A/"predictions"/f"{split_name}_{family}_scored.parquet",index=False)
            pr,rc,_=precision_recall_curve(y,s); ax.plot(rc,pr,label=family)
        ax.set(xlabel="Recall",ylabel="Precision",title=f"{split_name.title()} precision-recall comparison"); ax.legend(); fig.tight_layout(); fig.savefig(A/"plots"/f"{split_name}_precision_recall_comparison.png"); plt.close(fig)
    for kind in ("roc","calibration"):
        fig,ax=plt.subplots()
        for family in models:
            d=pd.read_parquet(A/"predictions"/f"test_{family}.parquet"); y=d.late_delivery; s=d.score
            x,z=(roc_curve(y,s)[0:2] if kind=="roc" else calibration_curve(y,s,n_bins=10,strategy="quantile")[::-1]); ax.plot(x,z,marker="." if kind=="calibration" else None,label=family)
        ax.set(xlabel="False positive rate" if kind=="roc" else "Mean predicted probability",ylabel="True positive rate" if kind=="roc" else "Observed fraction positive",title=f"Test {kind} comparison"); ax.legend(); fig.tight_layout(); fig.savefig(A/"plots"/f"test_{kind}_comparison.png"); plt.close(fig)
    prev=[.06606765,.12493476,.06328807]; fig,ax=plt.subplots(); ax.bar(["Train","Validation","Test"],prev); ax.set(ylabel="Late-delivery prevalence",title="Split prevalence comparison"); fig.tight_layout(); fig.savefig(A/"plots/split_prevalence_comparison.png"); plt.close(fig)
    importance={}; lr=joblib.load(A/"models/logistic_regression.joblib"); importance["logistic_regression"]={f:float(v) for f,v in zip(FEATURES,lr.named_steps["model"].coef_[0])}
    from catboost import CatBoostClassifier; cb=CatBoostClassifier(); cb.load_model(A/"models/catboost.cbm"); importance["catboost"]={f:float(v) for f,v in zip(FEATURES,cb.get_feature_importance())}
    import lightgbm as lgb; lb=lgb.Booster(model_file=str(A/"models/lightgbm.txt")); importance["lightgbm"]={f:float(v) for f,v in zip(FEATURES,lb.feature_importance(importance_type="gain"))}; (R/"feature_importance.json").write_text(json.dumps(importance,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,3,figsize=(14,5));
    for ax,(family,vals) in zip(axes,importance.items()):
        keys=sorted(vals,key=lambda k:abs(vals[k]))[-9:]; ax.barh(keys,[vals[k] for k in keys]); ax.set_title(family)
    fig.tight_layout(); fig.savefig(A/"plots/feature_importance_comparison.png"); plt.close(fig)
    rows=[]
    for f in models:
        v=manifest["selected"][f]; t=test[f]; rows.append(f"| {f} | {v['metrics']['average_precision']:.6f} | {v['metrics']['roc_auc']:.6f} | {v['metrics']['brier_score']:.6f} | {t['average_precision']:.6f} | {t['roc_auc']:.6f} | {t['brier_score']:.6f} | {t['average_precision']/t['prevalence']:.3f} | {v['metrics']['research_threshold']['threshold']:.6f} | {t['training_duration_seconds']:.3f}s |")
    report="# Phase 2A Olist Strict-Core Benchmark Report\n\nStatus: **COMPLETE**. Validation selected **CatBoost** before Test access. Test results did not change that locked decision.\n\n| Model | Val AP | Val ROC-AUC | Val Brier | Test AP | Test ROC-AUC | Test Brier | Test AP lift | Locked threshold | Refit time |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"+"\n".join(rows)+"\n\nValidation prevalence was 12.493476%, versus 6.328807% on Test. CatBoost AP fell from 0.144503 to 0.079478; Logistic Regression achieved the highest descriptive Test AP (0.089085), but Test was not used to revise selection. This indicates temporal instability and weak strict-core ranking signal. Confidence intervals in `final_test_metrics.json` represent historical Test resampling uncertainty only.\n\nFeature importance is associative, not causal. Every predictor is a purchase/approval calendar or approval-duration signal and may encode period-specific Olist operations. No conditional feature, target component, identifier, resampling, calibration, or class weighting was used.\n\nLegal/licensing readiness: **NO-GO**. Production readiness: **NO-GO**. Egyptian-market external validity: **NO-GO**. This is local retrospective research only.\n"; (ROOT/"docs/phase2a_olist_benchmark_report.md").write_text(report,encoding="utf-8")
    artifacts={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in A.rglob("*") if p.is_file()}; (R/"artifact_manifest.json").write_text(json.dumps(artifacts,indent=2),encoding="utf-8")
if __name__=="__main__": main()
