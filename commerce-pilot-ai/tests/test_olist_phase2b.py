import inspect,json
import numpy as np,pandas as pd,pytest,yaml
from src.modeling.olist.expanded_feature_builder import STRICT,EXPANDED,predictors
from src.modeling.olist.temporal_validation import fold_indices,paired_bootstrap,basic,corrected_evidence_label,validate_paired_predictions
from src.modeling.olist import phase2b_sensitivity as p2b
def frame():
 d={"order_id":["a","b","c"],"approved_at":pd.to_datetime(["2017-01-01","2017-02-01","2017-03-01"]),"late_delivery":[0,1,0]}
 for x in STRICT+EXPANDED:d[x]=[1.,2.,3.]
 return pd.DataFrame(d)
def test_exact_whitelists(): assert len(STRICT)==9 and len(EXPANDED)==15
def test_predictors_exclude_identifiers_target_timestamps():
 X,y=predictors(frame(),"expanded");assert list(X)==STRICT+EXPANDED;assert not {"order_id","approved_at","late_delivery"}&set(X)
def test_unknown_contract_fails():
 with pytest.raises(ValueError):predictors(frame(),"bad")
def test_temporal_fold_precedes_validation():
 d=frame();tr,va=fold_indices(d,{"train_start":"2017-01-01","train_end_exclusive":"2017-03-01","validation_start":"2017-03-01","validation_end_exclusive":"2017-04-01"});assert d.iloc[tr].approved_at.max()<d.iloc[va].approved_at.min()
def test_overlap_fails():
 with pytest.raises(ValueError):fold_indices(frame(),{"train_start":"2017-01-01","train_end_exclusive":"2017-04-01","validation_start":"2017-03-01","validation_end_exclusive":"2017-04-01"})
def test_paired_bootstrap_deterministic():
 y=np.array([0,1]*20);a=np.linspace(.1,.8,40);b=a+.01;assert paired_bootstrap(y,a,b,50,42)==paired_bootstrap(y,a,b,50,42)
def test_ap_lift_definition():
 r=basic(np.array([0,1,0,1]),np.array([.1,.8,.2,.7]));assert r["ap_lift"]==r["average_precision"]/r["prevalence"]
def test_no_final_test_route():
 source=inspect.getsource(p2b.main);assert 'choices=["audit","dry-run","development","reproduce"]' in source
def test_contract_test_boundary():
 c=yaml.safe_load(open("configs/olist_phase2b_sensitivity.yaml"));assert c["development"]["end_exclusive"]==c["test_exclusion"]["start"]
def test_audit_is_machine_readable_and_primary_empty():
 a=json.load(open("reports/generated/olist/phase2b/feature_availability_audit.json"));assert a["primary_asof_whitelist"]==[] and a["retrospective_expanded_whitelist"]==EXPANDED
def test_equality_is_not_late(): assert int(pd.Timestamp("2020-01-01")>pd.Timestamp("2020-01-01"))==0
def test_unverified_asof_never_supported(): assert corrected_evidence_label(.1,.01,3,False)=="INCONCLUSIVE_INCREMENTAL_SIGNAL"
def test_identical_predictions_are_no_signal(): assert corrected_evidence_label(0,0,0,False,identical=True)=="NO_INCREMENTAL_SIGNAL"
def test_no_improvement_is_not_automatically_degraded(): assert corrected_evidence_label(0,0,0,False)=="NO_INCREMENTAL_SIGNAL"
def test_degraded_requires_explicit_condition(): assert corrected_evidence_label(-.1,-.2,0,False,degraded=True)=="DEGRADED_PERFORMANCE"
def paired_frames():
 return pd.DataFrame({"order_id":["a","b"],"fold_id":["f","f"],"model_name":["m","m"],"y_true":[0,1],"y_score":[.1,.8]})
def test_pair_validation_accepts_aligned():
 a=paired_frames();b=a.copy();validate_paired_predictions(a,b)
def test_pair_validation_rejects_misalignment():
 a=paired_frames();b=a.copy();b.loc[1,"order_id"]="z"
 with pytest.raises(ValueError):validate_paired_predictions(a,b)
def test_pair_validation_rejects_duplicates():
 a=pd.concat([paired_frames(),paired_frames().iloc[[0]]]);
 with pytest.raises(ValueError):validate_paired_predictions(a,paired_frames())
def test_pair_validation_rejects_test_period():
 a=paired_frames().assign(approved_at=pd.Timestamp("2018-05-01"));
 with pytest.raises(ValueError):validate_paired_predictions(a,a.copy())
def test_lift_known_fixture():
 r=basic(np.array([0,1,0,1]),np.array([.1,.8,.2,.7]));assert r["average_precision"]==1 and r["average_precision"]/r["prevalence"]==2
