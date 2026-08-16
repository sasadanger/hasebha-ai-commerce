"""Fixed expanding-window development folds and paired uncertainty."""
import hashlib,numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
def fold_indices(d,spec):
    t=d.approved_at; tr=(t>=spec["train_start"])&(t<spec["train_end_exclusive"]); va=(t>=spec["validation_start"])&(t<spec["validation_end_exclusive"])
    if t[tr].max()>=t[va].min() or (tr&va).any(): raise ValueError("Temporal fold overlap")
    return np.flatnonzero(tr),np.flatnonzero(va)
def fold_hash(order_ids): return hashlib.sha256("\n".join(order_ids).encode()).hexdigest()
def basic(y,s):
    s=np.clip(np.asarray(s),1e-15,1-1e-15); p=float(np.mean(y));
    return {"average_precision":average_precision_score(y,s),"prevalence":p,"ap_lift":average_precision_score(y,s)/p,"roc_auc":roc_auc_score(y,s),"brier_score":brier_score_loss(y,s),"log_loss":log_loss(y,s)}
def paired_bootstrap(y,a,b,n=1000,seed=42):
    y=np.asarray(y); a=np.asarray(a); b=np.asarray(b); rng=np.random.default_rng(seed); values={k:[] for k in ["delta_ap","delta_roc_auc","delta_brier","delta_log_loss"]}; failed=0
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y)); yy=y[ix]
        if len(np.unique(yy))<2: failed+=1; continue
        x=basic(yy,a[ix]); z=basic(yy,b[ix]); values["delta_ap"].append(z["average_precision"]-x["average_precision"]); values["delta_roc_auc"].append(z["roc_auc"]-x["roc_auc"]); values["delta_brier"].append(z["brier_score"]-x["brier_score"]); values["delta_log_loss"].append(z["log_loss"]-x["log_loss"])
    observed_a=basic(y,a); observed_b=basic(y,b)
    points={"delta_ap":observed_b["average_precision"]-observed_a["average_precision"],"delta_roc_auc":observed_b["roc_auc"]-observed_a["roc_auc"],"delta_brier":observed_b["brier_score"]-observed_a["brier_score"],"delta_log_loss":observed_b["log_loss"]-observed_a["log_loss"]}
    return {k:{"point":float(points[k]),"lower":float(np.quantile(v,.025)),"upper":float(np.quantile(v,.975))} for k,v in values.items()}|{"estimand":"pooled out-of-fold metric difference","pairing_unit":"same order_id, fold_id, model family, and target","resampling_unit":"rows globally across concatenated folds","fold_structure_preserved":False,"method":"paired row percentile bootstrap; temporal dependence may be understated","confidence_level":0.95,"resamples":n,"valid":n-failed,"failed":failed,"single_class_samples":"rejected","seed":seed}

def corrected_evidence_label(delta_ap, ci_lower, folds_improved, asof_proven, identical=False, degraded=False):
    """Scientific label requires both performance evidence and proven as-of semantics."""
    if identical: return "NO_INCREMENTAL_SIGNAL"
    if degraded: return "DEGRADED_PERFORMANCE"
    if delta_ap > 0 and (ci_lower > 0 or folds_improved > 0):
        return "SUPPORTED_INCREMENTAL_SIGNAL" if asof_proven else "INCONCLUSIVE_INCREMENTAL_SIGNAL"
    return "NO_INCREMENTAL_SIGNAL"

def validate_paired_predictions(strict, expanded, test_start="2018-05-01"):
    keys=["order_id","fold_id","model_name"]
    for name,d in (("strict",strict),("expanded",expanded)):
        if d.duplicated(keys).any(): raise ValueError(f"Duplicate {name} paired prediction")
        if "approved_at" in d and (d["approved_at"]>=test_start).any(): raise ValueError("Test-period prediction detected")
    a=strict.sort_values(keys).reset_index(drop=True); b=expanded.sort_values(keys).reset_index(drop=True)
    if not a[keys].equals(b[keys]): raise ValueError("Strict/Expanded prediction rows are misaligned")
    if not np.array_equal(a.y_true.to_numpy(),b.y_true.to_numpy()): raise ValueError("Strict/Expanded targets differ")
    return a,b
