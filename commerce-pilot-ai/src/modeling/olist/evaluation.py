"""Canonical Phase 2A metrics and deterministic threshold selection."""
from __future__ import annotations
import hashlib, json
import numpy as np
from sklearn.metrics import (average_precision_score,roc_auc_score,brier_score_loss,log_loss,
 precision_recall_curve,confusion_matrix,precision_score,recall_score,f1_score,balanced_accuracy_score)

def prediction_hash(scores): return hashlib.sha256(np.asarray(scores,dtype="<f8").tobytes()).hexdigest()
def select_threshold(y,s):
    p,r,t=precision_recall_curve(y,s); candidates=[]
    for i,x in enumerate(t):
        f=0 if p[i]+r[i]==0 else 2*p[i]*r[i]/(p[i]+r[i]); candidates.append((f,r[i],-x,x))
    return float(max(candidates)[3])
def threshold_metrics(y,s,t):
    pred=(np.asarray(s)>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"threshold":float(t),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"specificity":tn/(tn+fp),"balanced_accuracy":balanced_accuracy_score(y,pred),"predicted_positive_rate":float(pred.mean()),"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}}
def capacities(y,s):
    y=np.asarray(y); s=np.asarray(s); order=np.lexsort((np.arange(len(s)),-s)); prev=y.mean(); out={}
    for pct in (1,5,10,20):
        n=max(1,int(np.ceil(len(y)*pct/100))); selected=y[order[:n]]; tp=int(selected.sum())
        out[str(pct)]={"number_flagged":n,"precision":tp/n,"recall":tp/y.sum(),"lift_over_prevalence":(tp/n)/prev}
    return out
def metrics(y,s,threshold=None):
    s=np.clip(np.asarray(s,dtype=float),1e-15,1-1e-15); threshold=select_threshold(y,s) if threshold is None else threshold
    return {"average_precision":average_precision_score(y,s),"roc_auc":roc_auc_score(y,s),"brier_score":brier_score_loss(y,s),"log_loss":log_loss(y,s),"prevalence":float(np.mean(y)),"prediction_hash":prediction_hash(s),"fixed_threshold":threshold_metrics(y,s,.5),"research_threshold":threshold_metrics(y,s,threshold),"capacity":capacities(y,s)}
def bootstrap_ci(y,s,seed=42,n=1000):
    y=np.asarray(y); s=np.asarray(s); rng=np.random.default_rng(seed); vals={k:[] for k in ("average_precision","roc_auc","brier_score")}
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y)); yy=y[ix]; ss=s[ix]
        if len(np.unique(yy))<2: continue
        vals["average_precision"].append(average_precision_score(yy,ss)); vals["roc_auc"].append(roc_auc_score(yy,ss)); vals["brier_score"].append(brier_score_loss(yy,ss))
    return {k:{"lower":float(np.quantile(v,.025)),"upper":float(np.quantile(v,.975)),"replicates":len(v)} for k,v in vals.items()}
