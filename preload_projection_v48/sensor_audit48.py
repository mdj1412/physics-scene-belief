"""Source-estimated ternary classifier used only to audit sensor identifiability."""
import argparse,json
import os
import numpy as np
from scipy.stats import rankdata,t
from sensor48 import readings
from world48 import ROOT, SOURCE, FAMILIES, STATES, HEIGHTS

ARMS=("local","fixed19","fixed31")
P=np.asarray([.5,.25,.25])
THRESHOLDS=np.arange(.0,.1801,.002)


def all_signals(catalog,source,arm,height,transfer=.60):
    folder=ROOT.parent/"factorial_projection_v47" if source else ROOT
    with np.load(folder/f"worlds/panel_{catalog:02d}.npz") as manifest:
        site=height if arm=="local" else int(arm[5:])
        return np.stack([np.concatenate([readings(manifest,f,height,s,site,transfer).ravel()
                    for f in (SOURCE if source else FAMILIES)]) for s in STATES])


def fit():
    model={}
    for arm in ARMS:
        for h in HEIGHTS:
            signals=np.concatenate([all_signals(i,True,arm,h) for i in range(12)],axis=1)
            # Fixed grids with catalog-independent threshold selection. Separate boundaries
            # allow a calibrated overall bias. Objective is the design's state-weighted accuracy.
            grids=np.sort(signals,axis=1)
            def cdf(i,values):
                return np.searchsorted(grids[i],values,side="left")/grids.shape[1]
            left=-THRESHOLDS;right=THRESHOLDS
            l=left[np.argmax(.25*cdf(1,left)-.5*cdf(0,left))]
            r=right[np.argmax(.5*cdf(0,right)-.25*cdf(2,right))]
            model[f"{arm}_{h}"]=[float(l),float(r)]
    return model


def auc(a,b):
    scores=np.concatenate((a,b))
    r=rankdata(scores)[len(a):]
    return float((r.sum()-len(b)*(len(b)+1)/2)/(len(a)*len(b)))


def ci(x):
    x=np.asarray(x);m=x.mean(0);e=x.std(0,ddof=1)*t.ppf(.975,len(x)-1)/np.sqrt(len(x))
    return np.stack((m,m-e,m+e),axis=-1).tolist()


def evaluate(model,transfer):
    result={}
    for arm in ARMS:
        confusion=[];aucs=[];direction=[];coupling=[]
        for i in range(12):
            cm=np.zeros((3,3));au=[];sign=[];ca=[]
            for h in HEIGHTS:
                sig=all_signals(i,False,arm,h,transfer)
                l,r=model[f"{arm}_{h}"]
                cls=np.where(sig<l,1,np.where(sig>=r,2,0))
                for actual in range(3):
                    cm[actual]+=np.bincount(cls[actual],minlength=3)/sig.shape[1]/2
                mag=abs(sig-(l+r)/2.)
                au.append(auc(mag[0],np.r_[mag[1],mag[2]]))
                sign.append(((sig[1]<(l+r)/2).mean()+(sig[2]>=(l+r)/2).mean())/2)
                ca.append(((cls[0]==0).mean()+.5*((cls[1]!=0).mean()+(cls[2]!=0).mean()))/2)
            confusion.append(cm);aucs.append(np.mean(au));direction.append(np.mean(sign));coupling.append(np.mean(ca))
        diag=np.diagonal(np.array(confusion),axis1=1,axis2=2)
        result[arm]=dict(confusion_row_true_col_pred=np.array(confusion).mean(0).tolist(),accuracy=ci(diag@P),
          balanced_ternary_accuracy=ci(diag.mean(1)),coupling_auc=ci(aucs),coupling_accuracy=ci(coupling),
          active_direction_accuracy=ci(direction))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("fit","target"));a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or os.getenv("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("Run this analysis on allocated CPU worker")
    artifact=ROOT/"diagnostic_thresholds48.json"
    if a.mode=="fit":
        artifact.write_text(json.dumps(fit(),indent=2)+"\n")
    else:
        model=json.loads(artifact.read_text())
        (ROOT/"sensor_identification48.json").write_text(json.dumps(
           dict(source_thresholds_mm=model,order=STATES,primary=evaluate(model,.60),perfect_transfer=evaluate(model,1.)),
           indent=2)+"\n")
