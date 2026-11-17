"""Paired re-scoring of the two observation engines, two terminal engines, and old tapes."""
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

H=Path(__file__).resolve().parent
OLD=H.parent/"law_axis_v67"
ROUTES=H.parent/"invariance_audit_v66"/"locked"/"routes"
ENGINES=("prescribed","compliant")
SOURCES=("archived","prescribed_raw","compliant_raw","prescribed_matched","compliant_matched",
         "prescribed_swapped","compliant_swapped")
FAMS=("wendland","dispersive")
LAWS=("strengthening","weakening")
STAGES=("full","soft","hard","dropped","restored")


def archive(path):
    with np.load(path,allow_pickle=False) as z:
        items={k:z[k] for k in ("p","shift") if k in z}
        items["ix"]={str(name):i for i,name in enumerate(z["cells"])}
    return items


def routed(raw, shift, gate):
    a=raw.astype(float)
    if shift is not None:
        factor=np.exp(shift)[None,None,:,None,None]
        a=a*factor/(1.-a+a*factor)
    # [seed, decoder, offset, environment, object, repeat]
    a=np.broadcast_to(a[:,:,None],(*a.shape[:2],3,*a.shape[2:])).copy()
    a[:,1:3]=np.where(gate[None,None,...,None],a[:,1:3],a[:,:1])
    return a


def fn(f,s,l,v,h):return f"{f}_{s}_{l}_{v}_h{h}.npz"


def loss(p,y):
    # Reduce probabilities across source restarts and measurement replicates only after squaring.
    q=p.mean((0,5))
    sq=(p*p).mean((0,5))
    return (sq+(1-2*q)*y.mean(0)[None]).mean((1,2,3))


def ci(x):
    m=x.mean(0); half=x.std(0,ddof=1)*t.ppf(.975,len(x)-1)/np.sqrt(len(x))
    return np.stack((m,m-half,m+half),axis=-1).tolist()


def main():
    assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    scores=np.zeros((12,2,2,len(SOURCES),2,5))
    prevalence=np.zeros((12,2,2,2))
    disagreements=np.zeros((12,2,2))
    identity_error=0.
    calibration_delta=np.zeros((12,2,2,2,3)) # nominal vs swapped |shift|, absolute probability change, iso-raw identical fraction
    for k in range(12):
        sub=f"panel_{k:02d}"
        prev=archive(OLD/"forecasts"/(sub+".npz"))
        new=[archive(H/"forecasts"/e/(sub+".npz")) for e in ENGINES]
        for f,s,l,v,h in product(FAMS,("inlay","woven"),LAWS,("base","hollow","bulge"),(19,31)):
            a=FAMS.index(f);b=LAWS.index(l)
            name=fn(f,s,l,v,h)
            with np.load(ROUTES/sub/f"{f}_{v}_h{h}.npz") as z: gate=z["route"]
            labels=[]
            for e in ENGINES:
                with np.load(H/("labels_"+e)/sub/name) as z: labels.append(z["labels"].astype(float))
            y0,y1=labels
            prevalence[k,a,b]+=np.mean(labels,axis=(1,2,3,4))/12
            disagreements[k,a,b]+=np.mean(y0!=y1)/12
            def put(i,p,weight=1):
                r0,r1=loss(p,y0),loss(p,y1)
                # Check both soft-full and restored-dropped on each cell.
                weighted_y=(y1-y0).mean(0)
                for ma,mb in ((1,0),(4,3)):
                    delta=(p[:,ma]-p[:,mb]).mean((0,4))
                    vdiff=(r1[ma]-r1[mb])-(r0[ma]-r0[mb])
                    nonlocal identity_error
                    identity_error=max(identity_error,abs(vdiff+2*(delta*weighted_y).mean()))
                scores[k,a,b,i]+=np.stack((r0,r1))*weight/12
            for oldlaw in ("rational","gaussian_rate"):
                put(0,routed(prev["p"][prev["ix"][fn(f,s,oldlaw,v,h)]],None,gate),.5)
            for j,record in enumerate(new):
                idx=record["ix"][name]
                other=record["ix"][fn(f,s,LAWS[1-b],v,h)]
                raw=record["p"][idx]
                shift=record["shift"][idx];other_shift=record["shift"][other]
                put(1+j,routed(raw,None,gate))
                matched=routed(raw,shift,gate); swapped=routed(raw,other_shift,gate)
                put(3+j,matched);put(5+j,swapped)
                delta=abs(matched[:,0]-swapped[:,0])
                calibration_delta[k,a,b,j]+=np.array([abs(shift-other_shift).mean(),
                    delta.mean(),(delta<1e-12).mean()])/12
    assert identity_error<5e-12,identity_error
    effects=np.stack((scores[...,1]-scores[...,0],scores[...,4]-scores[...,3]),-1)
    A=effects[:,:,:,0,0];B=effects[:,:,:,0,1]
    C=effects[:,:,:,2,0];D=effects[:,:,:,2,1]
    history=((C-A)+(D-B))/2
    terminal=((B-A)+(D-C))/2
    np.testing.assert_allclose(D-A,history+terminal,rtol=0,atol=1e-12)
    oldscore=np.array(json.loads((OLD/"score.json").read_text())["panels"])[:,:,2:].mean(3)
    v67_error=float(np.max(np.abs(scores[:,:,:,0,0]-oldscore)))
    results=dict(families=FAMS,laws=LAWS,forecast_sources=SOURCES,label_sources=ENGINES,methods=STAGES,
       by_catalog=scores.tolist(),mean=scores.mean(0).tolist(),label_prevalence=ci(prevalence),
       label_disagreement=ci(disagreements),identity_max_abs=float(identity_error),
       archived_risk_max_diff=v67_error,
       calibration_only_swap=ci(calibration_delta),
       decomposition=dict(methods=["soft_minus_full","restored_minus_dropped"],history=ci(history),
                          terminal=ci(terminal),total=ci(D-A)),job=os.environ["SLURM_JOB_ID"])
    (H/"crossover68.json").write_text(json.dumps(results,indent=2)+"\n")
if __name__=="__main__":main()
