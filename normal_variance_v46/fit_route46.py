"""Source-only calibration, with catalog- and family-held-out checks on v45."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"image_pressure_v45"
XGRID=np.array((18.5,19.,19.5,30.5,31.,31.5))
FIELDS=("cost","x","v","tx","tv","panel","family")

def read_source():
    rows={k:[] for k in FIELDS}
    for p,s,v,h in itertools.product(range(12),("fold","ripple"),("base","hollow","bulge"),(19,31)):
        sub=f"panel_{p:02d}"
        with np.load(ROOT/"source_camera46"/sub/f"{s}_{v}_h{h}.npz") as a:
            x, vv, tx,tv=[a[k] for k in ("x_hat_mm","variance_hat","true_x_mm","true_variance")]
        d=np.zeros(x.shape)
        for sf,law in itertools.product(("inlay","woven"),("rational","gaussian_rate")):
            stem=f"{s}_{sf}_{law}_{v}_h{h}.npz"
            with np.load(OLD/"results"/sub/stem) as a,np.load(OLD/"labels45"/sub/stem) as b:
                o,g=a["pred"][:2,0]
                # Cost is the mean per-object difference in Brier on narrow AND broad contacts.
                yy=b["labels"][1:].mean(0)
                d+=((g*g-o*o).mean(-1)-2*(g-o).mean(-1)*yy)*.25
        for key,val in zip(FIELDS,(d,x,vv,tx,tv,np.full(d.shape,p),np.full(d.shape,int(s=="ripple")))):
            rows[key].append(val.ravel())
    return {k:np.concatenate(vals) for k,vals in rows.items()}

def xbin(x):
    return abs(x[...,None]-XGRID).argmin(-1)

def fit(x,v,d):
    bins=xbin(x); cut=[]; base=[]; effect=[]; n=[]
    for k in range(6):
        vv,yy = v[bins==k],d[bins==k]
        cc=np.quantile(vv,[.25,.5,.75])
        j=np.searchsorted(cc,vv,side="left")
        counts=np.bincount(j,minlength=4)
        mean=yy.mean()
        strata=np.divide(np.bincount(j,weights=yy,minlength=4),counts,
                         out=np.full(4,mean),where=counts>0)
        cut.append(cc.tolist());base.append(mean);n.append(counts.tolist())
        effect.append((.75*strata+.25*mean).tolist())
    return dict(x_grid=XGRID.tolist(),thresholds=cut,risk_x=base,risk_joint=effect,counts=n)

def scores(model,x,v,d):
    i=xbin(x)
    j=(v[:,None]>np.asarray(model["thresholds"])[i]).sum(-1)
    a=np.asarray(model["risk_x"])[i]<0
    b=np.asarray(model["risk_joint"])[i,j]<0
    legacy=np.isin(i,(3,4))
    return np.stack((legacy*d,a*d,b*d,(b.astype(float)-a)*d,b.astype(float),a.astype(float)),axis=-1)

def ci(x):
    m=x.mean(0);w=t.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
    return np.stack((m,m-w,m+w),-1).tolist()

def main():
    rows=read_source()
    out=dict(source="v45 source-axis paired Brier, narrow and broad mean, areas/laws/surfaces/noise balanced",
             population=len(rows["cost"]),shrinkage_to_position=.25,feature_units="rad^2",
             scoring_order=["v45_baseline_minus_original","refit_x_minus_original","joint_minus_original","joint_minus_refit_x","joint_projection_frequency","refit_x_frequency"])
    cv={}
    for tag,xkey,vkey in (("visual","x","v"),("oracle","tx","tv")):
        x,v,d,p,f=[rows[k] for k in (xkey,vkey,"cost","panel","family")]
        table=fit(x,v,d);out[tag]=table
        test=[]
        for panel in range(12):
            keep=p!=panel
            rule=fit(x[keep],v[keep],d[keep])
            test.append(scores(rule,x[~keep],v[~keep],d[~keep]).mean(0))
        lofo=[]
        for fam in range(2):
            keep=f!=fam
            rule=fit(x[keep],v[keep],d[keep])
            vals=scores(rule,x[~keep],v[~keep],d[~keep])
            lofo.append(ci(np.stack([vals[p[~keep]==i].mean(0) for i in range(12)])))
        cv[tag]=dict(catalog_held_out=ci(np.stack(test)),family_held_out=lofo,
                     in_sample=scores(table,x,v,d).mean(0).tolist(),catalog_test_panel=np.asarray(test).tolist())
    (ROOT/"router46.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    (ROOT/"source_cv46.json").write_text(json.dumps(cv,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":
    main()
