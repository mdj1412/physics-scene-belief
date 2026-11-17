"""Compare the pixel-derived local angular variance to analytic geometry on held-out features."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t
from camera46 import ROOT, shape_function

def read(source):
    project=ROOT.parent/"image_pressure_v45" if source else ROOT
    folder="source_camera46" if source else "camera46"
    families=("fold","ripple") if source else ("terrace","dimple")
    records=[]
    for p in range(12):
        sub=f"panel_{p:02d}"
        with np.load(project/"results"/"worlds"/f"{sub}.npz") as w:
            th=w["theta"].reshape(-1,6)
        for shape,var,h in itertools.product(families,("base","hollow","bulge"),(19,31)):
            with np.load(ROOT/folder/sub/f"{shape}_{var}_h{h}.npz") as a:
                x,v,tx,tv=[a[key].reshape(3,-1) for key in ("x_hat_mm","variance_hat","true_x_mm","true_variance")]
            absolute=tx[:,:,:,None] if tx.ndim==3 else tx[...,None]
            xs=absolute+np.arange(-1.5,1.75,.5)
            normals=th[None,:,2,None]+th[None,:,3,None]*xs/25.+shape_function(shape)(xs,shape,var)[1]
            tang=np.arctan(normals).var(axis=-1)
            rows=(x,v,tx,tv,tang,np.full_like(x,p),np.full_like(x,families.index(shape)))
            records.append(np.stack(rows,-1).reshape(-1,7))
    return np.concatenate(records)

def metric(ar):
    x,v,tx,tv,d,p,f=ar.T
    return dict(clips=len(ar),x_mae=np.mean(abs(x-tx)),var_mae=np.mean(abs(v-tv)),
                var_median=np.median(v),truth_median=np.median(tv),rho=spearmanr(v,tv).statistic,
                differential_rho=spearmanr(v,d).statistic,true_secant_vs_differential_rho=spearmanr(tv,d).statistic)

def main():
    src=read(True);tar=read(False)
    out={}
    for data,name,fams in ((src,"source",("fold","ripple")),(tar,"target",("terrace","dimple"))):
        section=dict(pooled=metric(data),families={},cells=[])
        rho=[metric(data[data[:,5]==p])["rho"] for p in range(12)]
        section["catalog_rho_minmax"]=[min(rho),max(rho)]
        for j,shape in enumerate(fams):
            section["families"][shape]=metric(data[data[:,6]==j])
            for h in sorted(set(data[:,2])):
                z=data[(data[:,2]==h)&(data[:,6]==j)]
                section["cells"].append(dict(x_mm=h,family=shape,**metric(z)))
        # Relative rank within each true location removes the difference among heights.
        rs=[c["rho"] for c in section["cells"]]
        section["cell_rho_minmax_mean"]=[min(rs),max(rs),np.mean(rs)]
        out[name]=section
    table=json.loads((ROOT/"router46.json").read_text())
    bins=np.array(table["visual"]["x_grid"])
    cut=np.array(table["visual"]["thresholds"])
    tarbin=abs(tar[:,0,None]-bins).argmin(-1); srcbin=abs(src[:,0,None]-bins).argmin(-1)
    sizes=[]
    for i in range(6):
        lo,hi=np.quantile(src[srcbin==i,1],[.01,.99]);v=tar[tarbin==i,1]
        groups=(v[:,None]>cut[i]).sum(-1)
        sizes.append(dict(x=float(bins[i]),source_q01q99=[lo,hi],target_outside=np.mean((v<lo)|(v>hi)),
                      target_quartile_weight=(np.bincount(groups,minlength=4)/len(v)).tolist()))
    out["target_source_support"]=sizes
    (ROOT/"vision_audit46.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":
    main()
