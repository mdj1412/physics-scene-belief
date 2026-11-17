"""Score fixed routes and pressure interventions with inference clustered at catalog."""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
SHAPES = ("fold","ripple")
METHODS = ("original","projected","nominal","oracle","visual","shuffled","drop","bias_minus","bias_plus")
source_x = np.array([18.5,19,19.5,30.5,31,31.5])
TRANSFER = np.array([False,False,False,True,True,False])

def gate(x, switches):
    return switches[np.abs(x[...,None]-source_x).argmin(-1)]

def risk(p,truth):
    # p [environment,object,paid repetition]; y [area,offset,environment,object].
    return np.square(p).mean(-1) + truth*(1.-2.*p.mean(-1))

def routes(estimated,exact,damaged,nominal,switches):
    mask=gate(estimated,switches)
    return np.stack([np.zeros_like(mask),np.ones_like(mask),np.full_like(mask,gate(np.array(nominal),switches)),
                     gate(exact,switches),mask,np.broadcast_to(mask.mean(axis=0),mask.shape),
                     gate(damaged,switches),gate(estimated-.35,switches),gate(estimated+.35,switches)],axis=-1)

def interval(v, comparisons=1):
    v=np.asarray(v)
    hw=t.ppf(1-.025/comparisons,len(v)-1)*v.std(axis=0,ddof=1)/np.sqrt(len(v))
    m=v.mean(0)
    return np.stack((m,m-hw,m+hw),-1)

def summarize(cells):
    # cells [panel,shape,height,area,method]; equal weight within each stratum.
    pooled=cells[:,:,:,1:,:].mean(axis=(1,2,3))
    pairs=[(4,0),(4,3),(4,2)]
    comparisons=np.stack([pooled[...,a]-pooled[...,b] for a,b in pairs],-1)
    shape_area=cells.mean(axis=2)
    d=shape_area[...,1]-shape_area[...,0]
    area=d[:,:,2]-d[:,:,1]
    return dict(pooled=interval(pooled),pooled_panel=pooled,shape_area=interval(shape_area),
                h_shape_area=interval(cells),primary_comparisons=interval(comparisons,3),
                visual_minus_shuffled=interval(pooled[:,4]-pooled[:,5]),
                sensitivity_vs_original=interval(pooled[:,6:]-pooled[:,:1],3),
                projection_by_area=interval(d[:,:,1:],4),projection_point=interval(d[:,:,0],2),
                pressure_contrasts=interval(np.stack([area[:,0],area[:,1],area[:,0]-area[:,1]],-1),3),
                oracle_minus_original=interval(pooled[:,3]-pooled[:,0]),
                oracle_minus_nominal=interval(pooled[:,3]-pooled[:,2]),
                shape_visual_net=interval((shape_area[...,4]-shape_area[...,0])[:,:,1:].mean(2),2),
                visual_minus_original_panelwins=int(np.sum(pooled[:,4]<pooled[:,0])))

def evaluate(switches=TRANSFER):
    out=np.zeros((2,12,2,2,3,len(METHODS)))
    flips=np.zeros((12,2,2)); prevalence=np.zeros((12,2,2,3))
    accuracy=np.zeros((12,2,2,4)); raw_errors=[[] for _ in range(12)]
    within_by_x=np.zeros((12,2,2,3,3,len(METHODS)))
    for pan,i,hidx,var in itertools.product(range(12),range(2),range(2),("base","hollow","bulge")):
        h=(19,31)[hidx]; shap=SHAPES[i]; sub=f"panel_{pan:02d}"
        with np.load(ROOT/"camera45"/sub/f"{shap}_{var}_h{h}.npz") as cam:
            x=cam["x_hat_mm"].astype(float); exact=np.broadcast_to(cam["oracle_x_mm"],x.shape)
            assert x.shape==(3,4,128) and np.isfinite(x).all()
            for off, ref in enumerate((h-.5,h,h+.5)):
                assert np.allclose(exact[off],ref)
            actions=routes(x,exact,cam["x_drop_mm"].astype(float),h,switches)
            if np.array_equal(switches,TRANSFER):
                np.testing.assert_array_equal(actions[...,4],cam["visual_project"])
            abserr=abs(x-exact)
            raw_errors[pan].append(abserr.ravel())
            near=lambda arr:abs(arr[...,None]-source_x).argmin(-1)
            accuracy[pan,i,hidx]+=np.array([abserr.mean(),(near(x)==near(exact)).mean(),
                  (actions[...,3]==actions[...,4]).mean(),actions[...,4].mean()])/3
        for sf,law in itertools.product(("inlay","woven"),("rational","gaussian_rate")):
            filename=f"{shap}_{sf}_{law}_{var}_h{h}.npz"
            with np.load(ROOT/"results"/sub/filename) as p, np.load(ROOT/"labels45"/sub/filename) as label:
                pred=p["pred"][:,0]; y=label["labels"].astype(float)
                assert pred.shape==(3,4,128,4) and y.shape==(3,3,4,128)
                assert np.isfinite(pred).all() and (pred.min()>=-1e-12) and (pred.max()<=1+1e-12)
                r=risk(pred[0],y)
                np.testing.assert_allclose(r,((pred[0]-y[...,None])**2).mean(-1),atol=2e-12,rtol=1e-11)
                flips[pan,i,hidx]+=(y[1]!=y[2]).mean()/12.
                prevalence[pan,i,hidx]+=y.mean(axis=(1,2,3))/12.
                for di,pr in enumerate(pred[1:]):
                    mix=r[...,None]+(risk(pr,y)-r)[...,None]*actions
                    b=mix.mean(axis=(2,3)) # area, offset, method
                    out[di,pan,i,hidx]+=b.mean(axis=1)/12.
                    if di==0:
                        within_by_x[pan,i,hidx]+=b/12.
    all_errors=np.concatenate([np.concatenate(e) for e in raw_errors])
    return dict(methods=METHODS,source_axis=summarize(out[0]),privileged=summarize(out[1]),
                error_mae_p90=[all_errors.mean(),np.quantile(all_errors,.9)],accuracy=interval(accuracy.mean((1,2))),
                visual_metrics_by_shape_height=interval(accuracy),peak_area_label_flip=interval(flips.mean(2),2),
                prevalence=interval(prevalence.mean(2)),within_by_x=interval(within_by_x),
                note="CI fields [mean,lower,upper]. Paired across 12 panels; finite-primary pooled narrow+wide only.")

def write_results(path,result):
    path.write_text(json.dumps(result,default=lambda x: np.asarray(x).tolist(),indent=1,allow_nan=False)+"\n")

if __name__=="__main__":
    import argparse
    flags=argparse.ArgumentParser()
    flags.add_argument("--source-table",help="Six comma-separated 0/1 source-only route decisions")
    args=flags.parse_args()
    switches=TRANSFER if args.source_table is None else np.array([int(s) for s in args.source_table.split(",")],bool)
    assert switches.shape==(6,)
    dest="score45.json" if args.source_table is None else "score45_matched.json"
    results=evaluate(switches)
    results["lookup_project"]=switches
    write_results(ROOT/dest,results)
