"""Paired, catalog-clustered analysis of fixed decision rules on two unseen shape families."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
SHAPES=("terrace","dimple")
VARS=("base","hollow","bulge")
XGRID=np.array([18.5,19.,19.5,30.5,31.,31.5])
METHODS=("original","v45","position","joint","oracle_position","oracle_joint","projected","shuffled")
PAIRS=((3,0),(3,1),(3,2))
GRID=(2,3,2)
def ci(a,m=1):
    v=np.asarray(a,dtype=np.float64)
    mean=v.mean(0)
    half=t.ppf(1-.025/m,len(v)-1)*v.std(0,ddof=1)/np.sqrt(len(v))
    return np.stack((mean,mean-half,mean+half),axis=-1)

def load_routes(panel):
    actions=np.empty(GRID+(3,4,128,len(METHODS)))
    bins=np.empty(actions.shape[:-1],np.int8)
    for i,v,h in np.ndindex(GRID):
        p=f"panel_{panel:02d}/{SHAPES[i]}_{VARS[v]}_h{(19,31)[h]}.npz"
        with np.load(ROOT/"camera46"/p) as cm, np.load(ROOT/"route46"/p) as gate:
            x=cm["x_hat_mm"]
            assert x.shape==(3,4,128) and np.isfinite(x).all()
            b=np.abs(x[...,None]-XGRID).argmin(-1)
            bins[i,v,h]=b
            actions[i,v,h,...,0]=0.
            for j,key in enumerate(METHODS[1:6],1):
                z=gate[key]
                assert z.shape==x.shape and np.isin(z,[0,1]).all()
                actions[i,v,h,...,j]=z
            np.testing.assert_array_equal(np.isin(b,[3,4]),actions[i,v,h,...,1])
            actions[i,v,h,...,6]=1.
    # Exact mean over all permutations within the same panel, environment and measured x bin.
    for e in range(4):
        b=bins[...,e,:]
        sel=actions[...,e,:,3]
        rates=np.bincount(b.ravel(),weights=sel.ravel(),minlength=6)/np.maximum(1,np.bincount(b.ravel(),minlength=6))
        actions[...,e,:,7]=rates[b]
    np.testing.assert_allclose(actions[...,7].mean(),actions[...,3].mean(),atol=1e-12)
    return actions

def risk(pred,y):
    return np.square(pred).mean(-1)+y*(1.-2.*pred.mean(-1))

def eval_run(folder):
    rigid="bullet" in folder.lower()
    count=1 if rigid else 3
    scores=np.zeros((12,)+GRID+(count,3,len(METHODS)))
    rates=np.zeros((12,)+GRID+(3,len(METHODS)))
    effects=np.zeros((12,)+GRID+(count,3,2))
    reroutes=np.zeros((12,)+GRID+(3,2))
    material_changes=np.zeros(12)
    samples=0
    filters=(slice(None) if rigid else slice(1,None))
    for panel in range(12):
        acts=load_routes(panel)
        for i,v,h in np.ndindex(GRID):
            a=acts[i,v,h]; nH=(19,31)[h]; sp=f"panel_{panel:02d}"
            rates[panel,i,v,h]=a.mean(axis=(1,2))
            # How much loss and routing change are attributable to new vs removed projections?
            add=(a[...,3]>a[...,2]);remove=(a[...,3]<a[...,2])
            reroutes[panel,i,v,h]=np.stack((add.mean((1,2)),remove.mean((1,2))),-1)
            for sf,law in itertools.product(("inlay","woven"),("rational","gaussian_rate")):
                stem=f"{SHAPES[i]}_{sf}_{law}_{VARS[v]}_h{nH}.npz"
                with np.load(ROOT/"results"/sp/stem) as p, np.load(ROOT/folder/sp/stem) as raw:
                    pp=p["pred"]
                    assert pp.shape==(2,1,4,128,4)
                    assert np.isfinite(pp).all() and pp.min()>=-1e-12 and pp.max()<=1+1e-12
                    assert list(p["decoders"])==["original","source_axis"]
                    p0,p1=pp[:,0]
                    y=raw["labels"].astype(float)
                    if rigid:
                        assert raw["valid"].all()
                        y=y[None]
                    assert y.shape==(count,3,4,128)
                    r0=risk(p0,y);delta=risk(p1,y)-r0
                    check=np.where(a[None,...,3,None].astype(bool),p1,p0)
                    risk_each=r0+delta*a[...,3]
                    np.testing.assert_allclose(((check-y[...,None])**2).mean(-1),risk_each,atol=3e-12,rtol=0.)
                    b=r0[...,None]+delta[...,None]*a
                    scores[panel,i,v,h]+=b.mean(axis=(2,3))/4.
                    effects[panel,i,v,h]+=np.stack(((delta*add).mean((2,3)),(-delta*remove).mean((2,3))),-1)/4.
                    if not rigid:material_changes[panel]+=(y[1]!=y[2]).mean()/48
                    samples+=y.size
    grouped=scores[:,:,:,:,filters].mean(axis=(2,3,4,5))
    avg=grouped.mean(1)
    shape_diff=np.stack([grouped[...,a]-grouped[...,b] for a,b in PAIRS],-1)
    diff=shape_diff.mean(1)
    knobs=effects[:,:,:,:,filters].mean(axis=(1,2,3,4,5))
    np.testing.assert_allclose(knobs.sum(-1),diff[:,2],atol=5e-14,rtol=0.)
    result=dict(law=folder,methods=METHODS,physical_cases=samples,absolute=ci(avg),absolute_panel=avg,
      primary=ci(diff,3),negative_panels=(diff<0).sum(0),family_loss=ci(grouped),family_contrasts=ci(shape_diff,6),
      by_area=ci(scores.mean((1,2,3,5))),by_height=ci(scores[:,:,:,:,filters].mean((1,2,4,5))),
      by_variant=ci(scores[:,:,:,:,filters].mean((1,3,4,5))),by_offset=ci(scores[:,:,:,:,filters].mean((1,2,3,4))),
      shape_variant_height_offsets=ci(scores[:,:,:,:,filters].mean(4)),
      oracle_diagnostic=ci(np.stack((avg[:,5]-avg[:,4],avg[:,3]-avg[:,5]),-1),2),
      shuffled_diagnostic=ci(avg[:,3]-avg[:,7]),rate=ci(rates.mean((1,2,3,4))),
      family_rate=ci(rates.mean((2,3,4))),reroutes=ci(reroutes.mean((1,2,3,4))),
      added_removed_d=ci(knobs,2),narrow_broad_label_change=(ci(material_changes) if not rigid else None))
    dest=ROOT/("score46_bullet.json" if rigid else "score46.json")
    dest.write_text(json.dumps(result,default=lambda x:np.asarray(x).tolist(),allow_nan=False,indent=1)+"\n")
    print(dest.name,"n=",samples,"joint differences:",diff.mean(0))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--folder",default="labels46")
    eval_run(p.parse_args().folder)
