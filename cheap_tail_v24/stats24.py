"""Crossed, factorial-stratified paired percentile bootstrap for the prespecified estimands."""
from pathlib import Path
import json,os,hashlib
import numpy as np
R=Path("results")
def read(name):return json.loads((R/(name+".json")).read_text())
def draw_worlds(rng,groups,n):
    return np.concatenate([g[rng.integers(len(g),size=(n,len(g)))] for g in groups],axis=1)
def boot_means(values,cells,n=10000):
    # B,W,C,J. Share world/bank draws across every calibration and policy.
    rng=np.random.default_rng(240017);b,w,c,j=values.shape
    groups=[np.flatnonzero(cells==k) for k in range(6)]
    out=[]
    for start in range(0,n,50):
        size=min(50,n-start);idx=draw_worlds(rng,groups,size)
        bank=rng.multinomial(b,np.ones(b)/b,size)/b
        counts=np.zeros((size,w));np.add.at(counts,(np.arange(size)[:,None],idx),1/w)
        avg=(counts@values.transpose(1,0,2,3).reshape(w,-1)).reshape(size,b,c,j)
        out.append(np.einsum("sb,sbcj->scj",bank,avg))
    return np.concatenate(out)
def boot_latency(times,cells):
    # B,C,W,R,J. Repeat is a randomized timing block; preserve each method's paired observation.
    rng=np.random.default_rng(249017);b,c,w,reps,j=times.shape
    layout=times.transpose(0,2,3,1,4)
    groups=[np.flatnonzero(cells==k) for k in range(6)]
    out=[]
    for _ in range(500):
        size=10
        bank=rng.integers(b,size=(size,b,1,1))
        world=draw_worlds(rng,groups,size)[:,None,:,None]
        repeat=rng.integers(reps,size=(size,1,1,reps))
        sample=layout[bank,world,repeat].reshape(size,-1,c,j)
        p90=np.quantile(sample,.9,axis=1)
        out.append(p90/p90[:,:,[1]])
    return np.concatenate(out)
def main():
    assert os.environ.get("SLURM_JOB_ID")
    for n in ("sources","plan","choices_freeze"):
        d=read(n)
        for p,h in d.get("files",d).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    cfg=read("selected")
    cal=["camera","probe","balanced"]
    choices=[np.load(R/f"choices_{i}.npz") for i in range(5)]
    a=np.stack([z["acts"] for z in choices])-1
    fire=choices[0]["fire"]
    assert all(np.array_equal(fire,z["fire"]) for z in choices)
    assert np.array_equal(a[...,0],np.where(fire,a[...,3],a[...,2]))
    ev=[np.load(R/f"eval_{i}.npz") for i in range(5)]
    ts=np.stack([z["times"] for z in ev]);idx=ev[0]["ids"]
    assert all(np.array_equal(idx,z["ids"]) for z in ev)
    costs=np.stack([z["loss"].mean(-1) for z in ev])
    assert np.isfinite(costs).all() and np.isfinite(ts).all()
    pp=np.load(R/"test/private.npz");cell=pp["cell"]
    assert np.all(np.bincount(cell)==192) and np.all(np.bincount(cell[idx])==12)
    pick=np.take_along_axis(costs,a,axis=3)
    prop=fire.mean(0)
    blind=pick[...,2]*(1-prop)+pick[...,3]*prop
    risk=np.concatenate([pick,blind[...,None]],axis=-1)
    gap=risk-risk[...,1,None]
    boot=boot_means(gap,cell)
    tboot=boot_latency(ts,cell[idx]);rows={}
    for j,name in enumerate(cal):
        times=ts[:,j].reshape(-1,5)
        p50,p90,p95=np.quantile(times,[.5,.9,.95],axis=0)
        ub=np.quantile(boot[:,j],.95,axis=0)
        tub=np.quantile(tboot[:,j],.95,axis=0)
        delta=gap[:,:,j].mean((0,1))
        ratios=p90/p90[1]
        ci=np.quantile(boot[:,j,:,]-boot[:,j,:1],[.025,.975],axis=0)
        # Gate minus each other policy, with paired bootstrap intervals.
        gdiff=np.stack([-delta+delta[0],-ci[1],-ci[0]],axis=-1)
        flip=a[:,:,j,0]!=a[:,:,j,1]
        on=fire[:,j]
        dg=pick[:,:,j,2]-pick[:,:,j,3]
        def masked(v,m):return float(v[:,m].mean()) if np.any(m) else None
        strata=[]
        for k in range(6):
            mask=cell==k
            strata.append(dict(n=int(mask.sum()),high=float(on[mask].mean()),stick=pp["label"][mask].mean(),
                       loss=risk[:,mask,j,:2].mean((0,1)).tolist(),diff=masked(gap[:,:,j,0],mask),
                       flip=masked(flip,mask)))
        rows[name]=dict(loss=risk[:,:,j].mean((0,1)).tolist(),delta=delta.tolist(),ub=ub.tolist(),
             p50=p50.tolist(),p90=p90.tolist(),p95=p95.tolist(),ratio=ratios.tolist(),ratio_ub=tub.tolist(),
             gate_comparisons=gdiff.tolist(),brier_pass=bool(ub[0]<.003),p90_pass=bool(tub[0]<.95),
             rate=on.mean(),mean_k=(cfg["low"]+(cfg["high"]-cfg["low"])*on.mean()),strata=strata,
             flip=float(flip.mean()),world_flips=int(flip.any(0).sum()),
             flip_delta=float(gap[:,:,j,0][flip].mean()) if flip.any() else None,
             low_minus_high_on=masked(dg,on),low_minus_high_off=masked(dg,~on),
             bank_delta=gap[:,:,j,0].mean(1).tolist(),
             bank_ratio=(np.quantile(ts[:,j],.9,axis=(1,2))[:,0]/np.quantile(ts[:,j],.9,axis=(1,2))[:,1]).tolist())
    result=dict(seed=241037,rows=rows,pooled_delta=gap.mean((0,1,2)).tolist(),
        pooled_ub=np.quantile(boot.mean(1),.95,axis=0).tolist(),joint=all(v["brier_pass"] and v["p90_pass"] for v in rows.values()),
        correlation=np.corrcoef(pp["theta"][:,[2,3,4]].T).tolist(),n_cells=np.bincount(cell).tolist())
    (R/"summary.json").write_text(json.dumps(result,indent=2))
    for n,v in rows.items():print(n,[(k,v[k][0]) for k in ("loss","delta","ub","ratio","ratio_ub")],flush=True)
if __name__=="__main__":main()
