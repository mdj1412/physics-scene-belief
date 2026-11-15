"""Paired, post-protocol decomposition of changes in policy and in the physical loss table."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from physics26 import V25Engine, Engine, torch, t, LEVELS
from physics25 import generate as generate_v25
from stats26 import crossed_means, expanded, interval, verify_records

R=Path("results")


def max_error(a,b):
    if a.shape!=b.shape:
        raise ValueError((a.shape,b.shape))
    return float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))


def origin_check(design, xs, label, cell, theta, sensors):
    base=Path.cwd()
    out={}
    # Cell/latent/noise RNG stream is v25's; CPU yields floating-point roundoff versus CUDA.
    with TemporaryDirectory(prefix="v26_origin_") as tmp:
        try:
            os.chdir(tmp)
            generate_v25("origin",len(cell),int(design["seed"]))
            with np.load(R/"origin/public.npz") as f:out["factual"]=max_error(f["x"][:,0],xs[:,0])
            with np.load(R/"origin/private.npz") as f:
                out.update(theta=max_error(f["theta"],theta[:,0]),labels=max_error(f["label"],label[:,0]),
                           cell=max_error(f["cell"],cell))
                for a in (1,2,3):out[f"obs{a}"]=max_error(f[f"obs{a}"][:,0],sensors[a][:,0])
        finally:os.chdir(base)
    assert max(out.values())<1e-10,out
    pair=tuple(design["banks"][0])
    orig=V25Engine(pair)
    current=Engine(pair,1.,1.)
    bank_keys={}
    for bank in ("inner","outer"):
        x,y=getattr(orig,bank),getattr(current,bank)
        bank_keys[bank]={"theta":torch.equal(x.theta,y.theta),"label":torch.equal(x.label,y.label),
            "mu":all(torch.equal(x.mu[j],y.mu[j]) for j in range(4))}
        assert all(bank_keys[bank].values())
    ids=np.concatenate([np.flatnonzero(cell==i)[:2] for i in range(4)])
    diff_cpu=0.;diff_gpu=0.;n=0
    recorded=np.load(R/"eval_0.npz")["loss"]
    for i in ids:
        # Recover GPU probabilities from squared loss and binary target: p in [0,1].
        y=float(label[i,0]);xx=t(xs[i,0])
        wo=orig.inner.fit(xx);wc=current.inner.fit(xx)
        for j,a in enumerate((1,3)):
            sensor=t(sensors[a][i,0])
            po=(orig.inner.update(wo,a,sensor)@orig.inner.label[:,0]/np.sqrt(2.)).numpy()
            pc=(current.inner.update(wc,a,sensor)@current.inner.label[:,0]/np.sqrt(2.)).numpy()
            residual=np.sqrt(recorded[i,0,j])
            pgpu=residual if y==0 else 1.-residual
            diff_cpu=max(diff_cpu,max_error(po,pc));diff_gpu=max(diff_gpu,max_error(pgpu,pc))
            assert np.allclose(po,pc,rtol=0,atol=1e-12)
            assert np.allclose(pgpu,pc,rtol=0,atol=1e-8),(i,a,diff_gpu)
            n+=len(pc)
    return dict(max_abs_generated_origin_error=out,bitwise_cpu_bank_equality=bank_keys,
                max_abs_cpu_v25_vs_cpu_v26_probability=diff_cpu,max_abs_cpu_vs_logged_gpu_probability=diff_gpu,
                probability_checks=n,checked_world_ids=ids.tolist())


def pick(table,act):
    return np.take_along_axis(table,act,axis=-1)


def main():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("BELIEF_DEVICE")!="cpu":
        raise RuntimeError("Run on allocated CPU with BELIEF_DEVICE=cpu and CUDA_VISIBLE_DEVICES empty.")
    verify_records()
    torch.set_num_threads(1)
    design=json.loads((R/"design.json").read_text())
    with np.load(R/"test/private.npz") as f:
        label=f["label"].astype(bool);cell=f["cell"];theta=f["theta"];sensors={a:f[f"obs{a}"] for a in (1,2,3)}
    xs=np.load(R/"test/public.npz")["x"]
    origin=origin_check(design,xs,label,cell,theta,sensors)
    del sensors,xs
    raw_a=[];raw_l=[]
    for b,pair in enumerate(design["banks"]):
        with np.load(R/f"choices_{b}.npz") as f:raw_a.append((f["acts"][...,:2]==3).astype(int))
        with np.load(R/f"eval_{b}.npz") as f:raw_l.append(f["loss"].mean(-1))
    variants=np.load(R/"scenarios.npz")["variants"]
    act=expanded(np.stack(raw_a),variants)
    table=expanded(np.stack(raw_l),variants)
    unchanged=np.broadcast_to(act[:,:,0:1],act.shape)
    base=np.broadcast_to(table[:,:,:1],table.shape)
    # Corners (policy,environment); each is the same physical row's value.
    v00=pick(base,unchanged)
    v10=pick(base,act)
    v01=pick(table,unchanged)
    v11=pick(table,act)
    policy=.5*((v10-v00)+(v11-v01))
    environment=.5*((v01-v00)+(v11-v10))
    interaction=v11-v10-v01+v00
    drift=v11-v00
    assert np.max(np.abs(drift-policy-environment))<1e-12
    # Last object contrasts gate decomposition minus K256 decomposition.
    comp=np.stack([v00,v10,v01,v11,policy,environment,interaction,drift],axis=-1)
    comp=np.concatenate([comp,comp[:,:,:,:1]-comp[:,:,:,1:2]],axis=3)
    b,w,c,m,q=comp.shape
    boot=crossed_means(comp.reshape(b,w,-1),cell,n=5000).reshape(-1,c,m,q)
    points=comp.mean((0,1))
    names=["gate","k256","gate_minus_k256"]
    keys=["origin","shifted_actions_origin_environment","origin_actions_shifted_environment","physical_shift",
          "action_component","environment_component","interaction","drift"]
    rows=[]
    index=np.arange(w)
    for h,(ac,af) in enumerate(design["conditions"]):
        terms={name:{key:interval(boot[:,h,i,j],points[h,i,j]) for j,key in enumerate(keys)} for i,name in enumerate(names)}
        flip=label[index,variants[h]]!=label[:,0]
        row=dict(idx=h,alpha_c=ac,alpha_f=af,methods=terms,flip_labels=int(flip.sum()),
                 flip_label_fraction=float(flip.mean()),
                 unchanged_by_cell={str(k):dict(unchanged=int((~flip & (cell==k)).sum()),n=int((cell==k).sum())) for k in range(4)})
        rows.append(row)
    variant_counts={}
    for v in range(4):
        f=label[:,v]!=label[:,0]
        variant_counts[str(v)]={str(k):dict(changed=int((f&(cell==k)).sum()),n=int((cell==k).sum())) for k in range(4)}
    assert not (label[:,2]!=label[:,0]).any() and np.array_equal(label[:,3],label[:,1])
    result=dict(origin_check=origin,variant_label_flips=variant_counts,rows=rows,
            exact_recomposition_max_error=float(np.max(np.abs(drift-policy-environment))),
            component_bootstrap_draws=len(boot),comparisons="exploratory paired percentile 95% intervals")
    tmp=R/"counterfactual.tmp";tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(R/"counterfactual.json")
    print("Origin check",origin["max_abs_cpu_vs_logged_gpu_probability"],"decomposition",result["exact_recomposition_max_error"],flush=True)

if __name__=="__main__":main()
