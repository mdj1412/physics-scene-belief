"""Fixed-budget release routing diagnostic and independent product entry point."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
V70 = PROJECT / "racket_transfer_v70"
V71 = PROJECT / "racket_calibration_v71"
sys.path[:0] = [str(V71 / "baselines"), str(V71 / "analysis")]
import compare
import elastic

R = .02
BUDGETS = (4, 8)
METHODS = ("majority", "continuous_loss_gate", "single_affine_eb", "elastic_impulse")


def idle_device():
    def query(*options):
        return subprocess.check_output(["nvidia-smi", *options], text=True).strip()
    gpu = query("--query-gpu=name,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits")
    procs = query("--query-compute-apps=pid", "--format=csv,noheader")
    if len(gpu.splitlines()) != 1 or procs or int(gpu.split(",")[-1]) or int(gpu.split(",")[-2]) > 50:
        raise RuntimeError(f"GPU is occupied: {gpu} {procs}")
    return gpu


def as_cuda(a):
    return torch.as_tensor(a, device="cuda", dtype=torch.float64)


def read_external(path):
    """Canonical conversion to m/s, radians/s, 0.02-m radius must precede import."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    ids = [r["impact_id"] for r in rows]
    if not ids or any(not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("impact_id must be unique within split")
    def state(a, b, w):
        v = as_cuda([[float(r[a]), float(r[b]), R * float(r[w])] for r in rows])
        if not torch.isfinite(v).all():
            raise ValueError("Encountered nonfinite state")
        return v
    return ids, state("v_in_x","v_in_z","spin_before_y"), state("v_out_x","v_out_z","spin_after_y")


def route_predictions(x, fit):
    f = compare.matrix((x-as_cuda(fit["mean"])) / as_cuda(fit["std"]))
    route = f @ as_cuda(fit["logistic"]) > 0
    plus, minus = (f @ as_cuda(fit["expert_"+side]) for side in ("plus","minus"))
    return route, torch.where(route[:,None], plus, minus), torch.where(route[:,None], minus, plus)


def errors(p, y):
    diff = p-y
    return torch.linalg.vector_norm(diff[..., :2],dim=-1), diff[..., 2].abs()/R


def release(y):
    return y[...,0]-y[...,2] > 0


def elastic_predictions(source, x, y, idx, tx):
    # Source fitting reuses the v71 reference. Target posterior and prediction use CUDA.
    src = [(a.cpu().numpy(),b.cpu().numpy()) for a,b in source]
    center, sd, prior, precision, noise_precision = map(as_cuda,elastic.source_fit(src))
    def coordinates(v):
        return (torch.stack([v[:,1].abs(),(v[:,0]-v[:,2]).abs()],1)-center)/sd
    z=coordinates(x)
    slip=x[:,0]-x[:,2]
    h=x.new_zeros((len(x),3,4))
    h[:,0,2]=-slip; h[:,0,3]=-slip*z[:,1]
    h[:,2,2]=1.5*slip; h[:,2,3]=1.5*slip*z[:,1]
    h[:,1,0]=-x[:,1]; h[:,1,1]=-x[:,1]*z[:,0]
    j=h.mT@noise_precision
    gram=j@h
    rhs=j@(y-x*x.new_tensor([1,0,1])-h@prior).unsqueeze(-1)
    z=coordinates(tx); slip=tx[:,0]-tx[:,2]
    for k in BUDGETS:
        paid=idx[:,:k]
        beta=prior+torch.linalg.solve(precision+gram[paid].sum(1),rhs[paid].sum(1)).squeeze(-1)
        e=(beta[:,0,None]+beta[:,1,None]*z[:,0]).clamp(0,1)
        a=(beta[:,2,None]+beta[:,3,None]*z[:,1]).clamp(0,.8)
        yield k,torch.stack((tx[:,0]-a*slip,-e*tx[:,1],tx[:,2]+1.5*a*slip),dim=-1)


def evaluate(data, groups, x,y, tx,ty, tape):
    src=[compare.xy(data,"train",compare.GROUPS[g]) for g in groups]
    source_x,source_y=(torch.cat([s[j] for s in src]) for j in (0,1))
    _,fit=compare.old.empirical(source_x,source_y,source_x[:1])
    croute,orig,alt=route_predictions(x,fit)
    troute,t_orig,t_alt=route_predictions(tx,fit)
    sign_mismatch=croute!=release(y)
    ec,sc=errors(orig,y);ea,sa=errors(alt,y)
    loss_delta=ea+R*sa-ec-R*sc
    t_v,t_w=errors(t_orig,ty);t_a,t_b=errors(t_alt,ty)
    held_cbetter=(t_a+R*t_b).mean()<(t_v+R*t_w).mean()
    count=int((troute!=release(ty)).sum())
    held_orientation=None if 2*count==len(ty) else 2*count>len(ty)
    idx=torch.as_tensor(tape,device="cuda")
    def pack(p,flip=None):
        v,w=errors(p,ty)
        metrics=[v.mean(1),w.mean(1),(release(p)==release(ty)).double().mean(1)]
        if flip is not None:
            route=troute.unsqueeze(0)^flip[:,None]
            metrics.extend([flip.double(), (route==release(ty)).double().mean(1),
                            (flip==held_cbetter).double()])
        return torch.stack(metrics,1).cpu().numpy()
    results={}
    for k in BUDGETS:
        paid=idx[:,:k]
        for m,flag in [("majority",2*sign_mismatch[paid].sum(1)>k),
                       ("continuous_loss_gate",loss_delta[paid].sum(1)<0)]:
            p=torch.where(flag[:,None,None],t_alt,t_orig)
            results[m,k]=pack(p,flag)
    # make_prior bypasses the unused expensive inclined-rigid fit when given any physical anchor.
    prior=compare.make_prior(data,groups,[0.,0.,0.])
    for k,_,p in compare.affine_models(prior,x,y,tape,tx,[.1],True):
        if k in BUDGETS:results["single_affine_eb",k]=pack(p)
    for k,p in elastic_predictions(src,x,y,idx,tx):
        results["elastic_impulse",k]=pack(p)
    info=dict(target_train=len(x),target_test=len(tx), held_mismatches=count,
              held_flip_by_frequency=held_orientation,held_flip_by_loss=bool(held_cbetter),
              original=[float(t_v.mean()),float(t_w.mean())],flipped=[float(t_a.mean()),float(t_b.mean())])
    return results,info


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--external",type=Path,help="Directory of normalized calibration.csv and test.csv")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    gpu=idle_device()
    torch.set_default_dtype(torch.float64)
    data,hashes=compare.old.load(V70/"data"/"upstream")
    groups=list(compare.GROUPS)
    draw_ids={}; records=[]; folds={}
    ext_hash={}
    if args.external:
        cal, test = (args.external/(s+".csv") for s in ("calibration","test"))
        cid,x,y=read_external(cal);tid,tx,ty=read_external(test)
        if set(cid)&set(tid) or len(cid)<8:
            raise ValueError("Disjoint splits and at least eight calibration shots required")
        ext_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (cal,test)}
        products=["external_product"]
    else:
        products=groups
    for product in products:
        if args.external:
            src=groups
            tape=np.stack([np.random.default_rng(np.random.SeedSequence([720072,i])).choice(len(x),8,replace=False) for i in range(256)])
        else:
            cfg=compare.GROUPS[product]
            src=[g for g in groups if g!=product]
            x,y=compare.xy(data,"train",cfg);tx,ty=compare.xy(data,"test",cfg)
            tape=np.stack([np.random.default_rng(np.random.SeedSequence([710071,min(cfg),i])).choice(len(x),32,replace=False)[:8] for i in range(256)])
        compare.BUDGETS=BUDGETS
        draw_ids[product]=tape
        metrics,folds[product]=evaluate(data,src,x,y,tx,ty,tape)
        for (m,k),arr in metrics.items():
            for d,vals in enumerate(arr):
                records.append([product,m,k,d,*vals,*([None]*(6-len(vals)))])
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    with (out/"draws.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["rubber","method","k","draw","velocity","spin","output_sign_acc","flip","route_sign_acc","choose_min_held_loss"])
        w.writerows(records)
    np.savez_compressed(out/"draw_indices.npz",**draw_ids)
    (out/"run.json").write_text(json.dumps(dict(mode="external" if args.external else "retrospective",
        node=platform.node(),job=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID"),
        torch=torch.__version__,gpu_before=gpu,source_inputs=hashes,external_hashes=ext_hash,folds=folds),indent=2))

if __name__=="__main__":
    main()
