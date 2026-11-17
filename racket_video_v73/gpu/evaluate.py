"""Compare paid calibration methods at matched incoming-speed coverage on SI outcomes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "racket_external_v72/method"))
import bench

CONTROL = "abs_ii_soft"
MARGINS = (0., .2, .5)
SPEED_QUANTILE = .75


def write_table(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def split_control(data):
    x, y, ids = data["train", 10]
    tx, ty, tids = data["test", 10]
    key = lambda r: tuple(map(int, r))
    existing = set(map(key, ids + tids))
    assert len(existing) == len(ids) + len(tids)
    orig = bench.V70 / "data/upstream/original/racket_10.csv"
    with orig.open() as f:
        rows = list(csv.DictReader(f))
    keys = [(int(r["racket_angle"]), int(r["traj_dir"])) for r in rows]
    assert len(set(keys)) == len(rows) and existing.issubset(keys)
    extra = [r for r, k in zip(rows, keys) if k not in existing]
    def si(prefix1, prefix2, names):
        return bench.as_cuda([[float(r[prefix1 + "x"]), float(r[prefix1 + "z"]),
                               bench.R * float(r[prefix2 + "y"])] for r in names])
    if extra:
        tx = torch.cat((tx, si("v_in_", "spin_before_", extra)))
        ty = torch.cat((ty, si("v_out_", "spin_after_", extra)))
    def show(k):
        return "%d/%03d" % key(k)
    return x,y,tx,ty,list(map(show,ids)),list(map(show,tids))+[show((r["racket_angle"],r["traj_dir"])) for r in extra],len(tids)


def balanced_tape(x, cutoff):
    fast = (x[:,1].abs() >= cutoff).cpu().numpy()
    f, s = np.flatnonzero(fast), np.flatnonzero(~fast)
    if min(len(f), len(s)) < 4:
        raise ValueError("Need four distinct calibration impacts on each side of source speed cutoff.")
    rows=[]
    for i in range(256):
        rng = np.random.default_rng(np.random.SeedSequence([730073,i]))
        rows.append(np.column_stack([rng.choice(f,4,replace=False),rng.choice(s,4,replace=False)]).ravel())
    return np.stack(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=HERE/"control")
    ap.add_argument("--external",type=Path)
    args=ap.parse_args()
    gpu = bench.idle_device()
    torch.set_default_dtype(torch.float64)
    bench.compare.BUDGETS = bench.BUDGETS
    data,hashes = bench.compare.old.load(bench.V70/"data/upstream")
    src = [n for n in bench.compare.GROUPS if args.external or n != CONTROL]
    source = [bench.compare.xy(data,"train",bench.compare.GROUPS[n]) for n in src]
    sx,sy = (torch.cat([pair[j] for pair in source]) for j in (0,1))
    cut = float(torch.quantile(sx[:,1].abs(),SPEED_QUANTILE))
    ext_hash = {}
    if args.external:
        q, r = args.external/"calibration.csv", args.external/"test.csv"
        cid,x,y = bench.read_external(q)
        tid,tx,ty = bench.read_external(r)
        if set(cid)&set(tid): raise ValueError("Overlapping impact IDs")
        official_n = len(tid)
        ext_hash = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (q,r)}
    else:
        x,y,tx,ty,cid,tid,official_n=split_control(data)
    official = torch.arange(len(tid),device="cuda") < official_n
    fast = tx[:,1].abs() >= cut
    masks = {"all":official,"fast":fast&official,"other":~fast&official}
    if (~official).any(): masks["raw_only"]=~official
    _,fit=bench.compare.old.empirical(sx,sy,sx[:1])
    rule,co,ca=bench.route_predictions(x,fit)
    _,orig,alt=bench.route_predictions(tx,fit)
    mis=rule!=bench.release(y)
    def loss(pred,target):
        v,w=bench.errors(pred,target)
        return v+bench.R*w
    delta=loss(ca,y)-loss(co,y)
    prior=bench.compare.make_prior(data,src,[0.,0.,0.])
    cal_slip=y[:,0]-y[:,2]; observed=ty[:,0]-ty[:,2]
    tapes={"balanced":balanced_tape(x,cut)}
    if not args.external:
        tapes["v72_uniform"]=np.stack([np.random.default_rng(np.random.SeedSequence([710071,10,i])).choice(len(x),32,replace=False)[:8] for i in range(256)])
    all_predictions={}
    summary=[]; draw_rows=[]; impact=[]; paid_list=[]; exact=[]; stats={}
    for sn,sub in masks.items():
        count=int(sub.sum())
        if not count: continue
        stats[sn]={"n":count,"normal_speed_range_mps":[float(tx[sub,1].abs().min()),float(tx[sub,1].abs().max())],
                   "resolved":{str(m):int(((observed.abs()>m)&sub).sum()) for m in MARGINS}}
    for schedule,tape in tapes.items():
        idx=torch.as_tensor(tape,device="cuda"); draws=len(tape)
        affine={k:p for k,_,p in bench.compare.affine_models(prior,x,y,tape,tx,[.1],True)}
        elastic=dict(bench.elastic_predictions(source,x,y,idx,tx))
        for k in bench.BUDGETS:
            paid=idx[:,:k]
            flip_c=delta[paid].sum(1)<0
            for margin in MARGINS:
                if schedule=="v72_uniform" and margin!=0:continue
                cal_ok=cal_slip[paid].abs()>margin
                flip_m=2*(mis[paid]&cal_ok).sum(1)>cal_ok.sum(1)
                labels=observed.abs()>margin
                if schedule=="balanced":
                    paid_list.append(dict(K=k,margin=margin,mean_sign_votes=float(cal_ok.double().sum(1).mean()),
                        no_vote_draws=int((cal_ok.sum(1)==0).sum()),majority_flips=int(flip_m.sum()),
                        continuous_flips=int(flip_c.sum()),draws=draws))
                method_preds={"majority":torch.where(flip_m[:,None,None],alt,orig),
                    "continuous_loss_gate":torch.where(flip_c[:,None,None],alt,orig),
                    "single_affine_eb":affine[k],"elastic_impulse":elastic[k]}
                for method,pred in method_preds.items():
                    verr,werr=bench.errors(pred,ty)
                    right=bench.release(pred)==bench.release(ty)
                    for group,mask in masks.items():
                        n=int(mask.sum())
                        if not n:continue
                        ok=mask&labels; ns=int(ok.sum())
                        vc=verr[:,mask].mean(1); wc=werr[:,mask].mean(1)
                        hits=right[:,ok].sum(1)
                        summary.append(dict(schedule=schedule,group=group,K=k,method=method,margin=margin,
                            held=n,resolved=ns,positive=int(((observed>0)&ok).sum()),draws=draws,
                            correct_draw_impact=int(hits.sum()),trials_draw_impact=ns*draws,
                            sign_coverage=ns/n,sign_accuracy=float(hits.double().mean()/ns) if ns else None,
                            velocity=float(vc.mean()),spin=float(wc.mean())))
                        if margin==.2 and schedule=="balanced":
                            for d,(v,w,h) in enumerate(zip(vc.tolist(),wc.tolist(),hits.tolist())):
                                draw_rows.append(dict(group=group,K=k,method=method,draw=d,velocity=v,spin=w,hits=h,eligible=ns))
                    if schedule=="balanced" and margin==.2:
                        all_predictions[f"{method}_k{k}"]=pred.cpu().numpy()
                        per_v=verr.mean(0).tolist(); per_w=werr.mean(0).tolist(); per_h=right.sum(0).tolist()
                        for j,label in enumerate(tid):
                            impact.append(dict(id=label,official=int(official[j]),fast=int(fast[j]),
                                incident_normal=float(tx[j,1].abs()),release=float(observed[j]),resolved=int(labels[j]),
                                K=k,method=method,draws=draws,correct=per_h[j] if bool(labels[j]) else None,
                                velocity=per_v[j],spin=per_w[j]))
                    if schedule=="v72_uniform":
                        exact.append((method,k,verr[:,official].mean(1).cpu().numpy(),werr[:,official].mean(1).cpu().numpy(),
                                      right[:,official].double().mean(1).cpu().numpy()))
    err={}
    if exact:
        ref=REPO/"racket_external_v72/method/retrospective/draws.csv"
        with ref.open() as f:old={(r['method'],int(r['k']),int(r['draw'])):r for r in csv.DictReader(f) if r['rubber']==CONTROL}
        for field,i in [('velocity',0),('spin',1),('output_sign_acc',2)]:
            err[field]=max(abs(values[i][d]-float(old[m,k,d][field])) for m,k,*values in exact for d in range(len(values[i])))
            assert err[field]<1e-8
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    write_table(out/"metrics.csv",summary);write_table(out/"draws.csv",draw_rows)
    write_table(out/"impacts.csv",impact);write_table(out/"paid.csv",paid_list)
    np.savez_compressed(out/"predictions.npz",target=ty.cpu().numpy(),incident=tx.cpu().numpy(),**all_predictions)
    np.savez_compressed(out/"splits.npz",**tapes,calibration_ids=np.asarray(cid),target_ids=np.asarray(tid))
    meta=dict(source=src,source_train=len(sx),product="external" if args.external else CONTROL,
        target_calibration=len(x),calibration_fast=int((x[:,1].abs()>=cut).sum()),
        target_held=official_n,raw_unassigned=len(tid)-official_n,source_speed_q=SPEED_QUANTILE,fast_cutoff_mps=cut,
        endpoint_counts=stats,v72_max_abs_error=err,source_hashes=hashes,external_hashes=ext_hash,
        raw_control_sha256=None if args.external else hashlib.sha256((bench.V70/'data/upstream/original/racket_10.csv').read_bytes()).hexdigest(),
        gpu_before=gpu,torch=torch.__version__,node=platform.node(),job=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID"))
    (out/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({k:v for k,v in meta.items() if k not in ("source_hashes","external_hashes")},indent=2))
if __name__=="__main__":main()
