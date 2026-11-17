"""Evaluate measured tangential speed and spin without pseudo-observations of normal velocity."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch
from bench import BUDGETS, HERE, V70, as_cuda, compare, elastic, idle_device, route_predictions

R = .019
OUT = HERE / "feint"


def marginal_precision(full_precision, ix):
    cov = torch.linalg.inv(full_precision)
    return torch.linalg.inv(cov[ix][:,ix])


def affine_sufficient(prior, x, y, ix):
    a=compare.matrix((x-prior["mean"])/prior["sd"])
    h=torch.einsum("ni,op->noip",a,torch.eye(3,device=x.device)[ix]).reshape(len(x),len(ix),12)
    slope=prior["slope"].flatten()
    ht=h.mT @ marginal_precision(prior["noise_inv"],ix)
    return h,slope,.1*prior["precision"],ht@h,ht@(y-h@slope).unsqueeze(-1)


def impulse_sufficient(source, x, y, ix):
    center,scale,mu,prior,noise=map(as_cuda,elastic.source_fit([(a.cpu().numpy(),b.cpu().numpy()) for a,b in source]))
    slip=x[:,0]-x[:,2]
    z=torch.stack([(x[:,1].abs()-center[0])/scale[0],(slip.abs()-center[1])/scale[1]],dim=1)
    h=x.new_zeros((len(x),3,4))
    h[:,0,2]=-slip;h[:,0,3]=-slip*z[:,1]
    h[:,2,2]=1.5*slip;h[:,2,3]=1.5*slip*z[:,1]
    h[:,1,0]=-x[:,1];h[:,1,1]=-x[:,1]*z[:,0]
    h=h[:,ix]
    ht=h.mT@marginal_precision(noise,ix)
    return z,mu,prior,ht@h,ht@(y-(x*x.new_tensor([1,0,1]))[:,ix]-h@mu).unsqueeze(-1)


def canonical_rows(path, full):
    with path.open() as f: rows=sorted(csv.DictReader(f),key=lambda r:float(r["n_in_rps"]))
    ids=[r["id"] for r in rows]
    if len(set(ids))!=len(rows) or len(rows)<9:
        raise ValueError("Require at least nine unique linked measurements")
    y=as_cuda([[float(r["u_out_mps"]),*([float(r["e_normal"])*30/3.6] if full else []),R*math.tau*float(r["n_out_rps"])] for r in rows])
    w=as_cuda([math.tau*float(r["n_in_rps"]) for r in rows])
    if not torch.isfinite(y).all() or not torch.isfinite(w).all():
        raise ValueError("Nonfinite measurement")
    return ids,w,y


def estimate_subsets(x,y,sg,y0,y1,aff,imp,tape,held,ix,threshold):
    idx=torch.as_tensor(tape,device=x.device)
    weights=as_cuda(held)
    denom=weights.sum(1)
    assert (denom>0).all() and ((weights.gather(1,idx))==0).all()
    sign=y[:,0]>y[:,-1]
    valid=(y[:,0]-y[:,-1]).abs()>threshold
    mismatch=sg!=sign
    use_flip=2*(mismatch[idx]&valid[idx]).sum(1)>valid[idx].sum(1)
    def cost(p):
        d=(p-y).abs()
        return torch.linalg.vector_norm(d[:,:-1],dim=-1)+d[:,-1]
    risk=cost(y1)-cost(y0)
    risk_flip=risk[idx].sum(1)<0
    h,b0,p0,gram,v=aff
    b=b0+torch.linalg.solve(p0+gram[idx].sum(1),v[idx].sum(1)).squeeze(-1)
    affine=torch.einsum("nop,dp->dno",h,b)
    z,b0,p0,gram,v=imp
    b=b0+torch.linalg.solve(p0+gram[idx].sum(1),v[idx].sum(1)).squeeze(-1)
    alpha=(b[:,2,None]+b[:,3,None]*z[:,1]).clamp(0,.8)
    slip=x[:,0]-x[:,2]
    e=(b[:,0,None]+b[:,1,None]*z[:,0]).clamp(0,1)
    impulse=torch.stack([x[:,0]-alpha*slip,-e*x[:,1],x[:,2]+1.5*alpha*slip],-1)[:,:,ix]
    out={}
    for name,flip,pred in [("majority",use_flip,None),("continuous_loss_gate",risk_flip,None),
                            ("single_affine_eb",None,affine),("elastic_impulse",None,impulse)]:
        if flip is not None:
            pred=torch.where(flip[:,None,None],y1,y0)
        delta=(pred-y).abs()
        def average(value):return (value*weights).sum(1)/denom
        sign_weights=weights*valid
        sign_count=sign_weights.sum(1)
        def signed(value):
            return torch.where(sign_count>0,(value*sign_weights).sum(1)/sign_count,torch.nan)
        vals={"velocity":average(torch.linalg.vector_norm(delta[:,:,:-1],dim=-1)),"spin":average(delta[:,:,-1])/R,
              "output_sign_acc":signed((pred[:,:,0]>pred[:,:,-1])==sign)}
        if flip is not None:
            disagreement=(mismatch*sign_weights).sum(1)
            match=flip==(2*disagreement>sign_count)
            vals.update(flip=flip.double(),route_sign_acc=signed((sg^flip[:,None])==sign),
                        orientation_acc=torch.where(2*disagreement!=sign_count,match.double(),torch.nan))
        out[name]={m:v.cpu().numpy() for m,v in vals.items()}
    return out


def evaluate(source,prior,fit,ids,w,y,normal_speed,tangent,primary_only,ix,threshold):
    x=torch.stack([torch.full_like(w,tangent),torch.full_like(w,-normal_speed/3.6),R*w],dim=1)
    sg,y0,y1=route_predictions(x,fit)
    y0,y1=y0[:,ix],y1[:,ix]
    aff=affine_sufficient(prior,x,y,ix);imp=impulse_sufficient(source,x,y,ix)
    n=len(ids);pool=np.rint(np.linspace(0,n-1,8)).astype(int)
    common=np.setdiff1d(np.arange(n),pool)
    folds={};records=[]
    signs=(y[:,0]-y[:,-1]).cpu().numpy()
    labels=[dict(id=sample,measured_s_out=float(s),source_positive=bool(g)) for sample,s,g in zip(ids,signs,sg.cpu())]
    modes=["common_holdout"] if primary_only else ["common_holdout","all_disjoint_subsets"]
    for mode in modes:
        pool_now=pool if mode=="common_holdout" else np.arange(n)
        for k in BUDGETS:
            comb=list(itertools.combinations(pool_now,k))
            rows={}
            for start in range(0,len(comb),512):
                tape=np.array(comb[start:start+512])
                if mode=="common_holdout":
                    mask=np.zeros((len(tape),n));mask[:,common]=1
                else:
                    mask=np.ones((len(tape),n));np.put_along_axis(mask,tape,0,axis=1)
                metrics=estimate_subsets(x,y,sg,y0,y1,aff,imp,tape,mask,ix,threshold)
                for name,vals in metrics.items():
                    for field,arr in vals.items():rows.setdefault(name,{}).setdefault(field,[]).extend(arr.tolist())
                    if mode=="common_holdout" and not primary_only:
                        for d in range(len(tape)):
                            records.append(dict(k=k,method=name,paid="|".join(ids[j] for j in tape[d]),
                                held="|".join(ids[j] for j in common),**{a:float(v[d]) for a,v in vals.items()}))
            fits={}
            for name,vals in rows.items():
                stats={}
                for field,arr in vals.items():
                    finite=np.asarray(arr)[np.isfinite(arr)]
                    stats[field]={"mean":float(finite.mean()),"min":float(finite.min()),"max":float(finite.max()),"count":len(finite)} if len(finite) else dict(mean=None,count=0)
                fits[name]=stats
            folds.setdefault(mode,{})[k]=dict(paid_assignments=len(comb),held_points=len(common) if mode=="common_holdout" else n-k,methods=fits)
    return dict(n=n,pool=[ids[i] for i in pool],held=[ids[i] for i in common],nominal_kmh=normal_speed,assumed_tangent_mps=tangent,
                row_signs=labels,results=folds),records


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csv",type=Path)
    ap.add_argument("--output",type=Path,default=OUT)
    ap.add_argument("--derived-normal",action="store_true",help="Secondary planar endpoint using e normal times nominal incident speed")
    args=ap.parse_args();gpu=idle_device()
    torch.set_default_dtype(torch.float64)
    data,hashes=compare.old.load(V70/"data"/"upstream"); groups=list(compare.GROUPS)
    source=[compare.xy(data,"train",compare.GROUPS[g]) for g in groups]
    sx,sy=(torch.cat([s[j] for s in source]) for j in (0,1))
    _,fit=compare.old.empirical(sx,sy,sx[:1])
    prior=compare.make_prior(data,groups,[0.,0.,0.])
    ids,w,y=canonical_rows(args.csv,args.derived_normal)
    ix=[0,1,2] if args.derived_normal else [0,2]
    res,rows=evaluate(source,prior,fit,ids,w,y,30.,0.,False,ix,0.)
    sens={}
    for normal,tangent,threshold in ((27.,0.,0.),(33.,0.,0.),(30.,-.1,0.),(30.,.1,0.),(30.,0.,-.01),(30.,0.,.20)):
        sens[str((normal,tangent,threshold))]=evaluate(source,prior,fit,ids,w,y,normal,tangent,True,ix,threshold)[0]["results"]
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    res.update(endpoint="approximate_planar" if args.derived_normal else "measured_tangential",sensitivity=sens,job=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID"),host=platform.node(),gpu_before=gpu,
        torch=torch.__version__,source_products=groups,source_train_count=len(sx),
        normalized_csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),source_hashes=hashes)
    (out/"result.json").write_text(json.dumps(res,indent=2)+'\n')
    with (out/"primary_rows.csv").open("w") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    print("points",len(ids),"fixed held",res["held"])
    for mode,fold in res["results"].items():
        for k,meta in fold.items():
            print(mode,k,[(m,round(s["velocity"]["mean"],4),round(s["spin"]["mean"],3),
                           round(s["output_sign_acc"]["mean"],4)) for m,s in meta["methods"].items()])
if __name__=="__main__":main()
