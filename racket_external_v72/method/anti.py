"""Dimensionless transfer evaluation with unknown high-speed clock. Never outputs SI errors."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform

import numpy as np
import torch
from bench import HERE, V70, as_cuda, compare, idle_device
from pathlib import Path

SCALE=8.0
BUDGETS=(4,8)
METHODS=('majority','continuous_loss_gate','single_affine_eb','elastic_impulse')

def normalize(x,y):
    if torch.any(x[:,1]>=-1e-10) or torch.any(y[:,1]<=0):
        raise ValueError("Require directed incidence vz < 0 and rebound vz > 0")
    c=SCALE/x[:,1:2].abs()
    return x*c,y*c

def design(x,mean,sd):
    return compare.matrix((x[:,[0,2]]-mean)/sd)

def make_prior(src):
    xx,yy=(torch.cat([s[j] for s in src]) for j in (0,1))
    mean=xx[:,[0,2]].mean(0);sd=xx[:,[0,2]].std(0,correction=0)
    a=design(xx,mean,sd)
    sign=yy[:,0]>yy[:,2]
    def ols(v,w):return torch.linalg.lstsq(v,w).solution
    beta=a.new_zeros(3); reg=torch.diag(a.new_tensor([0,1,1]))
    for _ in range(50):
        pp=torch.sigmoid((a@beta).clamp(-35,35))
        step=torch.linalg.solve(a.T@((pp*(1-pp))[:,None]*a)+reg,a.T@(pp-sign.double())+reg@beta)
        beta-=step
        if float(step.norm())<1e-10:break
    heads=[ols(a[sign==s],yy[sign==s]) for s in (False,True)]
    B=ols(a,yy);ds=[];noises=[];elastic=[];est_noise=[]
    slip=(xx[:,0]-xx[:,2]).abs()
    center,scale=slip.mean(),slip.std(correction=0)
    for x,y in src:
        a=design(x,mean,sd);b=ols(a,y);e=y-a@b
        ds.append((b-B).ravel());noises.append(e.T@e/(len(x)-3))
        h=impact_design(x,center,scale)
        b=ols(h.reshape(-1,3),(y-x*x.new_tensor([1,0,1])).reshape(-1,1)).ravel()
        e=y-x*x.new_tensor([1,0,1])-h@b
        elastic.append(b);est_noise.append(e.T@e/len(x))
    ds=torch.stack(ds);cov=ds.T@ds/len(ds)
    prec=.1*torch.linalg.inv(.85*cov+.15*torch.diag(cov.diag().clamp(min=.0001)))
    noise=torch.stack(noises).mean(0)+.0001*torch.eye(3,device=xx.device)
    elastic=torch.stack(elastic)
    ep=torch.linalg.inv(torch.cov(elastic.T)+.0004*torch.eye(3,device=xx.device))
    en=(torch.stack(est_noise).mean(0)+.0025*torch.eye(3,device=xx.device))
    return mean,sd,beta,heads,B,prec,torch.linalg.inv(noise),center,scale,elastic.mean(0),ep,torch.linalg.inv(en)

def impact_design(x,center,scale):
    s=x[:,0]-x[:,2];z=(s.abs()-center)/scale
    h=x.new_zeros((len(x),3,3))
    h[:,1,0]=-x[:,1]
    h[:,0,1]=-s;h[:,0,2]=-s*z
    h[:,2,1]=1.5*s;h[:,2,2]=1.5*s*z
    return h

def inputs(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids) or len(ids)<9:raise ValueError("Require 9 unique impacts with direction labels")
    def cols(t):return as_cuda([[float(r['vx_'+t]),float(r['vz_'+t]),float(r['omega_'+t])*float(r['radius_px'])] for r in rows])
    x,y=normalize(cols('in'),cols('out'))
    order=torch.argsort(x[:,0]-x[:,2],stable=True).cpu().numpy()
    ids=[ids[i] for i in order];x=x[order];y=y[order]
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):raise ValueError("nonfinite input")
    return ids,x,y,[rows[i] for i in order]

def features(prior,x,y):
    mean,sd,beta,heads,b,prec,inv,center,scale,mu,ep,en=prior
    a=design(x,mean,sd);positive=a@beta>0
    pred=torch.where(positive[:,None],a@heads[1],a@heads[0]);alt=torch.where(positive[:,None],a@heads[0],a@heads[1])
    ah=torch.einsum('ni,ab->naib',a,torch.eye(3,device=x.device)).reshape(len(x),3,9)
    hta=ah.mT@inv
    eh=impact_design(x,center,scale)
    hte=eh.mT@en
    return (positive,pred,alt),(ah,b.ravel(),prec,hta@ah,hta@(y-a@b).unsqueeze(-1)),(
        (x[:,0]-x[:,2]).abs().sub(center)/scale,mu,ep,hte@eh,hte@(y-x*x.new_tensor([1,0,1])-eh@mu).unsqueeze(-1))

def compute(x,y,parts,d,mask,sign_valid):
    t=torch.as_tensor(d,device=x.device);w=as_cuda(mask)
    assert torch.all(w.gather(1,t)==0)
    tw=sign_valid*w
    sign=y[:,0]>y[:,2]
    def mean(v,mask):return torch.where(mask.sum(1)>0,(v*mask).sum(1)/mask.sum(1),torch.nan)
    def loss(p):
        dif=p-y
        return torch.linalg.vector_norm(dif[:,:2],dim=-1)+dif[:,2].abs()
    (route,orig,alt),(h,b,P,G,tb),(z,mu,Ep,eG,eb)=parts
    mismatch=route!=sign
    maj=2*(mismatch[t]&sign_valid[t]).sum(1)>sign_valid[t].sum(1)
    risk=(loss(alt)[t]-loss(orig)[t]).sum(1)<0
    co=b+torch.linalg.solve(P+G[t].sum(1),tb[t].sum(1)).squeeze(-1)
    pa=torch.einsum('nok,dk->dno',h,co)
    em=mu+torch.linalg.solve(Ep+eG[t].sum(1),eb[t].sum(1)).squeeze(-1)
    alpha=(em[:,1,None]+z*em[:,2,None]).clamp(0,.8);e=em[:,0,None].clamp(0,1)
    s=x[:,0]-x[:,2]
    pe=torch.stack([x[:,0]-alpha*s,-e*x[:,1],x[:,2]+1.5*alpha*s],-1)
    result={};preds={}
    for policy,flip,p in zip(METHODS,[maj,risk,None,None],[None,None,pa,pe]):
        if flip is not None:p=torch.where(flip[:,None,None],alt,orig)
        diff=(p-y).abs()
        metrics=dict(norm_velocity=mean(torch.linalg.vector_norm(diff[:,:,:2],dim=-1),w)/SCALE,
                     norm_spin=mean(diff[:,:,2],w)/SCALE,output_sign_acc=mean((p[:,:,0]>p[:,:,2])==sign,tw))
        if flip is not None:
            n=tw.sum(1);votes=(tw*mismatch).sum(1)
            metrics.update(flip=flip.double(),route_sign_acc=mean((route^flip[:,None])==sign,tw),
                           orientation_acc=torch.where(n!=2*votes,(flip==(2*votes>n)).double(),torch.nan))
        result[policy]={m:v.cpu().numpy() for m,v in metrics.items()}
        preds[policy]=(p/SCALE).cpu().numpy()
    return result,preds

def main():
    p=argparse.ArgumentParser()
    p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,default=HERE/'anti_normalized')
    args=p.parse_args();gpu=idle_device();torch.set_default_dtype(torch.float64)
    data,_=compare.old.load(V70/'data'/'upstream')
    src=[normalize(*compare.xy(data,'train',cfg)) for cfg in compare.GROUPS.values()]
    prior=make_prior(src)
    ids,x,y,rows=inputs(args.csv);n=len(ids);parts=features(prior,x,y)
    pool=np.rint(np.linspace(0,n-1,8)).astype(int);held=np.setdiff1d(np.arange(n),pool)
    random=np.stack([np.random.default_rng(np.random.SeedSequence([720073,i])).choice(n,8,replace=False) for i in range(256)])
    have_uncertainty=all('sign_resolved' in r and str(r['sign_resolved']) in ('0','1') for r in rows)
    observed=torch.abs(y[:,0]-y[:,2])>0
    consensus=as_cuda([int(r['sign_resolved']) for r in rows]).bool()&observed if have_uncertainty else observed
    validity=[('tracking_resolved',consensus),('all_nonzero',observed)] if have_uncertainty else [('all_nonzero',observed)]
    res={};pred_rows=[]; detailed=[]
    for variant,valid in validity:
        for mode in ('fixed','random'):
            for k in BUDGETS:
                if mode=='fixed':
                    d=np.array(list(itertools.combinations(pool,k)))
                    mask=np.zeros((len(d),n));mask[:,held]=1
                else:
                    d=random[:,:k]
                    mask=np.ones((len(d),n));np.put_along_axis(mask,random,0,axis=1)
                m,pp=compute(x,y,parts,d,mask,valid)
                sub={}
                for name,q in m.items():
                    sub[name]={}
                    for metric,arr in q.items():
                        a=arr[np.isfinite(arr)]
                        sub[name][metric]=dict(mean=float(a.mean()) if len(a) else None,n=len(a))
                    if variant==validity[0][0]:
                        for i in range(len(d)):
                            detailed.append(dict(mode=mode,k=k,method=name,draw=i,**{a:float(z[i]) for a,z in q.items()}))
                        if mode=='fixed' and k==8:
                            for t in held:
                                yp=pp[name][0,t];yy=(y[t]/SCALE).cpu().numpy()
                                pred_rows.append(dict(id=ids[t],method=name,sign_resolved=bool(valid[t]),
                                   vx=yy[0],vz=yy[1],rim=yy[2],pred_vx=yp[0],pred_vz=yp[1],pred_rim=yp[2]))
                res.setdefault(variant,{}).setdefault(mode,{})[k]=sub
    metadata=dict(rows=n,held=[ids[i] for i in held],pool=[ids[i] for i in pool],held_valid=int(consensus[held].sum()),
        held_route_mismatches=int(((parts[0][0]!=(y[:,0]>y[:,2]))&consensus)[held].sum()),
        results=res,gpu_before=gpu,host=platform.node(),job=os.getenv('SLURM_JOB_ID'),step=os.getenv('SLURM_STEP_ID'),
        sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),torch=torch.__version__)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'result.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for name,rr in [('k8_held.csv',pred_rows),('draws.csv',detailed)]:
        fs=list(dict.fromkeys(k for row in rr for k in row))
        with (args.output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows(rr)
    print('fixed',n,len(held),metadata['held_valid'],res[validity[0][0]]['fixed'])
if __name__=='__main__':main()
