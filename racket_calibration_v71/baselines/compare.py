"""Target-shot paired calibration baselines. Physical and affine models retain their v70 form."""
from pathlib import Path
import csv
import json
import os
import platform
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
V70=PROJECT/'racket_transfer_v70'
sys.path.insert(0,str(V70/'solver'))
import benchmark as old

R=.02
GROUPS=old.GROUPS
BUDGETS=[0,1,2,4,8,16,32]
PENALTIES={'inclined_rigid':[.25,1.,4.,16.],
           'single_affine_ridge':[.03,.3,3.,30.],
           'single_affine_eb':[.1,1.,10.]}


def matrix(z):
    return torch.cat((torch.ones_like(z[...,:1]),z),dim=-1)


def xy(data, split, configs):
    return tuple(torch.cat([data[split,c][j] for c in configs]) for j in (0,1))


def anchor(x,y):
    """Fit v70 (positive-grid) inclined rigid with the original two-component criterion."""
    mus=old.tensor(old.MUS)
    best=(float('inf'),)
    for deg in range(36):
        n=old.rotate(x,deg)[1]
        if torch.any(n>=0):continue
        e=float((-n@old.rotate(y,deg)[1]/(n@n)).clamp(0,1))
        outs=old.grid_predictions(x,deg,mus,e)
        loss=(outs[:,:,[0,2]]-y[:,[0,2]]).square().mean((1,2))
        i=int(loss.argmin())
        best=min(best,(float(loss[i]),deg,float(mus[i]),e))
    return list(best[1:])


def make_prior(data,groups,frozen):
    configs=sorted(c for g in groups for c in GROUPS[g])
    x,y=xy(data,'train',configs)
    m=x.mean(0);sd=x.std(0,unbiased=False)
    a=matrix((x-m)/sd)
    slope=torch.linalg.lstsq(a,y).solution
    coef=[]; noise=[]
    for g in groups:
        gx,gy=xy(data,'train',GROUPS[g]);ga=matrix((gx-m)/sd)
        b=torch.linalg.lstsq(ga,gy).solution
        coef.append((b-slope).flatten())
        res=gy-ga@b
        noise.append(res.T@res/(len(res)-4))
    d=torch.stack(coef)
    cov=d.T@d/len(groups)
    diag=torch.diag(cov.diag().clamp(min=.0001))
    precision=torch.linalg.inv(.85*cov+.15*diag)
    nc=torch.stack(noise).mean(0)
    noise_inv=torch.linalg.inv(nc+torch.eye(3,device='cuda')*.0001)
    # Input-only behavior anchor. A reproducible pseudo-sample from this source mixture.
    rng=np.random.default_rng(710690)
    ref=x[rng.choice(len(x),128,replace=False)]
    ph=frozen if frozen is not None else anchor(x,y)
    return dict(mean=m,sd=sd,slope=slope,precision=precision,noise_inv=noise_inv,ref=ref,physical=ph)


def physics(par,x):
    """Par [C,3]=angle,mu,e; x [N,3]; return [C,N,3]."""
    a=torch.deg2rad(par[:,0,None]);co=a.cos();si=a.sin()
    mu,e=par[:,1,None],par[:,2,None]
    vx,vz,rim=x.T
    t,n=co*vx+si*vz,-si*vx+co*vz
    limit=mu*(1+e)*(-n).clamp(min=0)
    j=torch.clamp((rim-t)/2.5,min=-limit,max=limit)
    no=torch.where(n<0,-e*n,n)
    return torch.stack((co*(t+j)-si*no,si*(t+j)+co*no,rim-1.5*j),dim=-1)


def physical_models(prior,x,y,indices,tx,lambdas):
    ph=prior['physical']
    degs=old.tensor(sorted(set(range(0,36,5))|{ph[0]}))
    mus=old.tensor(sorted(set([i*.05 for i in range(21)]+[1.1,1.25,1.5,2.,ph[1]])))
    es=old.tensor(sorted(set([i*.05 for i in range(21)]+[ph[2]])))
    pars=torch.cartesian_prod(degs,mus,es)
    src=old.tensor([ph])
    ref=prior['ref']
    original=physics(src,ref)
    # Only selected calibration inputs and labels enter the per-draw objective.
    used,reverse=np.unique(indices,return_inverse=True)
    reverse=torch.as_tensor(reverse.reshape(indices.shape),device='cuda')
    cx,cy=x[used],y[used]
    costs=[]; shifts=[]
    for p in pars.split(400):
        costs.append((physics(p,cx)-cy).square().sum(-1))
        shifts.append((physics(p,ref)-original).square().sum(-1).mean(1))
    cal_loss=torch.cat(costs)
    penalty=torch.cat(shifts)
    n_draws=indices.shape[0]
    total=torch.zeros((len(pars),n_draws),device='cuda')
    prev=0
    for k in BUDGETS:
        if k==0:
            p=physics(src,tx).expand(n_draws,-1,-1)
            for lam in lambdas:yield k,lam,p
            continue
        for j in range(prev,k):
            total+=cal_loss[:,reverse[:,j]]
        prev=k
        for lam in lambdas:
            best=(total+penalty[:,None]*lam).argmin(0)
            yield k,lam,physics(pars[best],tx)


def affine_models(prior,x,y,indices,tx,lambdas,eb):
    tc=matrix((tx-prior['mean'])/prior['sd'])
    b=prior['slope']
    cc=matrix((x[indices]-prior['mean'])/prior['sd'])
    res=y[indices]-cc@b
    eye=torch.eye(4,device='cuda')
    for k in BUDGETS:
        if k==0:
            for lam in lambdas:yield k,lam,(tc@b).expand(len(indices),-1,-1)
            continue
        a=cc[:,:k]
        gram=a.mT@a
        rhs=a.mT@res[:,:k]
        if eb:
            W=prior['noise_inv']
            gram=torch.einsum('bij,kl->bikjl',gram,W).reshape(-1,12,12)
            rhs=(rhs@W).reshape(-1,12,1)
        for lam in lambdas:
            P=prior['precision'] if eb else eye
            update=torch.linalg.solve(gram+lam*P,rhs)
            pred=tc@(b+update.reshape(-1,4,3))
            yield k,lam,pred


def run_methods(prior,cx,cy,idx,tx,choices):
    for method,h in choices.items():
        gen=physical_models(prior,cx,cy,idx,tx,h) if method=='inclined_rigid' else affine_models(prior,cx,cy,idx,tx,h,method.endswith('_eb'))
        for k,lam,yp in gen:yield method,k,lam,yp


def main():
    start=time.monotonic()
    torch.set_default_dtype(torch.float64)
    assert torch.cuda.is_available()
    data,_=old.load(V70/'data'/'upstream')
    v70=json.loads((V70/'solver'/'metrics.json').read_text())['folds']
    priors={}
    def fit(groups,targ=None):
        key=tuple(sorted(groups))
        if key not in priors:
            p=None
            if targ is not None:
                f=v70[targ]['physical']['inclined'];p=[f['deg'],f['mu'],f['e']]
            priors[key]=make_prior(data,key,p)
        return priors[key]
    params={};metrics=[]
    draws_saved={}; test_mean={}
    for fold,(target,held) in enumerate(GROUPS.items()):
        source=[s for s in GROUPS if s!=target]
        validation={m:{h:0. for h in vals} for m,vals in PENALTIES.items()}
        for pseudo in source:
            p=fit([g for g in source if g!=pseudo])
            x,y=xy(data,'train',GROUPS[pseudo])
            rng=np.random.default_rng(712000+min(GROUPS[pseudo]))
            perm=rng.permutation(len(x))
            verify=perm[:192];pool=perm[192:]
            idx=np.stack([rng.choice(pool,32,replace=False) for i in range(24)])
            for m,k,h,yp in run_methods(p,x,y,idx,x[verify],PENALTIES):
                if k:validation[m][h]+=float((yp-y[verify]).square().mean())/(6*5)
        best={m:[min(v,key=v.get)] for m,v in validation.items()}
        p=fit(source,target)
        x,y=xy(data,'train',held);tx,ty=xy(data,'test',held)
        idx=np.stack([np.random.default_rng(np.random.SeedSequence([710071,min(held),i])).choice(len(x),32,replace=False) for i in range(256)])
        draws_saved[target]=idx
        params[target]=dict(inner=validation,selected=best,anchor=p['physical'])
        print(target,best,flush=True)
        for m,k,h,yp in run_methods(p,x,y,idx,tx,best):
            delta=yp-ty
            ve=delta[:,:,:2].norm(dim=-1)
            se=delta[:,:,2].abs()/R
            probs=(yp[:,:,0]-yp[:,:,2]>0)==(ty[:,0]-ty[:,2]>0)
            arr=torch.stack((ve.mean(1),se.mean(1),probs.double().mean(1)),-1).cpu().numpy()
            for i,(v,s,acc) in enumerate(arr):
                metrics.append([target,i,k,m,v,s,acc])
            test_mean[f'{target}__{m}__{k}']=torch.stack((ve.mean(0),se.mean(0)),-1).cpu().numpy()
            if k==0:
                key='inclined' if m=='inclined_rigid' else 'single_affine'
                ref=v70[target]['metrics'][key]['test']
                assert abs(arr[0,0]-ref['velocity_mae'])<1e-8 and abs(arr[0,1]-ref['spin_mae'])<1e-8
    with (HERE/'draw_metrics.csv').open('w') as f:
        wr=csv.writer(f);wr.writerow(['rubber','draw','budget','method','velocity','spin','sign_accuracy']);wr.writerows(metrics)
    np.savez_compressed(HERE/'draw_indices.npz',**draws_saved)
    np.savez_compressed(HERE/'test_mean_errors.npz',**test_mean)
    (HERE/'fit.json').write_text(json.dumps(dict(fits=params,gpu=torch.cuda.get_device_name(),node=platform.node(),
        torch=torch.__version__,job=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start),indent=2)+'\n')
if __name__=='__main__':main()
