"""Post-hoc strong baseline: infer collision rounding continuously.

Same impact law, five continuous latent variables, b ~ Uniform(0, .006).
This is not a new method. It checks whether discrete proxy support caused error.
"""
import argparse,json,time
from pathlib import Path
import torch
import numpy as np
from engine import tensor,decode
from inference import loglike,systematic,predict,metrics
from run import plain

def smc_continuous(family,data,n=1536,seed=0,moves=12):
    torch.manual_seed(seed);start=time.perf_counter();device='cuda'
    x=torch.rand((n,5),device=device,dtype=torch.float64)
    def likelihood(xx):return loglike(xx[:,:4],xx[:,4]*.006,family,data)
    ll=likelihood(x);beta=0.;calls=n*len(data);stages=accepted=attempts=0
    while beta<1-1e-10:
        low=beta;high=1.
        for _ in range(24):
            mid=(low+high)/2;w=torch.softmax((mid-beta)*ll,0)
            if float(1/(w@w))<.75*n:high=mid
            else:low=mid
        nxt=1. if high>1-1e-7 else low
        if nxt<=beta+1e-12:raise RuntimeError('tempering stalled')
        w=torch.softmax((nxt-beta)*ll,0);mean=w@x;xc=x-mean
        cov=xc.T@(xc*w[:,None])+torch.eye(5,device=device,dtype=x.dtype)*1e-5
        chol=torch.linalg.cholesky(cov)*(2.38/np.sqrt(5))
        ids=systematic(w);x=x[ids];ll=ll[ids];beta=nxt
        for j in range(moves):
            if j%3==0:xp=x+torch.randn_like(x)@chol.T
            elif j%3==1:xp=x+torch.randn_like(x)*.07
            else:xp=torch.rand_like(x)
            valid=((xp>=0)&(xp<=1)).all(1);lp=likelihood(xp.clamp(0,1));calls+=n*len(data)
            ok=valid&(torch.log(torch.rand(n,device=device))<beta*(lp-ll));x[ok]=xp[ok];ll[ok]=lp[ok]
            accepted+=int(ok.sum());attempts+=n
        stages+=1
        if stages>100:raise RuntimeError('too many stages')
    torch.cuda.synchronize();atoms=(decode(x[:,:4]),x[:,4]*.006,torch.full((n,),1/n,device=device,dtype=x.dtype))
    return atoms,dict(calls=calls,stages=stages,acceptance=accepted/attempts,seconds=time.perf_counter()-start,n=n,seed=seed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);args=p.parse_args()
    out=Path('results/continuous_proxy');out.mkdir(parents=True,exist_ok=True)
    for case in range(args.worker,72,2):
        dest=out/f'case_{case:03d}.json'
        if dest.exists():continue
        d=json.loads((Path('results/main')/dest.name).read_text());truth={k:np.array(v) for k,v in d['truth'].items()}
        data=[(a,d['observations'][a],1.) for a in ['factual','view','center']]
        atoms,cost=smc_continuous(d['family'],data,seed=100900+case)
        result=dict(case=case,family=d['family'],metrics=metrics(atoms,predict(atoms,d['family']),truth),cost=cost,
                    rounding_mean=float(atoms[1].mean()),rounding_q05_q95=torch.quantile(atoms[1],tensor([.05,.95])).cpu().tolist())
        dest.write_text(json.dumps(plain(result),allow_nan=False));print('continuous',case,flush=True)
