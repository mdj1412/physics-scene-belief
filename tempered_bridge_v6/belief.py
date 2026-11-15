import json,time
from pathlib import Path
import numpy as np
import torch
from engine import tensor,decode,observation_means,ACTIONS,SIM_COUNTS
from base_inference import systematic,predict,metrics

class Belief:
    def __init__(self,family,quality='high',mode='boundary'):
        self.family=family;self.quality=quality;self.mode=mode
        c=json.loads((Path('results/calibration')/f'{quality}.json').read_text())[family][quality]
        self.grid=tensor(c['b_grid']);self.curve=tensor(c['mean']);self.view_cov=tensor(c['cov'])
    def mean_cov(self,theta,b,action):
        if action=='view':
            idx=(torch.searchsorted(self.grid,b)-1).clamp(0,len(self.grid)-2)
            frac=(b-self.grid[idx])/(self.grid[idx+1]-self.grid[idx])
            bias=self.curve[idx]+frac[:,None]*(self.curve[idx+1]-self.curve[idx])
            mu=torch.stack([theta[:,0]*1e3,theta[:,3]*1e3,torch.zeros_like(b)],1)+bias
            dims=3 if self.mode=='boundary' else 2
            return mu[:,:dims],self.view_cov[:dims,:dims]
        mu=observation_means(theta,b,self.family,action)
        if action=='factual':mu=mu[:,:1]
        return mu,torch.eye(mu.shape[1],device=mu.device,dtype=mu.dtype)*.025**2
    def likelihood(self,x,data):
        theta=decode(x[:,:4]);b=x[:,4]*.006;ll=torch.zeros(len(x),device=x.device,dtype=x.dtype)
        for a,y in data:
            mu,cov=self.mean_cov(theta,b,a);delta=mu-tensor(y)[:mu.shape[1]]
            ll-=.5*torch.einsum('nd,dk,nk->n',delta,torch.linalg.inv(cov),delta)
        return ll
    def smc(self,data,n=1536,seed=0,moves=12):
        torch.manual_seed(seed);start=time.perf_counter();rollouts_before=SIM_COUNTS['trajectories'];x=torch.rand((n,5),device='cuda',dtype=torch.float64)
        ll=self.likelihood(x,data);beta=0.;stages=accepted=attempts=0;calls=n*len(data)
        while beta<1-1e-10:
            low=beta;high=1.
            for _ in range(24):
                mid=(low+high)/2;w=torch.softmax((mid-beta)*ll,0)
                if float(1/(w@w))<.75*n:high=mid
                else:low=mid
            nxt=1. if high>1-1e-7 else low
            if nxt<=beta+1e-12:raise RuntimeError('SMC tempering stalled')
            w=torch.softmax((nxt-beta)*ll,0);mean=w@x;xc=x-mean
            cov=xc.T@(xc*w[:,None])+torch.eye(5,device=x.device,dtype=x.dtype)*1e-5
            chol=torch.linalg.cholesky(cov)*(2.38/np.sqrt(5))
            ids=systematic(w);x=x[ids];ll=ll[ids];beta=nxt
            for j in range(moves):
                if j%3==0:xp=x+torch.randn_like(x)@chol.T
                elif j%3==1:xp=x+torch.randn_like(x)*.07
                else:xp=torch.rand_like(x)
                valid=((xp>=0)&(xp<=1)).all(1);lp=self.likelihood(xp.clamp(0,1),data);calls+=n*len(data)
                ok=valid&(torch.log(torch.rand(n,device=x.device))<beta*(lp-ll));x[ok]=xp[ok];ll[ok]=lp[ok]
                accepted+=int(ok.sum());attempts+=n
            stages+=1
            if stages>100:raise RuntimeError('SMC too many stages')
        torch.cuda.synchronize();weights=torch.full((n,),1/n,device=x.device,dtype=x.dtype)
        stat=dict(calls=calls,dynamic_calls=int(calls*sum(a!='view' for a,y in data)/len(data)),trajectory_rollouts=SIM_COUNTS['trajectories']-rollouts_before,stages=stages,
            acceptance=accepted/attempts,seconds=time.perf_counter()-start,n=n,moves=moves,seed=seed)
        return (decode(x[:,:4]),x[:,4]*.006,weights),stat
    def acquisition(self,atoms,pred=None,samples=512,seed=42):
        torch.manual_seed(seed);theta,b,w=atoms;pred=predict(atoms,self.family) if pred is None else pred
        z=pred['velocity'];current=torch.einsum('qnd,n->qd',z,w);ids=torch.multinomial(w,samples,replacement=True)
        gains=[];eigs=[]
        for action_i,action in enumerate(ACTIONS):
            mu,cov=self.mean_cov(theta,b,action);chol=torch.linalg.cholesky(cov)
            # Identical dynamic-action MC noise across geometry variants.
            torch.manual_seed(seed+100*(action_i+1))
            noise=torch.randn((samples,3),device=theta.device,dtype=theta.dtype)[:,:mu.shape[1]]
            yy=mu[ids]+noise@chol.T
            delta=yy[:,None,:]-mu[None,:,:]
            ll=-.5*torch.einsum('snd,dk,snk->sn',delta,torch.linalg.inv(cov),delta)
            logw=w.log();post=torch.softmax(ll+logw[None,:],1)
            means=torch.einsum('sn,qnd->qsd',post,z)
            gains.append(((means-current[:,None,:])**2).sum(2).mean(1))
            eigs.append((ll[torch.arange(samples,device=theta.device),ids]-torch.logsumexp(ll+logw,1)).mean())
        return torch.stack(gains,1).cpu().numpy(),torch.stack(eigs).cpu().numpy()

def parameter_summary(atoms):
    theta,b,w=atoms
    return dict(mean=torch.cat([theta,b[:,None]],1).mean(0).cpu().tolist(),
        b_interval=torch.quantile(b,tensor([.05,.95])).cpu().tolist())
