"""Two-axis budget allocation with cached likelihood prefix sums."""
import torch
from model import *
from acquisition import plan
KS=[8,16,32,64,128];MS=[128,256,512,1024]
def plans(seed=91531,mmax=1024):
    return [plan(mmax,seed+100*j) for j in range(4)]
def rep_plan(seed=91531,Kmax=128):
    return torch.quasirandom.SobolEngine(4,scramble=True,seed=seed+1000).draw(Kmax*2,dtype=DT).to(DEV).T.reshape(2,2,Kmax*2)
def padded(x):return torch.nn.functional.pad(x,(0,7-x.shape[-1]))
class Engine:
    def __init__(self,b,w,z,o,ow,oz,outerplans,reps,Kmax=128,mmax=1024,global_mode=False):
        self.Kmax=Kmax;self.mmax=mmax;self.k=0;self.m=0;self.work=0;self.updates=0
        active=[];omitted=0.;ids=[torch.where((z[:,0]>0)==bool(r))[0] for r in [0,1]];masses=torch.stack([w[ix].sum() for ix in ids]);massval=masses.tolist()
        for r in [0,1]:
            if massval[r]<=1e-10:omitted+=massval[r]
            else:active.append((r,ids[r],masses[r],(w[ids[r]]/masses[r]).cumsum(0)))
        self.R=len(active);self.omitted=omitted;self.mu=[];self.labels=[];self.base=[]
        for c in [0,1]:
            if global_mode:
                n=Kmax*self.R;ix=torch.searchsorted(w.cumsum(0),reps[c,0,:n].contiguous()).clamp_max(len(w)-1);base=torch.ones(n,device=DEV,dtype=DT)
            else:
                ix=[];base=[]
                for r,idx,m,cdf in active:
                    ix.append(idx[torch.searchsorted(cdf,reps[c,r,:Kmax].contiguous()).clamp_max(len(idx)-1)]);base.append(m.expand(Kmax))
                ix=torch.stack(ix,1).flatten();base=torch.stack(base,1).flatten()
            self.mu.append(torch.stack([padded(b.white[a][ix]) for a in [1,2,3]]));self.labels.append(z[ix,0]>0);self.base.append(base)
        self.mu=torch.stack(self.mu);self.labels=torch.stack(self.labels);self.base=torch.stack(self.base);self.norm=.5*(self.mu*self.mu).sum(-1)
        ys=[];truth=[]
        for u,eps in outerplans:
            ix=torch.searchsorted(ow.cumsum(0),u[:mmax].contiguous()).clamp_max(o.N-1)
            ys.append(torch.stack([padded(o.white[a][ix]+eps[:mmax,:o.white[a].shape[1]]) for a in [1,2,3]]));truth.append(oz[ix,0])
        self.y=torch.stack(ys,1);self.truth=torch.stack(truth)
        self.logs={k:torch.empty((2,2,3,4,mmax),device=DEV,dtype=DT) for k in KS if k<=Kmax}
    def block(self,lo,hi,start,end):
        # C,A,G,m,N; reuse class evidence sums, not entire M-by-K matrices.
        mu=self.mu[:,:,lo:hi];y=self.y[:,:,start:end]
        ll=torch.matmul(y[None],mu.transpose(-1,-2)[:,:,None])-self.norm[:,:,None,None,lo:hi]+self.base[:,None,None,None,lo:hi].log()
        pos=ll.masked_fill(~self.labels[:,None,None,None,lo:hi],float('-inf'))
        self.work+=2*3*4*(end-start)*(hi-lo);self.updates+=1
        return torch.stack([torch.logcumsumexp(ll,-1),torch.logcumsumexp(pos,-1)])
    def ensure(self,K,m):
        assert K>=self.k and m>=self.m and K<=self.Kmax and m<=self.mmax
        if K>self.k and self.m:
            lo=self.k*self.R;v=self.block(lo,K*self.R,0,self.m)
            for k in KS:
                if self.k<k<=K:self.logs[k][...,:self.m]=torch.logaddexp(self.logs[self.k][...,:self.m],v[...,k*self.R-lo-1])
        if m>self.m:
            v=self.block(0,K*self.R,self.m,m)
            for k in KS:
                if k<=K:self.logs[k][...,self.m:m]=v[...,k*self.R-1]
        self.k=K;self.m=m
    def metrics(self,K,m):
        logs=self.logs[K][...,:m];pred=2**.5*torch.exp(logs[1]-logs[0]);loss=.5*(pred-self.truth[None,None,:,:m])**2
        return loss.mean((0,2,3)),loss.mean((0,3)).T,loss.mean((2,3))
    def bytes(self):return sum(x.numel()*x.element_size() for x in [self.mu,self.labels,self.base,self.norm,self.y,self.truth]+list(self.logs.values()))
def route(mode,K,m,ec,eo):
    if mode=='component_only':return (min(K*2,128),m,0) if K<128 else (K,m,-1)
    if mode=='outer_only':return (K,min(m*2,1024),1) if m<1024 else (K,m,-1)
    if mode=='balanced':return (min(K*2,128),min(m*2,1024),2) if K<128 or m<1024 else (K,m,-1)
    if K==128 and m==1024:return K,m,-1
    useK=(ec>=eo and K<128) or m==1024
    return (min(K*2,128),m,0) if useK else (K,min(m*2,1024),1)
def select(engine,mode='adaptive',fixed=None):
    K,m=fixed if fixed else (128 if mode=='outer_only' else 16,128);hist=[];k_steps=0;m_steps=0
    while True:
        # A shared offline engine may already contain larger prefixes.
        engine.ensure(max(K,engine.k),max(m,engine.m))
        risk,group,rep=engine.metrics(K,m);coarse=engine.metrics(K//2,m)[0];best=int(risk.argmin());g=risk-risk[best]
        ec=(g-(coarse-coarse[best])).abs()+.5*((rep[0]-rep[0,best])-(rep[1]-rep[1,best])).abs()
        eo=2*(group-group[:,best,None]).std(0,unbiased=True)/2
        others=torch.tensor([a for a in range(3) if a!=best],device=DEV);worst=others[(g-ec-eo)[others].argmin()]
        vals=torch.stack([risk.max()-risk.min(),(g-ec-eo)[worst],ec[worst],eo[worst]]).tolist();tie=vals[0]<=1e-12;resolved=vals[1]>0
        nk,nm,step=route(mode,K,m,vals[2],vals[3])
        reason=3 if fixed else (2 if tie else (0 if resolved else (1 if step<0 else -1)))
        hist.append(torch.cat([risk,t([K,m]),torch.stack([g[worst],ec[worst],eo[worst],(g-ec-eo)[worst]]),t([step])]))
        if reason>=0:break
        k_steps+=int(nk>K);m_steps+=int(nm>m);K,m=nk,nm
    action=1 if tie else best+1
    meta=t([K,m,len(hist),reason,k_steps,m_steps,engine.R,engine.omitted])
    hh=torch.full((8,10),float('nan'),device=DEV,dtype=DT);hh[:len(hist)]=torch.stack(hist)
    return action,risk,meta,hh
def full_prefix(b,w,z,o,ow,oz,preplans,levels=MS):
    """Full inner decoder; equal allocation across four independent outer scrambles."""
    result=[];lp=w.clamp_min(1e-300).log();M=max(levels)
    for u,eps in preplans:
        ix=torch.searchsorted(ow.cumsum(0),u[:M].contiguous()).clamp_max(o.N-1);truth=oz[ix];actions=[]
        for a in [1,2,3]:
            loss=[]
            for start in range(0,M,128):
                y=o.mu[a][ix[start:start+128]]+eps[start:start+128,:o.mu[a].shape[1]]*noise(a)
                post=torch.softmax(lp+b.ll(a,y),-1);pred=torch.stack([post@z[:,j] for j in range(z.shape[1])],-1)
                loss.append(.5*(pred-truth[start:start+128])**2)
            loss=torch.cat(loss);actions.append(torch.stack([loss[:m].mean(0) for m in levels]))
        result.append(torch.stack(actions,1))
    return torch.stack(result).mean(0) # levels, actions, tasks
