import time
import numpy as np
import torch
from engine import tensor,decode,observation_means,NOISE,batch,SHIFTS,ACTIONS

MODELS={'sharp':[0.],'broad':[0.,.001,.003],'outside':[.005]}

def loglike(x,m,family,data):
    theta=decode(x);out=torch.zeros(len(x),device=x.device,dtype=x.dtype)
    for action,y,scale in data:
        pred=observation_means(theta,m,family,action)
        sd=tensor(NOISE[action]*scale,x.device)
        out-=.5*(((pred-tensor(y,x.device))/sd)**2).sum(1)
    return out

def systematic(w,n=None):
    n=len(w) if n is None else n
    u=(torch.arange(n,device=w.device,dtype=w.dtype)+torch.rand((),device=w.device))/n
    return torch.searchsorted(w.cumsum(0).contiguous(),u).clamp_max(len(w)-1)

def smc(family,data,model='broad',n=512,seed=0,moves=6):
    torch.manual_seed(seed);start=time.perf_counter();device='cuda'
    vals=tensor(MODELS[model],device);x=torch.rand((n,4),device=device,dtype=torch.float64)
    mi=torch.randint(len(vals),(n,),device=device);ll=loglike(x,vals[mi],family,data)
    beta=0.;calls=n*len(data);stages=0;accepted=0;attempts=0
    while beta<1-1e-10:
        low=beta;high=1.
        for _ in range(24):
            mid=(low+high)/2;w=torch.softmax((mid-beta)*ll,0)
            if float(1/(w@w))<.75*n:high=mid
            else:low=mid
        nxt=1. if high>1-1e-7 else low
        if nxt<=beta+1e-12:raise RuntimeError('SMC tempering stalled')
        w=torch.softmax((nxt-beta)*ll,0);mean=w@x;xc=x-mean
        cov=xc.T@(xc*w[:,None])+torch.eye(4,device=device,dtype=x.dtype)*1e-5
        chol=torch.linalg.cholesky(cov)*(2.38/2)
        ids=systematic(w);x=x[ids];mi=mi[ids];ll=ll[ids];beta=nxt
        for j in range(moves):
            # Mixture of covariance, local independent, and global prior moves.
            if j%3==0:xp=x+torch.randn_like(x)@chol.T
            elif j%3==1:xp=x+torch.randn_like(x)*.07
            else:xp=torch.rand_like(x)
            # Reject out-of-domain proposals; reflecting correlated proposals
            # would not in general preserve proposal symmetry.
            valid=((xp>=0)&(xp<=1)).all(1)
            mp=mi.clone();refresh=torch.rand(n,device=device)<.25
            mp[refresh]=torch.randint(len(vals),(int(refresh.sum()),),device=device)
            lp=loglike(xp.clamp(0,1),vals[mp],family,data);calls+=n*len(data)
            ok=valid&(torch.log(torch.rand(n,device=device))<beta*(lp-ll))
            x[ok]=xp[ok];mi[ok]=mp[ok];ll[ok]=lp[ok]
            accepted+=int(ok.sum());attempts+=n
        stages+=1
        if stages>100:raise RuntimeError('too many SMC stages')
    torch.cuda.synchronize()
    return (decode(x),vals[mi],torch.full((n,),1/n,device=device,dtype=x.dtype)),dict(
        calls=calls,stages=stages,acceptance=accepted/max(attempts,1),seconds=time.perf_counter()-start,
        n=n,model=model,seed=seed,loglike_max=float(ll.max()))

def predict(atoms,family):
    t,m,w=atoms;out=batch(t,m,family,SHIFTS)
    if bool(out['overflow'].any()):raise RuntimeError('contact event budget exceeded')
    return out

def reweight(atoms,family,action,y,scale=1.):
    t,m,w=atoms;mu=observation_means(t,m,family,action)
    ll=-.5*(((mu-tensor(y))/tensor(NOISE[action]*scale))**2).sum(1)
    ww=torch.softmax(w.clamp_min(1e-300).log()+ll,0)
    return (t,m,ww)

def acquisition(atoms,family,pred=None,samples=192,seed=42):
    torch.manual_seed(seed);t,m,w=atoms
    pred=predict(atoms,family) if pred is None else pred
    z=pred['velocity'];mean=torch.einsum('qnd,n->qd',z,w)
    ids=torch.multinomial(w,samples,replacement=True)
    noise=torch.randn((samples,2),device=t.device,dtype=t.dtype)
    gains=[];eigs=[]
    for action in ACTIONS:
        mu=observation_means(t,m,family,action);sd=tensor(NOISE[action])
        yy=mu[ids]+noise*sd
        ll=-.5*(((yy[:,None,:]-mu[None,:,:])/sd)**2).sum(2)
        post=torch.softmax(w.clamp_min(1e-300).log()[None,:]+ll,1)
        means=torch.einsum('sn,qnd->qsd',post,z)
        gains.append(((means-mean[:,None,:])**2).sum(2).mean(1))
        # Mutual information between joint latent state/model and measurement.
        eigs.append(float((ll[torch.arange(samples,device=t.device),ids]-torch.logsumexp(ll+w.clamp_min(1e-300).log(),1)).mean()))
    return torch.stack(gains,1).cpu().numpy(),np.array(eigs)

def metrics(atoms,pred,truth):
    t,m,w=atoms;z=pred['velocity'];target=tensor(truth['velocity'])
    mean=torch.einsum('qnd,n->qd',z,w);mse=((mean-target)**2).mean(1)
    event=(pred['code']>0).double();cp=event@w;true_event=tensor(truth['code']>0)
    sorted_z,order=z.sort(1);sw=w[order];cdf=sw.cumsum(1)
    lo=torch.gather(sorted_z,1,(cdf>=.05).long().argmax(1)[:,None,:]).squeeze(1)
    hi=torch.gather(sorted_z,1,(cdf>=.95).long().argmax(1)[:,None,:]).squeeze(1)
    crps=((z-target[:,None,:]).abs()*w[None,:,None]).sum(1)-(sw*sorted_z*(2*cdf-sw-1)).sum(1)
    return dict(mse=mse.cpu().numpy(),crps=crps.mean(1).cpu().numpy(),
        coverage90=((target>=lo)&(target<=hi)).double().mean(1).cpu().numpy(),
        width90=(hi-lo).mean(1).cpu().numpy(),brier=((cp-true_event)**2).cpu().numpy(),
        ess=np.repeat(float(1/(w@w)),len(z)),mean=mean.cpu().numpy())

def compress(atoms,pred,n,method,seed):
    torch.manual_seed(seed);t,m,w=atoms
    if method=='systematic':idx=systematic(w,n);nw=torch.full((n,),1/n,device=t.device,dtype=t.dtype)
    elif method=='branch':
        # Stratify on event sequences for the same public query pool. No truth.
        signatures=pred['code'].T
        _,inverse=torch.unique(signatures,dim=0,return_inverse=True)
        groups=[torch.where(inverse==i)[0] for i in range(int(inverse.max())+1)]
        masses=torch.stack([w[g].sum() for g in groups]);k=len(groups)
        if k>n:return compress(atoms,pred,n,'systematic',seed)
        allocation=torch.ones(k,device=t.device,dtype=torch.long)
        extra=torch.multinomial(masses,n-k,replacement=True)
        allocation+=torch.bincount(extra,minlength=k)
        ids=[];weights=[]
        for g,mass,count in zip(groups,masses,allocation):
            count=int(count);ids.append(g[systematic(w[g]/mass,count)])
            weights.append(torch.full((count,),float(mass)/count,device=t.device,dtype=t.dtype))
        idx=torch.cat(ids);nw=torch.cat(weights)
    else:raise ValueError(method)
    return (t[idx],m[idx],nw),{k:v[:,idx] for k,v in pred.items()}
