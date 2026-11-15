"""Batched, independently tempered SMC populations, bounded uniform prior.

bridge: L_f(x) L_a(x)^beta; full: (L_f(x)L_a(x))^beta.
Every stage resamples, then applies invariant bounded MH transitions.
"""
import time
import numpy as np
import torch
from engine import tensor, decode, BOUNDS, batch, SHIFTS, SIM_COUNTS

def normalize(theta,b):
    bounds=tensor(BOUNDS)
    return torch.cat([(theta-bounds[:,0])/(bounds[:,1]-bounds[:,0]),b[:,None]/.006],1)

def resample(w):
    h,n=w.shape
    u=(torch.arange(n,device=w.device,dtype=w.dtype)[None,:]+torch.rand((h,1),device=w.device,dtype=w.dtype))/n
    cdf=w.cumsum(1);cdf[:,-1]=1.
    return torch.searchsorted(cdf.contiguous(),u.contiguous()).clamp_max(n-1)

def sample(loglike,h,n,method,seed,initial=None,moves=12,threshold=.75,safe_invalid=False):
    torch.manual_seed(seed);t0=time.perf_counter();before=SIM_COUNTS['trajectories']
    if method=='bridge':
        if initial is None:raise ValueError('bridge requires factual posterior')
        if initial.ndim==2:
            x=initial[torch.randint(len(initial),(h,n),device='cuda')].clone()
        else:
            assert initial.shape[0]==h and initial.shape[2]==5
            x=initial[torch.arange(h,device='cuda')[:,None],torch.randint(initial.shape[1],(h,n),device='cuda')].clone()
    elif method=='full':x=torch.rand((h,n,5),device='cuda',dtype=torch.float64)
    else:raise ValueError(method)
    ids=torch.arange(h,device=x.device);lf,ln=loglike(x,ids)
    beta=torch.zeros(h,device=x.device,dtype=x.dtype);logz=beta.clone()
    stages=torch.zeros(h,device=x.device,dtype=torch.int64)
    miness=torch.ones_like(beta)*n;maxweight=torch.zeros_like(beta)
    accepted=torch.zeros_like(beta);attempts=accepted.clone();cost=torch.ones_like(beta)*n
    ancestors=torch.arange(n,device=x.device)[None,:].repeat(h,1)
    # One-step importance weighting diagnostic before any tempering.
    inc=ln if method=='bridge' else lf+ln
    one_w=torch.softmax(inc,1);one_ess=1/(one_w*one_w).sum(1)
    history=[]
    while bool((beta<1-1e-10).any()):
        active=torch.where(beta<1-1e-10)[0]
        xx=x[active];ff=lf[active];aa=ln[active];old=beta[active]
        inc=aa if method=='bridge' else ff+aa
        lo=old.clone();hi=torch.ones_like(old)
        for _ in range(26):
            mid=(lo+hi)/2;w=torch.softmax((mid-old)[:,None]*inc,1)
            too_small=1/(w*w).sum(1)<threshold*n
            hi=torch.where(too_small,mid,hi);lo=torch.where(too_small,lo,mid)
        nxt=torch.where(hi>1-1e-7,torch.ones_like(hi),lo)
        if bool((nxt<=old+1e-12).any()):raise RuntimeError('tempering stalled')
        lw=(nxt-old)[:,None]*inc;w=torch.softmax(lw,1)
        ess=1/(w*w).sum(1);miness[active]=torch.minimum(miness[active],ess)
        maxweight[active]=torch.maximum(maxweight[active],w.max(1).values)
        logz[active]+=torch.logsumexp(lw,1)-np.log(n)
        mean=(xx*w[:,:,None]).sum(1);xc=xx-mean[:,None,:]
        cov=torch.einsum('hni,hn,hnj->hij',xc,w,xc)+torch.eye(5,device=x.device,dtype=x.dtype)[None,:,:]*1e-5
        chol=torch.linalg.cholesky(cov)*(2.38/np.sqrt(5))
        rr=resample(w);rr5=rr[:,:,None].expand(-1,-1,5)
        xx=torch.gather(xx,1,rr5);ff=torch.gather(ff,1,rr);aa=torch.gather(aa,1,rr)
        ancestors[active]=torch.gather(ancestors[active],1,rr)
        for j in range(moves):
            if j%3==0:xp=xx+torch.bmm(torch.randn_like(xx),chol.transpose(1,2))
            elif j%3==1:xp=xx+torch.randn_like(xx)*.07
            else:xp=torch.rand_like(xx)
            valid=((xp>=0)&(xp<=1)).all(2)
            # Invalid proposals are rejected either way. Re-evaluating current
            # states avoids artificial boundary trajectories from clamping.
            proposal_eval=torch.where(valid[:,:,None],xp,xx) if safe_invalid else xp.clamp(0,1)
            fp,ap=loglike(proposal_eval,active)
            delta=(fp-ff)+nxt[:,None]*(ap-aa) if method=='bridge' else nxt[:,None]*(fp+ap-ff-aa)
            ok=valid&(torch.log(torch.rand_like(ff))<delta)
            xx=torch.where(ok[:,:,None],xp,xx);ff=torch.where(ok,fp,ff);aa=torch.where(ok,ap,aa)
            accepted[active]+=ok.sum(1);attempts[active]+=n;cost[active]+=n
        x[active]=xx;lf[active]=ff;ln[active]=aa;beta[active]=nxt;stages[active]+=1
        history.append(dict(populations=active.cpu().tolist(),beta=nxt.cpu().tolist(),ess=ess.cpu().tolist()))
        if len(history)>120:raise RuntimeError('too many stages')
    torch.cuda.synchronize()
    unique=[int(torch.unique(a).numel()) for a in ancestors]
    stat=dict(method=method,n=n,h=h,moves=moves,seed=seed,safe_invalid=safe_invalid,seconds=time.perf_counter()-t0,
        trajectory_rollouts=SIM_COUNTS['trajectories']-before,stages=stages.cpu().tolist(),
        acceptance=(accepted/attempts).cpu().tolist(),one_step_ess=one_ess.cpu().tolist(),
        min_stage_ess=miness.cpu().tolist(),max_stage_weight=maxweight.cpu().tolist(),
        surviving_initial_ancestors=unique,likelihood_candidates=cost.cpu().tolist(),history=history)
    return x,logz,stat

def physics_likelihood(model,yf,action,yy):
    factual=tensor(yf)
    def evaluate(x,ids):
        h,n,_=x.shape;flat=x.reshape(-1,5);theta=decode(flat[:,:4]);b=flat[:,4]*.006
        fm,fc=model.mean_cov(theta,b,'factual');am,ac=model.mean_cov(theta,b,action)
        target=factual[:1] if factual.ndim==1 else factual[ids,None,:1]
        fd=fm.reshape(h,n,-1)-target
        ad=am.reshape(h,n,-1)-yy[ids,None,:]
        lf=-.5*torch.einsum('hnd,dk,hnk->hn',fd,torch.linalg.inv(fc),fd)
        ln=-.5*torch.einsum('hnd,dk,hnk->hn',ad,torch.linalg.inv(ac),ad)
        return lf,ln
    return evaluate

def summaries(x,family):
    h,n,_=x.shape;flat=x.reshape(-1,5)
    out=batch(decode(flat[:,:4]),flat[:,4]*.006,family,SHIFTS)
    if bool(out['overflow'].any()):raise RuntimeError('query overflow')
    z=out['velocity'].reshape(6,h,n,2).permute(1,2,0,3)
    contact=(out['code']>0).reshape(6,h,n).permute(1,2,0).double()
    return dict(mean=z.mean(1).cpu().numpy(),contact=contact.mean(1).cpu().numpy(),
        variance=z.var(1).cpu().numpy(),parameter_mean=x.mean(1).cpu().numpy())
