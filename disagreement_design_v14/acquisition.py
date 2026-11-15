"""Query-weighted pair discrimination; no reference posterior dependency."""
import torch
from physics import t,DT,DEV,noise
from flexible import measure,response

def indices(weights,K):
    u=(torch.arange(K,device=DEV,dtype=DT)+.5)/K
    return torch.searchsorted(weights.cumsum(-1).contiguous(),u.expand(len(weights),-1).contiguous()).clamp_max(weights.shape[-1]-1)

def committee(method,state,K=256):
    if hasattr(method,'theta'):
        idx=indices(state,K);return method.theta[idx]
    _,th=method.points(state);weights=(state.logp.exp()[:,:,None]*method.weights).flatten(1,2)
    idx=indices(weights,K);return th.flatten(1,2).gather(1,idx[:,:,None].expand(-1,-1,7))

def distance(x):
    n=(x*x).sum(-1);return (n[:,:,None]+n[:,None,:]-2*x@x.transpose(-1,-2)).clamp_min(0)

def pair_from_theta(th,x,mode='pair'):
    h=measure(th,0);ll=-.5*(((h-t(x)[:,None,:])/noise(0))**2).sum(-1)
    good=ll>=ll.max(-1,keepdim=True).values-8
    eligible=good[:,:,None]&good[:,None,:]&((ll[:,:,None]-ll[:,None,:]).abs()<=2)
    if mode=='unfiltered':eligible=torch.ones_like(eligible)
    D=distance(response(th))*.5
    if mode=='agnostic':D=torch.ones_like(D)
    w=D*eligible;den=eligible.sum((-1,-2)).clamp_min(1);scores=[torch.zeros(len(th),device=DEV,dtype=DT)]
    for a in [1,2,3]:
        sep=distance(measure(th,a)/noise(a));scores.append((w*(-torch.expm1(-sep/8))).sum((-1,-2))/den)
    scores=torch.stack(scores,1)
    diagnostics=dict(eligible_fraction=eligible.to(DT).mean((-1,-2)),query_disagreement=(D*eligible).sum((-1,-2))/den)
    return scores,diagnostics

def pair_select(method,state,x,mode='pair',K=256):
    scores,diag=pair_from_theta(committee(method,state,K),x,mode)
    fallback=scores.max(1).values<=1e-14
    if fallback.any():
        native=method.gains(state);scores=torch.where(fallback[:,None],native,scores)
    diag['fallback']=fallback
    return scores,diag

def nested_plan(M=128,seed=7919):
    sobol=torch.quasirandom.SobolEngine(8,scramble=True,seed=seed).draw(M,dtype=DT).to(DEV)
    return sobol[:,0],torch.special.ndtri(sobol[:,1:].clamp(1e-9,1-1e-9))

def nested_scores(inner,state,outer,outer_state,M=128,seed=7919,plan=None):
    """Cross-fitted EFL: independent outer bank, same inner predictive decoder."""
    u,eps=nested_plan(M,seed) if plan is None else plan;M=len(u)
    ix=torch.searchsorted(outer_state.cumsum(-1).contiguous(),u.expand(len(state),-1).contiguous()).clamp_max(outer.N-1)
    z2=.5*(inner.z*inner.z).sum(1);mean=inner.predict(state);var=state@z2-.5*(mean*mean).sum(1)
    scores=[torch.zeros(len(state),device=DEV,dtype=DT)];lp=state.clamp_min(1e-300).log()
    for a in [1,2,3]:
        mu=outer.mu[a][ix];y=mu+eps[None,:,:mu.shape[-1]]*noise(a)
        ll=inner.ll(a,y.flatten(0,1)).reshape(len(state),M,inner.N)
        w=torch.softmax(ll+lp[:,None,:],-1);pred=w@inner.z;conditional=w@z2-.5*(pred*pred).sum(-1)
        scores.append(var-conditional.mean(1))
    return torch.stack(scores,1)
