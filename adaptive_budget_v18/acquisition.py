"""Acquisition only: public evidence, candidate beliefs, and public context."""
import torch
from model import t,DT,DEV,noise,group,SIZES

def lmmse(bank,w,z,sizes=SIZES):
    dz=z-w@z;scores=[torch.zeros(len(sizes),device=DEV,dtype=DT)]
    for a in [1,2,3]:
        y=bank.mu[a];dy=y-w@y;S=dy.T@(w[:,None]*dy)+torch.diag(noise(a)**2);C=dz.T@(w[:,None]*dy)
        scores.append(group((C*torch.linalg.solve(S,C.T).T).sum(-1),sizes))
    return torch.stack(scores)
def distance(x):
    n=(x*x).sum(-1);return (n[:,None]+n[None,:]-2*x@x.T).clamp_min(0)
def pair(bank,w,z,x,sizes=SIZES,K=256,mode='pair'):
    u=(torch.arange(K,device=DEV,dtype=DT)+.5)/K
    ix=torch.searchsorted(w.cumsum(0),u).clamp_max(bank.N-1);ll=bank.ll(0,x)[ix]
    good=ll>=ll.max()-8;eligible=good[:,None]&good[None,:]&((ll[:,None]-ll[None,:]).abs()<=2)
    if mode=='unfiltered':eligible=torch.ones_like(eligible)
    Ds=torch.stack([distance(v)*.5 for v in torch.split(z[ix],sizes,dim=-1)],-1)
    if mode=='agnostic':Ds=torch.ones_like(Ds)
    weights=Ds*eligible[:,:,None];den=eligible.sum().clamp_min(1)
    scores=[torch.zeros(len(sizes),device=DEV,dtype=DT)]
    for a in [1,2,3]:
        B=-torch.expm1(-distance(bank.white[a][ix])/8)
        scores.append((weights*B[:,:,None]).sum((0,1))/den)
    scores=torch.stack(scores);fallback=scores[1:].max(0).values<=1e-14
    if fallback.any():scores=torch.where(fallback[None,:],lmmse(bank,w,z,sizes),scores)
    return scores,dict(eligible=float(eligible.to(DT).mean()),fallback=fallback.cpu().tolist())
def plan(M=512,seed=81531):
    u=torch.quasirandom.SobolEngine(8,scramble=True,seed=seed).draw(M,dtype=DT).to(DEV)
    return u[:,0],torch.special.ndtri(u[:,1:].clamp(1e-9,1-1e-9))
def nested(bank,w,z,outer,ow,oz,sizes=SIZES,M=512,seed=81531,preplan=None,return128=False):
    u,eps=plan(M,seed) if preplan is None else preplan;ix=torch.searchsorted(ow.cumsum(0),u).clamp_max(outer.N-1)
    truth=oz[ix];base=group((truth-w@z)**2,sizes).mean(0)
    risks=[base];variances=[group(w@(z*z)-(w@z)**2,sizes)];risks128=[base]
    lp=w.clamp_min(1e-300).log()
    for a in [1,2,3]:
        allrisk=[];allvar=[]
        for start in range(0,M,128):
            idx=ix[start:start+128];y=outer.mu[a][idx]+eps[start:start+128,:outer.mu[a].shape[-1]]*noise(a)
            weights=torch.softmax(lp+bank.ll(a,y),-1);pred=weights@z
            allrisk.append(group((pred-truth[start:start+128])**2,sizes));allvar.append(group(weights@(z*z)-pred*pred,sizes))
        rs=torch.cat(allrisk);vs=torch.cat(allvar)
        risks.append(rs.mean(0));variances.append(vs.mean(0));risks128.append(rs[:128].mean(0))
    out=dict(risk=torch.stack(risks),variance=torch.stack(variances))
    if return128:out['risk128']=torch.stack(risks128)
    return out
def choose(scores):return scores[1:].argmax(0)+1
def choose_risk(risk):return risk[1:].argmin(0)+1
