"""Public-belief adaptive compression. Error scales are heuristic, not confidence bounds."""
import torch
from model import *
from mixture import posterior,mv,payload
from acquisition import plan
LEVELS=[8,16,32,64,128]
def phases(seed):
    return torch.quasirandom.SobolEngine(4,scramble=True,seed=seed+1000).draw(1,dtype=DT).to(DEV).reshape(2,2)
def compact(b,w,z,K,phase,global_mode=False):
    ids=[];weights=[];active=[];omitted=0.
    for r in [0,1]:
        ix=torch.where((z[:,0]>0)==bool(r))[0];m=w[ix].sum()
        if float(m)<=1e-10:omitted+=float(m);continue
        active.append((r,ix,m))
    if global_mode:
        n=K*len(active);u=(torch.arange(n,device=DEV,dtype=DT)+phase[0])/n
        idx=torch.searchsorted(w.cumsum(0),u).clamp_max(len(w)-1);cw=torch.full((n,),1/n,device=DEV,dtype=DT)
    else:
        for r,ix,m in active:
            u=(torch.arange(K,device=DEV,dtype=DT)+phase[r])/K
            idx=ix[torch.searchsorted((w[ix]/m).cumsum(0),u).clamp_max(len(ix)-1)]
            ids.append(idx);weights.append((m/K).expand(K))
        idx=torch.cat(ids);cw=torch.cat(weights);cw=cw/cw.sum()
    return dict(w=cw,z=z[idx],mu={a:b.white[a][idx] for a in [1,2,3]},components=len(idx),omitted_mass=omitted)
def outer_samples(o,ow,oz,preplan):
    u,eps=preplan;idx=torch.searchsorted(ow.cumsum(0),u.contiguous()).clamp_max(o.N-1)
    return dict(truth=oz[idx],y={a:o.white[a][idx]+eps[:,:o.white[a].shape[1]] for a in [1,2,3]})
def losses(comp,sample):
    return torch.stack([.5*((mv(posterior(comp,a,sample['y'][a]),comp['z'])-sample['truth'])**2).sum(-1) for a in [1,2,3]],-1)
def select(b,w,z,sample,phase,global_mode=False,fixed=None):
    previous=None;history=[];comps=None;payload_total=0
    for K in ([fixed] if fixed else LEVELS):
        comps=[compact(b,w,z,K,phase[j],global_mode) for j in [0,1]]
        ll=torch.stack([losses(c,sample) for c in comps]);per=ll.mean(0);risk=per.mean(0);rep=ll.mean(1);best=int(risk.argmin());g=risk-risk[best]
        paired=per-per[:,best,None];se=paired.std(0,unbiased=True)/len(per)**.5
        drift=torch.zeros_like(g) if previous is None else (g-(previous-previous[best])).abs()
        spread=((rep[0]-rep[0,best])-(rep[1]-rep[1,best])).abs()/2
        error=2*se+drift+spread;others=torch.tensor([a for a in range(3) if a!=best],device=DEV);worst=int(others[(g-error)[others].argmin()])
        tie=float(risk.max()-risk.min())<=1e-12
        resolved=previous is not None and bool((g[others]>error[others]).all())
        history.append(torch.cat([risk,t([K,float(g[worst]),float(2*se[worst]),float(drift[worst]),float(spread[worst]),float((g-error)[worst])])]))
        if fixed or tie or resolved or K==LEVELS[-1]:break
        previous=risk
    action=1 if tie else best+1
    # reason 0: gap separated; 1: cap unresolved; 2: numerical tie; 3: fixed size
    reason=3 if fixed else (2 if tie else (0 if resolved else 1))
    meta=t([K,len(history),reason,comps[0]['components'],sum(payload(c) for c in comps),max(c['omitted_mass'] for c in comps)])
    hist=torch.full((len(LEVELS),9),float('nan'),device=DEV,dtype=DT);hist[:len(history)]=torch.stack(history)
    return action,risk,meta,hist
