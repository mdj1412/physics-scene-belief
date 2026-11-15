"""Equivalent implementation: cache region CDFs and avoid logging-related GPU synchronization.
No change to objective, thresholds, sampling phases, K schedule, or stopping decisions.
"""
from adaptive import LEVELS,losses
from model import *
from mixture import payload
def prepare_compact(w,z):
    regions=[];omitted=0.
    ids=[torch.where((z[:,0]>0)==bool(r))[0] for r in [0,1]]
    masses=torch.stack([w[ix].sum() for ix in ids]);values=masses.tolist()
    for r in [0,1]:
        if values[r]<=1e-10:omitted+=values[r]
        else:regions.append((r,ids[r],masses[r],(w[ids[r]]/masses[r]).cumsum(0)))
    return regions,omitted,w.cumsum(0)
def make_compact(b,w,z,K,phase,prepared,global_mode):
    regions,omitted,cdf=prepared
    if global_mode:
        n=K*len(regions);u=(torch.arange(n,device=DEV,dtype=DT)+phase[0])/n
        idx=torch.searchsorted(cdf,u).clamp_max(len(w)-1);cw=torch.full((n,),1/n,device=DEV,dtype=DT)
    else:
        ids=[];weights=[];ar=torch.arange(K,device=DEV,dtype=DT)
        for r,ix,m,rcdf in regions:
            u=(ar+phase[r])/K;ids.append(ix[torch.searchsorted(rcdf,u).clamp_max(len(ix)-1)]);weights.append((m/K).expand(K))
        idx=torch.cat(ids);cw=torch.cat(weights);cw=cw/cw.sum()
    return dict(w=cw,z=z[idx],mu={a:b.white[a][idx] for a in [1,2,3]},components=len(idx),omitted_mass=omitted)
def select(b,w,z,sample,phase,global_mode=False,fixed=None):
    prepared=prepare_compact(w,z);previous=None;history=[];others_table=torch.tensor([[1,2],[0,2],[0,1]],device=DEV)
    for K in ([fixed] if fixed else LEVELS):
        comps=[make_compact(b,w,z,K,phase[j],prepared,global_mode) for j in [0,1]]
        ll=torch.stack([losses(c,sample) for c in comps]);per=ll.mean(0);risk=per.mean(0);rep=ll.mean(1);best=int(risk.argmin());g=risk-risk[best]
        paired=per-per[:,best,None];se=paired.std(0,unbiased=True)/len(per)**.5
        drift=torch.zeros_like(g) if previous is None else (g-(previous-previous[best])).abs()
        spread=((rep[0]-rep[0,best])-(rep[1]-rep[1,best])).abs()/2
        error=2*se+drift+spread;others=others_table[best];worst=others[(g-error)[others].argmin()]
        tie=float(risk.max()-risk.min())<=1e-12
        resolved=previous is not None and bool((g[others]>error[others]).all())
        history.append(torch.cat([risk,t([K]),torch.stack([g[worst],2*se[worst],drift[worst],spread[worst],(g-error)[worst]])]))
        if fixed or tie or resolved or K==LEVELS[-1]:break
        previous=risk
    action=1 if tie else best+1;reason=3 if fixed else (2 if tie else (0 if resolved else 1))
    meta=t([K,len(history),reason,comps[0]['components'],sum(payload(c) for c in comps),prepared[1]])
    hist=torch.full((len(LEVELS),9),float('nan'),device=DEV,dtype=DT);hist[:len(history)]=torch.stack(history)
    return action,risk,meta,hist
