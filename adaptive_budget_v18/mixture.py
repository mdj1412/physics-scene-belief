"""Compact conditional predictive mixtures; no private data access."""
import torch
from model import *

def normal_plan(count=2048,seed=91631):
    u=torch.quasirandom.SobolEngine(7,scramble=True,seed=seed).draw(count,dtype=DT).to(DEV)
    return torch.special.ndtri(u.clamp(1e-9,1-1e-9))
def resample(w,K):
    return torch.searchsorted(w.cumsum(0).contiguous(),(torch.arange(K,device=DEV,dtype=DT)+.5)/K).clamp_max(len(w)-1)
def region_ids(w,z,event):
    if event:return (z[:,0]>0).long()
    order=z[:,0].argsort();q=z[order,0][torch.searchsorted(w[order].cumsum(0),t([.25,.5,.75])).clamp_max(len(w)-1)]
    return torch.bucketize(z[:,0].contiguous(),q)
def compress(bank,w,z,event,K=32,global_mode=False):
    regions=region_ids(w,z,event);ids=[];masses=[];omitted=0.;active=[]
    for r in range(2 if event else 4):
        idx=torch.where(regions==r)[0];mass=w[idx].sum()
        if float(mass)<=1e-10:omitted+=float(mass);continue
        active.append((idx,mass))
    count=K*len(active)
    if global_mode:
        ix=resample(w,count);cw=torch.full((count,),1/count,device=DEV,dtype=DT)
    else:
        for idx,mass in active:
            ids.append(idx[resample(w[idx]/mass,K)]);masses.append((mass/K).expand(K))
        ix=torch.cat(ids);cw=torch.cat(masses);cw=cw/cw.sum()
    return dict(w=cw,z=z[ix],mu={a:bank.white[a][ix] for a in [1,2,3]},omitted_mass=omitted,components=len(ix),regions=len(active))
def mv(A,Z):return torch.stack([A@Z[:,j] for j in range(Z.shape[1])],-1)
def posterior(comp,a,ywhite):
    mu=comp['mu'][a];ll=ywhite@mu.T-.5*(mu*mu).sum(-1)
    return torch.softmax(ll+comp['w'].log(),-1)
def risks(comp,eps,Q=4,outer=None):
    source=comp if outer is None else outer;cw=source['w'];K=len(cw);z=comp['z'];oz=source['z'];means=comp['w']@z
    base=.5*(cw*((oz-means)**2).sum(-1)).sum();rs=[base]
    for a in [1,2,3]:
        mu=source['mu'][a];ys=(mu[:,None,:]+eps[:K*Q,:mu.shape[1]].reshape(K,Q,-1)).flatten(0,1)
        post=posterior(comp,a,ys);pred=mv(post,z)
        if outer is None:loss=.5*(post@(z*z).sum(-1)-(pred*pred).sum(-1)).clamp_min(0)
        else:loss=.5*((pred-oz[:,None,:].expand(-1,Q,-1).reshape(-1,z.shape[-1]))**2).sum(-1)
        rs.append((cw*loss.reshape(K,Q).mean(1)).sum())
    return torch.stack(rs)
def predict(comp,a,y):return mv(posterior(comp,a,t(y)/noise(a)),comp['z'])
def payload(comp):return sum(v.numel()*v.element_size() for v in [comp['w'],comp['z']]+list(comp['mu'].values()))
