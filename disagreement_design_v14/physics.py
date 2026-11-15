"""Forward equations only. No reference posterior or dataset imports."""
import numpy as np
import torch
DT=torch.float64
DEV='cuda'
L=.025
FY=np.linspace(-.010,.010,9)
VYS=[np.linspace(.019,.031,7),np.linspace(-.008,.008,7)]
ACTIONS=['none','query_view','factual_view','probe']
CONDITIONS=['matched','boundary_only','contact_only']
def t(x):return torch.as_tensor(x,dtype=DT,device=DEV)
def basis(y,g,derivative=False):
    z=y/L
    if derivative:return torch.where(g==0,z,torch.where(g==1,z-1.4*z**3,z-.6*z**3+.36*z**5))
    return L*torch.where(g==0,.5*z*z,torch.where(g==1,.5*z*z-.35*z**4,.5*z*z-.15*z**4+.06*z**6))
def profile(th,ys):
    g,c,s,a,e=th.unbind(-1);y=t(ys)
    return s[...,None]*y+a[...,None]*basis(y,g[...,None])
def velocity(th,y,vin,viscous=None):
    g,c,s,a,e=th.unbind(-1);slope=s+a*basis(t(y),g,True);norm=torch.sqrt(1+slope*slope)
    vn=(vin[0]-slope*vin[1])/norm;vt=(vin[0]*slope+vin[1])/norm;jn=(1+e)*vn
    jt=torch.sign(vt)*torch.minimum(vt.abs(),.22*c*jn)
    if viscous is not None:jt=t(viscous)*c*vt
    return torch.stack([vin[0]-jn/norm-jt*slope/norm,vin[1]+jn*slope/norm-jt/norm],-1)
def target(th):return velocity(th,L,(1,.9))
def measure(th,action):
    if action==4:return profile(th,FY)
    if action==5:return velocity(th,0,(1,.35))
    if action==0:return torch.cat([profile(th,FY),velocity(th,0,(1,.35))],-1)
    if action in [1,2]:return profile(th,VYS[action-1])
    return velocity(th,0,(1,-.7))
def noise(action):
    if action==4:return t([.0012]*9)
    if action==5:return t([.07]*2)
    if action==0:return t([.0012]*9+[.07]*2)
    if action in [1,2]:return t([.001 if action==1 else .00035]*7)
    return t([.05]*2)
def latent_to_theta(u):
    probs=torch.special.ndtr(u)
    pars=torch.stack([.8*probs[...,0]-.4,.2*probs[...,1]-.1,.35+.55*probs[...,2]],-1)
    gc=t([[0,0],[0,1],[1,0],[1,1]])
    shape=[1]*(u.ndim-3)+[4,1,2]
    return torch.cat([gc.reshape(shape).expand(*u.shape[:-1],2),pars],-1)
def make_world(seed,conditions=True,repeats=16):
    rng=np.random.default_rng(seed)
    base=np.array([rng.integers(2),rng.integers(2),rng.uniform(-.4,.4),rng.uniform(-.1,.1),rng.uniform(.35,.9)])
    lam=rng.uniform(.25,.65)
    eps0=rng.normal(size=11);eps={a:rng.normal(size=(repeats,len(noise(a)))) for a in [1,2,3]}
    publics=[];truths=[];targets=[];observations={a:[] for a in [1,2,3]}
    for condition in range(3 if conditions else 1):
        th=base.copy()
        if condition==1:th[0]=2
        visc=lam if condition==2 else None;theta=t(th)
        factual=torch.cat([profile(theta,FY),velocity(theta,0,(1,.35),visc)],-1)
        publics.append(factual.cpu().numpy()+eps0*noise(0).cpu().numpy())
        targets.append(velocity(theta,L,(1,.9),visc).cpu().numpy());truths.append(th)
        for a in [1,2,3]:
            mean=profile(theta,VYS[a-1]) if a<3 else velocity(theta,0,(1,-.7),visc)
            observations[a].append(mean.cpu().numpy()+eps[a]*noise(a).cpu().numpy())
    return np.array(publics),dict(theta=np.array(truths),lam=lam,target=np.array(targets),**{f'obs{a}':np.array(v) for a,v in observations.items()})
