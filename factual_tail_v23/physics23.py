"""Physical world law, candidate observation model and independent paired worlds."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"tail_recovery_v22"))
from policy22 import torch, t, DT, RecoveryEngine
from model import FY, VYS, Bank
from common import R, save
import numpy as np

# Columns: family (0 curvature, 1 friction), offset, slope, curvature, friction, restitution.
def parameters(u):
    f=(u[:,0]>=.5).to(DT)
    sign=torch.where(u[:,1]<.5,-1.,1.).to(DT)
    slope=(f*2-1)*.115+.14*(u[:,2]-.5)
    curvature=(1-f)*(.30*sign+.10*(u[:,3]-.5))+f*.08*(u[:,3]-.5)
    friction=(1-f)*(.215+.08*(u[:,4]-.5))+f*(.22+.145*sign+.06*(u[:,4]-.5))
    return torch.stack([f,.008*(u[:,5]-.5),slope,curvature,friction,.58+.26*(u[:,6]-.5)],dim=1)

def profile(th,y):
    _,offset,s,c,_,_=th.unbind(-1)
    y=t(y)
    return offset[...,None]+s[...,None]*y+c[...,None]*(y*y/.05)

def impact(th,y,incoming):
    _,_,s,c,mu,e=th.unbind(-1)
    slope=s+c*y/.025
    norm=(1+slope.square()).sqrt()
    vn=(1-slope*incoming)/norm
    vt=(slope+incoming)/norm
    jn=(1+e)*vn.clamp_min(0)
    jt=vt.sign()*torch.minimum(vt.abs(),mu*jn)
    return torch.stack([1-(jn+jt*slope)/norm,incoming+(jn*slope-jt)/norm],-1),vt.abs()<=mu*jn

def noiseless(th):
    return [torch.cat([profile(th,FY),impact(th,0,-.10)[0]],-1),
            profile(th,VYS[0]),profile(th,VYS[1]),impact(th,0,-.85)[0]]

def sigmas(setting):
    multiplier=(1,24)[setting]
    return [t([.0025]*9+[.16]*2),t([.00035]*7)*multiplier,
            t([.001]*7)*multiplier,t([.025]*2)*multiplier]

class PhysicalBank(Bank):
    def __init__(self,seed,setting):
        self.N=16384
        unit=torch.quasirandom.SobolEngine(7,scramble=True,seed=seed).draw(self.N,dtype=DT).to(t(0).device)
        self.theta=parameters(unit)
        self.sigma=sigmas(setting)
        self.mu=dict(enumerate(noiseless(self.theta)))
        self.white={a:m/self.sigma[a] for a,m in self.mu.items()}
        self.norm={a:.5*m.square().sum(-1) for a,m in self.white.items()}
        self.label=impact(self.theta,.025,.425)[1].to(DT)[:,None]*np.sqrt(2.)
    def ll(self,a,y):
        return (y/self.sigma[a])@self.white[a].T-self.norm[a]

def generate(split,count,seed,repeats=32):
    rng=np.random.default_rng(seed)
    u=rng.uniform(size=(count,7))
    theta=parameters(t(u))
    signals=noiseless(theta)
    x=signals[0]+t(rng.normal(size=(count,11)))*sigmas(0)[0]
    data={"label":impact(theta,.025,.425)[1].cpu().numpy(),"family":theta[:,0].cpu().numpy()}
    for a in (1,2,3):
        eps=t(rng.normal(size=(count,repeats,signals[a].shape[-1])))
        data[f"obs{a}"]=torch.stack([signals[a][:,None,:]+eps*sigmas(s)[a] for s in (0,1)],dim=1).cpu().numpy()
    save(R/split/"public.npz",x=x.cpu().numpy())
    save(R/split/"private.npz",**data)
