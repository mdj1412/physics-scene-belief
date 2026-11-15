"""Candidate prior, observations, and public-context query targets. No data access."""
import os
import numpy as np
import torch
DEV=os.environ.get('BELIEF_DEVICE','cuda');DT=torch.float64
TASKS=['smooth_velocity','contact_velocity','contact_event','slip_velocity','slip_event']
SIZES=[2,2,1,2,1]
ACTIONS=['none','query_view','factual_view','probe']
FY=np.linspace(-.010,.010,9);VYS=[np.linspace(.019,.031,7),np.linspace(-.008,.008,7)]
def t(x):return torch.as_tensor(x,dtype=DT,device=DEV)
def unpack(th):return th.unbind(-1)
def basis(y,g,derivative=False):
    z=y/.025
    if derivative:return torch.where(g==0,z,z-1.4*z**3)
    return .025*torch.where(g==0,.5*z*z,.5*z*z-.35*z**4)
def profile(th,ys):
    g,c,d,s,a,e,b0,b1=unpack(th);y=t(ys)
    return d[...,None]+s[...,None]*y+a[...,None]*basis(y,g[...,None])
def velocity(th,y,vin,law='nominal',details=False):
    g,c,d,s,a,e,b0,b1=unpack(th);slope=s+a*basis(t(y),g,True);norm=torch.sqrt(1+slope*slope)
    vn=(vin[0]-slope*vin[1])/norm;vt=(vin[0]*slope+vin[1])/norm;jn=(1+e)*vn
    jt=torch.sign(vt)*torch.minimum(vt.abs(),.22*(c==1)*jn.clamp_min(0))
    if law=='nominal':frac=torch.sigmoid(b0+b1*torch.log((vt.abs()+.02)/.5))
    else:
        scale=.1+.7*torch.sigmoid(b1);frac=torch.sigmoid(b0)*(.2+.8/(1+(vt.abs()/scale)**2))
    jt=torch.where(c==2,vt*frac,jt);out=torch.stack([vin[0]-jn/norm-jt*slope/norm,vin[1]+jn*slope/norm-jt/norm],-1)
    return (out,vt-jt,vn) if details else out
def measure(th,a,law='nominal'):
    if a==0:return torch.cat([profile(th,FY),velocity(th,0,(1,.35),law)],-1)
    if a in [1,2]:return profile(th,VYS[a-1])
    return velocity(th,0,(1,-.7),law)
def noise(a):
    return t(([.0012]*9+[.07]*2) if a==0 else ([.001 if a==1 else .00035]*7 if a in [1,2] else [.05]*2))
def context(factual):
    x=t(factual)[:9];s=(x*t(FY)).sum()/(t(FY)**2).sum();end=x.mean()+s*.025;s=s.clamp(-.6,.6);r=.22*1.625
    return torch.stack([end,(r-s)/(1+r*s)])
def targets(th,ctx,law='nominal',task=None):
    which=TASKS if task is None else [task];out=[]
    for name in which:
        if name=='smooth_velocity':z=velocity(th,.025,(1,.9),law)
        elif name.startswith('contact'):
            event=profile(th,[.025])[...,0]<=ctx[0]
            if name=='contact_event':z=event.to(DT)[...,None]*2**.5
            else:z=torch.where(event[...,None],velocity(th,.025,(1,.9),law),t([1,.9]))
        else:
            v,tangent,vn=velocity(th,.025,(1,ctx[1]),law,True)
            z=v if name=='slip_velocity' else (tangent.abs()<=1e-8).to(DT)[...,None]*2**.5
        out.append(z)
    return torch.cat(out,-1)
def from_unit(u):
    idx=torch.bucketize(u[:,0].contiguous(),t([.125,.25,.375,.5,.75]));gc=t([[0,0],[0,1],[1,0],[1,1],[0,2],[1,2]])[idx]
    pars=t([-.004,-.4,-.1,.35,-4,-2])+u[:,1:]*t([.008,.8,.2,.55,6,4])
    return torch.cat([gc,pars],-1)
def group(x,sizes=SIZES):return torch.stack([v.sum(-1) for v in torch.split(x,sizes,dim=-1)],-1)*.5

class Bank:
    def __init__(self,N=16384,seed=2026111201,law='nominal'):
        self.N=N;self.seed=seed;self.law=law
        self.theta=from_unit(torch.quasirandom.SobolEngine(7,scramble=True,seed=seed).draw(N,dtype=DT).to(DEV))
        self.mu={a:measure(self.theta,a,law) for a in range(4)}
        self.white={a:m/noise(a) for a,m in self.mu.items()};self.norm={a:.5*(m*m).sum(-1) for a,m in self.white.items()}
    def ll(self,a,y):return (t(y)/noise(a))@self.white[a].T-self.norm[a]
    def fit(self,x):return torch.softmax(self.ll(0,x),-1)
    def update(self,w,a,y):return torch.softmax(w.clamp_min(1e-300).log()+self.ll(a,y),-1)
    def cache_bytes(self):return sum(x.numel()*x.element_size() for x in [self.theta]+list(self.mu.values())+list(self.white.values())+list(self.norm.values()))
