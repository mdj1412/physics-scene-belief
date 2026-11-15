"""Continuous boundary, discrete geometry/contact families, quadrature inference."""
import math
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

DT=torch.float64
DEV='cuda'
L=.025
AB=.10
QVIN=(1,.9)
FY=np.linspace(-.010,.010,9)
VYS=[np.linspace(.019,.031,7),np.linspace(-.008,.008,7)]
FSD=.0012
VSD=[.001,.00035,.05]
ACTIONS=['none','query_view','factual_view','probe']

def t(x): return torch.as_tensor(x,dtype=DT,device=DEV)
def basis(y,g,derivative=False):
    z=y/L
    if derivative:
        return torch.where(g==0,z,torch.where(g==1,z-1.4*z**3,z-.6*z**3+.36*z**5))
    return L*torch.where(g==0,.5*z*z,torch.where(g==1,.5*z*z-.35*z**4,.5*z*z-.15*z**4+.06*z**6))
def profile(theta,ys):
    g,c,s,a,e=theta.unbind(-1);ys=t(ys)
    return s[...,None]*ys+a[...,None]*basis(ys,g[...,None])
def velocity(theta,y,vin,mu_override=None):
    g,c,s,a,e=theta.unbind(-1)
    slope=s+a*basis(t(y),g,True);d=torch.sqrt(1+slope*slope)
    vn=(vin[0]-slope*vin[1])/d;vt=(slope*vin[0]+vin[1])/d
    mu=.22*c if mu_override is None else t(mu_override)
    normal=(1+e)*vn
    tang=torch.sign(vt)*torch.minimum(vt.abs(),mu*normal)
    return torch.stack([vin[0]-normal/d-tang*slope/d,vin[1]+normal*slope/d-tang/d],-1)
def sensor(theta,action):
    if action in [1,2]:return profile(theta,VYS[action-1])
    return velocity(theta,0,(1,-.7))
def likelihood(y,mu,sd):
    yy=t(y)
    if yy.ndim==1:yy=yy[None,:]
    # The observation-only constant is omitted: posterior normalization cancels it.
    return (yy@mu.T-.5*(mu*mu).sum(1)[None,:])/sd**2

class Bank:
    def __init__(self,n):
        self.n=n;x,w=leggauss(n)
        s=.4*x;a=AB*x;e=.625+.275*x
        grid=np.stack(np.meshgrid([0,1],[0,1],s,a,e,indexing='ij'),-1).reshape(-1,5)
        ww=np.prod(np.stack(np.meshgrid(w/2,w/2,w/2,indexing='ij'),-1),-1)
        self.theta=t(grid);self.prior=t(np.tile(ww.ravel()/4,4));self.lp=self.prior.log()
        self.fimage=profile(self.theta,FY);self.fvel=velocity(self.theta,0,(1,.35))
        self.z=velocity(self.theta,L,QVIN);self.z2=.5*(self.z*self.z).sum(1)
        self.mu={a:sensor(self.theta,a) for a in [1,2,3]}
        self.group=(2*self.theta[:,0]+self.theta[:,1]).long()
        # Group moments make updated plug-in prediction efficient.
        self.mom=torch.cat([torch.nn.functional.one_hot(self.group,4).to(DT)[:,:,None]*torch.cat([torch.ones((len(grid),1),device=DEV,dtype=DT),self.theta[:,2:]],1)[:,None,:]],0).reshape(len(grid),16)
    def posterior(self,public):
        ll=likelihood(public['profile'],self.fimage,FSD)[0]+likelihood(public['velocity'],self.fvel,.07)[0]+self.lp
        return torch.softmax(ll,0)
    def factorize(self,w):
        q=w.reshape(2,2,self.n,self.n,self.n)
        geom=q.sum((1,4));contact=q.sum((0,2,3))
        return (geom[:,None,:,:,None]*contact[None,:,None,None,:]).reshape(-1)
    def decode(self,w,all_modes=False):
        if w.ndim==1:w=w[None,:]
        mean=w@self.z;mom=(w@self.mom).reshape(-1,4,4)
        group=mom[:,:,0].argmax(1);selected=mom[torch.arange(len(w),device=DEV),group]
        pars=selected[:,1:]/selected[:,:1].clamp_min(1e-300)
        th=torch.cat([(group//2)[:,None],(group%2)[:,None],pars],1)
        plug=velocity(th,L,QVIN)
        if all_modes:
            probabilities=mom[:,:,0]
            parameters=mom[:,:,1:]/probabilities[:,:,None].clamp_min(1e-300)
            groups=torch.arange(4,device=DEV).expand(len(w),-1)
            components=torch.cat([(groups//2)[:,:,None],(groups%2)[:,:,None],parameters],2)
            mix=(probabilities[:,:,None]*velocity(components,L,QVIN)).sum(1)
            return mean,plug,mix
        return mean,plug
    def update(self,w,action,obs):
        return torch.softmax(likelihood(obs,self.mu[action],VSD[action-1])+w.clamp_min(1e-300).log()[None,:],1)
    def outer(self,w,H,seed):
        gen=torch.Generator(device=DEV).manual_seed(seed)
        ids=torch.multinomial(w,H,replacement=True,generator=gen)
        return {a:self.mu[a][ids]+VSD[a-1]*torch.randn((H,self.mu[a].shape[1]),generator=gen,device=DEV,dtype=DT) for a in [1,2,3]}
    def risks(self,w,outer):
        mean,plug,mix=self.decode(w,True);var=w@self.z2-.5*(mean*mean).sum()
        joint=[var];plugin=[var+.5*((plug-mean)**2).sum()]
        mixture=[var+.5*((mix-mean)**2).sum()]
        ess=[]
        for a in [1,2,3]:
            post=self.update(w,a,outer[a]);mm,pp,mp=self.decode(post,True)
            variance=post@self.z2-.5*(mm*mm).sum(1)
            joint.append(variance.mean());plugin.append((variance+.5*((pp-mm)**2).sum(1)).mean())
            mixture.append((variance+.5*((mp-mm)**2).sum(1)).mean())
            ess.append(float(torch.median(1/(post*post).sum(1))))
        self.last_mixture_risk=torch.stack(mixture).cpu().numpy()
        return torch.stack(joint).cpu().numpy(),torch.stack(plugin).cpu().numpy(),ess
    def snis(self,w,outer,seed,N=8192):
        gen=torch.Generator(device=DEV).manual_seed(seed);ids=torch.multinomial(w,N,replacement=True,generator=gen)
        z=self.z[ids];n=N//2;m0=(w@self.z)[None,:];std=[0.];cf=[0.];ess=[]
        for a in [1,2,3]:
            ll=likelihood(outer[a],self.mu[a][ids],VSD[a-1]);p=torch.softmax(ll,1)
            pa=torch.softmax(ll[:,:n],1);pb=torch.softmax(ll[:,n:],1)
            d=p@z-m0;da=pa@z[:n]-m0;db=pb@z[n:]-m0
            std.append(float(.5*(d*d).sum(1).mean()));cf.append(float(.5*(da*db).sum(1).mean()))
            ess.append(float(torch.median(1/(p*p).sum(1))))
        return np.array(std),np.array(cf),ess
    def diagnostics(self,w):
        q=w.reshape(2,2,self.n,self.n,self.n);fac=self.factorize(w)
        mi=float((w*(w.clamp_min(1e-300).log()-fac.clamp_min(1e-300).log())).sum())
        pars=self.theta[:,2:];mean=w@pars;cov=(pars-mean).T@(w[:,None]*(pars-mean))
        corr=float(cov[0,2]/torch.sqrt(cov[0,0]*cov[2,2]))
        return dict(mutual_information=mi,s_e_correlation=corr,p_geometry=q.sum((1,2,3,4)).cpu().tolist(),p_contact=q.sum((0,2,3,4)).cpu().tolist())

def draw_truth(seed,misspec=False):
    r=np.random.default_rng(seed)
    th=np.array([r.integers(2),r.integers(2),r.uniform(-.4,.4),r.uniform(-AB,AB),r.uniform(.35,.9)],float)
    mu=.22*th[1]
    if misspec:th[0]=2;th[1]=1;mu=r.uniform(.1,.3)
    theta=t(th)
    fp=profile(theta,FY).cpu().numpy();fv=velocity(theta,0,(1,.35),mu).cpu().numpy()
    public=dict(profile=(fp+r.normal(0,FSD,9)).tolist(),velocity=(fv+r.normal(0,.07,2)).tolist())
    target=velocity(theta,L,QVIN,mu).cpu().numpy()
    obs={}
    for a in [1,2,3]:
        mean=profile(theta,VYS[a-1]) if a<3 else velocity(theta,0,(1,-.7),mu)
        obs['obs'+str(a)]=mean.cpu().numpy()+r.normal(0,VSD[a-1],(32,len(mean)))
    return public,dict(theta=th,mu=mu,target=target,**obs)
