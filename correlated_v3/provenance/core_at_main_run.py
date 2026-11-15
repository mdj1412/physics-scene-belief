"""Exact frictionless sphere/box impact and factual-conditioned inference.

All quantities are SI except normalized outgoing velocity. The initial x
velocity is positive and the ball always reaches the front of the box.
No hidden truth is passed to inference or acquisition.
"""
import numpy as np
from scipy.special import ndtr, logsumexp
from scipy.stats import truncnorm

GLO,GHI=.032,.0498
ELO,EHI=.1,.9
RADIUS=.02
HALF_Y=.03
NOISES=[.03,.01,.003,.001]
SEED=20261104

def impact(g,e):
    g,e=np.broadcast_arrays(g,e)
    u=np.clip((abs(g)-HALF_Y)/RADIUS,0,1)
    A=1-u*u
    vx=1-(1+e)*A
    vy=-(1+e)*np.sqrt(A)*u*np.sign(g)
    return vx,vy,(abs(g)<HALF_Y+RADIUS).astype(float)

def normalize(logw):
    return np.exp(logw-logsumexp(logw))

def expand_sign(g,e,w,signed):
    if signed:return np.r_[g,-g],np.r_[e,e],np.r_[w,w]/2
    return g,e,w

def conditional_params(g,y,sigma):
    A=1-((g-HALF_Y)/RADIUS)**2
    mean=(1-y)/A-1
    sd=sigma/A
    aa=(ELO-mean)/sd;bb=(EHI-mean)/sd
    # Stable normal interval probability; both upper-tail endpoints reflect.
    mass=np.where(aa>0,ndtr(-aa)-ndtr(-bb),ndtr(bb)-ndtr(aa))
    return mean,sd,aa,bb,np.maximum(mass,1e-300)/A

def support(y,sigma):
    alo=np.clip((1-y-8*sigma)/(1+EHI),0,1)
    ahi=np.clip((1-y+8*sigma)/(1+ELO),0,1)
    lo=max(GLO,HALF_Y+RADIUS*np.sqrt(1-ahi))
    hi=min(GHI,HALF_Y+RADIUS*np.sqrt(1-alo))
    assert hi>lo
    return lo,hi

def conditional_quad(y,sigma,signed,ng=256,ne=16):
    lo,hi=support(y,sigma)
    x,w=np.polynomial.legendre.leggauss(ng)
    g=lo+(x+1)*(hi-lo)/2
    mean,sd,aa,bb,density=conditional_params(g,y,sigma)
    pg=w*density;pg/=pg.sum()
    xe,we=np.polynomial.legendre.leggauss(ne)
    e=truncnorm.ppf((xe[None,:]+1)/2,aa[:,None],bb[:,None],loc=mean[:,None],scale=sd[:,None])
    gg=np.broadcast_to(g[:,None],e.shape)
    ww=pg[:,None]*we[None,:]/2
    return expand_sign(gg.ravel(),e.ravel(),ww.ravel(),signed),dict(factual_evals=ng,conditional_quantiles=ng*ne)

def fixed_grid(y,sigma,signed,ng,ne):
    g=np.linspace(GLO,GHI,ng);e=np.linspace(ELO,EHI,ne)
    gg,ee=np.meshgrid(g,e,indexing='ij')
    w=np.ones_like(gg);w[[0,-1],:]*=.5;w[:,[0,-1]]*=.5
    p=normalize(np.log(w.ravel())-.5*((impact(gg,ee)[0].ravel()-y)/sigma)**2)
    return expand_sign(gg.ravel(),ee.ravel(),p,signed),dict(factual_evals=ng*ne)

def prior_particles(y,sigma,signed,rng,n=512):
    g=rng.uniform(GLO,GHI,n);e=rng.uniform(ELO,EHI,n)
    if signed:g*=rng.choice([-1,1],n)
    w=normalize(-.5*((impact(g,e)[0]-y)/sigma)**2)
    return (g,e,w),dict(factual_evals=n)

def posterior_particles(y,sigma,signed,rng,n=512):
    lo,hi=support(y,sigma)
    gline=np.linspace(lo,hi,4097)
    den=conditional_params(gline,y,sigma)[-1]
    cdf=np.r_[0,np.cumsum((den[1:]+den[:-1])/2)];cdf/=cdf[-1]
    g=np.interp(rng.random(n),cdf,gline)
    mean,sd,aa,bb,_=conditional_params(g,y,sigma)
    e=truncnorm.ppf(rng.random(n),aa,bb,loc=mean,scale=sd)
    if signed:g*=rng.choice([-1,1],n)
    return (g,e,np.full(n,1/n)),dict(factual_evals=4097,conditional_quantiles=n)

def systematic(w,rng):
    return np.minimum(np.searchsorted(np.cumsum(w),(rng.random()+np.arange(len(w)))/len(w)),len(w)-1)

def smc(y,sigma,signed,rng,n=512):
    # Adaptive tempering, systematic resampling and covariance-scaled MH.
    # Work in unit-square coordinates; symmetric proposals reject outside.
    x=rng.random((n,2));beta=0.;calls=n;stages=0;accept=0;attempts=0
    def loglike(xx):
        g=GLO+(GHI-GLO)*xx[:,0];e=ELO+(EHI-ELO)*xx[:,1]
        return -.5*((impact(g,e)[0]-y)/sigma)**2
    ll=loglike(x)
    while beta<1-1e-12:
        lo=beta;hi=1.
        for _ in range(32):
            mid=(lo+hi)/2;w=normalize((mid-beta)*ll)
            if 1/(w@w)<.8*n:hi=mid
            else:lo=mid
        nxt=1. if hi>1-1e-9 else lo
        assert nxt>beta+1e-12
        w=normalize((nxt-beta)*ll)
        cov=np.cov(x.T,aweights=w)+np.eye(2)*1e-10
        idx=systematic(w,rng);x=x[idx];ll=ll[idx];beta=nxt
        chol=np.linalg.cholesky(cov)*(2.38/np.sqrt(2))
        for _ in range(10):
            proposal=x+rng.standard_normal(x.shape)@chol.T
            valid=((proposal>=0)&(proposal<=1)).all(1)
            lp=loglike(np.clip(proposal,0,1));calls+=n
            ok=valid&(np.log(rng.random(n))<beta*(lp-ll))
            x[ok]=proposal[ok];ll[ok]=lp[ok]
            accept+=ok.sum();attempts+=n
        stages+=1
        assert stages<200
    g=GLO+(GHI-GLO)*x[:,0];e=ELO+(EHI-ELO)*x[:,1]
    if signed:g*=rng.choice([-1,1],n)
    return (g,e,np.full(n,1/n)),dict(factual_evals=calls,stages=stages,mh_accept=float(accept/attempts))

def factorized(atoms,n=64):
    g,e,w=atoms
    u=(np.arange(n)+.5)/n
    def quantile(x):
        idx=np.argsort(x);return np.interp(u,np.cumsum(w[idx]),x[idx])
    gg,ee=np.meshgrid(quantile(g),quantile(e),indexing='ij')
    return (gg.ravel(),ee.ravel(),np.full(n*n,1/(n*n))),dict(factual_evals=0,reference_derived=True)

def diagnostic(atoms,y,sigma):
    g,e,w=atoms
    def corr(a,b):
        a=a-w@a;b=b-w@b
        return float(w@(a*b)/max(np.sqrt((w@(a*a))*(w@(b*b))),1e-30))
    hist=np.histogram2d(abs(g),e,bins=[np.linspace(GLO,GHI,33),np.linspace(ELO,EHI,33)],weights=w)[0]
    independent=hist.sum(1)[:,None]*hist.sum(0)[None,:]
    ok=hist>0
    return dict(corr_g_e=corr(g,e),corr_abs_g_e=corr(abs(g),e),mi_hist=float(np.sum(hist[ok]*np.log(hist[ok]/independent[ok]))),
        ess=float(1/(w@w)),positive_mass=float(w[g>0].sum()),
        factual_rmse=float(np.sqrt(w@((impact(g,e)[0]-y)**2))),
        in_three_sigma=float(w[abs(impact(g,e)[0]-y)<3*sigma].sum()))
