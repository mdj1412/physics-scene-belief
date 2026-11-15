"""Known-family linear image sensor and exact single-contact response."""
import math
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

DT=torch.float64
DEV='cuda' if torch.cuda.is_available() else 'cpu'
SD=np.array([.001,.0025,.0025])
BOUND=.004
QUERIES=np.array([-.033,-.025,-.017,.017,.025,.033])
ACTIONS=['none','left_view','right_view','probe']
FY=np.linspace(-.044,.044,81)
VY=[np.linspace(-.041,-.009,17),np.linspace(.009,.041,17)]
F_DX=.0005; F_W=48; F_PIXEL_SD=.15
V_DX=.00025; V_W=64; V_PIXEL_SD=.10
F_SD=F_DX*math.sqrt(F_W)*F_PIXEL_SD
V_SD=V_DX*math.sqrt(V_W)*V_PIXEL_SD

def t(x):return torch.as_tensor(x,dtype=DT,device=DEV)
def basis(y,derivative=False):
    y=np.asarray(y);u=(y[...,None]-np.array([-.025,.025]))/.018
    return np.where(abs(u)<1,-np.pi/(2*.018)*np.sin(np.pi*u) if derivative else .5*(1+np.cos(np.pi*u)),0.)
QB=basis(QUERIES,True)
def design(y):return np.column_stack([np.ones(len(y)),basis(y)])
def response(g,e):
    s=np.asarray(g)[...,1:]@QB.T
    k=(1+np.asarray(e)[...,None])/(1+s*s)
    return np.stack([1-k,k*s],-1)
def sensor(g,ys,dx,w,sd,rng):
    boundary=design(ys)@np.asarray(g)
    xs=(np.arange(w)+.5-w/2)*dx
    clean=np.clip((xs[None,:]+dx/2-boundary[:,None])/dx,0,1)
    raw=clean+rng.normal(0,sd,clean.shape)
    extracted=w*dx/2-dx*raw.sum(1)
    return raw,extracted,clean
def prior_draw(rng,n):
    out=[]
    while sum(len(a) for a in out)<n:
        x=rng.normal(0,SD,(max(n,32),3));out.append(x[(abs(x)<=BOUND).all(1)])
    g=np.concatenate(out)[:n]
    e=[]
    while sum(len(a) for a in e)<n:
        x=rng.normal(.65,.12,max(n,32));e.append(x[(x>=.45)&(x<=.85)])
    return g,np.concatenate(e)[:n]
def info(public,mode='boundary',obs=None,action='none'):
    J=np.diag(1/SD**2);h=np.zeros(3)
    ys=np.asarray(public['ys']);v=np.asarray(public['features']);D=design(ys)
    if mode=='coarse':D=D.mean(0,keepdims=True);v=np.array([v.mean()]);noise=F_SD/math.sqrt(len(ys))
    else:noise=F_SD
    J+=D.T@D/noise**2;h+=D.T@v/noise**2
    ep=1/.12**2+1/.06**2;eh=.65/.12**2+float(public['e_observed'])/.06**2
    if action in ['left_view','right_view']:
        d=design(VY[ACTIONS.index(action)-1]);v=np.asarray(obs)
        if mode=='coarse':d=d.mean(0,keepdims=True);v=np.array([v.mean()]);noise=V_SD/math.sqrt(len(obs))
        else:noise=V_SD
        J+=d.T@d/noise**2;h+=d.T@v/noise**2
    elif action=='probe':ep+=1/.015**2;eh+=float(obs)/.015**2
    return t(J),t(h),t(ep),t(eh)
def trunc_mean(p,h):
    sd=torch.rsqrt(p);mu=h/p;a=(.45-mu)/sd;b=(.85-mu)/sd
    phi=lambda z:torch.exp(-z*z/2)/math.sqrt(2*math.pi)
    Z=(torch.special.ndtr(b)-torch.special.ndtr(a)).clamp_min(1e-300)
    return mu+sd*(phi(a)-phi(b))/Z
def draw(J,h,ep,eh,n,seed):
    gen=torch.Generator(device=DEV).manual_seed(seed)
    C=torch.linalg.inv(J);mu=C@h;chol=torch.linalg.cholesky(C)
    out=[];count=0
    while count<n:
        x=mu+torch.randn((max(n,256),3),device=DEV,dtype=DT,generator=gen)@chol.T
        x=x[(x.abs()<=BOUND).all(1)];out.append(x);count+=len(x)
    g=torch.cat(out)[:n]
    sd=torch.rsqrt(ep);em=eh/ep
    lo=torch.special.ndtr((.45-em)/sd);hi=torch.special.ndtr((.85-em)/sd)
    u=lo+(hi-lo)*torch.rand(n,device=DEV,dtype=DT,generator=gen)
    e=em+sd*torch.special.ndtri(u)
    return g,e
def predictions(g,e):
    s=g[:,1:]@t(QB).T;k=(1+e[:,None])/(1+s*s)
    return torch.stack([1-k,k*s],-1)

class Quad:
    def __init__(self,order=64):
        nodes,weights=leggauss(order);nodes*=BOUND
        a,b=np.meshgrid(nodes,nodes,indexing='ij');wa,wb=np.meshgrid(weights,weights,indexing='ij')
        self.a=t(np.stack([a.ravel(),b.ravel()],1));self.lw=t(np.log((wa*wb).ravel()))
        s=self.a@t(QB).T
        self.f=torch.stack([1/(1+s*s),s/(1+s*s)],-1).reshape(-1,12)
    def predict(self,J,h,ep,eh,return_geometry=False):
        if h.ndim==1:h=h[None,:]
        if J.ndim==2:J=J[None,:,:].expand(len(h),-1,-1)
        ja=J[:,0,1:];j0=J[:,0,0]
        K=J[:,1:,1:]-ja[:,:,None]*ja[:,None,:]/j0[:,None,None]
        v=h[:,1:]-ja*h[:,0,None]/j0[:,None]
        # Integrate bounded x0 analytically, retaining covariance coupling.
        xm=(h[:,0,None]-ja@self.a.T)/j0[:,None];sd=torch.rsqrt(j0)[:,None]
        z=torch.special.ndtr((BOUND-xm)/sd)-torch.special.ndtr((-BOUND-xm)/sd)
        ll=-.5*torch.einsum('nd,bdk,nk->bn',self.a,K,self.a)+v@self.a.T+torch.log(z.clamp_min(1e-300))+self.lw
        w=torch.softmax(ll,1);f=(w@self.f).reshape(-1,6,2)
        em=trunc_mean(ep,eh).reshape(-1,1)
        result=torch.stack([1-(1+em)*f[:,:,0],(1+em)*f[:,:,1]],-1)
        if return_geometry:return result,w@self.a
        return result

def update_batch(J,h,ep,eh,action,observations,mode='boundary'):
    y=t(observations)
    if action=='probe':return J,h[None,:].expand(len(y),-1),ep+1/.015**2,eh+y.reshape(-1)/.015**2
    D=t(design(VY[ACTIONS.index(action)-1]));sd=V_SD
    if mode=='coarse':y=y.mean(1,keepdim=True);sd/=math.sqrt(len(D));D=D.mean(0,keepdim=True)
    return J+D.T@D/sd**2,h[None,:]+y@D/sd**2,ep,eh

def action_params(g,e,action,mode='boundary'):
    if action=='probe':return e[:,None],.015
    D=t(design(VY[ACTIONS.index(action)-1]));sd=V_SD
    if mode=='coarse':sd/=math.sqrt(len(D));D=D.mean(0,keepdim=True)
    return g@D.T,sd

def hypothetical(g,e,n,seed,mode='boundary'):
    gen=torch.Generator(device=DEV).manual_seed(seed)
    ids=torch.randint(len(g),(n,),device=DEV,generator=gen);out={}
    for a in ACTIONS[1:]:
        mu,sd=action_params(g[ids],e[ids],a,mode)
        out[a]=(mu+sd*torch.randn(mu.shape,dtype=DT,device=DEV,generator=gen)).cpu().numpy()
    return out

def utilities(g,e,outer,current,mode='boundary'):
    z=predictions(g,e).reshape(len(g),12);n=len(g)//2
    standard=[];cross=[];ess=[];eig=[]
    for a in ACTIONS[1:]:
        mu,sd=action_params(g,e,a,mode);y=t(outer[a])
        if a=='probe':y=y.reshape(-1,1)
        # squared distance expansion avoids H*N*pixel-count allocation.
        ll=(-.5*(y*y).sum(1)[:,None]+y@mu.T-.5*(mu*mu).sum(1)[None,:])/sd**2
        w=torch.softmax(ll,1);wa=torch.softmax(ll[:,:n],1);wb=torch.softmax(ll[:,n:],1)
        pooled=(w@z).reshape(-1,6,2)-current
        da=(wa@z[:n]).reshape(-1,6,2)-current;db=(wb@z[n:]).reshape(-1,6,2)-current
        standard.append(.5*(pooled*pooled).sum(2).mean(0));cross.append(.5*(da*db).sum(2).mean(0))
        ess.append(float(torch.median(1/(w*w).sum(1))))
        eig.append(float((w*(ll-torch.logsumexp(ll,1)[:,None]+math.log(len(g)))).sum(1).mean()))
    zero=torch.zeros(6,device=DEV,dtype=DT)
    return torch.stack([zero]+standard,1).cpu().numpy(),torch.stack([zero]+cross,1).cpu().numpy(),ess,eig
