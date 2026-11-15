"""CUDA numerical Bayes design using a shared simulator response model.

True parameter values and realized measurements are not used in acquisition.
Expected risks integrate the observation likelihood on a 1-D trapezoid grid.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import truncnorm


class Forward:
    def __init__(self,path=Path('data')):
        d=np.load(path/'physics_table.npz')
        self.velocity=RegularGridInterpolator((d['offsets'],d['es']),d['values'][...,:2])
        self.contact=RegularGridInterpolator((d['offsets'],d['es']),d['values'][...,2],method='nearest')
        s=np.load(path/'sensor_tables.npz')
        self.sg,self.sgm,self.se,self.sem=s['g'],s['gmu'],s['e'],s['emu']

    def outcomes(self,offset,e):
        shape=np.broadcast_shapes(np.shape(offset),np.shape(e))
        offset=np.broadcast_to(offset,shape);e=np.broadcast_to(e,shape)
        points=np.c_[abs(offset).ravel(),e.ravel()]
        v=self.velocity(points).reshape(shape+(2,))
        v[...,1]*=np.sign(offset)
        return v,self.contact(points).reshape(shape)

    def observation(self,g,e):
        return np.stack([np.interp(g,self.sg,self.sgm),np.interp(e,self.se,self.sem),np.full_like(g,64.)],axis=-1)


def model_atoms(cfg,kind='dense'):
    mu,sd=cfg['geometry_mean'],cfg['geometry_sd']
    em,es=cfg['restitution_mean'],cfg['restitution_sd']
    if kind=='particles512':
        rng=np.random.default_rng([20261104,cfg['family'],777])
        g=truncnorm.rvs(-4,4,loc=mu,scale=sd,size=512,random_state=rng)
        e=truncnorm.rvs((.1-em)/es,(.9-em)/es,loc=em,scale=es,size=512,random_state=rng)
        return g,e,np.full(512,1/512),None
    if kind=='quadrature16':
        x,w=np.polynomial.legendre.leggauss(16)
        u=(x+1)/2;w=w/2
        g=truncnorm.ppf(u,-4,4,loc=mu,scale=sd)
        e=truncnorm.ppf(u,(.1-em)/es,(.9-em)/es,loc=em,scale=es)
        pg=w;pe=w
    else:
        ng,ne={'dense':(129,33),'grid33':(33,17),'fine':(257,65)}[kind]
        g=np.linspace(mu-4*sd,mu+4*sd,ng);e=np.linspace(.1,.9,ne)
        pg=np.exp(-.5*((g-mu)/sd)**2);pe=np.exp(-.5*((e-em)/es)**2)
        pg[[0,-1]]*=.5;pe[[0,-1]]*=.5
        pg/=pg.sum();pe/=pe.sum()
    gg,ee=np.meshgrid(g,e,indexing='ij')
    return gg.ravel(),ee.ravel(),np.outer(pg,pe).ravel(),(len(g),len(e))


class BeliefModel:
    def __init__(self,cfg,forward,kind='dense',device='cuda',integration=257):
        self.cfg=cfg;self.device=device;self.kind=kind
        g,e,prior,self.shape=model_atoms(cfg,kind)
        self.g=g;self.e=e
        self.to=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
        self.prior=self.to(prior)
        self.mu=self.to(forward.observation(g,e))
        self.sd=self.to([cfg['sigma_view_px'],2**.5*cfg['sigma_probe_px'],cfg['sigma_view_px']])
        delta=np.array(cfg['delta'])[:,None]
        v,c=forward.outcomes(g[None,:]+delta,e[None,:])
        self.z=self.to(v[...,0]);self.vy=self.to(v[...,1]);self.contact=self.to(c)
        self.order=torch.argsort(self.z,dim=1)
        self.sorted_z=torch.gather(self.z,1,self.order)
        self.likelihood=[];self.integration_weights=[]
        for a in range(3):
            obs=np.linspace(float(self.mu[:,a].min().cpu())-7*float(self.sd[a].cpu()),
                            float(self.mu[:,a].max().cpu())+7*float(self.sd[a].cpu()),integration)
            weights=np.full(integration,obs[1]-obs[0]);weights[[0,-1]]*=.5
            self.likelihood.append(torch.exp(-.5*((self.to(obs)[:,None]-self.mu[None,:,a])/self.sd[a])**2)
                                   /(self.sd[a]*(2*np.pi)**.5))
            self.integration_weights.append(self.to(weights))

    def collapse(self,p):
        ng,ne=self.shape
        pg=p.reshape(-1,ng,ne).sum(-1)
        idx=pg.argmax(-1)
        mask=torch.arange(ng,device=self.device)[None,:]==idx[:,None]
        out=p.reshape(-1,ng,ne)*mask[:,:,None]
        return (out/out.sum((1,2),keepdim=True)).reshape_as(p)

    def gains(self,p,qindex,criterion='evsi'):
        z=self.z[qindex]
        mean=(p*z).sum(-1)
        gains=[]
        for a,(L,w) in enumerate(zip(self.likelihood,self.integration_weights)):
            density=p@L.T
            if criterion=='evsi':
                numerator=(p*z)@L.T
                second=(numerator.square()/density.clamp_min(1e-300)*w).sum(-1)
                gain=second-mean.square()
            else:
                gain=-(density*density.clamp_min(1e-300).log()*w).sum(-1)-.5*torch.log(2*torch.pi*torch.e*self.sd[a]**2)
            gains.append(gain.clamp_min(0))
        return torch.stack(gains,-1)

    def variance_rule(self,p,qindex):
        ng,ne=self.shape
        pp=p.reshape(-1,ng,ne);z=self.z[qindex].reshape(-1,ng,ne)
        pg=pp.sum(-1)
        zm=(pp*z).sum(-1)/pg.clamp_min(1e-300)
        mean=(pp*z).sum((1,2));second=(pp*z*z).sum((1,2))
        geometry_var=(pg*zm.square()).sum(-1)-mean.square()
        material_var=second-(pg*zm.square()).sum(-1)
        action=torch.where(geometry_var>=material_var,0,1)
        return torch.where(second-mean.square()>.005,action,-1)

    def update(self,p,action,observation):
        logp=p.clamp_min(1e-300).log()
        for a in range(3):
            ii=torch.where(action==a)[0]
            if len(ii):
                logp[ii]+=-.5*((observation[ii,None]-self.mu[None,:,a])/self.sd[a])**2
        return torch.softmax(logp,-1)

    def metrics(self,p,qindex,true_v,true_contact):
        z=self.z[qindex];vy=self.vy[qindex];c=self.contact[qindex]
        mean=(p*z).sum(-1);cp=(p*c).sum(-1)
        sp=torch.gather(p,1,self.order[qindex]);sz=self.sorted_z[qindex]
        cdf=sp.cumsum(-1)
        first=(p*(z-true_v[:,0,None]).abs()).sum(-1)
        half_pair=(sp*sz*(2*cdf-sp-1)).sum(-1)
        lo=sz.gather(1,(cdf>=.05).to(torch.int64).argmax(-1)[:,None])[:,0]
        hi=sz.gather(1,(cdf>=.95).to(torch.int64).argmax(-1)[:,None])[:,0]
        return dict(mse=(mean-true_v[:,0]).square(),vy_mse=((p*vy).sum(-1)-true_v[:,1]).square(),
            brier=(cp-true_contact).square(),crps=first-half_pair,
            posterior_risk=((p*z.square()).sum(-1)-mean.square()).clamp_min(0),
            coverage90=((true_v[:,0]>=lo-1e-10)&(true_v[:,0]<=hi+1e-10)).double(),
            width90=hi-lo,ess=1/p.square().sum(-1),prediction=mean,contact_probability=cp)
