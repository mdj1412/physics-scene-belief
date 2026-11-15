"""Independent strong baselines. Never imported by compact.py."""
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from physics import t,DT,DEV,measure,noise,target

class Bank:
    def __init__(self,kind='qmc',size=16384,moments=True):
        self.kind=kind;self.size=size
        if kind=='quad':
            x,w=leggauss(size);coords=np.stack(np.meshgrid(.4*x,.1*x,.625+.275*x,indexing='ij'),-1).reshape(-1,3)
            ww=np.prod(np.stack(np.meshgrid(w/2,w/2,w/2,indexing='ij'),-1),-1).ravel()/4
            self.prior=t(np.tile(ww,4));pars=t(coords).repeat(4,1)
        else:
            assert size%4==0
            u=torch.quasirandom.SobolEngine(3,scramble=True,seed=20261112).draw(size//4,dtype=DT).to(DEV)
            coords=torch.stack([.8*u[:,0]-.4,.2*u[:,1]-.1,.35+.55*u[:,2]],1);pars=coords.repeat(4,1)
            self.prior=torch.full((size,),1/size,device=DEV,dtype=DT)
        n=len(pars)//4;gc=t([[0,0],[0,1],[1,0],[1,1]]).repeat_interleave(n,0)
        self.theta=torch.cat([gc,pars],1);self.N=len(pars);self.lp=self.prior.log()
        self.mu={a:measure(self.theta,a) for a in [0,1,2,3]};self.z=target(self.theta)
        self.white={a:self.mu[a]/noise(a) for a in [0,1,2,3]}
        self.norm={a:.5*(self.white[a]**2).sum(1) for a in [0,1,2,3]}
        self.features={}
        if moments:
            for a in [1,2,3]:
                y=self.mu[a];self.features[a]=torch.cat([y,torch.einsum('ni,nj->nij',y,y).flatten(1),torch.einsum('ni,nj->nij',self.z,y).flatten(1)],1)
    def ll(self,action,observed):
        y=t(observed)/noise(action)
        return y@self.white[action].T-self.norm[action]
    def fit(self,public):return torch.softmax(self.ll(0,public)+self.lp,1)
    def update(self,state,action,observed):return torch.softmax(self.ll(action,observed)+state.clamp_min(1e-300).log(),1)
    def predict(self,state):return state@self.z
    def mixture(self,state):
        q=state.reshape(len(state),4,-1);pars=self.theta[:,2:].reshape(4,-1,3)
        prob=q.sum(2);means=torch.einsum('bkn,knd->bkd',q,pars)/prob[:,:,None].clamp_min(1e-300)
        gc=t([[0,0],[0,1],[1,0],[1,1]]).expand(len(state),-1,-1)
        point=target(torch.cat([gc,means],2));return (prob[:,:,None]*point).sum(1)
    def gains(self,state):
        zm=self.predict(state);gains=[torch.zeros(len(state),device=DEV,dtype=DT)]
        for a in [1,2,3]:
            d=self.mu[a].shape[1];mom=state@self.features[a];ym=mom[:,:d]
            yy=mom[:,d:d+d*d].reshape(-1,d,d);zy=mom[:,d+d*d:].reshape(-1,2,d)
            cyy=yy-ym[:,:,None]*ym[:,None,:]+torch.diag(noise(a)**2)
            czy=zy-zm[:,:,None]*ym[:,None,:]
            gains.append(.5*torch.einsum('bij,bji->b',czy,torch.linalg.solve(cyy,czy.transpose(1,2))))
        return torch.stack(gains,1)
    def choose(self,state):return self.gains(state).argmax(1)
    def cache_bytes(self):
        tensors=[self.theta,self.prior,self.lp,self.z]+list(self.mu.values())+list(self.white.values())+list(self.norm.values())+list(self.features.values())
        return sum(x.numel()*x.element_size() for x in tensors)
    def state_bytes(self):return self.N*8
