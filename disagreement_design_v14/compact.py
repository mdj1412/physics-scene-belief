"""Standalone Gaussian-sum assumed-density filter. No full posterior inputs."""
from dataclasses import dataclass
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from physics import t,DT,DEV,latent_to_theta,measure,noise,target

@dataclass
class State:
    mean:torch.Tensor
    cov:torch.Tensor
    logp:torch.Tensor
    def repeat(self,n):return State(self.mean.repeat_interleave(n,0),self.cov.repeat_interleave(n,0),self.logp.repeat_interleave(n,0))
    def slice(self,i,j):return State(self.mean[i:j],self.cov[i:j],self.logp[i:j])

class Compact:
    def __init__(self,order=3,steps=1,diagonal=False):
        self.order=order;self.steps=steps;self.diagonal=diagonal
        x,w=hermgauss(order);grid=np.stack(np.meshgrid(x,x,x,indexing='ij'),-1).reshape(-1,3)*np.sqrt(2)
        weight=np.prod(np.stack(np.meshgrid(w,w,w,indexing='ij'),-1),-1).ravel()/np.pi**1.5
        self.nodes=t(grid);self.weights=t(weight)
    def prior(self,B):
        return State(torch.zeros((B,4,3),device=DEV,dtype=DT),torch.eye(3,device=DEV,dtype=DT).expand(B,4,3,3).clone(),torch.full((B,4),-np.log(4),device=DEV,dtype=DT))
    def points(self,state):
        chol=torch.linalg.cholesky(state.cov)
        u=state.mean[:,:,None,:]+torch.einsum('bkij,qj->bkqi',chol,self.nodes)
        return u,latent_to_theta(u)
    def update(self,state,action,observed):
        y=t(observed)/noise(action);dim=y.shape[-1]
        I=torch.eye(dim,device=DEV,dtype=DT)
        for _ in range(self.steps):
            u,theta=self.points(state);h=measure(theta,action)/noise(action)
            hm=torch.einsum('q,bkqd->bkd',self.weights,h);dh=h-hm[:,:,None,:];du=u-state.mean[:,:,None,:]
            cyy=torch.einsum('q,bkqi,bkqj->bkij',self.weights,dh,dh)+self.steps*I
            cuy=torch.einsum('q,bkqi,bkqj->bkij',self.weights,du,dh)
            chol=torch.linalg.cholesky(cyy);gain=torch.cholesky_solve(cuy.transpose(-1,-2),chol).transpose(-1,-2)
            res=y[:,None,:]-hm
            mean=state.mean+torch.einsum('bkij,bkj->bki',gain,res)
            cov=state.cov-gain@cuy.transpose(-1,-2);cov=(cov+cov.transpose(-1,-2))*.5
            if self.diagonal:cov=torch.diag_embed(torch.diagonal(cov,dim1=-2,dim2=-1).clamp_min(1e-8))
            else:
                ev,U=torch.linalg.eigh(cov);cov=(U*ev.clamp_min(1e-8)[:,:,None,:])@U.transpose(-1,-2)
            solved=torch.cholesky_solve(res[:,:,:,None],chol).squeeze(-1)
            loglik=-.5*((res*solved).sum(-1)+2*torch.diagonal(chol,dim1=-2,dim2=-1).log().sum(-1))
            logp=state.logp+loglik;logp-=torch.logsumexp(logp,1,keepdim=True)
            state=State(mean,cov,logp)
        return state
    def fit(self,public):return self.update(self.prior(len(public)),0,public)
    def predict(self,state):
        _,th=self.points(state);z=target(th);prob=state.logp.exp()
        component=torch.einsum('q,bkqd->bkd',self.weights,z)
        mean=(prob[:,:,None]*component).sum(1)
        thmean=torch.einsum('q,bkqd->bkd',self.weights,th)
        point=target(thmean);mixture=(prob[:,:,None]*point).sum(1)
        hard=point[torch.arange(len(point),device=DEV),prob.argmax(1)]
        return dict(predictive=mean,mixture=mixture,hard=hard)
    def gains(self,state):
        _,th=self.points(state);z=target(th).flatten(1,2);w=(state.logp.exp()[:,:,None]*self.weights).flatten(1,2)
        zm=(w[:,:,None]*z).sum(1);dz=z-zm[:,None,:];gains=[torch.zeros(len(z),device=DEV,dtype=DT)]
        for a in [1,2,3]:
            y=measure(th,a).flatten(1,2);ym=(w[:,:,None]*y).sum(1);dy=y-ym[:,None,:]
            cyy=torch.einsum('bq,bqi,bqj->bij',w,dy,dy)+torch.diag(noise(a)**2)
            czy=torch.einsum('bq,bqi,bqj->bij',w,dz,dy)
            gain=.5*torch.einsum('bij,bji->b',czy,torch.linalg.solve(cyy,czy.transpose(1,2)))
            gains.append(gain)
        return torch.stack(gains,1)
    def choose(self,state):return self.gains(state).argmax(1)
    def cache_bytes(self):return sum(x.numel()*x.element_size() for x in [self.nodes,self.weights])
    @staticmethod
    def state_bytes():return 52*8
