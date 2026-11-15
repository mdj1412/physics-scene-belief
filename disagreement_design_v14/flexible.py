"""Candidate model expansion. No held-out-law generators or reference imports."""
import numpy as np
import torch
from physics import t,DT,DEV,profile,basis,L,FY,VYS,noise
from compact import State,Compact
from numpy.polynomial.hermite import hermgauss

def velocity(th,y,vin):
    g,c,s,a,e,b0,b1=th.unbind(-1);slope=s+a*basis(t(y),g,True);norm=torch.sqrt(1+slope*slope)
    vn=(vin[0]-slope*vin[1])/norm;vt=(vin[0]*slope+vin[1])/norm;jn=(1+e)*vn
    jt=torch.sign(vt)*torch.minimum(vt.abs(),.22*(c==1)*jn)
    frac=torch.sigmoid(b0+b1*torch.log((vt.abs()+.02)/.5))
    jt=torch.where(c==2,vt*frac,jt)
    return torch.stack([vin[0]-jn/norm-jt*slope/norm,vin[1]+jn*slope/norm-jt/norm],-1)
def response(th):return velocity(th,L,(1,.9))
def measure(th,a):
    if a==0:return torch.cat([profile(th[...,:5],FY),velocity(th,0,(1,.35))],-1)
    if a in [1,2]:return profile(th[...,:5],VYS[a-1])
    if a==4:return profile(th[...,:5],FY)
    return velocity(th,0,(1,.35) if a==5 else (1,-.7))
def transform(u):
    p=torch.special.ndtr(u);lo=t([-.4,-.1,.35,-4,-2]);width=t([.8,.2,.55,6,4]);pars=lo+width*p
    gc=t([[0,0],[0,1],[1,0],[1,1],[0,2],[1,2]])
    return torch.cat([gc.reshape(1,6,1,2).expand(*u.shape[:-1],2),pars],-1)

class Expanded(Compact):
    def __init__(self):
        self.order=3;self.steps=4;self.diagonal=False
        x,w=hermgauss(3);self.nodes=t(np.stack(np.meshgrid(*([x]*5),indexing='ij'),-1).reshape(-1,5)*np.sqrt(2))
        self.weights=t(np.prod(np.stack(np.meshgrid(*([w]*5),indexing='ij'),-1),-1).ravel()/np.pi**2.5)
    def prior(self,B):
        return State(torch.zeros((B,6,5),device=DEV,dtype=DT),torch.eye(5,device=DEV,dtype=DT).expand(B,6,5,5).clone(),t([.125]*4+[.25]*2).log().expand(B,-1).clone())
    def points(self,state):
        u=state.mean[:,:,None,:]+torch.einsum('bkij,qj->bkqi',torch.linalg.cholesky(state.cov),self.nodes)
        return u,transform(u)
    def update(self,state,action,observed):
        y=t(observed)/noise(action);I=torch.eye(y.shape[-1],device=DEV,dtype=DT)
        for _ in range(self.steps):
            u,th=self.points(state);h=measure(th,action)/noise(action)
            hm=torch.einsum('q,bkqd->bkd',self.weights,h);dh=h-hm[:,:,None,:];du=u-state.mean[:,:,None,:]
            S=torch.einsum('q,bkqi,bkqj->bkij',self.weights,dh,dh)+self.steps*I
            C=torch.einsum('q,bkqi,bkqj->bkij',self.weights,du,dh);ch=torch.linalg.cholesky(S)
            K=torch.cholesky_solve(C.transpose(-1,-2),ch).transpose(-1,-2);res=y[:,None,:]-hm
            mu=state.mean+torch.einsum('bkij,bkj->bki',K,res);cov=state.cov-K@C.transpose(-1,-2);cov=(cov+cov.transpose(-1,-2))*.5
            ev,U=torch.linalg.eigh(cov);cov=(U*ev.clamp_min(1e-8)[:,:,None,:])@U.transpose(-1,-2)
            loglik=-.5*((res*torch.cholesky_solve(res[:,:,:,None],ch).squeeze(-1)).sum(-1)+2*torch.diagonal(ch,dim1=-2,dim2=-1).log().sum(-1))
            lp=state.logp+loglik;lp-=torch.logsumexp(lp,1,keepdim=True);state=State(mu,cov,lp)
        return state
    def predict(self,state):
        _,th=self.points(state);return torch.einsum('bk,q,bkqd->bd',state.logp.exp(),self.weights,response(th))
    def gains(self,state):
        _,th=self.points(state);z=response(th).flatten(1,2);w=(state.logp.exp()[:,:,None]*self.weights).flatten(1,2)
        zm=(w[:,:,None]*z).sum(1);dz=z-zm[:,None,:];gains=[torch.zeros(len(z),device=DEV,dtype=DT)]
        for a in [1,2,3]:
            y=measure(th,a).flatten(1,2);dy=y-(w[:,:,None]*y).sum(1)[:,None,:]
            S=torch.einsum('bq,bqi,bqj->bij',w,dy,dy)+torch.diag(noise(a)**2);C=torch.einsum('bq,bqi,bqj->bij',w,dz,dy)
            gains.append(.5*torch.einsum('bij,bji->b',C,torch.linalg.solve(S,C.transpose(1,2))))
        return torch.stack(gains,1)
    @staticmethod
    def state_bytes():return 6*(1+5+25)*8
