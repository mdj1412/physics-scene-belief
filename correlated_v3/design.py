import numpy as np
import torch
from core import impact

class Design:
    def __init__(self,atoms,deltas,sd,points=257):
        g,e,w=atoms
        self.to=lambda x:torch.as_tensor(x,dtype=torch.float64,device='cuda')
        self.p=self.to(w);self.mu=self.to(np.stack([g,e],1));self.sd=self.to(sd)
        z,vy,c=impact(g[None,:]+np.array(deltas)[:,None],e[None,:])
        self.z=self.to(z);self.c=self.to(c);self.Q=len(deltas)
        self.L=[];self.iw=[]
        for a in range(2):
            grid=np.linspace(float(self.mu[:,a].min())-8*sd[a],float(self.mu[:,a].max())+8*sd[a],points)
            ww=np.full(points,grid[1]-grid[0]);ww[[0,-1]]*=.5
            L=torch.exp(-.5*((self.to(grid)[:,None]-self.mu[:,a])/sd[a])**2)/(sd[a]*np.sqrt(2*np.pi))
            self.L.append(L);self.iw.append(self.to(ww))

    def gains(self):
        p=self.p;mean=self.z@p;out=[]
        for L,w in zip(self.L,self.iw):
            density=L@p;numerator=(self.z*p)@L.T
            out.append(((numerator.square()/density.clamp_min(1e-300)*w).sum(1)-mean.square()).clamp_min(0))
        gains=torch.stack(out,1)
        scores=torch.cat([gains-.0005,torch.zeros((self.Q,1),device='cuda')],1)
        return gains.cpu().numpy(),scores.argmax(1).cpu().numpy()

    def evaluate(self,observed,truth,contact):
        # Return all choices (view/probe/stop) for matched-information analyses.
        post=[torch.softmax(self.p.clamp_min(1e-300).log()-.5*((observed[a]-self.mu[:,a])/self.sd[a])**2,0) for a in range(2)]+[self.p]
        out=[]
        order=torch.argsort(self.z,1);sz=self.z.gather(1,order)
        for p in post:
            mean=self.z@p;cp=self.c@p;var=self.z.square()@p-mean.square()
            sp=p[order];cdf=sp.cumsum(1)
            lo=sz.gather(1,(cdf>=.05).long().argmax(1)[:,None])[:,0]
            hi=sz.gather(1,(cdf>=.95).long().argmax(1)[:,None])[:,0]
            crps=(p*(self.z-self.to(truth)[:,None]).abs()).sum(1)-(sp*sz*(2*cdf-sp-1)).sum(1)
            out.append(dict(prediction=mean.cpu().numpy(),mse=(mean.cpu().numpy()-truth)**2,
                brier=(cp.cpu().numpy()-contact)**2,crps=crps.cpu().numpy(),risk=var.cpu().numpy(),
                coverage90=((self.to(truth)>=lo)&(self.to(truth)<=hi)).double().cpu().numpy(),
                width90=(hi-lo).cpu().numpy()))
        return out
