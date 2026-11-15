import torch
from physics import t,DT,DEV,noise
from flexible import measure,response

class Bank:
    def __init__(self,expanded=False,N=16384,seed=20261112):
        self.expanded=expanded;self.N=N;self.seed=seed
        components=[(0,0),(0,1),(1,0),(1,1)]+([(0,2),(1,2)] if expanded else [])
        theta=[]
        u=torch.quasirandom.SobolEngine(5 if expanded else 3,scramble=True,seed=seed).draw(N//4,dtype=DT).to(DEV)
        if not expanded:u=torch.cat([u,torch.full((len(u),2),.5,device=DEV,dtype=DT)],1)
        for g,c in components:
            count=N//(8 if c<2 else 4) if expanded else N//4
            pars=t([-.4,-.1,.35,-4,-2])+u[:count]*t([.8,.2,.55,6,4]);gc=t([g,c]).expand(count,2)
            theta.append(torch.cat([gc,pars],1))
        self.theta=torch.cat(theta);assert len(self.theta)==N
        self.z=response(self.theta);self.mu={a:measure(self.theta,a) for a in [0,1,2,3]}
        self.white={a:v/noise(a) for a,v in self.mu.items()};self.norm={a:.5*(v*v).sum(1) for a,v in self.white.items()}
        self.features={a:torch.cat([y,torch.einsum('ni,nj->nij',y,y).flatten(1),torch.einsum('ni,nj->nij',self.z,y).flatten(1)],1) for a,y in self.mu.items() if a}
    def ll(self,a,y):return (t(y)/noise(a))@self.white[a].T-self.norm[a]
    def fit(self,x):return torch.softmax(self.ll(0,x),1)
    def update(self,s,a,y):return torch.softmax(s.clamp_min(1e-300).log()+self.ll(a,y),1)
    def predict(self,s):return s@self.z
    def gains(self,s):
        zm=self.predict(s);g=[torch.zeros(len(s),device=DEV,dtype=DT)]
        for a in [1,2,3]:
            d=self.mu[a].shape[1];mom=s@self.features[a];ym=mom[:,:d]
            S=mom[:,d:d+d*d].reshape(-1,d,d)-ym[:,:,None]*ym[:,None,:]+torch.diag(noise(a)**2)
            C=mom[:,d+d*d:].reshape(-1,2,d)-zm[:,:,None]*ym[:,None,:]
            g.append(.5*torch.einsum('bij,bji->b',C,torch.linalg.solve(S,C.transpose(1,2))))
        return torch.stack(g,1)
    def choose(self,s):return self.gains(s).argmax(1)
    def state_bytes(self):return self.N*8
    def cache_bytes(self):return sum(x.numel()*x.element_size() for x in [self.theta,self.z]+list(self.mu.values())+list(self.white.values())+list(self.norm.values())+list(self.features.values()))
