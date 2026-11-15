"""Equivalent kernels: block logsumexp instead of dense logcumsumexp; direct fixed mixtures."""
from budget import Engine as Base,KS
from model import *
class Engine(Base):
    def raw(self,lo,hi,start,end):
        mu=self.mu[:,:,lo:hi];y=self.y[:,:,start:end]
        ll=torch.matmul(y[None],mu.transpose(-1,-2)[:,:,None])-self.norm[:,:,None,None,lo:hi]+self.base[:,None,None,None,lo:hi].log()
        self.work+=2*3*4*(end-start)*(hi-lo);self.updates+=1
        return ll,ll.masked_fill(~self.labels[:,None,None,None,lo:hi],float('-inf'))
    def extend(self,oldK,newK,start,end):
        lo=oldK*self.R;ll,pos=self.raw(lo,newK*self.R,start,end);acc=self.logs[oldK][...,start:end] if oldK else None;left=0
        for k in KS:
            if oldK<k<=newK:
                right=k*self.R-lo;v=torch.stack([torch.logsumexp(ll[...,left:right],-1),torch.logsumexp(pos[...,left:right],-1)])
                acc=v if acc is None else torch.logaddexp(acc,v);self.logs[k][...,start:end]=acc;left=right
    def ensure(self,K,m):
        assert K>=self.k and m>=self.m and K<=self.Kmax and m<=self.mmax
        if K>self.k and self.m:self.extend(self.k,K,0,self.m)
        if m>self.m:self.extend(0,K,self.m,m)
        self.k=K;self.m=m
    def fixed(self,K,m):
        self.logs={};ll,pos=self.raw(0,K*self.R,0,m)
        pr=2**.5*torch.exp(torch.logsumexp(pos,-1)-torch.logsumexp(ll,-1));loss=.5*(pr-self.truth[None,None,:,:m])**2;return loss.mean((0,2,3))
