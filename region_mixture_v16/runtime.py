import torch
from model import *
from baselines import prepare,native,nested
from acquisition import pair,plan
from mixture import compress,risks,normal_plan,predict,payload
class Runtime:
    def __init__(self,name):
        self.name=name;self.bank=Bank(65536,2026111203) if name=='qmc64_native' else Bank();self.outer=None;self.comp=None
        if name in ['native','qmc64_native']:prepare(self.bank)
        if name.startswith('nested') or name=='region_cf32':self.outer=Bank(seed=2026111202)
        self.nplan=plan(int(name[6:])) if name.startswith('nested') else None;self.eps=normal_plan() if name.startswith(('region','global')) else None
    def begin(self,x,ctx,task):
        b=self.bank;w=b.fit(x);z=targets(b.theta,ctx,task=task);self.comp=None
        if self.name in ['query','probe']:a=1 if self.name=='query' else 3
        elif self.name in ['native','qmc64_native']:a=int(native(b,w,z,[z.shape[1]])[1:].argmax())+1
        elif self.name=='pair':a=int(pair(b,w,z,x,[z.shape[1]])[0][1:].argmax())+1
        elif self.name.startswith('nested'):
            ow=self.outer.fit(x);oz=targets(self.outer.theta,ctx,task=task);r,_=nested(b,w,z,self.outer,ow,oz,self.nplan,[z.shape[1]]);a=int(r[1:].argmin())+1
        else:
            K=64 if self.name=='region64' else 32;self.comp=compress(b,w,z,task.endswith('event'),K,global_mode=self.name=='global32');outer=None
            if self.name=='region_cf32':
                ow=self.outer.fit(x);oz=targets(self.outer.theta,ctx,task=task);outer=compress(self.outer,ow,oz,task.endswith('event'))
            r=risks(self.comp,self.eps,outer=outer);a=int(r[1:].argmin())+1
        return a,w,z
    def finish(self,state,y):
        a,w,z=state
        return predict(self.comp,a,t(y)[None,:])[0] if self.name=='region32_compact' else self.bank.update(w,a,y)@z
    def cache_bytes(self):
        values=self.bank.cache_bytes()+(self.outer.cache_bytes() if self.outer else 0)
        if hasattr(self.bank,'features'):values+=sum(v.numel()*v.element_size() for v in self.bank.features.values())
        if self.eps is not None:values+=self.eps.numel()*8
        if self.nplan is not None:values+=sum(v.numel()*8 for v in self.nplan)
        return values
