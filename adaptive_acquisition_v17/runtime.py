from model import *
from baselines import prepare,native,nested
from acquisition import plan
from mixture import compress,risks,normal_plan,payload
from adaptive import select,phases,outer_samples
from run_policy import ADAPT,config
class Runtime:
    def __init__(self,name):
        self.name=name;self.bank=Bank(65536,2026111203) if name=='qmc64_native' else Bank();self.outer=None;self.comp=None;self.last_meta=None
        if name in ['native','qmc64_native']:prepare(self.bank)
        if name.startswith('nested') or name in ADAPT:self.outer=Bank(seed=2026111202)
        self.pre=plan(int(name[6:]) if name.startswith('nested') else 512) if self.outer else None
        self.phase=phases(81531) if name in ADAPT else None
        self.eps=normal_plan() if name in ['region32','global32'] else None
    def begin(self,x,ctx,task):
        b=self.bank;w=b.fit(x);z=targets(b.theta,ctx,task=task);self.comp=None;self.last_meta=None
        if self.name in ['query','probe']:a=1 if self.name=='query' else 3
        elif self.name in ['native','qmc64_native']:a=int(native(b,w,z,[1])[1:].argmax())+1
        elif self.name.startswith('nested'):
            ow=self.outer.fit(x);oz=targets(self.outer.theta,ctx,task=task);r,_=nested(b,w,z,self.outer,ow,oz,self.pre,[1]);a=int(r[1:].argmin())+1
        elif self.name in ADAPT:
            ow=self.outer.fit(x);oz=targets(self.outer.theta,ctx,task=task);sample=outer_samples(self.outer,ow,oz,self.pre)
            a,r,self.last_meta,h=select(b,w,z,sample,self.phase,**config(self.name))
        else:
            self.comp=compress(b,w,z,True,global_mode=self.name=='global32');r=risks(self.comp,self.eps);a=int(r[1:].argmin())+1
        return a,w,z
    def finish(self,state,y):
        a,w,z=state;return self.bank.update(w,a,y)@z
    def cache_bytes(self):
        v=self.bank.cache_bytes()+(self.outer.cache_bytes() if self.outer else 0)
        if hasattr(self.bank,'features'):v+=sum(x.numel()*x.element_size() for x in self.bank.features.values())
        for x in [self.eps,self.phase]:
            if x is not None:v+=x.numel()*x.element_size()
        if self.pre:v+=sum(x.numel()*x.element_size() for x in self.pre)
        return v
