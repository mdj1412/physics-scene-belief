from model import *
from baselines import prepare,native
from budget import plans,rep_plan,full_prefix
from run_policy import MIX,FIXED,NEST,make_engine,execute
class Runtime:
    def __init__(self,name):
        self.name=name;self.bank=Bank(65536,2026111203) if name=='qmc64_native' else Bank();self.outer=None;self.pp=None;self.rp=None;self.engine=None
        if name in ['native','qmc64_native']:prepare(self.bank)
        if name in MIX or name in NEST:
            self.outer=Bank(seed=2026111202);m=FIXED[name][1] if name in FIXED else (NEST[name] if name in NEST else (128 if name=='component_only' else 1024));self.pp=plans(mmax=m)
        if name in MIX:self.rp=rep_plan(Kmax=FIXED[name][0] if name in FIXED else 128)
    def begin(self,x,ctx,task):
        self.engine=None;b=self.bank;w=b.fit(x);z=targets(b.theta,ctx,task=task)
        if self.name in ['query','probe']:a=1 if self.name=='query' else 3
        elif self.name in ['native','qmc64_native']:a=int(native(b,w,z,[1])[1:].argmax())+1
        else:
            ow=self.outer.fit(x);oz=targets(self.outer.theta,ctx,task=task)
            if self.name in NEST:r=full_prefix(b,w,z,self.outer,ow,oz,self.pp,[NEST[self.name]])[0];a=int(r.argmin())+1
            else:self.engine=make_engine(self.name,b,w,z,self.outer,ow,oz,self.pp,self.rp);a,r,meta,h=execute(self.name,self.engine)
        return a,w,z
    def finish(self,state,y):
        a,w,z=state;return self.bank.update(w,a,y)@z
    def cache_bytes(self):
        v=self.bank.cache_bytes()+(self.outer.cache_bytes() if self.outer else 0)
        if hasattr(self.bank,'features'):v+=sum(x.numel()*x.element_size() for x in self.bank.features.values())
        if self.rp is not None:v+=self.rp.numel()*self.rp.element_size()
        if self.pp is not None:v+=sum(x.numel()*x.element_size() for pp in self.pp for x in pp)
        return v
