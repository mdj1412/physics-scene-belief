from runtime import Runtime as Base
from model import *
from budget import select
from budget_fast import Engine
from run_policy import MIX,FIXED
def engine_for(name,b,w,z,o,ow,oz,pp,rp):
    K,m,g=FIXED[name] if name in FIXED else (128,128 if name=='component_only' else 1024,False)
    return Engine(b,w,z,o,ow,oz,pp,rp,K,m,g)
def execute_fast(name,e):
    if name in FIXED:
        r=e.fixed(*FIXED[name][:2]);a=1 if float(r.max()-r.min())<=1e-12 else int(r.argmin())+1
        return a,r,None,None
    return select(e,mode=name)
class Runtime(Base):
    def begin(self,x,ctx,task):
        if self.name not in MIX:return super().begin(x,ctx,task)
        b=self.bank;w=b.fit(x);z=targets(b.theta,ctx,task=task);ow=self.outer.fit(x);oz=targets(self.outer.theta,ctx,task=task)
        self.engine=engine_for(self.name,b,w,z,self.outer,ow,oz,self.pp,self.rp);a,r,meta,h=execute_fast(self.name,self.engine);return a,w,z
