from runtime import Runtime as Original
from model import *
from adaptive import outer_samples
from adaptive_fast import select
from run_policy import ADAPT,config
class Runtime(Original):
    def begin(self,x,ctx,task):
        if self.name not in ADAPT:return super().begin(x,ctx,task)
        b=self.bank;w=b.fit(x);z=targets(b.theta,ctx,task=task);self.comp=None
        ow=self.outer.fit(x);oz=targets(self.outer.theta,ctx,task=task);sample=outer_samples(self.outer,ow,oz,self.pre)
        a,r,self.last_meta,h=select(b,w,z,sample,self.phase,**config(self.name))
        return a,w,z
