from physics23 import PhysicalBank, RecoveryEngine, torch, t, DT
from policy22 import TASK,action
from model import DEV


class TailGate(RecoveryEngine):
    """Reuse the v22 acquisition kernels on the v23 physical model and sensor calibration."""
    def __init__(self,pair,setting,cutoff=0):
        # Plan and sampler setup is offline for all methods.
        super().__init__(pair)
        self.inner=PhysicalBank(pair[0],setting)
        self.outer=PhysicalBank(pair[1],setting)
        self.cutoff=cutoff
        for k in (6144,7168):
            n=k//8
            jitter=torch.rand(2,n,dtype=DT,generator=torch.Generator().manual_seed(798311+k))
            self.strata[k]=((torch.arange(n,dtype=DT)+jitter)/n).to(DEV)

    def fit(self,x,ctx=None,task=None,need_outer=True):
        return self.inner.fit(x),self.inner.label,self.outer.fit(x),self.outer.label

    def score(self,state):
        w,z=state[2:]
        labels=(z[:,0]>0).to(DT)
        # Estimate pressure on compact support from factual evidence at a fixed diagnostic resolution.
        head,ix=w.topk(512,sorted=False)
        head_mass=head.sum()
        p=w@labels
        pos=head@labels[ix]
        excluded=(1-head_mass).clamp_min(0)
        # Removed mass times uncertainty + discrepancy of head prediction from uncompressed factual p.
        return 4*p*(1-p)*excluded+(p-pos/head_mass).abs()

    def risk(self,state,name):
        if name.startswith("gate"):
            family="keep" if float(self.score(state))>=self.cutoff else "top"
            return super().risk(state,family+name[4:])
        return super().risk(state,name)
