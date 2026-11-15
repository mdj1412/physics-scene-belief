from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"external_label_v21"))
from v21_policy import StudyEngine,TASK,t,DT,DEV,torch,padded,action

PAIRS=[(22310001+10009*i,22440001+12007*i) for i in range(5)]
PILOT=(23610001,23770001)
KGRID=tuple(range(256,5377,256))

class RecoveryEngine(StudyEngine):
    def __init__(self,pair):
        super().__init__(pair)
        self.strata={}
        for k in KGRID:
            n=k//8
            rand=torch.rand(2,n,dtype=DT,generator=torch.Generator().manual_seed(798311+k))
            self.strata[k]=((torch.arange(n,dtype=DT)+rand)/n).to(DEV)

    def support(self,ow,oz,k):
        slots=k//8
        head_w,head=torch.topk(ow,k-2*slots,sorted=False)
        residual=ow.clone().scatter_(0,head,0.)
        label=(oz[:,0]>0).to(DT)
        by_label=torch.stack([residual*(1-label),residual*label])
        cdf=by_label.cumsum(-1)
        total=cdf[:,-1:]
        draw=total*self.strata[k]
        ix=torch.searchsorted(cdf.contiguous(),draw.contiguous(),right=True).clamp_max(len(ow)-1)
        weight=(total/slots).expand(-1,slots).reshape(-1)
        return torch.cat([head,ix.reshape(-1)]),torch.cat([head_w,weight])

    def risk(self,state,name):
        if not name.startswith("keep"):
            return super().risk(state,name)
        k,m=map(int,name[4:].split("m"))
        w,z,ow,oz=state
        idx,weight=self.support(ow,oz,k)
        compact_ix=torch.searchsorted(w.cumsum(0),self.reps).clamp_max(len(w)-1)
        labels=(z[compact_ix,0]>0).to(DT)
        compact=torch.stack([padded(self.inner.white[a][compact_ix]) for a in (1,2,3)],1)
        oix=torch.searchsorted(ow.cumsum(0),self.u[:,:m//4].contiguous()).clamp_max(len(ow)-1)
        y=torch.stack([padded(self.outer.white[a][oix]) for a in (1,2,3)])+self.eps[None,:,:m//4]
        q_logits=y[None]@compact[:,:,None].transpose(-1,-2)-.5*compact.square().sum(-1)[:,:,None,None]
        q=(q_logits.softmax(-1)@labels[:,None,None,:,None]).squeeze(-1)
        mu=torch.stack([padded(self.outer.white[a][idx]) for a in (1,2,3)])
        log_prior=weight.log()[None]-.5*mu.square().sum(-1)
        p=((y@mu[:,None].transpose(-1,-2)+log_prior[:,None,None]).softmax(-1)@(oz[idx,0]>0).to(DT)).clamp(0,1)
        return (p*(1-q).square()+(1-p)*q.square()).mean((0,2,3))
