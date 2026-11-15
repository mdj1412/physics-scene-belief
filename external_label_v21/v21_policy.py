from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"factual_switch_v20"))
from policy import Factorial, Selector, CELLS, TASK, action, torch, DT, DEV, padded
from model import t

PAIRS=[(3301001+i*10009,4401007+i*12007) for i in range(5)]
PILOT=(5501009,6601019)
CANDIDATES=[f"top{k}m{m}" for k in (128,512,2048) for m in (512,2048)]


class StudyEngine(Factorial):
    def __init__(self,pair):
        Selector.__init__(self,651701,*pair)
        self.uniforms=torch.stack([torch.rand(128,generator=torch.Generator().manual_seed(673303+j),
                                               dtype=DT).to(DEV) for j in range(4)])
        self.u512=self.u[:,:128]
        self.noise512=self.eps[:,:128]

    def compressed(self,state,k,m):
        w,z,ow,oz=state
        compact_ix=torch.searchsorted(w.cumsum(0),self.reps).clamp_max(len(w)-1)
        compact_labels=(z[compact_ix,0]>0).to(DT)
        compact=torch.stack([padded(self.inner.white[a][compact_ix]) for a in (1,2,3)],1)
        outer_ix=torch.searchsorted(ow.cumsum(0),self.u[:,:m//4].contiguous()).clamp_max(len(ow)-1)
        y=torch.stack([padded(self.outer.white[a][outer_ix]) for a in (1,2,3)])+self.eps[None,:,:m//4]
        logq=torch.matmul(y[None],compact[:,:,None].transpose(-1,-2))-.5*compact.square().sum(-1)[:,:,None,None]
        q=(logq.softmax(-1)@compact_labels[:,None,None,:,None]).squeeze(-1)
        weight,idx=torch.topk(ow,k,sorted=False)
        centers=torch.stack([padded(self.outer.white[a][idx]) for a in (1,2,3)])
        log_prior=weight.clamp_min(1e-300).log()[None]-.5*centers.square().sum(-1)
        logits=torch.matmul(y,centers[:,None].transpose(-1,-2))+log_prior[:,None,None]
        p=(logits.softmax(-1)@(oz[idx,0]>0).to(DT)).clamp(0,1)
        risk=(p*(1-q).square()+(1-p)*q.square()).mean((0,2,3))
        return risk,1-weight.sum()

    def risk(self,state,name):
        if name=="fixed":return self.baseline(state,"v18_global")
        if name in CELLS:return self.risks(state,[name])[name]
        k,m=map(int,name[3:].split("m"))
        return self.compressed(state,k,m)[0]

    def online(self,x,ctx,name):
        state=self.fit(x,ctx,TASK,name!="probe" and not name.startswith("C-C"))
        return (3 if name=="probe" else action(self.risk(state,name))),state[0],state[1]
