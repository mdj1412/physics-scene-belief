"""Exploratory held-out recovery check at the chosen K. Does not refit the gate."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
from run23 import PAIRS, R, read, public, private, TailGate, torch, np, require, sha, save
from policy23 import DT

def audit():
    require()
    cfg=read(R/"design.json"); k=int(cfg["gate"][4:].split("m")[0])
    data=private("test");x=public("test")
    for b,pair in enumerate(PAIRS):
        freeze_record=read(R/"choices_freeze.json")["files"]
        file=R/f"choices_{b}.npz"
        assert sha(file)==freeze_record[str(file)]
        choices=np.load(file)
        results=np.empty((len(x),3,6))
        eng=TailGate(pair,0,cfg["cutoff"])
        for i,v in enumerate(x):
            state=eng.fit(v);_,_,w,z=state
            idx=w.argsort(descending=True);head=idx[:k];tail=idx[k:]
            ix,weight=eng.support(w,z,k)
            label=(z[:,0]>0).to(DT)
            prior=w.clamp_min(1e-300).log()
            for a in (1,2,3):
                yy=data[f"obs{a}"][i,0]
                ll=eng.outer.ll(a,yy)
                logits=ll+prior
                post=logits.softmax(-1)
                m=post[:,tail].sum(-1)
                base=w[tail].sum()
                p=post@label
                ph=logits[:,head].softmax(-1)@label[head]
                pk=(ll[:,ix]+weight.log()).softmax(-1)@label[ix]
                recovered=((base<=.05)&(m>=.1)&(m>=5*base)).to(DT).mean()
                values=torch.stack([base,m.mean(),m.max(),(p-ph).abs().mean(),(p-pk).abs().mean(),recovered])
                results[i,a-1]=values.cpu().numpy()
        assert np.isfinite(results).all(); save(R/f"audit_{b}.npz",detail=results)
        print("audit",b+1,flush=True)
if __name__=="__main__":audit()
