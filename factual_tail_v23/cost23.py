"""Supplementary online timings for the routing ablations, after the primary test."""
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
from run23 import R, PAIRS, TailGate, require, read, public, private, timing, torch,t,np,save
from model import FY

class CostEngine(TailGate):
    def __init__(self,*args):
        super().__init__(*args)
        self.slope_weights=t(FY)/t(FY).square().sum()
    def online(self,x,ctx,name):
        if name=="slope":
            a=1 if float(x[:9]@self.slope_weights)<0 else 3
            return a,self.inner.fit(x),self.inner.label
        return super().online(x,ctx,name)

if __name__=="__main__":
    require();torch.set_num_threads(2)
    cfg=read(R/"design.json");g=cfg["gate"]
    x=public("test");data=private("test");times=[]
    for b,pair in enumerate(PAIRS):
        eng=CostEngine(pair,0,cfg["cutoff"]);a=np.load(R/f"choices_{b}.npz")["acts"][:,0]
        simple=np.where((x[:,:9]@eng.slope_weights).cpu().numpy()<0,1,3)
        checks=np.column_stack([a[:,0],a[:,3],a[:,4],simple])
        names=[g,"top"+g[4:],"keep"+g[4:],"slope"]
        times.append(timing(eng,x,data,list(range(48)),0,names,3,checks))
    save(R/"cost_ablation.npz",times=np.stack(times),names=names)
