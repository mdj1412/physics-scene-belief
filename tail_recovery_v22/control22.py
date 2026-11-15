"""Exploratory same K, M, head and tail slots with no label stratification."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
from policy22 import RecoveryEngine,PAIRS,TASK,torch,DT,action
from run22 import public,R,require,save,write,freeze
import numpy as np

class Unstratified(RecoveryEngine):
    def support(self,w,z,k):
        n=k//4
        head_w,head=torch.topk(w,k-n,sorted=False)
        tail=w.clone().scatter_(0,head,0.)
        cdf=tail.cumsum(0);mass=cdf[-1]
        # Flatten independent jitters from the two pre-existing strata plans.
        jitter=self.strata[k]*(n//2)-torch.arange(n//2,device=w.device,dtype=DT)
        u=(torch.arange(n,device=w.device,dtype=DT)+jitter.flatten())/n
        idx=torch.searchsorted(cdf,(mass*u).contiguous(),right=True).clamp_max(len(w)-1)
        return torch.cat((head,idx)),torch.cat((head_w,(mass/n).expand(n)))

def main():
    require();torch.set_num_threads(2);x,ctx=public("test")
    directory=R/"unstratified";directory.mkdir(exist_ok=True)
    paths=[]
    for b,pair in enumerate(PAIRS):
        engine=Unstratified(pair);choice=np.empty((len(x),3),int)
        for i in range(len(x)):
            for l in range(3):
                state=engine.fit(x[i,l],ctx[i,l],TASK,True)
                choice[i,l]=action(engine.risk(state,"keep512m512"))
        path=directory/f"choice_{b}.npz";save(path,act=choice);paths.append(path)
        del engine
    freeze(directory/"freeze.json",paths+[Path("control22.py")])
    print("Blind same-budget unstratified choices saved",flush=True)
if __name__=="__main__":main()
