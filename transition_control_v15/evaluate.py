"""Post-freeze evaluation with private simulator outcomes."""
import time
import numpy as np
import torch
from model import *
from common import *
from run_policy import CELLS

def run():
    require();f=read(R/'policy_freeze.json');assert not f['private_outcomes_read']
    for p,h in f['files'].items():assert sha(p)==h
    pub=np.load(R/'data/test/public.npz');priv=np.load(R/'data/test/private.npz');n=len(pub['factual'])
    for cell,condition,law in CELLS:
        for N,seed in [(16384,2026111201)]+([(65536,2026111203)] if law=='nominal' else []):
            b=Bank(N,seed,law);pred=[];ess=[]
            for i in range(n):
                x=t(pub['factual'][i,condition]);ctx=t(pub['context'][i,condition]);w=b.fit(x);z=targets(b.theta,ctx,law)
                byaction=[(w@z).expand(16,-1)];esses=[(1/(w*w).sum()).expand(16)]
                for a in [1,2,3]:
                    updated=b.update(w,a,priv[f'obs{a}'][i,condition]);byaction.append(updated@z);esses.append(1/(updated*updated).sum(-1))
                pred.append(torch.stack(byaction).cpu().numpy());ess.append(torch.stack(esses).cpu().numpy())
            save(R/'evaluations'/f'{cell}_{N}.npz',pred=pred,ess=ess)
            print('Evaluated',cell,N,flush=True)
    write(R/'evaluation_done.json',dict(time=time.time(),policy_freeze=f['time']))
if __name__=='__main__':run()
