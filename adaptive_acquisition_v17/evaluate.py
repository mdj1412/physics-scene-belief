import time
import numpy as np
import torch
from model import *
from common import *
from run_policy import CELLS,EIDX
def run():
    require();f=read(R/'policy_freeze.json');assert not f['private_outcomes_read']
    for p,h in f['files'].items():assert sha(p)==h
    pub=np.load(R/'data/test/public.npz');priv=np.load(R/'data/test/private.npz')
    for cell,cond,law in CELLS:
        for N,seed in [(16384,2026111201),(65536,2026111203)]:
            b=Bank(N,seed,law);pred=[];ess=[]
            for i in range(len(pub['factual'])):
                w=b.fit(t(pub['factual'][i,cond]));z=targets(b.theta,t(pub['context'][i,cond]),law)[:,EIDX]
                ps=[(w@z).expand(16,-1)];es=[(1/(w*w).sum()).expand(16)]
                for a in [1,2,3]:
                    up=b.update(w,a,priv[f'obs{a}'][i,cond]);ps.append(up@z);es.append(1/(up*up).sum(-1))
                pred.append(torch.stack(ps).cpu().numpy());ess.append(torch.stack(es).cpu().numpy())
            save(R/'evaluations'/f'{cell}_{N}.npz',pred=pred,ess=ess);print('Evaluated',cell,N,flush=True)
    write(R/'evaluation_done.json',dict(time=time.time()))
if __name__=='__main__':run()
