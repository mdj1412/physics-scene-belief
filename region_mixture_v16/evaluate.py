import time
import numpy as np
import torch
from model import *
from mixture import compress,predict
from common import *
from run_policy import CELLS
def run():
    require();f=read(R/'policy_freeze.json');assert not f['private_outcomes_read']
    for p,h in f['files'].items():assert sha(p)==h
    pub=np.load(R/'data/test/public.npz');priv=np.load(R/'data/test/private.npz');n=len(pub['factual']);starts=np.cumsum([0]+SIZES)
    for cell,cond,law in CELLS:
        policy=np.load(R/'policies'/f'{cell}.npz')['region32']
        for N,seed in [(16384,2026111201)]+([(65536,2026111203)] if law=='nominal' else []):
            b=Bank(N,seed,law);pred=[];compactpred=[];ess=[]
            for i in range(n):
                x=t(pub['factual'][i,cond]);ctx=t(pub['context'][i,cond]);w=b.fit(x);z=targets(b.theta,ctx,law)
                byaction=[(w@z).expand(16,-1)];eses=[(1/(w*w).sum()).expand(16)]
                for a in [1,2,3]:
                    up=b.update(w,a,priv[f'obs{a}'][i,cond]);byaction.append(up@z);eses.append(1/(up*up).sum(-1))
                pred.append(torch.stack(byaction).cpu().numpy());ess.append(torch.stack(eses).cpu().numpy())
                if N==16384:
                    parts=[]
                    for k in range(5):
                        c=compress(b,w,z[:,starts[k]:starts[k+1]],k in [2,4]);a=int(policy[i,k]);parts.append(predict(c,a,priv[f'obs{a}'][i,cond]))
                    compactpred.append(torch.cat(parts,-1).cpu().numpy())
            save(R/'evaluations'/f'{cell}_{N}.npz',pred=pred,ess=ess)
            if N==16384:save(R/'evaluations'/f'{cell}_compact.npz',pred=compactpred)
            print('Evaluated',cell,N,flush=True)
    write(R/'evaluation_done.json',dict(time=time.time()))
if __name__=='__main__':run()
