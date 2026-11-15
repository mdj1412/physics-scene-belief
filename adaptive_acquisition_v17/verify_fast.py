import torch,time
import numpy as np
from model import *
from common import *
from run_policy import CELLS,EIDX,ADAPT,config
from adaptive import outer_samples,phases
from adaptive_fast import select
from acquisition import plan
def run():
    require();torch.set_num_threads(2);pub=np.load(R/'data/test/public.npz');pre=plan(512);phase=phases(81531);error=0.;count=0
    for cell,cond,law in CELLS:
        b=Bank(law=law);o=Bank(seed=2026111202,law=law);p=np.load(R/'policies'/f'{cell}.npz')
        for i in range(512):
            x=t(pub['factual'][i,cond]);ctx=t(pub['context'][i,cond]);w=b.fit(x);ow=o.fit(x);z=targets(b.theta,ctx,law)[:,EIDX];oz=targets(o.theta,ctx,law)[:,EIDX]
            for k in range(2):
                samp=outer_samples(o,ow,oz[:,k:k+1],pre)
                for m in ADAPT:
                    a,r,meta,h=select(b,w,z[:,k:k+1],samp,phase,**config(m));assert a==p[m][i,k],(cell,i,k,m)
                    assert np.array_equal(meta.cpu().numpy(),p[m+'_meta'][i,k])
                    assert np.allclose(h.cpu().numpy(),p[m+'_history'][i,k],rtol=0,atol=1e-14,equal_nan=True)
                    error=max(error,float(np.max(np.abs(r.cpu().numpy()-p[m+'_risk'][i,k]))));count+=1
        print('Fast equivalence verified',cell,flush=True)
    assert error<1e-14;write(R/'fast_equivalence.json',dict(cases=count,max_risk_error=error,source=sha('adaptive_fast.py'),time=time.time()))
if __name__=='__main__':run()
