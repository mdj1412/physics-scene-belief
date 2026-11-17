"""Pre-outcome regression test against the frozen v60 aggregates and exact weighted quantiles."""
import json
import os
import numpy as np
from score61 import (ROOT,C,B,target_assign,components,averages,make_weights,
                     paired_session_tail,weights,cdf_boot)
assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and os.getenv('SLURM_JOB_ID')
prior=ROOT.parent/'trigger_fault_v60'
previous=json.loads((prior/'score60.json').read_text())
cat,board,product=make_weights(20,6108)
with np.load(prior/'score60_cells.npz') as f:
    error=[]
    for target in ('independent','matched','reversed'):
        rate=target_assign(f['rates'],target).mean(1)
        cause=target_assign(f['cause'],target).mean(1)
        rate=np.repeat(rate[:,None],B,axis=1)
        cause=np.repeat(cause[:,None],B,axis=1)
        parts=averages(components(f['fine'],rate,cause),product)
        for new,old in [('mean_Brier','mean_Brier'),('AB_action_BA','action_BA'),('cause_BA','cause_BA'),('lower_loss_action_BA','lower_risk_action_BA')]:
            # Ignore privileged A/B previous label's separate handling.
            error.append(abs(parts[new][0,:4]-previous['uniform_delay'][target][old][:4]).max())
assert max(error)<1.e-12
# Stress both cluster sources using nonidentical simulated board routes and concentrated bootstrap weights.
rng=np.random.default_rng(610892)
r=rng.random((C,B,2,2,2))
y=np.sort(rng.random((C,2,3,2,2,23)).astype('float32'),axis=-1)
q=paired_session_tail(y,r,cat,board,1)
for i in range(len(cat)):
    percat=np.einsum('b,cbfza->cfza',board[i],r)
    w=weights(percat)[1]*cat[i,:,None,None,None,None]
    p=q[i]
    assert (w[...,None]*(y<p)).sum()/23<.900000000001
    assert (w[...,None]*(y<=p)).sum()/23>=.899999999999
print('v60 replay maximum discrepancy',max(error),'and exact two-way CDF verified in',len(q),'draws.')
