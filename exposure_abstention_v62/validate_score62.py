"""Cross-check v62 summaries with exact archived v61 numbers without using v62 outcomes."""
import os
import json
from pathlib import Path
import numpy as np
from score62 import (ROOT, C, B, board_average, log_odds, SOURCE_LOG_CUTOFF,
                     target_assign, components, averages, make_weights)

assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
old=ROOT.parent/'switch_diode_v61'
ref=json.loads((old/'score61.json').read_text())
with np.load(old/'score61_cells.npz') as f:
    rates,causes,fine=f['rates'],f['cause'],f['fine']
_,_,product=make_weights(10,620412)
point=product[:1]
errors=[]
for relation in ('independent','matched','reversed'):
    a=target_assign(rates,relation).mean(2);k=target_assign(causes,relation).mean(2)
    result=averages(components(fine,a,k),point)
    for key,val in result.items():
        errors.append(np.max(np.abs(val[0]-ref['uniform'][relation][key])))
with np.load(old/'pilots/panel_03.npz') as f:
    c=f['codes'][:,2,1]
odds,means=log_odds(c,1,True)
assert np.allclose(board_average(odds>SOURCE_LOG_CUTOFF),rates[3,...,0])
assert np.allclose(board_average(means>1e-12),causes[3,...,0])
# Confirm the 13-symbol valid camera bit from the raw two signs.
with np.load(old/'vision_audit/panel_03.npz') as f:
    camera=f['camera'][:,2]
assert np.array_equal(np.stack([c//13,c%13],axis=-1)>=5,(camera!=0).all(-1))
assert max(errors)<5e-14
print('v61 risks, both BAs, board interleaving, optical mask verified; max absolute difference',max(errors))
