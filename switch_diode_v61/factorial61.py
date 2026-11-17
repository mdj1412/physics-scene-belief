"""Estimate paired factorial timing effects with board and catalog resampling."""
import json
import os
import numpy as np
from score61 import ROOT, DELAYS, METHODS, target_assign, components, averages, make_weights

assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and os.getenv('SLURM_JOB_ID')
with np.load(ROOT/'score61_cells.npz') as source:
    rate=target_assign(source['rates'],'independent')
    cause=target_assign(source['cause'],'independent')
    fine=source['fine']
cat,board,product=make_weights(6000,6104358)
metrics=[averages(components(fine,rate[:,:,i],cause[:,:,i]),product) for i in range(4)]
isolates={
    'LED_effect_averaged_over_trigger':np.array([-.5,-.5,.5,.5]),
    'trigger_effect_averaged_over_LED':np.array([-.5,.5,-.5,.5]),
    'LED_by_trigger':np.array([1,-1,-1,1]),
}
out={}
for key,contrast in isolates.items():
    out[key]={}
    for endpoint in metrics[0]:
        vals=sum(w*cell[endpoint] for w,cell in zip(contrast,metrics))
        out[key][endpoint]={}
        for i in [0,2,4]:
            v=vals[:,i]
            out[key][endpoint][METHODS[i]]=[v[0],*np.quantile(v[1:],[.025,.975])]
(ROOT/'factorial61.json').write_text(json.dumps(out,indent=2)+'\n')
