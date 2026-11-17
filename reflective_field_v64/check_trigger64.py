"""Locate the recorded camera trigger transition in each real 80 MHz oscilloscope record."""
from pathlib import Path
import json,os
import numpy as np
from audit_opara64 import DATA, ROOT
assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
result=[]
for f in sorted(DATA.rglob('*80MHz.txt')):
    a=np.loadtxt(f,skiprows=3,usecols=(0,3)); w=(a[:,0]>=-.15)&(a[:,0]<=.35)
    t,v=a[w].T
    v0=np.median(v[t<-.05]); v1=np.median(v[(t>.12)&(t<.15)])
    raw=(v>(v0+v1)/2) if v1>v0 else (v<(v0+v1)/2)
    # Ten consecutive high-state samples span 0.125 us.
    transitions=np.flatnonzero(np.convolve(raw.astype('u1'),np.ones(10,'u1'),mode='valid')>=10)
    result.append(dict(test=f.parents[1].name,low_window_v=v0,post_window_v=v1,
        first_stable_change_after_scope_zero_us=1000*t[transitions[(t[transitions]>0)][0]] if len(transitions) and abs(v1-v0)>.5 else None,
        minmax_V=[v.min(),v.max()]))
(ROOT/'opara64_triggers.json').write_text(json.dumps(result,indent=2)+'\n')
