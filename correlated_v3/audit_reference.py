"""Independent uniform-grid posterior moment audit on development worlds."""
import json,time,platform
from pathlib import Path
import numpy as np
from core import *
from run import case_spec

out=Path('results/audit');out.mkdir(parents=True,exist_ok=True)
start=time.perf_counter();records=[]
for f in range(8):
    spec=case_spec(f)
    for sigma in [.03,.001]:
        y=float(impact(spec['g'],spec['e'])[0]+sigma*spec['factual_noise'])
        dense,_=fixed_grid(y,sigma,False,2049,1025)
        ref,_=conditional_quad(y,sigma,False)
        def moments(a):
            g,e,w=a
            return np.r_[w@g,w@e,[w@impact(g+d,e)[0] for d in spec['deltas']]]
        a=moments(dense);b=moments(ref)
        records.append(dict(family=f,sigma=sigma,g_error=float(abs(a[0]-b[0])),e_error=float(abs(a[1]-b[1])),vx_max_error=float(abs(a[2:]-b[2:]).max())))
        print(f,sigma,records[-1],flush=True)
(out/'uniform_grid_check.json').write_text(json.dumps(dict(host=platform.node(),seconds=time.perf_counter()-start,atoms=2049*1025,checks=records),indent=2))
