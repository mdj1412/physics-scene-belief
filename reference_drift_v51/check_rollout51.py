import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
exact=json.loads((ROOT/'score51.json').read_text())
with np.load(ROOT/'rollout_sessions51.npz') as archive:
    risk=archive['risk'] # catalog,epsilon,drift,environment,rollout,method
    methods=list(archive['method'])
table=np.array(exact['loss'])[...,[exact['methods'].index(m) for m in methods]]
observed=risk.mean(axis=(0,3,4))
diff=observed-table
contrasts=diff[...,:1]-diff[...,1:3]
out=dict(methods=methods,max_abs_policy_Brier_error=np.max(abs(diff),axis=(0,1)).tolist(),
         max_abs_soft_comparison_error=np.max(abs(contrasts),axis=(0,1)).tolist(),
         observed_minus_integrated=diff.tolist())
(ROOT/'rollout_replay51.json').write_text(json.dumps(out,indent=1))
