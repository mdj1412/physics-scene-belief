"""A target-label diagnostic lower bound for the impulse constraint, never used for prediction."""
import csv
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
with (root/'data'/'feint_ox_digitized.csv').open() as f: rows=list(csv.DictReader(f))
run=json.loads((root/'method'/'feint'/'result.json').read_text())
w=.019*2*np.pi
u=np.array([float(r['u_out_mps']) for r in rows])
dn=w*np.array([float(r['n_out_rps'])-float(r['n_in_rps']) for r in rows])
held=np.array([r['id'] in run['held'] for r in rows])
report={}
for name,kappa in [('v71_thin_shell',2/3),('araki_ball',.653)]:
    # An unconstrained single impulse obeys delta_u + kappa*delta_rim = 0.
    # Thus |err_u| + |err_rim| >= |delta_u + kappa*delta_rim| for 0<=kappa<=1.
    violation=np.abs(u+kappa*dn)
    report[name]={'all_mean_mps':float(violation.mean()),'held_mean_mps':float(violation[held].mean()),
                  'by_point_mps':dict(zip([r['id'] for r in rows],violation.tolist()))}
report['actual_common_loss_mps']={}
for k,v in run['results']['common_holdout'].items():
    report['actual_common_loss_mps'][k]={m:z['velocity']['mean']+.019*z['spin']['mean'] for m,z in v['methods'].items()}
(root/'analysis'/'impulse_bound.json').write_text(json.dumps(report,indent=2)+'\n')
