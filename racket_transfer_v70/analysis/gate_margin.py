"""Observed endpoint sign audit sensitivity, and error scale vs actual release magnitudes."""
from pathlib import Path
import csv
import json
import numpy as np
root=Path(__file__).resolve().parents[1]
fit=json.loads((root/'solver'/'metrics.json').read_text())['folds']
all_metrics={}
for group,entry in fit.items():
    rows=[]
    for c in entry['held']:
        with (root/'data'/'upstream'/'test'/f'racket_{c:02}.csv').open() as f: rows.extend(csv.DictReader(f))
    def col(k):return np.array([float(r[k]) for r in rows])
    x=np.column_stack([col('v_in_x'),col('v_in_z'),col('spin_before_y')*.02])
    y=np.column_stack([col('v_out_x'),col('v_out_z'),col('spin_after_y')*.02])
    s=y[:,0]-y[:,2]
    p=entry['empirical'];a=np.column_stack([np.ones(len(x)),(x-p['mean'])/p['std']])
    pred=a@p['logistic']>0
    gate={}
    for t in [0,.25,.5,1.,2.]:
        keep=abs(s)>t
        gate[str(t)]={'n':int(keep.sum()),'right':int(np.sum(pred[keep]==(s[keep]>0))),
                     'accuracy':float(np.mean(pred[keep]==(s[keep]>0))) if keep.any() else None}
    all_metrics[group]=dict(gate=gate,mean_speed=float(np.linalg.norm(y[:,:2],axis=1).mean()),mean_abs_spin=float(abs(y[:,2]).mean()/.02))
(root/'analysis'/'gate_margin.json').write_text(json.dumps(all_metrics,indent=2)+'\n')
