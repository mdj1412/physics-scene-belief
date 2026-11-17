"""Counts, source-marginal Brier baselines and analytic tiebreak audit."""
import csv
import json
from pathlib import Path

from statistics import mean

HERE=Path(__file__).resolve().parent
with (HERE/'predictions.csv').open() as f:
    rows=list(csv.DictReader(f))

def events(sel):
    pre=[float(r['vx_in'])+float(r['rim_in']) for r in sel]
    post=[float(r['vx_out'])+float(r['rim_out']) for r in sel]
    return [[abs(u)<=h for u in post] for h in [.25,.50,1.]] + [[a*b<0 for a,b in zip(pre,post)]]

out={}
for source in 'AB':
    rows_s=[r for r in rows if r['source']==source and r['method']=='pybullet']
    s=[r for r in rows_s if r['split']=='cal']
    t=[r for r in rows_s if r['split']=='test']
    freq={}
    for k,ec,et in zip(['025','050','100','rev'],events(s),events(t)):
        rate=(sum(ec)+.5)/(len(ec)+1)
        freq[k]={'cal_count':int(sum(ec)),'n_cal':len(ec),'target_count':int(sum(et)),
                 'n_target':len(et),'source_marginal_p':rate,'brier_source_marginal':mean((y-rate)**2 for y in et)}
    out[source]=freq
(HERE/'marginal_baselines.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
