"""Inspect estimation uncertainty using the pilot sensor readouts only."""
from pathlib import Path
import json
import numpy as np
from switch56 import W, BUDGETS, CROSSING, estimate, choose

ROOT=Path(__file__).resolve().parent
def main():
    sensor=[]
    for i in range(24):
        with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as f:
            sensor.append(np.stack([f['pilot_ref'],f['pilot_imp']]))
    s=np.stack(sensor,axis=1)
    out={}
    for k in BUDGETS:
        rows=[]
        for j,w in enumerate(W):
            r,i=[x[:,:,j,:,:k] for x in s]
            n,flips,q,wh=estimate(r,i)
            ok=n>0
            switched=choose(r,i)
            pool=(.125+.375*w)
            row=dict(w=float(w),q=pool,valid=n.mean(),empty=float(1-ok.mean()),
                     purchased_pairs=n.size*k,usable_pairs=int(n.sum()),fixed_probability=switched.mean())
            if np.any(ok):row.update(q_observed=flips.sum()/n.sum(),rmse_q=float(np.sqrt(np.mean((q[ok]-pool)**2))),
                                      mae_q=float(np.mean(abs(q[ok]-pool))),conditional_mean_w=wh[ok].mean(),
                                      rmse_w=float(np.sqrt(np.mean((wh[ok]-w)**2))),estimated_w_p10_p50_p90=np.quantile(wh[ok],[.1,.5,.9]).tolist(),
                                      n_p10_p50_p90=np.quantile(n,[.1,.5,.9]).tolist())
            rows.append(row)
        out[k]=rows
    (ROOT/'pilot_stats56.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
