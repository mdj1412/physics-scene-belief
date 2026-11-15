from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'compute_matched_stopping_v32'))
from stats32 import clean,allocations,simultaneous
read=lambda p:json.loads(p.read_text())
def main():
    rule=read(ROOT/'results_prefix/discovery_lock.json')['rule']
    laws=read(ROOT/'design_eval.json')['laws'];stage={2:0,4:1}[rule['stage']]
    feature={'posterior_friction_sd':3,'paid_tangential_spread':11,'paid_velocity_covariance':14}[rule['feature']]
    delta=np.empty((4,8,2,2));sums=[];shifts=[]
    for c in range(4):
        for b in range(8):
            for d in (0,1):
                law=laws[2*c+d]
                with np.load(ROOT/'results_eval'/law/f'block_{b:03d}'/'metrics.npz') as z:
                    score=z['uncertainty'].copy()
                    if law.endswith(rule['direction']):score[:,stage]=z['features'][:,stage,feature]*(1 if rule['half']=='upper' else -1)
                    k=allocations(score);assert k.sum()==32
                    r=z['oracle_stage_risk'][np.arange(8),np.searchsorted([2,4,8],k)].mean()
                    delta[c,b,d]=r-z['brier'][:,2:].mean(0)
                    sums.append(r);shifts.append(np.mean(k!=z['counts'][:,3]))
    v=delta.mean((0,1,2));ids=np.random.default_rng(3305333).integers(8,size=(20000,4,8))
    draw=delta[np.arange(4)[None,:,None],ids].mean((1,2,3));ci,_=simultaneous(v,draw)
    (ROOT/'shadow33.json').write_text(json.dumps(clean(dict(rule=rule,absolute=np.mean(sums),delta=v,simultaneous_ci=ci,
              fraction_counts_changed_vs_uncertainty=np.mean(shifts))),indent=2)+'\n')
if __name__=='__main__':main()
