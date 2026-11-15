from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'compute_matched_stopping_v32'))
from stats32 import clean,simultaneous
def main():
    root=ROOT/'results_prefix'
    rule=json.loads((root/'discovery_lock.json').read_text())['rule']
    laws=json.loads((ROOT/'design_eval.json').read_text())['laws']
    ix={'posterior_friction_sd':3,'paid_tangential_spread':11,'paid_velocity_covariance':14}[rule['feature']]
    out=np.zeros((4,8,4))
    for f in range(4):
        for b in range(8):
            for d in (0,1):
                with np.load(ROOT/'results_eval'/laws[2*f+d]/f'block_{b:03d}'/'metrics.npz') as z:
                    feats=z['features'];risk=z['oracle_stage_risk']
                    for s,k in enumerate((2,4)):
                        gain=risk[:,s]-risk[:,s+1]
                        def get(c,side):
                            ids=np.argsort(-feats[:,s,c],kind='stable');v,w=gain[ids[:4]].mean(),gain[ids[4:]].mean()
                            a,b=(v,w) if side=='upper' else (w,v)
                            return np.array([a,a-b])
                        out[f,b,:2]+=.25*get(3,'upper')
                        if k==rule['stage'] and laws[2*f+d].endswith(rule['direction']):
                            out[f,b,2:]=get(ix,rule['half'])
    values=out.mean((0,1));ids=np.random.default_rng(3304021).integers(8,size=(20000,4,8))
    draws=out[np.arange(4)[None,:,None],ids].mean((1,2));ci,_=simultaneous(values,draws)
    (ROOT/'transfer33.json').write_text(json.dumps(clean(dict(rule=rule,values=values,ci=ci)),indent=2)+'\n')
if __name__=='__main__':main()
