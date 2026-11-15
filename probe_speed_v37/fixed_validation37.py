"""Historical-validation constant shrinkage, trained without any v37 observation."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.optimize import minimize_scalar
R=Path(__file__).resolve().parent
V=R.parent/'query_matched_probe_v36'
BETA=.8289760825845313

def fit():
    # Every historical local regime, cell, mirror and repeat has equal weight.
    files=sorted(p for p in (V/'results').glob('*/*/*.npz') if not p.name.startswith('global_null'))
    assert len(files)==192
    scores=[]; truths=[]; cues=[]; hashes=[]
    # Pool multiple files to amortize disk access across optimizer evaluations.
    for p in files:
        hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
        with np.load(p) as f:
            scores.append(f['p_near']);truths.append(f['labels'][...,None]);cues.append(f['z'][:,None,None])
    p=np.stack(scores);y=np.stack(truths);z=np.stack(cues)
    def loss(t):
        w=np.exp(t*BETA*z);prob=p*w/(1-p+p*w)
        return float(np.square(prob-y).mean())
    opt=minimize_scalar(loss,bounds=(0.,1.),method='bounded',options={'xatol':1e-7})
    candidates=[(opt.x,opt.fun),(0.,loss(0.)),(1.,loss(1.))]
    best,val=min(candidates,key=lambda a:a[1])
    save=dict(scale=best,historical_brier=val,reference_brier=loss(.325),archives=len(files),
              input_sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest(),
              description='One constant fit at +0.50 using v36 local strong/weak/null; transported unchanged at all v37 speeds.')
    (R/'fixed_validation37.json').write_text(json.dumps(save,indent=2)+'\n')
    print(save)
if __name__=='__main__':fit()
