import json
from pathlib import Path
import numpy as np
from geometry import render,extract
from belief import Belief,predict
from engine import BOUNDS,tensor,simulate,observation_means
import torch

rows=[]
for family in ['corner','gate','deflection']:
    fs=[]
    for b in [.001,.005]:
        im=render(0,.027,b,family,'high',np.random.default_rng(123))
        fs.append(extract(im,family,'high'))
    assert np.allclose(fs[0][:2],fs[1][:2]),fs
    assert fs[1][2]-fs[0][2]>5,fs
    model=Belief(family);atoms,cost=model.smc([('factual',[.6])],n=512,seed=71)
    pred=predict(atoms,family);g0,e0=model.acquisition(atoms,pred,samples=128,seed=3)
    g1,e1=Belief(family,'high','bbox').acquisition(atoms,pred,samples=128,seed=3)
    assert np.allclose(g0[:,1:],g1[:,1:],atol=1e-12)
    assert np.allclose(e0[1:],e1[1:],atol=1e-12)
    rows.append(dict(family=family,features=[f.tolist() for f in fs],cost=cost,dynamic_gains_invariant=True))
out=Path('results/validation');out.mkdir(parents=True,exist_ok=True);(out/'checks.json').write_text(json.dumps(rows,indent=2));print('validation passed',flush=True)

rng=np.random.default_rng(2026110800+36);theta=BOUNDS[:,0]+rng.random(4)*(BOUNDS[:,1]-BOUNDS[:,0]);b=rng.uniform(0,.006)
audit=[]
for action,shift in [('factual',0.),('left',-.022),('right',.022),('center',.009)]:
    th=tensor(theta[None,:]);bb=tensor([b]);a=simulate(th,bb,'gate',shift);p=simulate(th,bb,'gate',shift,horizon=.75)
    finite=(a['position']-p['position'])/.05;pred=observation_means(th,bb,'gate',action)
    assert float((finite-pred).abs().max())<1e-9
    audit.append(dict(action=action,last_event=float(a['last_event'][0]),instantaneous=a['velocity'][0].cpu().tolist(),finite_frame=finite[0].cpu().tolist()))
assert any(r['last_event']>.75 for r in audit)
(out/'finite_frame_check.json').write_text(json.dumps(dict(case=36,theta=theta.tolist(),rounding=b,actions=audit),indent=2));print('late-contact finite-frame validation passed',flush=True)
