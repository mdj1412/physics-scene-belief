"""Independent endpoint scoring and matched hollow/bulge label perturbation."""
import json
from pathlib import Path
from itertools import product
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parent
FAMILIES=('wendland','dispersive')
record=json.loads((ROOT/'locked_results.json').read_text())
pooled=np.zeros((12,2,3))
flips=np.zeros((12,2))
for i,f,s,law,h in product(range(12),FAMILIES,('inlay','woven'),('rational','gaussian_rate'),(19,31)):
    sub=f'panel_{i:02d}'
    yy={}
    for v in ('base','hollow','bulge'):
        fn=f'{f}_{s}_{law}_{v}_h{h}.npz'
        with np.load(ROOT/'labels'/sub/fn) as z:
            y=z['labels'][1:].astype(float)
            assert ((z['slip_speed']<=.020)==z['labels']).all()
            assert z['normal_impulse_Ns'].min()>0
        with np.load(ROOT/'results'/sub/fn) as z:
            p=z['pred'][:,0]
            assert z['selected'].shape==(4,128)
        with np.load(ROOT/'routes'/sub/f'{f}_{v}_h{h}.npz') as z:g=z['route']
        risk=(p*p).mean(-1)[:,None,None]+y[None]*(1-2*p.mean(-1)[:,None,None])
        aggregate=risk[:1]+g[None,None]*(risk-risk[:1])
        pooled[i,FAMILIES.index(f)]+=aggregate.mean((1,2,3,4))/24
        yy[v]=y
    flips[i,FAMILIES.index(f)]+=(yy['hollow']!=yy['bulge']).mean()/8
diff=float(np.abs(pooled-np.asarray(record['panel_family'])).max())
assert diff<2e-12
mean=flips.mean(0)
hw=t.isf(.0125,11)*flips.std(0,ddof=1)/np.sqrt(12)
print(json.dumps(dict(max_independent_catalog_endpoint_brier_difference=diff,
     paired_shape_flip_rate=dict(zip(FAMILIES,np.stack((mean,mean-hw,mean+hw),-1).tolist())),
     unit='matched full-load solver replay on same prototype; CI adjusted across two families',
     forecast_files=576,optical_route_files=144,primary_label_files=576),indent=2))
