"""Partition actual paired Brier contrasts by source/target label agreement (descriptive)."""
from pathlib import Path
import json
import numpy as np
from stats32 import METHODS, LAWS, PAIRS, risk_differences

OUT=Path(__file__).resolve().parent/'results_eval'


def main():
    rows=[]
    for law in LAWS:
        r=[]
        for b in range(16):
            with np.load(OUT/law/f'block_{b:03d}/metrics.npz') as z:
                same=z['labels']==z['nominal_labels']
                cost=((z['probabilities']-z['labels'][:,:,None,None])**2).mean(-1)
                actual=np.take_along_axis(cost,(z['actions']==3).astype(int),axis=2)
                by=np.stack([(actual*k[:,:,None]).mean((0,1)) for k in (same,~same)])
                np.testing.assert_allclose(by.sum(0),z['brier'].mean(0),atol=1e-12,rtol=0)
                r.append(by)
        rows.append(r)
    v=risk_differences(np.array(rows))
    ix=np.random.default_rng(3276027).integers(16,size=(10000,3,16))
    pooled=v[np.arange(3)[None,:,None],ix].mean((1,2))
    obj=dict(methods=METHODS,laws=LAWS,contrasts=[[METHODS[a],METHODS[b]] for a,b in PAIRS],
             contribution_order=['unchanged','changed'],by_law=v.mean(1).tolist(),
             pooled=v.mean((0,1)).tolist(),paired95=np.moveaxis(np.quantile(pooled,[.025,.975],axis=0),0,-1).tolist())
    (OUT/'shift32.json').write_text(json.dumps(obj,indent=2))
if __name__=='__main__':main()
