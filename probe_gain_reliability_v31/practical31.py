"""Descriptive reproducibility of a predeclared Brier gain threshold and its sign."""
import json
from pathlib import Path
import numpy as np
from stats31 import LAWS, RESULTS, replicate_design


def per_block(g):
    _, i, j=replicate_design(4, 4, "both_new_1x1")
    v=g.reshape(2,8,16)
    v0, v1=v[:,:,i],v[:,:,j]
    def indicators(level):
        event=v0>level
        return np.stack([event.mean((1,2)),(event & (v1>level)).mean((1,2))],axis=-1)
    means=g.mean((-2,-1))
    return np.concatenate([indicators(0.),indicators(.003),
                           (means>0.).mean(1)[:,None],(means>.003).mean(1)[:,None]],axis=1)


def estimands(values):
    return np.stack([values[...,0],values[...,1]/values[...,0],values[...,2],values[...,3]/values[...,2],
                     values[...,4],values[...,5]],axis=-1)


def main():
    xs=[]
    for law in LAWS:
        rows=[]
        for path in sorted((RESULTS/law).glob("block_*/metrics.npz")):
            with np.load(path) as z:rows.append(per_block(z['gain']))
        assert len(rows)==6
        xs.append(rows)
    x=np.stack(xs)
    rng=np.random.default_rng(31784131)
    ids=rng.integers(6,size=(10000,3,6))
    boots=x[np.arange(3)[None,:,None],ids].mean(2).mean(1)
    point=estimands(x.mean((0,1)))
    values=dict(columns=['P(G>0)','P(G2>0|G1>0)','P(G>.003)','P(G2>.003|G1>.003)',
                         'observed P(mean_4x4>0)','observed P(mean_4x4>.003)'],
                stage_order=['2to4','4to8'],law_order=LAWS,pooled=point.tolist(),
                pooled95=np.moveaxis(np.quantile(estimands(boots),[.025,.975],axis=0),0,-1).tolist(),
                bylaw=estimands(x.mean(1)).tolist(),seed=31784131,
                note='Exploratory equal-law, equal-block, equal-prefix weighting before forming conditional probabilities. G1 and G2 have disjoint calibration future and deployment batch at the same paid prefix. One positive draw is evidence from unavailable held-out labels. P(mean_4x4>threshold) is the observed finite grid statistic, not the share of truly valuable environments.')
    (RESULTS/'practical31.json').write_text(json.dumps(values,indent=2,allow_nan=False)+'\n')
    print(values['pooled'])
if __name__=='__main__':
    main()
