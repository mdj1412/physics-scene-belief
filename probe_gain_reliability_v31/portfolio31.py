"""Stage-local allocation replay using only disjoint futures and deployment batches for scoring."""
from itertools import combinations, product
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LAWS = ("coulomb", "split_static", "speed_weak")
METRICS = ("one_selected_in", "one_selected_out", "two_by_two_selected_in",
           "two_by_two_selected_out", "frozen_learned", "uncertainty",
           "one_top_half_overlap", "two_by_two_top_half_overlap")


def top_half(scores):
    """Inclusion probabilities under uniform tie-breaking at the cutoff."""
    cutoff = np.sort(scores, axis=-1)[..., -4, None]
    winners = scores > cutoff
    tied = scores == cutoff
    extra = (4 - winners.sum(axis=-1, keepdims=True)) / tied.sum(axis=-1, keepdims=True)
    return winners + tied * extra


def folds(scores, size):
    # scores: stage,environment,future,batch
    subsets = list(combinations(range(4), size))
    scenario = list(product(subsets, repeat=2))
    vals = np.stack([scores[:, :, f, :][:, :, :, b].mean((2,3)) for f,b in scenario],axis=1)
    left, right = zip(*[(i,j) for i,(fi,bi) in enumerate(scenario) for j,(fj,bj) in enumerate(scenario)
                        if set(fi).isdisjoint(fj) and set(bi).isdisjoint(bj)])
    w=top_half(vals)
    # Ranks and their tie-breaking are allowed to vary independently between repeats.
    inside=((w[:,left]-.5)*vals[:,left]).mean((1,2))
    outside=((w[:,left]-.5)*vals[:,right]).mean((1,2))
    overlap=(w[:,left]*w[:,right]).sum(2).mean(1)/4
    return inside,outside,overlap


def pair_compare(x,y):
    i,j=np.triu_indices(8,1)
    return np.mean(.5+.5*np.sign(x[i]-x[j])*np.sign(y[i]-y[j]))


def new_prefix_repeats(risks):
    stats=[]
    for vals in risks.transpose(1,0,2):
        new=[pair_compare(vals[i],vals[j]) for i,j in combinations(range(len(vals)),2)]
        half=[]
        for included in combinations(range(len(vals)),len(vals)//2):
            other=sorted(set(range(len(vals)))-set(included))
            if 0 in included:
                half.append(pair_compare(vals[list(included)].mean(0),vals[other].mean(0)))
        stats.append([np.mean(new),np.mean(half)])
    return stats


def assess(gain,score,uncertainty):
    assert gain.shape==(2,8,4,4) and score.shape==uncertainty.shape==(2,8)
    a,b,o1=folds(gain,1)
    c,d,o2=folds(gain,2)
    mean = gain.mean((2,3))
    assert np.allclose(top_half(score).sum(-1), 4)
    replay=[((top_half(x)-.5)*mean).mean(-1) for x in (score,uncertainty)]
    return np.stack([a,b,c,d,*replay,o1,o2],axis=-1)


def main():
    from argparse import ArgumentParser
    parser=ArgumentParser()
    parser.add_argument("--panel",choices=["results","results_pool"],default="results")
    results=ROOT/parser.parse_args().panel
    vals=[];repeat=[]
    for law in LAWS:
        blocks=[];couplings=[]
        for path in sorted((results/law).glob("block_*/metrics.npz")):
            with np.load(path) as z:
                gain=z['gain']
                blocks.append(assess(gain,z['score'],z['uncertainty']))
                a=z['alpha']
                assert len(set(map(tuple,a)))==8
                order=np.lexsort((a[:,1],a[:,0]))
                couplings.append(gain[:,order].mean((2,3)))
        assert len(blocks)==6,law
        vals.append(blocks)
        repeat.append(new_prefix_repeats(np.stack(couplings)))
    arr=np.stack(vals)
    rng=np.random.default_rng(31398535)
    rep=np.stack([a[rng.integers(6,size=(10000,6))].mean(1) for a in arr]).mean(0)
    result=dict(metrics=METRICS,laws=LAWS,by_law=arr.mean(1).tolist(),pooled=arr.mean((0,1)).tolist(),
                pooled95=np.moveaxis(np.quantile(rep,[.025,.975],axis=0),0,-1).tolist(),
                independent_prefix_law_stage=np.array(repeat).tolist(),
                independent_prefix_pooled=np.mean(repeat,axis=0).tolist(),
                independent_prefix_columns=['one block vs independent block','three vs three disjoint blocks'],
                description='Gain over an exact uniform random top-four allocation, averaged over all eight environments; top-half overlap is recovery fraction with random reference 0.5.')
    (results/'portfolio31.json').write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
