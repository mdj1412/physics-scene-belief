"""Two disjoint catalog folds; fit fourth-sensor choice from six pre-decision sign patterns."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V53 = ROOT.parent / 'impact_witness_v53'
sys.path.insert(0, str(V53))
from score53 import E, G, KAPPA, N, R, quadratic, quantiles, V52
from policy52 import witnesses, readable

GROUPS = [(0,0), (1,0), (2,0), (2,1), (3,0), (3,1)]


def bins(weak, ref):
    if weak.shape != ref.shape or weak.shape[-1] != 3:
        raise ValueError("Three weak/reference pairs are required")
    active = readable(weak, ref)
    same = (active & ((weak > 0) == (ref > 0))).sum(-1)
    n = active.sum(-1)
    minority = np.minimum(same, n - same)
    group = np.full(n.shape, -1, dtype=int)
    for j,(count,minor) in enumerate(GROUPS):
        group[(n == count) & (minority == minor)] = j
    assert np.all(group >= 0)
    return group


def read_panel(i):
    name = f"panel_{i:02d}.npz"
    with np.load(V53/'decisions53'/name) as data:
        methods=list(data['methods'].astype(str))
        weights=data['alpha'][..., [methods.index(m) for m in ('reference4_joint','impact_joint')]]
    with np.load(V52/'risk52'/name) as data:
        exact = data['exact'][0]
        realized = data['realized']
    risk = quadratic(exact[...,None,None,:], weights, False)
    actual = quadratic(realized[...,None,:], weights, False)
    group = np.empty((2,5,4,R), dtype=int)
    for event, old in np.ndindex(2,5):
        new = 4-old if event else old
        _,_,w,f=witnesses(i,old,new,3)
        group[event,old]=bins(w,f)
    return risk,actual,group


def main():
    assert os.environ.get("SLURM_JOB_ID")
    ex, raw, group = [np.stack(arr) for arr in zip(*(read_panel(i) for i in range(N)))]
    weight = np.array([1-G.mean(),G.mean()])[None,None,:,None,None,None]
    g = np.broadcast_to(group[:,None],ex.shape[:-1])
    delta = ex[...,1] - ex[...,0]
    masks = np.zeros_like(g, dtype=bool)
    splits=[]
    for train,test in [(slice(0,6),slice(6,12)),(slice(6,12),slice(0,6))]:
        bin_deltas = []
        for j in range(len(GROUPS)):
            w = weight * (g[train]==j)
            bin_deltas.append(float((delta[train]*w).sum()/w.sum()) if w.sum() else
                              float((delta[train]*weight).mean()*2))
        rule = np.array(bin_deltas)<0
        masks[test] = rule[g[test]]
        splits.append(dict(impact_rule=rule.tolist(), training_mean_deltas=bin_deltas,
                           test_catalogs=list(range(N))[test]))
    chosen = np.where(masks,ex[...,1],ex[...,0])
    chosen_raw = np.where(masks,raw[...,1],raw[...,0])
    per_catalog=(chosen*weight).mean((1,2,3,4,5))*2
    fixed=(ex*weight[...,None]).mean((1,2,3,4,5))*2
    rate=(masks*weight).mean((1,2,3,4,5))*2
    q=quantiles(np.concatenate([chosen_raw[...,None],raw],-1),G.mean())
    for row in splits:
        i=row['test_catalogs']
        row['heldout_mean']=float(per_catalog[i].mean())
        row['heldout_gain_vs_fixed']=(fixed[i]-per_catalog[i,None]).mean(0).tolist()
        row['impact_fraction']=float(rate[i].mean())
    result=dict(groups=GROUPS, splits=splits, mean=per_catalog.mean(),
                mean_fixed=fixed.mean(0).tolist(), p90_rule_reference_impact=q.tolist(),
                impact_fraction=rate.mean(),catalog_gain_vs_fixed=(fixed-per_catalog[:,None]).tolist())
    (ROOT/'sign_bins54.json').write_text(json.dumps(result, indent=2))
    print(result['mean'],q[:3],result['impact_fraction'],flush=True)
if __name__ == '__main__':
    main()
