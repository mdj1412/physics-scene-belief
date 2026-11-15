"""Exploratory transfer of retrospectively optimized allocations across held-out world halves."""
import itertools
import json
from pathlib import Path

import numpy as np
from analyze30 import LAWS, RESULTS, quota, selected


def main():
    assignments = np.array(sorted(set(itertools.permutations([0]*4 + [1]*2 + [2]*2))))
    assert len(assignments) == 420
    rows=[]
    for law in LAWS:
        blocks=[]
        for file in sorted((RESULTS/law).glob('block_*/metrics.npz')):
            with np.load(file) as z:
                # Keep the jointly determined 13/128 observation quota decisions intact.
                potentials=selected(z['cost'],quota(z['gap']))[:,:,0]
                halves=[potentials[:,offset::2].mean(1) for offset in (0,1)]
            all_cost=np.stack([h[np.arange(8),assignments].mean(1) for h in halves])
            best=all_cost.argmin(1)
            hindsight=all_cost.mean(0).min()
            replay=(all_cost[1,best[0]]+all_cost[0,best[1]])*.5
            fixed=potentials[:,:,1].mean()
            blocks.append([fixed,hindsight,replay,replay-fixed])
        assert len(blocks)==16
        rows.append(blocks)
    v=np.array(rows)
    rng=np.random.default_rng(3058361)
    boots=np.stack([a[rng.integers(16,size=(10000,16))].mean(1) for a in v]).mean(0)
    result=dict(methods=['fixed4', 'optimistic full-batch hindsight','opposite-half replay','replay minus fixed4'],
                pooled=v.mean((0,1)).tolist(),bylaw=v.mean(1).tolist(),
                intervals=np.quantile(boots,[.025,.975],axis=0).T.tolist(),
                note='All 420 allocations preserve counts. Each half uses full-batch factual-only quota assignments established before labels. Calibration probes and calibration futures are shared. No deployable oracle or independently estimated upper bound is claimed.')
    (RESULTS/'half_replay.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    main()
