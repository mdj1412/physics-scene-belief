"""Enumerate the actions reachable at the same paid readout count, independently of scoring."""
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "sensor_reset_v59"))
from route59 import log_odds, SOURCE_LOG_CUTOFF
from score59 import MIX

NAMES = ["count_choice", "action_choice", "random_fixed", "confidence_fixed", "oracle_fixed",
         "gate_Brier", "random_Brier", "confidence_Brier", "oracle_Brier",
         "confidence_minus_random", "confidence_minus_oracle", "random_minus_oracle"]
def symbols(code):
    return np.stack([code // 13, code % 13], axis=-1).reshape(*code.shape[:-1], 8)

def route(s):
    paired = (13 * s[:, ::2] + s[:, 1::2]).astype(np.uint8)
    return log_odds(paired, 1, True)[0] > SOURCE_LOG_CUTOFF

def reachable(old, selective):
    valid, keep = old > 4, selective > 4
    blank = np.where(old > 0, 1 + (old.astype(int) - 1) % 4, 0).astype(np.uint8)
    np.testing.assert_array_equal(np.where(keep, old, blank), selective)
    n, k = valid.sum(1), keep.sum(1)
    selected = route(selective)
    prob = selected.astype(float)
    flexible = (k > 0) & (k < n)
    for ki in np.unique(k[flexible]):
        group = np.flatnonzero(flexible & (k == ki))
        ways = np.zeros(len(group), dtype=int)
        fixed = np.zeros(len(group), dtype=int)
        for subset in itertools.combinations(range(8), int(ki)):
            cols = list(subset)
            rows = valid[group][:, cols].all(1)
            if not rows.any():
                continue
            idx = group[rows]
            one = blank[idx].copy()
            one[:, cols] = old[np.ix_(idx, cols)]
            fixed[rows] += route(one)
            ways[rows] += 1
        assert np.array_equal(ways, [math.comb(int(ni), int(ki)) for ni in n[group]])
        prob[group] = fixed / ways
    return selected, prob, flexible

def one_catalog(c):
    with np.load(ROOT / 'pilots' / f'panel_{c:02d}.npz') as f:
        old, kept = (symbols(f['codes'][:, mode, 1]) for mode in (1, 2))
    shape = old.shape[:-1]
    sel, rnd, choice = [v.reshape(shape) for v in reachable(old.reshape(-1, 8),kept.reshape(-1, 8))]
    possible = (rnd > 0) & (rnd < 1)
    assert np.all(possible <= choice)
    assert np.array_equal(rnd[~possible], sel[~possible])
    material = ((np.arange(4) + c) % 4) // 2
    with np.load(ROOT/'scored63'/f'panel_{c:02d}.npz') as f:
        cell = np.stack([f['exact_cells'][:, :, material == j, :2].mean((1,2)) for j in (0,1)])
    bank = np.einsum('mea,ze->mza', cell, MIX)
    accum = np.zeros((4,8,len(NAMES)))
    for cause in (0,1):
        for regime in (0,1):
            potential = range(2) if cause else [regime]
            for hidden in potential:
                for m in (0,1):
                    i = (slice(None), cause, hidden, m)
                    gate, fixed = bank[m,regime]
                    delta = fixed - gate
                    p,s = rnd[i],sel[i]
                    o = np.where(possible[i], float(delta < 0), s)
                    values = np.stack([choice[i], possible[i], p, s, o,
                        np.full_like(p,gate), gate+p*delta, gate+s*delta, gate+o*delta,
                        (s-p)*delta,(s-o)*delta,(p-o)*delta], axis=-1)
                    assert values[...,10:].min() > -1e-15
                    accum += values.reshape(4,-1,8,len(NAMES)).mean(1) / (8*len(potential))
    return accum

def main():
    assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    arr = np.stack([one_catalog(c) for c in range(24)])
    rng = np.random.default_rng(630771)
    cw = rng.multinomial(24, np.full(24,1/24), size=6000)/24
    bw = rng.multinomial(8, np.full(8,1/8), size=6000)/8
    def summarize(grid):
        boot = np.einsum('tc,tb,cbm->tm',cw,bw,grid,optimize=True)
        out = np.vstack([grid.mean((0,1)),np.quantile(boot,[.025,.975],axis=0)])
        return dict(zip(NAMES,out.T.tolist()))
    result = {'pooled':summarize(arr.mean(1)), 'cells':[summarize(arr[:,i]) for i in range(4)],
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'interval':'pointwise paired crossed catalog×board bootstrap; privileged diagnostic'}
    (ROOT/'headroom63.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/'headroom63_cells.npz',metrics=NAMES,catalog_cell_board=arr)
if __name__=='__main__':
    main()
