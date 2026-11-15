"""Replay changes to a fixed 13-of-128 observation quota using common world costs."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
LAWS = ('coulomb', 'split_static', 'speed_weak')


def block_summary(before, after, cost, gain):
    if before.shape != (2, 8, 4, 128) or after.shape != (2, 8, 4, 4, 128):
        raise ValueError('unexpected stage/environment/future/batch/world axes')
    old = (before[:, :, None] == 3)
    new = (after == 3)
    assert np.all(old.sum(-1) == 13) and np.all(new.sum(-1) == 13)
    # Brier advantage for probe over query. Risk reduction has positive sign.
    advantage = (cost[..., 0] - cost[..., 1])[None, :, None]
    enter = (~old & new)
    leave = (old & ~new)
    n = enter.sum(-1)
    assert np.array_equal(n, leave.sum(-1))
    a = (enter * advantage).mean(-1)
    b = -(leave * advantage).mean(-1)
    np.testing.assert_allclose(a+b, gain, atol=1.e-12, rtol=0)
    # Difference in selected probe worlds due to changing the future only.
    different = np.stack([(new[:, :, i] != new[:, :, j]).sum(-1) / 2
                        for i, j in combinations(range(4),2)],axis=2)
    def avg(x):
        return x.mean(tuple(range(1, x.ndim)))
    return np.stack([avg(n), avg(n == 0), avg(a), avg(b), avg(gain < -1e-14),
                     avg(gain > 1.e-14), avg(different), avg(different == 0)], axis=1)


def main():
    from argparse import ArgumentParser
    parser=ArgumentParser()
    parser.add_argument('panel',choices=['results','results_pool'])
    args=parser.parse_args()
    panel=ROOT/args.panel
    blocks=[]
    for law in LAWS:
        rows=[]
        for k in range(6):
            key=Path(law)/f'block_{k:03d}'
            with (np.load(ROOT/'results'/key/'choices.npz') as x,
                  np.load(ROOT/'results'/key/'metrics.npz') as y,
                  np.load(panel/key/'choices.npz') as nx,
                  np.load(panel/key/'metrics.npz') as ny):
                rows.append(block_summary(x['action_before'],nx['action_after'],y['cost'],ny['gain']))
        blocks.append(rows)
    arr=np.array(blocks)
    obj=dict(metrics=['n_exchanged','no_switch_rate','gain_from_query_to_probe','gain_from_probe_to_query',
                      'negative_gain_fraction','positive_gain_fraction','n_exchanged_two_futures',
                      'identical_world_quota_two_futures'], pooled=arr.mean((0,1)).tolist(), by_law=arr.mean(1).tolist(),
             interpretation='Exact action replay. Negative/positive rates use zero tolerance 1e-14. Law-weighted stage summaries; descriptive.')
    (panel/'swap31.json').write_text(json.dumps(obj,indent=2,allow_nan=False))
    print(obj['pooled'])

if __name__=='__main__':
    main()
