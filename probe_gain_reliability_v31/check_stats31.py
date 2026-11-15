"""Deterministic fixture tests for the estimands; no simulator output needed."""
import numpy as np
from stats31 import block_metrics, ranking_and_repeat_noise, variance_components


def main():
    # Add-only rows and columns, centered, with different environment offsets.
    rng = np.random.default_rng(3109416)
    r=rng.normal(size=(2,8,4,1))
    b=rng.normal(size=(2,8,1,4))
    base=rng.normal(size=(2,8,1,1))
    gain=base+r+b
    c=variance_components(gain)
    np.testing.assert_allclose(c[...,0],r[...,0].var(axis=-1,ddof=1),rtol=1e-12)
    np.testing.assert_allclose(c[...,1],b[...,0,:].var(axis=-1,ddof=1),rtol=1e-12)
    np.testing.assert_allclose(c[...,2],0,atol=1e-12)
    block_metrics(gain)
    # No hidden convention puts ties into the success numerator.
    const=np.zeros((8,4,4))
    ordered=const+np.arange(8)[:,None,None]
    assert ranking_and_repeat_noise(const,"both_new_1x1")[0]==.5
    assert ranking_and_repeat_noise(ordered,"both_new_1x1")[0]==1.
    # Future-only noise, no between-environment effects: sharing the future
    # gives perfect agreement, independent futures give chance in expectation.
    x=rng.normal(size=(8,4,1))*np.ones((1,1,4))
    assert ranking_and_repeat_noise(x,"same_future_new_batch")[0]==1.
    block_metrics(np.stack([x, x.transpose(0,2,1)]))
    # Swapping axes exchanges their variance and the corresponding rank scores.
    a=rng.normal(size=(2,8,4,4))
    m,n=block_metrics(a),block_metrics(a.swapaxes(-2,-1))
    np.testing.assert_allclose(m['components'][:,[1,0,2]],n['components'])
    np.testing.assert_allclose(m['ranks'][:,[1,0,2,4,3,5]],n['ranks'])
    # Exact concordance must be invariant to arbitrary axis labels per environment.
    z=np.stack([[v[rng.permutation(4)][:,rng.permutation(4)] for v in s] for s in a])
    np.testing.assert_allclose(m['ranks'],block_metrics(z)['ranks'])
    print('PASS additive recovery, tie/rank extremes, direct crossed differences, axis swap, replicate relabeling.')
if __name__=='__main__':
    main()
