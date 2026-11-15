"""Block-paired inference for v38, preserving geometry/law cell and starting sign."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
CELLS = ('shoulder_weakening', 'shoulder_strengthening', 'bead_weakening', 'bead_strengthening')
REGIMES = ('strong', 'weak', 'null')
METHODS = ('fixed', 'binary', 'ternary', 'continuous')
NBOOT = 16000


def samples():
    rng = np.random.default_rng(381115)
    return (2 * rng.integers(4, size=(NBOOT, 4, 4, 2)) + np.arange(2)).reshape(NBOOT, 4, 8)


def draw(x, idx):
    return x[np.arange(4)[None, :, None], idx].mean(axis=(1, 2))


def limits(a):
    return np.moveaxis(np.quantile(a, [.025, .975], axis=0), 0, -1)


def simultaneous(estimate, reps):
    sd = reps.std(axis=0, ddof=1)
    mx = (abs(reps - estimate) / np.maximum(sd, 1.e-14)).max(axis=1)
    radius = np.quantile(mx, .95) * sd
    return np.stack((estimate - radius, estimate + radius), axis=-1)


def summarize(loss, idx):
    mu = loss.mean(axis=(0, 1))
    samp = draw(loss, idx)
    gain = mu[0] - mu[1]
    comparisons = [(2, 0), (2, 1), (2, 3), (3, 0), (1, 0)]
    contrasts = np.stack([loss[:, :, 1, x] - loss[:, :, 1, y] for x,y in comparisons], axis=-1)
    diff = contrasts.mean((0, 1))
    rep = draw(contrasts, idx)
    return dict(loss=mu,loss_ci=limits(samp),gain=gain,gain_ci=limits(samp[:,0]-samp[:,1]),
                comparisons=comparisons,diff=diff,diff_ci=limits(rep),primary_sim95=simultaneous(diff[:2],rep[:,:2]))


def timing_summary(ms):
    # Eight timing clusters, two per cell. Resample paired blocks with their three calls intact.
    q = np.quantile(ms,.9,axis=(0,1,2))
    rng = np.random.default_rng(381116)
    indices = rng.integers(2, size=(NBOOT,4,2))
    trials = ms[np.arange(4)[None,:,None],indices]
    qb = np.quantile(trials,.9,axis=(1,2,3))
    ratio = q[2]/q[[0,1]]
    qr = qb[:,2,None]/qb[:,[0,1]]
    leave = np.stack([np.quantile(np.delete(ms.reshape(8,3,4), i, axis=0),.9,axis=(0,1))
                      for i in range(8)])
    return dict(ms=q,ms_ci=limits(qb),ternary_ratios=ratio,ratio_sim95=simultaneous(ratio,qr),
                leave_one_cluster_p90_ratio_range=np.stack(((leave[:,2,None]/leave[:,[0,1]]).min(0),
                                                           (leave[:,2,None]/leave[:,[0,1]]).max(0)),axis=-1))


def main():
    out = ROOT/'results'
    meta=json.loads((out/'completion.json').read_text())
    assert meta['blocks'] == 32
    loss = np.empty((4,8,3,2,2,4))
    counts = np.empty((4,8,3,2,2,3))
    slip = np.empty((4,8,3,2))
    timing={}
    for c,cell in enumerate(CELLS):
        for b in range(8):
            folder=out/cell/f'block_{b:03d}'
            for g,regime in enumerate(REGIMES):
                for s,sign in enumerate(('plus','minus')):
                    with np.load(folder/f'{regime}_{sign}.npz') as f:
                        np.testing.assert_array_equal(f['methods'],METHODS)
                        loss[c,b,g,s]=f['risk'].mean(-1)
                        slip[c,b,g,s]=1-f['fourth_stick'].mean()
                        w=f['weights'][:,2]
                        counts[c,b,g,s]=np.stack([(w==v).mean(1) for v in (0.,.325,1.)],axis=-1)
                        if len(f['full_ms']):
                            assert (c,b) not in timing
                            timing[c,b]=f['full_ms']
    assert len(timing)==8 and np.isfinite(loss).all() and np.isfinite(slip).all()
    times=np.asarray([[timing[c,b] for b in range(2)] for c in range(4)])
    idx=samples()
    data=dict(methods=METHODS,cells=CELLS,regimes=REGIMES,
              primary=summarize(loss.mean((2,3)),idx),cell_loss=loss.mean((1,2,3)),
              cell_gain=(loss[:,:,:,:,0]-loss[:,:,:,:,1]).mean((1,2,3)),
              by_regime={name:summarize(loss[:,:,g].mean(2),idx) for g,name in enumerate(REGIMES)},
              ternary_frequencies=counts.mean((0,1,2,3)),frequency_by_regime=counts.mean((0,1,3)),
              fourth_slip=slip.mean((0,1,3)),timing=timing_summary(times))
    (ROOT/'policy_stats38.json').write_text(json.dumps(data,indent=2,default=lambda x:x.tolist(),allow_nan=False)+'\n')


if __name__=='__main__':
    main()
