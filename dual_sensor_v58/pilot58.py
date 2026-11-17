"""Paired factorial, order-randomized, dual instrument calibration pilot."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
N_CATALOG = 24
N_EPISODE = 2048
Q = .29
H = .26582278481012656
P_SIGN_ERROR = .02
W = [[.42, 0, .58], [0, .84, .16]]
E = np.array([0., .25, .5])
AXES = (0, 1, 2)
METRICS = ['complete', 'same_double', 'cross_double', 'cross01', 'cross10', 'q_first',
           'q_second', 'true_double', 'order0_n', 'order0_same', 'order0_cross',
           'order1_n', 'order1_same', 'order1_cross', 'n_no_complete', 'active', 'mismatch']

def counts(obs, active, truth, swapped):
    complete = active.all(axis=-1)
    first, second = obs[..., 0, :], obs[..., 1, :]
    same = (first & second).mean(axis=-1)
    cross = (first & second[..., ::-1]).mean(axis=-1)
    def total(x):
        return np.sum(np.where(complete, x, 0.))
    return np.array([complete.sum(), total(same), total(cross), total(first[...,0] & second[...,1]),
                     total(first[...,1] & second[...,0]), total(first.mean(-1)), total(second.mean(-1)),
                     total(truth.all(-1)), total(~swapped), total(same * ~swapped), total(cross * ~swapped),
                     total(swapped), total(same * swapped), total(cross * swapped),
                     (~complete.any(-1)).sum(), active.sum(), (obs & active[...,None]).sum()/2])

def make(i):
    rng = np.random.default_rng([58044271, i])
    shape = (2, N_EPISODE, 4, 2)
    swap = rng.random(shape[:-1]) < .5
    # These sacrificial identities, active states, directions and individual flip coins
    # are drawn before the independent randomized assignment to chronological slot.
    active_id = rng.random(shape) < .5
    negative_id = rng.random(shape) < .5
    coin_id = rng.random(shape)
    def chronological(values):
        return np.where(swap[...,None], values[...,::-1], values)
    active, negative, flip = map(chronological, (active_id, negative_id, coin_id))
    session_draw = rng.random(shape)
    latch_draw = rng.random(shape + (2,))
    nuisance_draw = rng.random(shape + (2,)) < Q
    err_r = rng.random(shape + (2,)) < P_SIGN_ERROR
    err_i = rng.random(shape + (2,)) < P_SIGN_ERROR
    ref = negative[...,None] ^ err_r
    grid = (2, 2, 2, 2)
    codes = np.empty(grid + shape[:3], dtype='u1')
    raw = np.empty(grid + shape, dtype='u1')
    counters = np.empty((2,) + grid + (len(METRICS),))
    for l in range(2):
        def session(values):
            return values if l == 0 else np.repeat(values[:,:,:,:1,:], 2, axis=3)
        z = session(nuisance_draw)
        for rho in range(2):
            env = session_draw if not (l and rho) else np.repeat(session_draw[...,:1],2,axis=-1)
            for g in range(2):
                e = E[np.searchsorted(np.cumsum(W[g]), env, side='right')]
                truth = flip < e
                for hi, hazard in enumerate([0., H]):
                    latch = session(latch_draw) < hazard
                    pre_error = np.where(latch, z, truth[...,None])
                    impact = negative[...,None] ^ pre_error ^ err_i
                    obs = ref ^ impact
                    s = np.where(active, 1 + obs[...,0].astype('u1') + 2*obs[...,1].astype('u1'), 0)
                    idx = (l, rho, hi, g)
                    codes[idx] = 5*s[...,0]+s[...,1]
                    raw[idx] = np.where(active, 16+ref[...,0].astype('u1')+2*ref[...,1].astype('u1')
                                        +4*impact[...,0].astype('u1')+8*impact[...,1].astype('u1'), 0)
                    counters[(1,)+idx] = counts(obs,active,truth,swap)
                    counters[(0,)+idx] = counts(pre_error,active,truth,swap)
    p = ROOT/'pilots'/f'panel_{i:02}.npz'
    audit = ROOT/'pilot_audit'/p.name
    p.parent.mkdir(exist_ok=True)
    audit.parent.mkdir(exist_ok=True)
    np.savez_compressed(p, codes=codes)
    np.savez_compressed(audit, raw_signed=raw, swap=swap, counts=counters, metrics=METRICS,
                        specimen_first=(~swap).sum(), n_coin=swap.size)
    # Recompute every code exclusively from the two signed R and I readouts.
    active_raw = raw >= 16
    b0 = (raw & 1) ^ ((raw >> 2) & 1)
    b1 = ((raw >> 1) & 1) ^ ((raw >> 3) & 1)
    s = np.where(active_raw, 1+b0+2*b1,0)
    np.testing.assert_array_equal(codes,5*s[...,0]+s[...,1])
    np.testing.assert_array_equal(codes[0,0],codes[0,1])
    np.testing.assert_array_equal(codes[0,0,0],codes[1,0,0])
    return counters

def summary():
    data=[]; files=[]; swap_total=first_total=0
    for i in range(N_CATALOG):
        name=f'panel_{i:02}.npz'
        with np.load(ROOT/'pilot_audit'/name) as a:
            data.append(a['counts'])
            first_total+=int(a['specimen_first']); swap_total+=int(a['n_coin'])
        files.append({k: hashlib.sha256((ROOT/k/name).read_bytes()).hexdigest() for k in ('pilots','pilot_audit')})
    data=np.stack(data)
    rng=np.random.default_rng(58077401)
    freq=rng.multinomial(N_CATALOG,np.ones(N_CATALOG)/N_CATALOG,size=4000)
    fields={k:j for j,k in enumerate(METRICS)}
    rows={}
    for key in np.ndindex(2,2,2,2,2):
        noisy,l,rho,hi,g=key
        a=data[(slice(None),)+key]
        avg=a.sum(axis=0)
        b=freq@a
        def col(k): return fields[k]
        def rate(a,k): return a[..., col(k)]/a[...,col('complete')]
        def cov(a,k): return rate(a,k)-rate(a,'q_first')*rate(a,'q_second')
        def interval(x): return np.quantile(x,[.025,.975]).tolist()
        hazard=[0.,H][hi]
        err=P_SIGN_ERROR if noisy else 0.
        d=2*err*(1-err)
        scale=(1-2*d)**2
        variance=[.145-Q**2,.0925-Q**2][g]
        marginal=d+(1-2*d)*Q
        phy=l*rho*variance
        within=l*hazard
        predicted=[marginal**2+scale*((1-within)*phy+within*Q*(1-Q)),
                   marginal**2+scale*(1-within)**2*phy]
        rows[','.join(map(str,key))] = {
            'observed_same_cross':[rate(avg,k) for k in ('same_double','cross_double')],
            'analytic_same_cross':predicted, 'analytic_q':marginal,
            'sample_q':avg[col('mismatch')]/avg[col('active')],
            'cross_cov':[cov(avg,'cross_double'),*interval(cov(b,'cross_double'))],
            'same_cov':[cov(avg,'same_double'),*interval(cov(b,'same_double'))],
            'same_minus_cross':rate(avg,'same_double')-rate(avg,'cross_double'),
            'same_minus_cross_ci':interval(rate(b,'same_double')-rate(b,'cross_double')),
            'order_cross_delta':avg[col('order1_cross')]/avg[col('order1_n')]-avg[col('order0_cross')]/avg[col('order0_n')],
            'order_cross_ci':interval(b[:,col('order1_cross')]/b[:,col('order1_n')]-b[:,col('order0_cross')]/b[:,col('order0_n')]),
            'cross01_cross10':[rate(avg,k) for k in ('cross01','cross10')],
            'true_double':rate(avg,'true_double'),'complete_groups':int(avg[col('complete')]),
            'no_complete_episode':int(avg[col('n_no_complete')])
        }
    # A shared instrument fault can exactly reproduce the pair distribution.
    c=(.145-Q**2)/(Q*(1-Q))
    p11=(1-c)*Q**2+c*Q
    np.testing.assert_allclose(p11,.145,rtol=0.,atol=1e-14)
    pair=[1-2*Q+p11, Q-p11, Q-p11,p11]
    def observation_law(p):
        d=2*P_SIGN_ERROR*(1-P_SIGN_ERROR)
        emit=np.zeros((2,5))
        for x in range(2):
            emit[x,0]=.5
            for symbol in range(1,5):
                bits=[(symbol-1)%2,(symbol-1)//2]
                emit[x,symbol]=.5*np.prod([1-d if x==y else d for y in bits])
        return emit.T @ np.reshape(p,(2,2)) @ emit
    physical=[1-2*Q+.145,Q-.145,Q-.145,.145]
    max_difference=float(np.abs(observation_law(pair)-observation_law(physical)).max())
    results=dict(catalogs=N_CATALOG,materials=N_CATALOG*2,episodes_per_cell=N_CATALOG*2*N_EPISODE,
                 p_first_original=first_total/swap_total,active_sign_error=P_SIGN_ERROR,
                 H=H,order_coin_count=swap_total,rows=rows,hashes=files,
                 common_mode=dict(c=c,physical_pair_distribution=physical,
                                  shared_fault_distribution=pair,max_25_symbol_difference=max_difference))
    (ROOT/'pilot_results58.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for i in range(N_CATALOG):
        make(i)
        print(i,flush=True)
    summary()
