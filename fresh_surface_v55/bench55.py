"""Paired resident-model timing on fresh physical and post-swap sessions."""
from itertools import permutations
from pathlib import Path
import json
import os
import socket
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'adaptive_witness_v54'))
from bench54 import mixed53, prev, terminal
from policy54 import select54
import generate55 as fresh

N = 24
METHODS = ('source_risk_gate', 'fixed_3RI', 'independent_quota_random')


def session_reads(i):
    d = fresh.raw55(i)
    env = np.arange(4)
    pre = (i + env) % 5
    drift = (i + env) % 2 == 1
    post = np.where(drift, 4 - pre, pre)
    eps = (i + env) % 3
    repeat = np.random.default_rng([550929, i]).integers(d['nw'].shape[2], size=4)
    raw = (d['ow'][pre, env, repeat], d['of'][pre, env, repeat],
           d['nw'][post, env, repeat], d['nf'][post, env, repeat],
           d['imp'][eps, post, env, repeat], None)
    assert all(a.shape == (4, 4) for a in raw[:4]) and raw[4].shape == (4,)
    return raw, np.stack((pre, post, eps, repeat))


def lottery_values(i, row):
    return np.random.default_rng([551319, i, row+1]).random(4)


def threshold(row, quota):
    values = np.concatenate([lottery_values(i, row) for i in range(N)])
    assert np.unique(values).size == 4*N
    if quota == 0:
        return 0.
    if quota == 4*N:
        return 1.
    values.sort()
    return (values[quota] + values[quota-1])/2


def choose(m, raw, i, row, cut):
    if m == 0:
        return select54(raw[2][:, :3], raw[3][:, :3], mode='robust')
    if m == 1:
        return np.ones(4, dtype=bool)
    return lottery_values(i, row) < cut


def serial_alpha(raw):
    """Both single-session endpoint weights, used only for post-timing verification."""
    rows = []
    for use_i in (False, True):
        n = 3 if use_i else 4
        values = []
        for j in range(4):
            part = tuple(x[j:j+1] for x in raw[:5])
            ow, of, nw, nf, imp = part
            sign = imp if use_i else np.zeros_like(imp)
            d = mixed53(ow, of, nw[:, :n], nf[:, :n], nw[:, 3], sign)
            values.append(float(d['alpha_joint'][0]))
        rows.append(values)
    return np.asarray(rows)


def measure(i, pair, cuts, engine, templates, signed):
    raw, scenario = pair
    measured = fresh.measured55(fresh.pop55(i), i)
    cache_path = ROOT/'forecasts'/f'panel_{i:02d}.npz'
    with np.load(cache_path) as cached:
        plus, minus, actions = [cached[key] for key in ('pplus', 'pminus', 'actions')]
    assert plus.shape == minus.shape == actions.shape == (4, 128)
    endpoints = serial_alpha(raw)
    # Excluded measurements are poisoned in both branch types.
    mixed = np.array([True, False, True, False])
    bad = [a.copy() if a is not None else None for a in raw]
    bad[3][mixed, 3] = np.nan
    bad[4][~mixed] = 99
    np.testing.assert_allclose(terminal(bad,mixed), endpoints[mixed.astype(int),np.arange(4)], atol=2e-12,rtol=0)
    rng = np.random.default_rng([551582, i])
    orders = rng.permutation(np.array(list(permutations(range(3)))))
    clock = np.empty((6, 3))
    routes = np.empty((6,3,4), dtype=bool)
    host = np.empty((51,3))
    max_error = np.zeros(3)
    for row, seq in enumerate([rng.permutation(3), *orders], start=-1):
        for m in seq:
            torch.cuda.synchronize()
            begin = time.perf_counter_ns()
            mask = choose(m, raw, i, row, cuts[row+1])
            alpha = terminal(raw, mask)
            pplus, pminus, sel = fresh.forward55(measured, engine, templates, signed)
            pred = alpha[:,None]*pplus + (1-alpha[:,None])*pminus
            torch.cuda.synchronize()
            duration = (time.perf_counter_ns()-begin)/1e6
            if row >= 0:
                clock[row,m] = duration
                routes[row,m] = mask
            weight = endpoints[mask.astype(int),np.arange(4)]
            np.testing.assert_allclose(alpha,weight,atol=2e-12,rtol=0)
            np.testing.assert_array_equal(sel,actions)
            expected = weight[:,None]*plus + (1-weight[:,None])*minus
            np.testing.assert_allclose(pred,expected,atol=2e-9,rtol=0)
            max_error[m] = max(max_error[m], np.max(np.abs(pred-expected)))
    for j in range(51):
        for m in rng.permutation(3):
            t = time.perf_counter_ns()
            choose(m, raw, i, j % 6, cuts[1+j%6])
            host[j,m] = (time.perf_counter_ns()-t)/1e6
    target = ROOT/'timing55'/f'panel_{i:02d}.npz'
    prev.r.v38.atomic_npz(target, ms=clock, routes=routes, host_ms=host, orders=orders,
                          scenario=scenario, maxerr=max_error, methods=METHODS)
    print('catalog', i, 'p90', np.quantile(clock,.9,axis=0), flush=True)
    return clock,routes,host,max_error


def main():
    assert torch.cuda.device_count() == 1 and os.getenv('SLURM_JOB_ID')
    torch.set_num_threads(1)
    (ROOT/'timing55').mkdir(exist_ok=True)
    start = time.time()
    setup = time.perf_counter_ns()
    pairs = [session_reads(i) for i in range(N)]
    quota = sum(np.count_nonzero(choose(0,pair[0],i,0,0.)) for i,pair in enumerate(pairs))
    cuts = [threshold(j,quota) for j in range(-1,6)]
    setup_ms = (time.perf_counter_ns()-setup)/1e6
    before = prev.snapshot()
    with torch.inference_mode():
        fresh.install()
        engine,templates = prev.engine50()
        signed,_ = prev.coefficients()
        rows = [measure(i,pair,cuts,engine,templates,signed) for i,pair in enumerate(pairs)]
    after = prev.snapshot()
    ms, routes, host, err = [np.stack(v) for v in zip(*rows)]
    counts = routes.sum(axis=(0,3))
    np.testing.assert_array_equal(counts,np.tile([quota,4*N,quota],(6,1)))
    p90 = np.quantile(ms,.9,axis=(0,1))
    delta = p90[0]-p90[1:]
    ix = np.random.default_rng(551622).integers(N,size=(16000,N))
    b = np.quantile(ms[ix],.9,axis=(1,2))
    ci = np.quantile(b[:,[0]]-b[:,1:],[.0125,.9875],axis=0).T
    data = dict(methods=METHODS,p90_ms=p90.tolist(),gate_delta_ms=delta.tolist(),
                simultaneous95_ci=ci.tolist(),marginal95_ci=np.quantile(b,[.025,.975],axis=0).T.tolist(),
                host_gate_only_p90_ms=np.quantile(host,.9,axis=(0,1)).tolist(),
                independent_lottery_count_per_repeat=counts.tolist(),rate_matched_fixtures=4*N,
                fixture_quota=int(quota),label_blind_quota_setup_including_read_io_ms=setup_ms,
                catalogs=N,session_size=128,batch=512,per_arm=int(ms.shape[0]*ms.shape[1]),
                max_replay_error=err.max(0).tolist(),job=os.getenv('SLURM_JOB_ID'),node=socket.gethostname(),
                gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
                before=before,after=after,utc_start=start,utc_end=time.time())
    prev.r.v38.atomic_json(ROOT/'timing55.json',data)
    print(json.dumps(data),flush=True)


if __name__ == '__main__':
    main()
