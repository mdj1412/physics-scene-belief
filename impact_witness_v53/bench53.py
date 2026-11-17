"""Session-cold, resident-model deployment timing on one guarded GPU."""
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
V50 = ROOT.parent / 'polarity_calibration_v50'
V52 = ROOT.parent / 'postswap_adaptation_v52'
sys.path.insert(0, str(V52))
from bench52 import coefficients, engine50, observe, r, routed_predictions, snapshot, world, world50
from policy52 import decide as reference_selector
from policy53 import choice53, load_model, witnesses53

METHODS = ('impact_joint', 'impact_selector', 'impact_vote_selector', 'impact_post_vote',
           'impact_joint_masked', 'reference4_joint', 'reference4_selector', 'reference4_vote_selector')
N = len(METHODS)


def actions(method, raw, model):
    ow, of, nw, nf, impact, noisy = raw
    if method in ('reference4_selector', 'reference4_vote_selector'):
        mode = 'likelihood' if method == 'reference4_selector' else 'vote'
        result = reference_selector(ow, of, nw, nf, mode, model)
        return result['action'], result['alpha']
    return choice53(method, ow, of, nw, nf, impact, noisy, model)


def scheduling(rng):
    # Each treatment occurs once at each position, and every directed transition once.
    first = np.array([0, 1, 7, 2, 6, 3, 5, 4])
    rows = (first[None, :] + np.arange(N)[:, None]) % N
    rows = rng.permutation(N)[rows[rng.permutation(N)]]
    np.testing.assert_array_equal(np.sort(rows, axis=0), np.broadcast_to(np.arange(N)[:, None], rows.shape))
    transitions = np.zeros((N, N), dtype=int)
    np.add.at(transitions, (rows[:, :-1], rows[:, 1:]), 1)
    np.testing.assert_array_equal(transitions, 1 - np.eye(N, dtype=int))
    return rows


def fixture(catalog, model):
    rng = np.random.default_rng(530442 + catalog)
    f, h = catalog % 4, (catalog // 4 + catalog // 2) % 2
    align, variant = (catalog // 4 + catalog) % 2, (catalog // 2 + catalog // 4) % 2
    family, height = world50.FAMILIES[f], world50.HEIGHTS[h]
    pop = world(catalog)
    xx, optics = r.apparatus(pop, family, height, r.VARIANTS[variant])
    optics = observe(optics, family, height, r.ALIGN[align], r.VARIANTS[variant])
    bought = [{key: row[key][:, :1] for key in (r.r.Q, r.r.P)} for row in optics]
    env, obj = np.indices((4, 128))
    state = rng.permutation(np.repeat(np.arange(3), (256, 128, 128))).reshape(4, 128)
    disp = np.stack([world50.readings(pop, family, height, s)[..., 0] for s in world50.STATES])[state, env, obj]
    # These indices are used exclusively for sampling fixture measurements.
    pre = (np.arange(4) + catalog) % 5
    event = (np.arange(4) + catalog) % 2
    post = np.where(event, 4 - pre, pre)
    err = (np.arange(4) + catalog) % 3
    repeat = rng.integers(256, size=4)
    sampled = []
    for j in range(4):
        obs = witnesses53(catalog, int(pre[j]), int(post[j]), int(err[j]))
        sampled.append(tuple(a[j, repeat[j]] for a in obs))
    raw = tuple(np.stack(x) for x in zip(*sampled))
    assert all(x.shape == (4, 4) for x in raw[:4]) and all(x.shape == (4,) for x in raw[4:])
    with np.load(V50 / 'forecast50' / f'panel_{catalog:02d}.npz') as cache:
        assert tuple(cache['methods'].astype(str)) == ('original', 'local_plus', 'local_minus', 'fixed_plus', 'fixed_minus', 'local_unsigned_fit')
        p = cache['p'][:, f, h, :, align, variant, ..., 0]
        plus, minus, blind = [p[i, state, env, obj] for i in (1, 2, 5)]
        selected = cache['selected'][f, h, variant]
    expected = []
    action_rows = []
    alpha_rows = []
    for method in METHODS:
        per_session = [actions(method, tuple(x[j:j+1] for x in raw), model) for j in range(4)]
        ac, alpha = [np.concatenate(x) for x in zip(*per_session)]
        if method.startswith('impact_'):
            # The replacement removes reference #4 from this acquisition design.
            impossible = list(raw)
            impossible[3] = raw[3].copy()
            impossible[3][:, -1] += 999.
            ac2, alpha2 = actions(method, impossible, model)
            np.testing.assert_array_equal(ac, ac2)
            np.testing.assert_allclose(alpha, alpha2, atol=1e-12, rtol=0)
        action_rows.append(ac)
        alpha_rows.append(alpha)
        expected.append(np.where(ac[:, None] == 2, blind, alpha[:, None]*plus + (1-alpha[:, None])*minus))
    return xx, bought, disp, h, raw, selected, expected, np.stack(action_rows), np.stack(alpha_rows), pre, post, err, repeat


def measure(catalog, engine, templates, signed, unsigned, model):
    xx, bought, disp, h, raw, selected, expected, action, alpha, pre, post, err, repeat = fixture(catalog, model)
    rng = np.random.default_rng(530821 + catalog)
    orders = scheduling(rng)
    elapsed = np.empty((len(orders), N))
    max_error = np.zeros(N)
    before = snapshot()
    for rep, order in enumerate([rng.permutation(N), *orders], start=-1):
        for m in order:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            ac, weight = actions(METHODS[m], raw, model)
            obs = r.infer.actions43(xx, engine)
            paid = r.infer.paid_sensors(bought, obs)
            p0 = r.infer.decode43(engine, templates[h], xx, obs, paid)[..., 0]
            arm = (obs == r.r.Q).astype(np.intp)
            pred = routed_predictions(p0, disp, arm, signed[1,h], unsigned[h], ac, weight)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            if rep >= 0:
                elapsed[rep, m] = (stop - start)*1e-6
            np.testing.assert_array_equal(ac, action[m])
            np.testing.assert_array_equal(obs, selected)
            np.testing.assert_allclose(pred, expected[m], atol=2e-9, rtol=0)
            max_error[m] = max(max_error[m], abs(pred - expected[m]).max())
    host = np.empty((101, N))
    for rep in range(-1, len(host)):
        for m in rng.permutation(N):
            start = time.perf_counter_ns()
            actions(METHODS[m], raw, model)
            if rep >= 0:
                host[rep, m] = (time.perf_counter_ns() - start)*1e-6
    path = ROOT / 'timing53' / f'panel_{catalog:02d}.npz'
    r.v38.atomic_npz(path, ms=elapsed, host_ms=host, method_order=orders, methods=np.asarray(METHODS),
                     max_replay_error=max_error, action=action, alpha=alpha, pre=pre, post=post,
                     impact_error_idx=err, repeat=repeat)
    r.v38.atomic_json(path.with_suffix('.json'), dict(before=before, after=snapshot()))
    print('finished catalog', catalog, flush=True)
    return elapsed, host, max_error


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    (ROOT/'timing53').mkdir(exist_ok=True)
    started = time.time()
    with torch.inference_mode():
        engine, templates = engine50()
        signed, unsigned = coefficients()
        model = load_model()
        rows = [measure(c, engine, templates, signed, unsigned, model) for c in range(12)]
    clocks, host, error = [np.stack(x) for x in zip(*rows)]
    rand = np.random.default_rng(530992).integers(12, size=(16000, 12))
    p90 = np.quantile(clocks, .90, axis=(0, 1))
    boot = np.quantile(clocks[rand], .90, axis=(1, 2))
    diff = boot[:, [0]] - boot
    # Prespecified three contrasts against mixed evidence selector, vote-selector, and post-only vote.
    half = .025/3
    data = dict(methods=METHODS, p90_ms=p90.tolist(), p90_ci=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
                joint_minus_ms=(p90[0]-p90).tolist(), joint_minus_ci=np.quantile(diff,[.025,.975],axis=0).T.tolist(),
                primary_joint_minus_simultaneous_ci=np.quantile(diff[:,1:4],[half,1-half],axis=0).T.tolist(),
                host_p90_ms=np.quantile(host,.9,axis=(0,1)).tolist(), replay_error=error.max(0).tolist(),
                job=os.getenv('SLURM_JOB_ID'), node=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                torch=str(torch.__version__), cuda=torch.version.cuda, batch=512, sessions_per_batch=4,
                catalogs=12, samples_per_policy=int(clocks.shape[0]*clocks.shape[1]), crossover='8-arm Williams square/catalog',
                utc_start=started, utc_end=time.time())
    r.v38.atomic_json(ROOT / 'timing53.json', data)
    print(json.dumps(data), flush=True)


if __name__ == '__main__':
    main()
