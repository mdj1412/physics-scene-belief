"""Paired finite-belief experiment, vectorized over world/noise realizations.

Policy functions receive belief and candidate likelihoods, never truth indices
or unselected measurement realizations. Truth enters only the environment,
evaluation, and explicitly named known-parameter diagnostics.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

ACTIONS = ['view', 'probe', 'repeat']
METHODS = [
    ('joint_evsi', 'joint', 'evsi'),
    ('plugin_fixed_evsi', 'fixed', 'evsi'),
    ('plugin_refit_evsi', 'refit', 'evsi'),
    ('joint_parameter_eig', 'joint', 'parameter_eig'),
    ('joint_query_eig', 'joint', 'query_eig'),
    ('joint_fixed_view', 'joint', 'fixed_view'),
    ('joint_fixed_probe', 'joint', 'fixed_probe'),
    ('joint_random', 'joint', 'random'),
    ('joint_query_rule', 'joint', 'query_rule'),
    ('plugin_refit_fixed_view', 'refit', 'fixed_view'),
    ('plugin_refit_fixed_probe', 'refit', 'fixed_probe'),
    ('joint_known_geometry', 'known_geometry', 'evsi'),
    ('joint_known_restitution', 'known_restitution', 'evsi'),
]


def normalize(logp):
    p = np.exp(logp - np.max(logp, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def entropy(p):
    return -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=-1)


def variance(p, z):
    return np.maximum(0, p @ (z*z) - (p @ z)**2)


def collapse(p, tie):
    geom = p.reshape(-1, 2, 2).sum(axis=-1)
    choice = np.where(np.abs(geom[:, 0]-geom[:, 1]) < 1e-12, tie, np.argmax(geom, axis=1))
    mask = (np.arange(4)[None, :] // 2 == choice[:, None])
    w = p * mask
    return w / w.sum(axis=-1, keepdims=True)


def design_cache(mu, sd, order):
    x, w = np.polynomial.hermite.hermgauss(order)
    # Gaussian quadrature conditional on each candidate world.
    caches = []
    for a in range(3):
        y = (mu[:, a, None] + np.sqrt(2)*sd[a]*x[None, :]).ravel()
        ll = -.5*((y[:, None]-mu[None, :, a])/sd[a])**2
        likelihood = np.exp(ll-ll.max(axis=1, keepdims=True))
        caches.append((likelihood, w/np.sqrt(np.pi)))
    return caches


def information_groups(z):
    _, indices = np.unique(np.round(z, 8), return_inverse=True)
    return np.eye(indices.max()+1)[indices]


def gains(p, z, caches, criterion):
    groups = information_groups(z)
    before = variance(p, z) if criterion == 'evsi' else entropy(p if criterion == 'parameter_eig' else p @ groups)
    answers = []
    for likelihood, ghw in caches:
        unnorm = p[:, None, :] * likelihood[None, :, :]
        after = unnorm / np.maximum(unnorm.sum(axis=-1, keepdims=True), 1e-300)
        if criterion == 'evsi':
            values = variance(after, z)
        else:
            values = entropy(after if criterion == 'parameter_eig' else after @ groups)
        integration = (p[:, :, None] * ghw[None, None, :]).reshape(len(p), -1)
        answers.append(before - np.sum(integration * values, axis=1))
    return np.stack(answers, axis=1)


def choose(p, z, caches, policy, q, step, lam, random_uniform):
    n = len(p)
    if policy in ['evsi', 'parameter_eig', 'query_eig']:
        score = gains(p, z, caches, policy)
        action = score.argmax(axis=1)
        # EIG penalties are not in MSE units: use only numerical-zero stop.
        threshold = lam if policy == 'evsi' else 1e-9
        return np.where(score.max(axis=1) > threshold + 1e-10, action, -1)
    if policy == 'fixed_view':
        return np.zeros(n, int)
    if policy == 'fixed_probe':
        return np.ones(n, int)
    if policy == 'random':
        return np.minimum((random_uniform * 3).astype(int), 2)
    if policy == 'query_rule':
        return np.full(n, -1 if q == 2 else (int(step > 0) if q == 0 else 1), int)
    raise ValueError(policy)


def evaluate(p, z, contact, traj, truth):
    estimate = p @ z
    cp = p @ contact
    crps = np.sum(p * np.abs(z[None, :]-z[truth, None]), axis=1)
    crps -= .5*np.einsum('ni,ij,nj->n', p, np.abs(z[:, None]-z[None, :]), p)
    order = np.argsort(z)
    cdf = np.cumsum(p[:, order], axis=1)
    lo = z[order[np.argmax(cdf >= .05-1e-12, axis=1)]]
    hi = z[order[np.argmax(cdf >= .95-1e-12, axis=1)]]
    flat = traj.reshape(4, -1)
    pred_t = p @ flat
    pairdist = np.linalg.norm(flat[:, None, :]-flat[None, :, :], axis=-1)/np.sqrt(flat.shape[1])
    energy = np.sum(p*pairdist[truth], axis=1) - .5*np.einsum('ni,ij,nj->n', p, pairdist, p)
    return dict(mse=(estimate-z[truth])**2, brier=(cp-contact[truth])**2,
                crps=crps, coverage90=((z[truth] >= lo-1e-10)&(z[truth] <= hi+1e-10)).astype(float),
                width90=hi-lo, trajectory_mse=np.mean((pred_t-flat[truth])**2, axis=1),
                energy=energy, posterior_risk=variance(p,z), prediction=estimate, contact_probability=cp)


def run(data, out, repeats=32, order=32, limit=60):
    out.mkdir(parents=True, exist_ok=True)
    arrays = np.load(data/'worlds.npz')
    specs = json.loads((data/'families.json').read_text())[:limit]
    config = dict(seed=481516, repeats=repeats, quadrature_order=order, lambda_mse=.005,
                  sigmas=[.5,1.,2.], budgets=[0,1,2], methods=METHODS, family_count=len(specs),
                  simulation_hash=hashlib.sha256((data/'worlds.npz').read_bytes()).hexdigest(),
                  host=platform.node(), acquisition='greedy myopic; equal action cost 1',
                  policy_input='belief, query, model likelihood; no hidden truth',
                  plugin_tie='geometry=family%2 when MAP geometry probabilities equal',
                  code_hashes={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['physics.py','experiment.py']})
    (out/'config.json').write_text(json.dumps(config, indent=2))
    truth = np.repeat(np.arange(4), repeats)
    n = len(truth)
    all_rows = []
    details = []
    timing = {method: 0. for method, _, _ in METHODS}
    started = time.perf_counter()
    for fam in specs:
        f = fam['family']
        rng = np.random.default_rng([config['seed'], f])
        # Matched random noise for every policy/query; separate draw per action and visit.
        noise = rng.standard_normal((n,3,2))
        random_uniform = rng.random((n,2))
        mu = arrays['observation_mean'][f]
        for sigma in config['sigmas']:
            sd = np.array([sigma, np.sqrt(2)*sigma, sigma])
            cache = design_cache(mu, sd, order)
            for q in range(3):
                z = arrays['velocity'][f,:,q]
                contact = arrays['contact'][f,:,q]
                traj = arrays['trajectory'][f,:,q]
                for name, belief, policy in METHODS:
                    t0 = time.perf_counter()
                    prior = np.full((n,4), .25)
                    if belief == 'fixed':
                        prior[:,np.arange(4)//2 != f%2] = 0
                    elif belief == 'known_geometry':
                        prior *= (np.arange(4)[None,:]//2 == truth[:,None]//2)
                    elif belief == 'known_restitution':
                        prior *= (np.arange(4)[None,:]%2 == truth[:,None]%2)
                    prior /= prior.sum(axis=1,keepdims=True)
                    logp = np.full_like(prior, -np.inf)
                    np.log(prior, out=logp, where=prior>0)
                    consumed = np.zeros(n)
                    histories = np.full((n,2), -2, int)
                    visits = np.zeros((n,3), int)
                    active = np.ones(n, bool)
                    for budget in range(3):
                        posterior = normalize(logp)
                        if belief == 'refit':
                            posterior = collapse(posterior, f%2)
                        metrics = evaluate(posterior,z,contact,traj,truth)
                        metrics['cost'] = consumed.copy()
                        metrics['penalized_mse'] = metrics['mse']+.005*consumed
                        for action, key in enumerate(ACTIONS):
                            metrics[f'first_{key}'] = (histories[:,0] == action).astype(float)
                        metrics['first_stop'] = (histories[:,0] == -1).astype(float)
                        for g in range(2):
                            take = truth//2 == g
                            row = dict(family=f, split=fam['split'], sigma=sigma, q=q+1,
                                       method=name, budget=budget, true_geometry=g, tie_geometry=f%2,
                                       n=int(take.sum()))
                            row.update({k: float(v[take].mean()) for k,v in metrics.items()})
                            all_rows.append(row)
                        if f >= 40 and sigma == 2 and budget in [0,1,2] and name in ['joint_evsi','plugin_refit_evsi','plugin_fixed_evsi']:
                            for i in range(n):
                                details.append(dict(family=f, world=int(truth[i]), repeat=i%repeats,
                                    q=q+1, method=name, budget=budget, truth_velocity=float(z[truth[i]]),
                                    truth_contact=int(contact[truth[i]]), first_action=int(histories[i,0]),
                                    second_action=int(histories[i,1]), **{k:float(v[i]) for k,v in metrics.items()}))
                        if budget == 2:
                            break
                        action = choose(posterior,z,cache,policy,q,budget,.005,random_uniform[:,budget])
                        action[~active] = -1
                        histories[:,budget] = action
                        active &= action >= 0
                        for a in range(3):
                            inds = np.where(active & (action == a))[0]
                            if not len(inds):
                                continue
                            obs = mu[truth[inds],a] + sd[a]*noise[inds,a,visits[inds,a]]
                            logp[inds] += -.5*((obs[:,None]-mu[None,:,a])/sd[a])**2
                            visits[inds,a] += 1
                            consumed[inds] += 1
                    timing[name] += time.perf_counter()-t0
        print(f'Evaluated family {f}/{len(specs)}', flush=True)
    for filename, rows in [('family_metrics.csv',all_rows),('test_episode_details.csv',details)]:
        if rows:
            with (out/filename).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    (out/'runtime.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,
        timing_seconds_by_method=timing, family_metric_rows=len(all_rows), episode_detail_rows=len(details),
        unique_world_query_episodes=len(specs)*4*3,
        noise_realizations_per_episode=repeats, gpu_used=False),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/pilot'))
    parser.add_argument('--out',type=Path,default=Path('results/pilot'))
    parser.add_argument('--repeats',type=int,default=32)
    parser.add_argument('--order',type=int,default=32)
    parser.add_argument('--families',type=int,default=60)
    args = parser.parse_args()
    run(args.data,args.out,args.repeats,args.order,args.families)
