"""Measure and match training-label wall compute on one isolated Slurm GPU."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_train'
V31 = ROOT.parent / 'probe_gain_reliability_v31'
sys.path.insert(0, str(V31))
from runner31 import (actions, batch_gap, Engine, PosteriorPredictiveEngine, ProbeShiftEstimator,
                      atomic_json, atomic_npz, calibration_data, digest, draw_theta, evaluation_data,
                      factual, features, jsonable, means, potential_costs, sigmas, stream, torch)


def make_environment(detector, base, env, alpha, futures, batches):
    home = base + 1000 * env
    pair = [base + 8901, base + 8902]
    x, prefix_ids, prefix_y = calibration_data(home + 1, alpha, 'coulomb')
    prefix_ids, prefix_y = prefix_ids[:4], prefix_y[:4]
    ll = detector.component_loglikelihood(x)
    joint = detector.joint_component_loglikelihood(x[prefix_ids], prefix_y)
    signal = means(draw_theta(len(x), home + 1, 10, alpha), 'coulomb')[3].cpu().numpy()
    probe_sd = sigmas()[3].cpu().numpy()
    batch_seeds = np.arange(batches) + home + 300
    xx = np.stack([factual(128, int(z), 20, alpha, 'coulomb') for z in batch_seeds])
    gaps = np.empty((2, 1 + futures, batches, 128))
    fs = []
    fields = {}
    for s,k in enumerate((2,4)):
        ids = prefix_ids[:k]
        est = detector.posterior_from_components(ll, ids, joint[:k])
        fs.append(features(est, x, ids, prefix_y[:k]))
        stages = [est]
        avail = np.setdiff1d(np.arange(len(x)), ids)
        samples, observed = [], []
        for j in range(futures):
            seed = home + 100 + 20*s + j
            extra = stream(seed,81).choice(avail, size=k, replace=False)
            y = signal[extra] + stream(seed,82).normal(size=(k,2))*probe_sd
            extra_joint = detector.joint_component_loglikelihood(x[extra], y)
            stages.append(detector.posterior_from_components(ll, np.r_[ids, extra],
                          np.vstack([joint[:k], extra_joint])))
            samples.append(extra)
            observed.append(y)
        fields[f'ids_{k}'] = samples
        fields[f'y_{k}'] = observed
        for n,est in enumerate(stages):
            engine = PosteriorPredictiveEngine(pair, est)
            gaps[s,n] = batch_gap(engine, xx.reshape(-1,11)).reshape(batches,128)
    selected = actions(gaps)
    action_sha = hashlib.sha256(selected.tobytes()).hexdigest()
    # Hidden deployment targets/observation outcomes are generated after all decisions.
    dec = Engine(pair,*alpha)
    costs = []
    for b,z in enumerate(batch_seeds):
        labels, _, obs, _ = evaluation_data(128,int(z),alpha,'coulomb')
        cost,_ = potential_costs(xx[b],labels,obs,[dec])
        costs.append(cost[:,0])
    cost = np.asarray(costs)
    assert hashlib.sha256(selected.tobytes()).hexdigest() == action_sha
    risk = np.where(selected==3, cost[None,None,:,:,1], cost[None,None,:,:,0]).mean(-1)
    return dict(features=fs, gain=risk[:,0:1]-risk[:,1:], action=selected, cost=cost,
                prefix_ids=prefix_ids, prefix_y=prefix_y, prefix_seed=home+1, alpha=alpha,
                deployment_seeds=batch_seeds, selection_sha256=action_sha, **fields)


def save_block(arm, block, base, rows, env_times, start, fingerprint):
    path = OUT/arm/f'block_{block:03d}'
    path.mkdir(parents=True,exist_ok=True)
    arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    for k in ('features','gain'):
        arrays[k] = arrays[k].swapaxes(0,1)
    atomic_npz(path/'data.npz', **arrays, env_elapsed_s=env_times)
    sha = digest(path/'data.npz')
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    meta = dict(arm=arm,block=block,base_seed=base,bank_pair=[base+8901,base+8902],
                environments=len(rows),elapsed_s=elapsed,data_sha256=sha,fingerprint=fingerprint,
                job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,gpu=torch.cuda.get_device_name())
    atomic_json(path/'meta.json',meta)
    print(arm,block,len(rows),'environments',round(elapsed,3),'s',flush=True)
    return elapsed


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    cfg = json.loads((ROOT/'training_plan.json').read_text())
    dirs = [ROOT/'training_plan.json',Path(__file__), V31/'batch31.py', V31/'runner31.py']
    dirs += [Path(sys.modules[x].__file__) for x in ('score30','contact30','runner29','runner28','estimator28','runner27','physics25')]
    fingerprint = hashlib.sha256('|'.join(digest(p) for p in dirs).encode()).hexdigest()
    OUT.mkdir(exist_ok=True)
    names = list(cfg['arms'])
    panel = np.array(cfg['alpha_panel'])
    cfg_arms = cfg['arms']
    torch.cuda.synchronize()
    start = time.perf_counter()
    detector = ProbeShiftEstimator(seed=cfg['detector_seed'],bank_size=8192)
    torch.cuda.synchronize()
    setup = time.perf_counter()-start
    warm = {}
    progress = {}
    for idx,arm in enumerate(names):
        c = cfg_arms[arm]
        st = time.perf_counter()
        make_environment(detector,388000000+1000000*idx,0,panel[0],c['futures'],c['batches'])
        torch.cuda.synchronize()
        warm[arm] = time.perf_counter()-st
        old = sorted((OUT/arm).glob('block_*/meta.json'))
        times, total, last_n = [],0.,8
        for b,p in enumerate(old):
            m = json.loads(p.read_text())
            assert m['fingerprint']==fingerprint and m['block']==b and digest(p.parent/'data.npz')==m['data_sha256']
            with np.load(p.parent/'data.npz') as saved:
                times.extend(saved['env_elapsed_s'])
            total += m['elapsed_s']; last_n = m['environments']
        progress[arm] = dict(total=total,times=times,block=len(old),done=last_n!=8)
    while not all(q['done'] for q in progress.values()):
        arm = min((a for a in names if not progress[a]['done']), key=lambda a:progress[a]['total'])
        p = progress[arm]; c = cfg_arms[arm]; b = p['block']; base = c['base_seed'] + 10000*b
        ordered = panel[np.random.default_rng(base+8903).permutation(8)]
        torch.cuda.synchronize()
        st = time.perf_counter(); rows=[]; env_times=[]
        for e,alpha in enumerate(ordered):
            recent = (p['times']+env_times)[-8:]
            if recent and cfg['label_time_cap_s_per_arm']-p['total']-(time.perf_counter()-st)<np.median(recent):
                p['done']=True
                break
            begin = time.perf_counter()
            rows.append(make_environment(detector,base,e,alpha,c['futures'],c['batches']))
            torch.cuda.synchronize()
            env_times.append(time.perf_counter()-begin)
        if rows:
            p['total'] += save_block(arm,b,base,rows,env_times,st,fingerprint)
            p['times'].extend(env_times);p['block'] += 1
    for arm,p in progress.items():
        md=dict(arm=arm,elapsed_s=p['total'],n_environments=len(p['times']),n_groups=p['block'],
                fingerprint=fingerprint,complete=True)
        atomic_json(OUT/arm/'manifest.json',md)
    atomic_json(OUT/'labels_complete.json',dict(fingerprint=fingerprint,shared_setup_s=setup,warmup_s=warm,
                arms={a:{'elapsed_s':p['total'],'n':len(p['times']),'blocks':p['block']} for a,p in progress.items()},
                gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID']))
    from fit32 import fit_and_lock
    fit_and_lock()


if __name__=='__main__':
    with torch.inference_mode():
        main()
