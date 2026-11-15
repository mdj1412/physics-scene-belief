"""Compare four portfolios under a sealed 32-interaction budget on eight environments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
V28 = ROOT.parent / 'calibration_probe_v28'
sys.path.insert(0, str(V28))
from runner28 import (Q, P, Engine, ProbeShiftEstimator, PosteriorPredictiveEngine,
                      atomic_json, atomic_npz, calibration_probe_measurements, digest,
                      evaluation_only, factual_batch, measure_online, potential_costs,
                      stream, t, timed_call, torch)
from policy29 import RankFlipScorer

METHODS = ('adaptive', 'fixed4', 'random', 'uncertainty', 'known')
BUDGETS = (2, 4, 8)
E = 8


class Calibration:
    """Simulator-owned potential responses; fit(k) only exposes the first k pairs."""
    def __init__(self, cfg, env, detector):
        self.detector = detector
        seed, n = env['seed'], cfg['n_cal']
        self.x = factual_batch(n, seed, 10, env['alpha'])
        self.order = stream(seed, 98).permutation(n)[:8]
        self.probe = calibration_probe_measurements(n, seed, env['alpha'], self.order)
        self.ll, self.ll_ms = timed_call(detector.component_loglikelihood, self.x)
        self.joint = {}
        self.joint_ms = {}
        self.posterior = {}
        self.posterior_ms = {}

    def fit(self, k):
        if k not in self.posterior:
            ids = self.order[:k]
            self.joint[k], self.joint_ms[k] = timed_call(
                self.detector.joint_component_loglikelihood, self.x[ids], self.probe[:k])
            self.posterior[k], self.posterior_ms[k] = timed_call(
                self.detector.posterior_from_components, self.ll, ids, self.joint[k])
        return self.posterior[k]

    def charge(self, path):
        # Factual log-likelihood is cached from the first stage. Each probe stage
        # recomputes joint factors for its complete acquired prefix on the deployable path.
        return self.ll_ms + sum(self.joint_ms[k] + self.posterior_ms[k] for k in path)


def successive_halving(views, scorer, name):
    active = list(range(E))
    counts = np.full(E, 2, dtype=int)
    paths = [[2] for _ in views]
    score_ms = np.zeros(E)
    logs = []
    for k, horizon in ((2, 2), (4, 4)):
        priorities = []
        details = []
        for i in active:
            cal = views[i]
            est = cal.fit(k)
            start = time.perf_counter()
            if name == 'adaptive':
                detail = scorer.score(est, horizon=horizon, factual_logs=cal.ll,
                                      probe_ids=cal.order[:k], joint_logs=cal.joint[k])
                priority = float(detail['priority'])
            else:
                priority = float(est['sd'][1])
                detail = {'priority': priority}
            score_ms[i] += 1000 * (time.perf_counter()-start)
            assert np.isfinite(priority)
            priorities.append(priority)
            details.append(detail)
        start = time.perf_counter()
        rank = np.argsort(-np.asarray(priorities), kind='stable')
        kept = sorted(active[int(j)] for j in rank[:len(active)//2])
        sort_ms = 1000*(time.perf_counter()-start)
        score_ms[active] += sort_ms/len(active)
        logs.append(dict(budget=k, horizon=horizon, active=active, retained=kept, details=details))
        for i in kept:
            counts[i] += horizon
            paths[i].append(k+horizon)
        active = kept
    return counts, paths, score_ms, logs


def allocate(views, scorer, seed):
    counts = np.zeros((E,len(METHODS)),dtype=int)
    fit = np.zeros_like(counts,dtype=float)
    scoring = np.zeros_like(fit)
    logs = {}
    # Stable tie-breaking uses only the block's randomized anonymous environment order.
    for col in (0, 3):
        name = METHODS[col]
        counts[:,col], paths, scoring[:,col], logs[name] = successive_halving(views, scorer, name)
        for i,path in enumerate(paths):
            views[i].fit(path[-1])
            fit[i,col] = views[i].charge(path)
    counts[:,1] = 4
    rng = stream(seed, 1017)
    counts[:,2] = rng.permutation([2,2,2,2,4,4,8,8])
    for i,cal in enumerate(views):
        for col in (1,2):
            k = int(counts[i,col])
            cal.fit(k)
            fit[i,col] = cal.charge([k])
    assert np.array_equal(counts.sum(0), [32,32,32,32,0])
    for col in (0,2,3):
        assert np.array_equal(np.sort(counts[:,col]), [2,2,2,2,4,4,8,8])
    return counts,fit,scoring,logs


def build_and_select(env, view, pair, public_x):
    engines = []
    build_ms = []
    for k in BUDGETS:
        engine, ms = timed_call(PosteriorPredictiveEngine,pair,view.fit(k))
        engines.append(engine)
        build_ms.append(ms)
    eng,ms = timed_call(Engine,pair,*env['alpha'])
    engines.append(eng)
    build_ms.append(ms)
    gaps = np.empty((len(public_x),len(engines)))
    for w,x in enumerate(public_x):
        xx=t(x)
        for j,engine in enumerate(engines):
            risk=engine.risk(engine.fit(xx),'keep256m2048')
            gaps[w,j] = float(risk[0]-risk[2])
    assert np.isfinite(gaps).all()
    return engines,np.array(build_ms),gaps,np.where(gaps>1.e-12,P,Q).astype(np.int8)


def block_probes(gaps, fraction):
    # A separate batch-deployment control: identical number of evaluation probes
    # as well as identical calibration spend. No future observation or label in ranking.
    n_env,n_world,n_method = gaps.shape
    n=int(np.floor(n_world*fraction+.5))
    order=np.argsort(-gaps,axis=1,kind='stable')
    decisions=np.full(gaps.shape,Q,dtype=np.int8)
    np.put_along_axis(decisions,order[:,:n,:],P,axis=1)
    assert np.all((decisions==P).sum(1)==n)
    return decisions,n*n_env


def jsonable(value):
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    return value


def run_block(cfg, catalog, rep, detector, scorer, detector_ms, scorer_ms, fingerprint):
    out = ROOT/'results'/f'block_{rep:03d}'
    out.mkdir(parents=True,exist_ok=True)
    info,cost_file,choice_file,allocation_file=[out/name for name in
        ('metrics.json','metrics.npz','choice_lock.npz','allocation_lock.json')]
    if info.is_file():
        m=json.loads(info.read_text())
        if (m['fingerprint']!=fingerprint or m['npz_sha256']!=digest(cost_file)
                or m['choices_sha256']!=digest(choice_file) or m['allocation_sha256']!=digest(allocation_file)):
            raise RuntimeError(f'Invalid completed block: {out}')
        return
    seed=cfg.get('block_seed',29019371)+rep*419
    permutation=stream(seed,1001).permutation(E)
    envs=[catalog[j] for j in permutation]
    assert all(e['rep']==rep for e in envs) and len(set(e['bank_index'] for e in envs))==1
    pair=cfg['bank_pairs'][envs[0]['bank_index']]
    views=[Calibration(cfg,e,detector) for e in envs]
    counts,fit_ms,score_ms,stages=allocate(views,scorer,seed)
    atomic_json(allocation_file,jsonable(dict(counts=counts,methods=METHODS,stages=stages,
                                              environment_ids=[e['id'] for e in envs])))
    allocation_hash=digest(allocation_file)
    # The following complete-prefix responses are available to the independent
    # evaluator for diagnostic contrasts only after every allocator has committed.
    idx = np.array([[BUDGETS.index(k) if k else 3 for k in row] for row in counts])
    public=[]; engines=[]; setups=[]; gaps=[]; audit_actions=[]
    for e,view in zip(envs,views):
        x=factual_batch(cfg['n_test'],e['seed'],20,e['alpha'])
        en,ms,g,a=build_and_select(e,view,pair,x)
        public.append(x);engines.append(en);setups.append(ms);gaps.append(g);audit_actions.append(a)
    all_gaps=np.stack(gaps);all_actions=np.stack(audit_actions)
    ix=np.broadcast_to(idx[:,None,:],(E,cfg['n_test'],len(METHODS)))
    actions=np.take_along_axis(all_actions,ix,axis=2)
    method_gaps=np.take_along_axis(all_gaps,ix,axis=2)
    quota_act,quota=block_probes(method_gaps,cfg.get('quota_fraction',.1))
    setup=np.take_along_axis(np.stack(setups),idx,axis=1)
    atomic_npz(choice_file,counts=counts,action=actions,quota_action=quota_act,
        audit_action=all_actions,audit_gap=all_gaps,x_cal=np.stack([v.x for v in views]),
        probe_order=np.stack([v.order for v in views]),probe_y=np.stack([v.probe for v in views]),
        x_test=np.stack(public),env_ids=[e['id'] for e in envs],bank_pair=pair)
    choice_hash=digest(choice_file)
    values={}
    for i,e in enumerate(envs):
        label,cell,obs=evaluation_only(cfg['n_test'],cfg['repeats'],e['seed'],e['alpha'])
        cc,pp=potential_costs(public[i],label,obs,engines[i])
        assert np.isfinite(cc).all() and (cc>=0).all() and (cc<=1+1.e-10).all()
        costs=cc[:,idx[i]];pred=pp[:,idx[i]];row=np.arange(len(label))[:,None]
        chosen=(actions[i]==P).astype(int);batch_chosen=(quota_act[i]==P).astype(int)
        decision_native=costs[row,np.arange(len(METHODS))[None,:],chosen]
        decision_common=cc[row,3,chosen]
        times,tids=measure_online(public[i],obs,[engines[i][j] for j in idx[i]],actions[i],
                                  pred,cfg,e['seed'])
        counter=(all_actions[i,:,:3]==P).astype(int)
        fields=dict(action=actions[i],loss=decision_native,common_known=decision_common,
            quota_action=quota_act[i],quota_native=costs[row,np.arange(len(METHODS))[None,:],batch_chosen],
            quota_common_known=cc[row,3,batch_chosen],audit_action=all_actions[i,:,:3],
            audit_native=cc[row,np.arange(3)[None,:],counter],audit_common=cc[row,3,counter],
            audit_gap=all_gaps[i,:,:3],known_gap=all_gaps[i,:,3],cost=costs,
            probabilities=pred,label=label,cell=cell,times_ms=times,timing_ids=tids)
        for name,v in fields.items():
            values.setdefault(name,[]).append(v)
    assert choice_hash==digest(choice_file) and allocation_hash==digest(allocation_file)
    atomic_npz(cost_file,**{k:np.stack(v) for k,v in values.items()},counts=counts,methods=METHODS,
               quota_probe_count=quota,audit_budgets=BUDGETS,env_ids=[e['id'] for e in envs])
    atomic_json(info,jsonable(dict(block=rep, fingerprint=fingerprint, allocation_sha256=allocation_hash,
        choices_sha256=choice_hash,npz_sha256=digest(cost_file),environments=envs, bank_pair=pair,
        methods=METHODS,counts=counts,quota_probe_count=quota,stages=stages,
        estimates=[{str(k):v.posterior[k] for k in BUDGETS} for v in views],
        setup_ms=dict(fit=fit_ms,score=score_ms,engines=setup,detector_static=detector_ms,scorer_static=scorer_ms),
        gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],host=os.environ.get('SLURMD_NODENAME'))))
    print('block',rep,'common',np.stack(values['common_known']).mean((0,1)).round(6).tolist(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'design.json'))
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    path=Path(args.config).resolve();cfg=json.loads(path.read_text())
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1
    assert 0<=args.shard<args.shards and cfg['n_cal']>=8 and 0<cfg.get('quota_fraction',.1)<1
    catalog=cfg['environments'];assert len({e['id'] for e in catalog})==len(catalog)
    blocks=sorted({e['rep'] for e in catalog})
    assert all(sum(e['rep']==b for e in catalog)==E for b in blocks)
    torch.set_num_threads(2)
    files=[path,Path(__file__).resolve(),ROOT/'policy29.py',V28/'runner28.py',V28/'estimator28.py',
           ROOT.parent/'population_shift_v27'/'runner27.py',ROOT.parent/'population_shift_v27'/'estimator27.py',
           ROOT.parent/'proxy_shift_v26'/'physics26.py',ROOT.parent/'within_calibration_v25'/'physics25.py']
    fingerprint=hashlib.sha256(''.join(digest(p) for p in files).encode()).hexdigest()
    with torch.inference_mode():
        detector,dms=timed_call(ProbeShiftEstimator,seed=cfg['estimator_seed'],
                                bank_size=cfg.get('estimator_bank_size',8192))
        scorer,sms=timed_call(RankFlipScorer,detector)
        print('scorer static ms',round(sms,2),flush=True)
        for i,b in enumerate(blocks):
            if i%args.shards==args.shard:
                run_block(cfg,[e for e in catalog if e['rep']==b],b,detector,scorer,dms,sms,fingerprint)

if __name__=='__main__':
    main()
