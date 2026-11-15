"""Correctly specified unpaired-factual/probe estimator and paired held-out comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from runner28 import (ROOT, ProbeShiftEstimator, PosteriorPredictiveEngine, atomic_json, atomic_npz,
                      digest, evaluation_only, potential_costs, select_all, t, torch)
from estimator28 import independent_masses, summaries


def unpaired_estimate(estimator, x, probe_ids, y):
    mask=np.ones(len(x),dtype=bool)
    mask[probe_ids]=False
    fact=estimator.posterior(estimator.component_loglikelihood(x[mask]))
    # Probe means depend on friction/restitution/slope, not offset or curvature
    # at y=0. Marginalizing the paired bank gives two Y-only components.
    mu=estimator.joint_centers[:,:, -2:]
    assert torch.equal(mu[0],mu[2]) and torch.equal(mu[1],mu[3])
    mu=mu[:2]
    whitened=t(y)/estimator.probe_sigma
    kernels=whitened@mu.transpose(1,2)-.5*mu.square().sum(-1)[:,None,:]
    component=torch.logsumexp(kernels,-1).T.cpu().numpy()
    component-=component.max(1,keepdims=True)
    ll=np.logaddexp(component[:,0,None]+estimator.log_f[0],
                    component[:,1,None]+estimator.log_f[1]).sum(0)
    fr=summaries(estimator.grid_f,np.exp(ll-ll.max()))
    alpha=[fact['alpha'][0],fr['mean']]
    return dict(alpha=alpha,ci95=[fact['ci95'][0],fr['ci95']],
                posterior_prob_negative=[fact['posterior_prob_negative'][0],fr['prob_negative']],
                posterior_kl_nats=[fact['posterior_kl_nats'][0],fr['kl']],
                sd=[fact['sd'][0],fr['sd']],variant_probabilities=independent_masses(alpha).tolist(),
                n_factual=int(mask.sum()),n_probes=len(y),covariance=0.)


def evaluate(cfg,env,estimator,fingerprint):
    source=ROOT/'results'/env['id']
    dest=ROOT/'results'/'marginal'/env['id']
    dest.mkdir(parents=True,exist_ok=True)
    raw=source/'choice_lock.npz'
    orig=json.loads((source/'metrics.json').read_text())
    assert orig['environment']==env and digest(raw)==orig['choices_sha256']
    info=dest/'metrics.json'; result=dest/'metrics.npz'; locked=dest/'choice_lock.npz'
    if info.exists():
        prev=json.loads(info.read_text())
        assert prev['fingerprint']==fingerprint and prev['main_npz_sha256']==orig['npz_sha256']
        assert prev['choices_sha256']==digest(locked) and prev['npz_sha256']==digest(result)
        return
    with np.load(raw) as data:
        x,y,ids,xt=[data[k] for k in ('x_cal','probe_y','probe_order','x_test')]
    estimate=unpaired_estimate(estimator,x,ids,y)
    engines=[PosteriorPredictiveEngine(orig['bank_pair'],estimate)]
    act=select_all(xt,engines)[:,0]
    atomic_npz(locked,action=act,probe_order=ids,alpha=estimate['alpha'])
    sealed=digest(locked)
    assert digest(source/'metrics.npz')==orig['npz_sha256']
    with np.load(source/'metrics.npz') as main:
        known=list(main['methods']).index('known')
        common_cost=main['cost'][:,known]
        yy,cc=main['label'],main['cell']
    labels,cells,views=evaluation_only(len(xt),cfg['repeats'],env['seed'],env['alpha'])
    assert np.array_equal(yy,labels) and np.array_equal(cc,cells)
    costs,_=potential_costs(xt,labels,views,engines)
    row=np.arange(len(xt));slot=(act==3).astype(int)
    loss=costs[row,0,slot]; common=common_cost[row,slot]
    assert np.isfinite(loss).all() and np.isfinite(common).all() and sealed==digest(locked)
    atomic_npz(result,action=act,loss=loss,common_known=common,cost=costs[:,0])
    atomic_json(info,dict(environment=env,estimate=estimate,main_npz_sha256=orig['npz_sha256'],
        choices_sha256=sealed,main_choices_sha256=digest(raw),npz_sha256=digest(result),
        fingerprint=fingerprint,method='marginal_probe16',job=os.environ['SLURM_JOB_ID']))
    print(env['id'],'marginal Brier',float(loss.mean()),'alpha_f',estimate['alpha'][1],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'design.json'))
    parser.add_argument('--shard',default=0,type=int)
    parser.add_argument('--shards',default=1,type=int)
    args=parser.parse_args()
    path=Path(args.config).resolve();cfg=json.loads(path.read_text())
    cases=[x for x in cfg['environments'] if x['target'] and x['rep']<3]
    assert len(cases)==24 and 0<=args.shard<args.shards
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    sources=[Path(__file__).resolve(),path,ROOT/'runner28.py',ROOT/'estimator28.py']
    fingerprint=hashlib.sha256(''.join(digest(p) for p in sources).encode()).hexdigest()
    torch.set_num_threads(2)
    with torch.inference_mode():
        estimator=ProbeShiftEstimator(seed=cfg['estimator_seed'],bank_size=cfg.get('estimator_bank_size',8192))
        for i,e in enumerate(cases):
            if i%args.shards==args.shard:
                evaluate(cfg,e,estimator,fingerprint)
if __name__=='__main__':main()
