"""Prospective supplementary allocation targeting the deployment probe-cap boundary."""
import json
import os
import time
from pathlib import Path

import numpy as np

from runner29 import (ProbeShiftEstimator, RankFlipScorer, Q, P, ROOT, atomic_json,
                      atomic_npz, block_probes, digest, jsonable, timed_call, torch)
from policy29 import interpolate, moments_for_horizon, posterior_grid

FRACTION = 13/128


def quota_membership(gaps, weights):
    """Fractional cutoff row enforces an exact reference-measure quota in each scenario."""
    gaps=np.broadcast_to(gaps,weights.shape)
    idx=np.argsort(-gaps,axis=1,kind='stable')
    w=np.take_along_axis(weights,idx,axis=1)
    cumulative=np.cumsum(w,axis=1)
    left=FRACTION-cumulative+w
    selected=np.clip(np.divide(left,w,out=np.zeros_like(w),where=w>0),0.,1.)
    membership=np.zeros_like(selected)
    np.put_along_axis(membership,idx,selected,axis=1)
    assert np.allclose((weights*membership).sum(1),FRACTION,rtol=0,atol=1.e-12)
    return membership


def priority(scorer,est,horizon,ll,ids,joint):
    mass=posterior_grid(ll,ids,joint)
    mf,mcf=moments_for_horizon(mass,horizon)
    coef=scorer.coefficients[horizon]
    ratio=coef@mf[:-1]
    pos=ratio>1e-14
    prob=ratio[pos]/ratio[pos].sum()
    ac=coef[pos]@mcf/ratio[pos]
    af=coef[pos]@mf[1:]/ratio[pos]
    gap0=interpolate(scorer.table,*est['alpha'])
    gap1=interpolate(scorer.table,ac,af)
    weights=np.maximum(0,1+ac[:,None]*scorer.x_score[None,:])
    weights/=weights.sum(1,keepdims=True)
    # Both rankings would be applied to the same future batch, so each must
    # use its 13/128 upper tail under this *same future* distribution of X.
    old=quota_membership(gap0,weights)
    new=quota_membership(gap1,weights)
    move=new-old
    frequency=float(prob@(weights*np.abs(move)).sum(1))
    # Equal mass constraints cancel the shadow price, giving the gain in posterior
    # Q-minus-P risk from swapping low-gap future worlds for high-gap ones.
    gain=(weights*move*gap1).sum(1)
    assert np.min(gain)>=-1e-13
    return dict(priority=max(0.,float(prob@gain)),flip_probability=frequency,horizon=horizon)


def allocate(detector,scorer,source,out):
    with np.load(source/'choice_lock.npz') as f:
        factual=f['x_cal'];order=f['probe_order'];potential=f['probe_y'];names=f['env_ids']
    # No test observations, outcomes, or precomputed future posteriors read.
    logs=[]
    ll=[];counts=np.full(len(names),2,dtype=int);time_score=np.zeros(len(names))
    time_fit=np.zeros(len(names))
    for e,x in enumerate(factual):
        v,ms=timed_call(detector.component_loglikelihood,x);ll.append(v);time_fit[e]+=ms
    active=list(range(len(names)))
    for k,h in [(2,2),(4,4)]:
        ds=[]
        for e in active:
            paid=order[e,:k]
            joint,ms=timed_call(detector.joint_component_loglikelihood,factual[e,paid],potential[e,:k])
            est,post_ms=timed_call(detector.posterior_from_components,ll[e],paid,joint)
            time_fit[e]+=ms+post_ms
            start=time.perf_counter()
            ds.append(priority(scorer,est,h,ll[e],paid,joint))
            time_score[e]+=1000*(time.perf_counter()-start)
        rank=np.argsort(-np.array([v['priority'] for v in ds]),kind='stable')
        kept=sorted(active[j] for j in rank[:len(active)//2])
        logs.append(dict(active=active,retained=kept,details=ds,budget=k,horizon=h))
        counts[kept]+=h;active=kept
    assert counts.sum()==32 and np.array_equal(np.sort(counts),[2,2,2,2,4,4,8,8])
    return dict(env_ids=names.tolist(),counts=counts.tolist(),stages=logs,
                score_ms=time_score.tolist(),allocation_fit_ms=time_fit.tolist())


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1 and torch.cuda.is_available()
    torch.set_num_threads(2)
    root=ROOT/'results'
    dest=root/'quota29'
    dest.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'design.json').read_text())
    budget=np.array([2,4,8])
    with torch.inference_mode():
        detector=ProbeShiftEstimator(seed=cfg['estimator_seed'],bank_size=cfg.get('estimator_bank_size',8192))
        scorer,static_ms=timed_call(RankFlipScorer,detector)
        blocks=sorted({e['rep'] for e in cfg['environments']})[:24]
        assert len(blocks)==24
        prepared=[allocate(detector,scorer,root/f'block_{b:03d}',dest) for b in blocks]
        lock=dest/'allocation_lock.json'
        atomic_json(lock,jsonable(dict(static_ms=static_ms,blocks=blocks,allocations=prepared,
                                       rule='fixed supplementary top-13/128 lookahead',source=digest(Path(__file__)))))
        allocation_hash=digest(lock)
    # Every supplementary allocation has now been sealed. Apply the fixed per-environment
    # ranking using held-out factual-only prefix risks from the primary choice locks.
    actions=[];streams=[]
    for block,row in zip(blocks,prepared):
        source=root/f'block_{block:03d}'
        with np.load(source/'choice_lock.npz') as f:
            assert list(f['env_ids'])==row['env_ids']
            k=np.searchsorted(budget,row['counts'])
            gap=np.take_along_axis(f['audit_gap'],k[:,None,None],axis=-1)
            aa,_=block_probes(gap,FRACTION)
            actions.append(aa[:,:,0])
            streams.append(np.where(gap[:,:,0]>1e-12,P,Q))
    choice=dest/'choices.npz';atomic_npz(choice,quota_action=actions,stream_action=streams)
    choice_hash=digest(choice)
    # Outcomes opened only by the evaluator after ALL allocations and choices are locked.
    vals=[];nchanges=[]
    for b,row,quota,stream in zip(blocks,prepared,actions,streams):
        d=root/f'block_{b:03d}';meta=json.loads((d/'metrics.json').read_text())
        assert digest(d/'metrics.npz')==meta['npz_sha256']
        with np.load(d/'metrics.npz') as f:
            assert list(f['methods'])[:2]==['adaptive','fixed4']
            assert f['methods'][-1]=='known' and f['quota_probe_count']==104
            cc=f['cost'][:,:,-1,:]
            entries=[]
            for arr,base in [(quota,'quota_common_known'),(stream,'common_known')]:
                own=np.take_along_axis(cc,(arr==P).astype(int)[:,:,None],axis=-1).mean()
                fixed,orig=f[base][:,:,[1,0]].mean((0,1))
                entries.append([own,fixed,orig])
            vals.append(entries)
            nchanges.append(dict(budget_diff_from_primary=np.mean(np.array(row['counts'])!=f['counts'][:,0]),
                                 quota_action_diff_from_primary=np.mean(quota!=f['quota_action'][:,:,0]),
                                 stream_probes=np.sum(stream==P),quota_probes=np.sum(quota==P)))
    assert digest(lock)==allocation_hash and digest(choice)==choice_hash
    x=np.asarray(vals);diff=x[:,:,:1]-x[:,:,1:]
    draws=np.random.default_rng(29059713).integers(len(x),size=(10000,len(x)))
    bd=diff[draws].mean(1)
    pt=diff.mean(0);sd=bd.std(0,ddof=1)
    crit=np.quantile((np.abs(bd-pt)/np.where(sd>0,sd,1.)).max((1,2)),.95)
    sim=np.stack([pt-crit*sd,pt+crit*sd],axis=-1)
    p=dest/'summary.json'
    out=dict(blocks=len(blocks),methods=['quota_adaptive','fixed4','original_adaptive'],endpoint_order=['quota_common','stream_common'],
        risk=x.mean(0),deltas_vs_fixed_original=pt,sim95=sim,pointwise95=np.quantile(bd,[.025,.975],axis=0).transpose(1,2,0),
        score_ms_per_env=np.mean([r['score_ms'] for r in prepared]),static_ms=static_ms,block_risks=x,
        mean_changes={k:np.mean([a[k] for a in nchanges]) for k in nchanges[0]},
        allocation_sha256=allocation_hash,choices_sha256=choice_hash)
    atomic_json(p,jsonable(out))
    print('Supplement quota/stream against fixed, original:',pt.tolist(),flush=True)
if __name__=='__main__':main()
