"""Balanced cold-session GPU timing and replay for the three target policies and hard vote."""
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
V50 = ROOT.parent / "polarity_calibration_v50"
sys.path.insert(0, str(V50))
from bench50 import coefficients, engine50, estimator50, observe, r, snapshot, unsigned_predict, update, world, world50
from controls50 import estimate

METHODS = ("likelihood_soft", "vote_soft", "unsigned_fit", "vote_hard")
WILLIAMS = np.array([[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]])
BLOCKS_PER_CATALOG = 2


def vote_weight(weak, reference):
    readable = (abs(weak) >= estimator50.DEAD_ZONE[0]) & (abs(reference) >= estimator50.DEAD_ZONE[1])
    agreeing = readable & ((weak > 0) == (reference > 0))
    return (1 + agreeing.sum(-1)) / (2 + readable.sum(-1))


def balanced_order(rng):
    order = rng.permutation(4)[WILLIAMS[rng.permutation(4)]]
    np.testing.assert_array_equal(np.sort(order, axis=0), np.broadcast_to(np.arange(4)[:, None], order.shape))
    pairs = np.zeros((4,4), dtype=int)
    np.add.at(pairs, (order[:, :-1], order[:, 1:]), 1)
    np.testing.assert_array_equal(pairs, 1-np.eye(4, dtype=int))
    return order


def panel(idx, engine, templates, signed, unsigned):
    gen = np.random.default_rng(510708+idx)
    scheduling = np.random.default_rng(510918+idx)
    f, hi = idx%4, (idx//4+idx//2)%2
    alignment, variant = (idx//4+idx)%2, (idx//2+idx//4)%2
    family, height = world50.FAMILIES[f], world50.HEIGHTS[hi]
    pop = world(idx)
    xx, optics = r.apparatus(pop, family, height, r.VARIANTS[variant])
    optics = observe(optics, family, height, r.ALIGN[alignment], r.VARIANTS[variant])
    one = [{key: row[key][:,:1] for key in (r.r.Q,r.r.P)} for row in optics]
    env, obj = np.indices((4,128))
    states = gen.permutation(np.repeat(np.arange(3), (256,128,128))).reshape(env.shape)
    displacement = np.stack([world50.readings(pop, family, height, state)[...,0] for state in world50.STATES])[states,env,obj]
    name = f"panel_{idx:02d}.npz"
    # Calibration station comes from the other physical surface under the same law in every catalog.
    origin = (np.arange(4) + 2)%4
    assert np.array_equal(pop['law'][origin],pop['law'])
    assert np.all(pop['surface'][origin]!=pop['surface'])
    from_kappa = (np.arange(4)+idx)%5
    repeat = gen.integers(256,size=4)
    with np.load(V50/"cal50"/name) as cal:
        weak = cal["weak_mm"][from_kappa,origin,repeat,:4]
        reference = cal["full_mm"][from_kappa,origin,repeat,:4]
        stored = cal["khat"][from_kappa,2,origin,repeat]
    h = estimator50.infer(weak,reference)[:,None]
    np.testing.assert_allclose(stored,h[:,0],rtol=0,atol=1e-13)
    # Independent vote replay uses the original controls50 vote expression.
    v = estimate(weak,reference,4,reference)[3][:,None]
    np.testing.assert_array_equal(v[:,0],vote_weight(weak,reference))
    with np.load(V50/"forecast50"/name) as archive:
        bank = archive["p"][:,f,hi,:,alignment,variant,...,0]
        plus, minus, blind = [bank[m,states,env,obj] for m in (1,2,5)]
        selected = archive["selected"][f,hi,variant]
    expected = [h*plus+(1-h)*minus, v*plus+(1-v)*minus, blind, np.where(v<.5,minus,plus)]
    orders = np.concatenate([balanced_order(scheduling) for _ in range(BLOCKS_PER_CATALOG)])
    elapsed = np.empty((len(orders),len(METHODS)))
    max_error = np.zeros(len(METHODS))
    start_snapshot = snapshot()
    for rep,order in enumerate([scheduling.permutation(4),*orders],start=-1):
        for m in order:
            method = METHODS[m]
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            if method=="likelihood_soft":
                a = estimator50.infer(weak,reference)[:,None]
            elif method in ("vote_soft","vote_hard"):
                a = vote_weight(weak,reference)[:,None]
            act = r.infer.actions43(xx,engine)
            purchased = r.infer.paid_sensors(one,act)
            p0 = r.infer.decode43(engine,templates[hi],xx,act,purchased)[...,0]
            arm = (act==r.r.Q).astype(np.intp)
            if method=="unsigned_fit":
                pred = unsigned_predict(p0,displacement,unsigned[hi][arm])
            elif method=="vote_hard":
                pred = update(p0,np.where(a<.5,-displacement,displacement),signed[1,hi][arm])
            else:
                coeff = signed[1,hi][arm]
                pred = a*update(p0,displacement,coeff) + (1.-a)*update(p0,-displacement,coeff)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            if rep>=0:
                elapsed[rep,m] = (stop-start)*1e-6
            np.testing.assert_array_equal(selected,act)
            np.testing.assert_allclose(pred,expected[m],rtol=0,atol=2e-9)
            max_error[m] = max(max_error[m],np.max(abs(pred-expected[m])))
    micro = np.empty((101,2))
    for rep in range(102):
        for ell in scheduling.permutation(2):
            timer=time.perf_counter_ns()
            if ell: vote_weight(weak,reference)
            else: estimator50.infer(weak,reference)
            if rep: micro[rep-1,ell]=(time.perf_counter_ns()-timer)*1e-6
    dest = ROOT/"timing51"/name
    r.v38.atomic_npz(dest,ms=elapsed,methods=METHODS,methods_idx=orders,max_error=max_error,estimator_ms=micro,
                     origin=origin,kappa_cal=from_kappa,repeat=repeat)
    r.v38.atomic_json(dest.with_suffix(".json"),dict(before=start_snapshot,after=snapshot()))
    print("finished catalog",idx,flush=True)
    return elapsed,max_error,micro


def main():
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count()!=1:
        raise RuntimeError("requires one Slurm-allocated GPU")
    torch.set_num_threads(1)
    (ROOT/"timing51").mkdir(exist_ok=True)
    first=time.time()
    with torch.inference_mode():
        engine,templates=engine50()
        signed,unsigned=coefficients()
        packets=[panel(i,engine,templates,signed,unsigned) for i in range(12)]
    times=np.stack([x[0] for x in packets])
    samples=np.random.default_rng(510779).integers(0,len(times),(16000,len(times)))
    p90=np.quantile(times,.9,axis=(0,1))
    bootstrap=np.quantile(times[samples],.9,axis=(1,2))
    diffs=bootstrap[:,0,None]-bootstrap
    raw=dict(methods=METHODS,p90_ms=p90.tolist(),p90_ci=np.quantile(bootstrap,[.025,.975],axis=0).T.tolist(),
             soft_minus_other_ms=(p90[0]-p90).tolist(),soft_minus_other_ci=np.quantile(diffs,[.025,.975],axis=0).T.tolist(),
             soft_vs_vote_unsigned_simultaneous_ci=np.quantile(diffs[:,1:3],[.0125,.9875],axis=0).T.tolist(),
             host_p90_ms=np.quantile(np.stack([x[2] for x in packets]),.9,axis=(0,1)).tolist(),
             max_replay_absolute_error=np.max([x[1] for x in packets],axis=0).tolist(),
             samples_per_method=int(times.shape[0]*times.shape[1]),shape=times.shape,bootstrap_blocks=12,
             n_bootstrap=16000,batch=512,cal_pairs_per_env=4,environments_per_batch=4,
             job=os.getenv("SLURM_JOB_ID"),node=socket.gethostname(),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
             torch=str(torch.__version__),utc_start=first,utc_finish=time.time(),
             scope="End-to-end synchronized K4 cold-session, 4 actual pairs for each of 4 environment sessions, one deployment noise; simulation labels and physical cycles excluded.")
    r.v38.atomic_json(ROOT/"timing51.json",raw)
    print(json.dumps(raw),flush=True)


if __name__=="__main__":
    main()
