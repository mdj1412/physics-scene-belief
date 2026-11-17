"""End-to-end acquisition and inference replay on a single idle GPU."""
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
sys.path.insert(0, str(ROOT.parent / 'impact_witness_v53'))
import bench53 as prev
from policy53 import mixed53
from policy54 import select54

METHODS = ('adaptive', 'fixed_4R', 'fixed_3R_I')


def fourth(method, weak3, ref3):
    if method == 'adaptive':
        return np.asarray(select54(weak3, ref3), dtype=bool)
    return np.full(weak3.shape[:-1], method == 'fixed_3R_I', dtype=bool)


def terminal(raw, route):
    """Read either reference four or impact four for each session after routing."""
    ow, of, nw, nf, impact, _ = raw
    weight = np.empty(route.shape)
    for use_impact in (False, True):
        rows = np.flatnonzero(route == use_impact)
        if not len(rows):
            continue
        # Index only purchased columns for this subset, including on mixed batches.
        n = 3 if use_impact else 4
        y = impact[rows] if use_impact else np.zeros(len(rows), dtype=np.int8)
        part = mixed53(ow[rows], of[rows], nw[rows, :n], nf[rows, :n], nw[rows, 3], y)
        weight[rows] = part['alpha_joint']
    return weight


def policy(method, raw):
    route = fourth(method, raw[2][:, :3], raw[3][:, :3])
    return route, terminal(raw, route)


def measure(catalog, engine, templates, signed, unsigned, model):
    xx, bought, disp, h, raw, selected, exp53, _, alpha53, pre, post, err, repeat = prev.fixture(catalog, model)
    rng = np.random.default_rng(540918 + catalog)
    orders = rng.permutation(np.array(list(permutations(range(3)))))
    np.testing.assert_array_equal(np.sort(orders, axis=0), np.repeat(np.arange(3), 2)[:, None].repeat(3, axis=1))
    transitions = np.zeros((3,3), dtype=int)
    np.add.at(transitions, (orders[:, :-1], orders[:, 1:]), 1)
    np.testing.assert_array_equal(transitions, 2*(1-np.eye(3,dtype=int)))
    expected, routes, weights = [], [], []
    for method in METHODS:
        route, weight = policy(method, raw)
        routes.append(route)
        weights.append(weight)
        expected.append(np.where(route[:, None], exp53[0], exp53[5]))
        np.testing.assert_allclose(weight, np.where(route, alpha53[0], alpha53[5]), atol=2e-12, rtol=0)
        # Poison only counterfactual readouts. Also test a deliberately mixed mask for
        # each hardware fixture even if an adaptive policy always chooses one sensor.
        for test_route in (route, np.array([True, False, True, False])):
            poisoned = [z.copy() for z in raw]
            poisoned[3][test_route, 3] = np.nan
            poisoned[4][~test_route] = 99
            np.testing.assert_allclose(terminal(poisoned,test_route),
                                       np.where(test_route,alpha53[0],alpha53[5]), atol=2e-12,rtol=0)
        test_route = fourth(method, raw[2][:,:3],raw[3][:,:3])
        np.testing.assert_array_equal(test_route,route)
    elapsed = np.empty((len(orders),3))
    max_error = np.zeros(3)
    before = prev.snapshot()
    for row, order in enumerate([rng.permutation(3), *orders], start=-1):
        for m in order:
            torch.cuda.synchronize()
            begin = time.perf_counter_ns()
            route, alpha = policy(METHODS[m],raw)
            obs = prev.r.infer.actions43(xx,engine)
            paid = prev.r.infer.paid_sensors(bought,obs)
            p0 = prev.r.infer.decode43(engine,templates[h],xx,obs,paid)[...,0]
            arm = (obs == prev.r.r.Q).astype(np.intp)
            # All forecasts are continuous mixtures in the shared terminal decoder.
            pred = prev.routed_predictions(p0,disp,arm,signed[1,h],unsigned[h],np.zeros(4,dtype=np.int8),alpha)
            torch.cuda.synchronize()
            if row >= 0:
                elapsed[row,m]=(time.perf_counter_ns()-begin)*1e-6
            np.testing.assert_array_equal(route,routes[m])
            np.testing.assert_array_equal(obs,selected)
            np.testing.assert_allclose(pred,expected[m],atol=2e-9,rtol=0)
            max_error[m]=max(max_error[m], abs(pred-expected[m]).max())
    host = np.empty((51,3)); gate=np.empty(51)
    for row in range(-1,51):
        for m in rng.permutation(3):
            begin = time.perf_counter_ns()
            policy(METHODS[m],raw)
            if row>=0: host[row,m]=(time.perf_counter_ns()-begin)*1e-6
        begin=time.perf_counter_ns()
        fourth('adaptive',raw[2][:,:3],raw[3][:,:3])
        if row>=0: gate[row]=(time.perf_counter_ns()-begin)*1e-6
    path=ROOT / 'timing54' / f'panel_{catalog:02d}.npz'
    prev.r.v38.atomic_npz(path,ms=elapsed,host_ms=host,gate_ms=gate,orders=orders,routes=routes,
                          alpha=weights,methods=METHODS,max_replay_error=max_error,
                          pre=pre,post=post,impact_error_idx=err,repeat=repeat)
    prev.r.v38.atomic_json(path.with_suffix('.json'),dict(before=before,after=prev.snapshot()))
    print('finished',catalog,flush=True)
    return elapsed,host,gate,max_error,routes[0]


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    (ROOT/'timing54').mkdir(exist_ok=True)
    with torch.inference_mode():
        engine,templates=prev.engine50()
        signed,unsigned=prev.coefficients()
        model=prev.load_model()
        rows=[measure(i,engine,templates,signed,unsigned,model) for i in range(12)]
    clock,host,gate,err,route=[np.stack(x) for x in zip(*rows)]
    indices=np.random.default_rng(540197).integers(12,size=(16000,12))
    p90=np.quantile(clock,.9,axis=(0,1))
    boot=np.quantile(clock[indices],.9,axis=(1,2))
    ci=np.quantile(boot[:,[0]]-boot[:,1:],[.0125,.9875],axis=0).T
    result=dict(methods=METHODS,p90_ms=p90.tolist(),paired_delta_ms=(p90[0]-p90[1:]).tolist(),
                simultaneous_ci=ci.tolist(),marginal_ci=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
                fixture_impact_fraction=route.mean(),fixture_routes=route.astype(int).tolist(),
                host_total_p90_ms=np.quantile(host,.9,axis=(0,1)).tolist(),host_gate_p90_ms=np.quantile(gate,.9),
                max_replay_error=err.max(0).tolist(),batch=512,catalogs=len(rows),samples_per_method=72,
                session_size=128,device=torch.cuda.get_device_name(),node=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'))
    prev.r.v38.atomic_json(ROOT/'timing54.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
