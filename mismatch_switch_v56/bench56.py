"""Synchronised K=8 pilot-to-forecast latency, with balanced policy order."""
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
sys.path.insert(0, str(ROOT.parent/'fresh_surface_v55'))
from bench55 import serial_alpha, terminal, prev
from policy54 import select54
from switch56 import choose
import generate56 as fresh

N = 24
K = 8
W = np.array([.1,.35,.4,.4348074444131813,.47,.52,.8])
METHODS = ('pilot_switch','always_v54_gate','fixed_3RI')


def fixture(i):
    """Two paid surface pilots, each shared across two held-out law sessions."""
    d = fresh.raw56(i)
    env = np.arange(4)
    surface = ((env+i)%4)//2
    wi = np.where((i+surface)%2==0,1,5)
    rng = np.random.default_rng([561518,i])
    pilot_rep = rng.integers(512,size=2)
    surf_env = np.array([np.flatnonzero(surface==j)[0] for j in range(2)])
    with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as z:
        ref = z['pilot_ref'][np.arange(2),wi[surf_env],pilot_rep,:K]
        imp = z['pilot_imp'][np.arange(2),wi[surf_env],pilot_rep,:K]
    pre = rng.integers(5,size=4)
    post = np.where(rng.random(4)<.4375,4-pre,pre)
    e = np.where(rng.random(4)<W[wi],2,rng.integers(2,size=4))
    rep = rng.integers(d['nw'].shape[2],size=4)
    raw = (d['ow'][pre,env,rep],d['of'][pre,env,rep],d['nw'][post,env,rep],d['nf'][post,env,rep],d['imp'][e,post,env,rep],None)
    return raw,(ref,imp,surface),np.stack((pre,post,e,rep,surface,wi,pilot_rep[surface]))


def fourth(arm, raw, pilot):
    weak, ref = raw[2][:,:3],raw[3][:,:3]
    if arm==2:
        return np.ones(4,bool)
    if arm==1:
        return select54(weak,ref,mode='robust')
    reference,impact,which_surface = pilot
    route = np.asarray(choose(reference, impact),bool)[which_surface]
    # Sessions committed to I do not execute the v54 downstream acquisition score.
    stay = ~route
    if stay.any():
        route[stay] = select54(weak[stay],ref[stay],mode='robust')
    return route


def measure(i,engine,templates,coef):
    raw,pilot,scenario=fixture(i)
    measured=fresh.measured56(fresh.pop56(i),i)
    with np.load(ROOT/'forecasts'/f'panel_{i:02d}.npz') as a:
        plus,minus,actions=[a[k] for k in ('pplus','pminus','actions')]
    assert plus.shape==minus.shape==actions.shape==(4,128)
    independent=serial_alpha(raw)
    switch=np.asarray(choose(*pilot[:2]),bool)[pilot[2]]
    nominal=select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
    masks=(np.logical_or(switch,nominal),nominal,np.ones(4,bool))
    mixed=np.array([True,False,True,False])
    poisoned=[x.copy() if x is not None else None for x in raw]
    poisoned[3][mixed,3]=np.nan
    poisoned[4][~mixed]=99
    np.testing.assert_allclose(terminal(poisoned,mixed),independent[mixed.astype(int),np.arange(4)],atol=2e-12,rtol=0)
    rng=np.random.default_rng([560479,i])
    orders=rng.permutation(np.array(list(permutations(range(3)))))
    clock=np.empty((6,3))
    host=np.empty((51,3))
    maxerr=np.zeros(3)
    for row,order in enumerate([rng.permutation(3), *orders],start=-1):
        for arm in order:
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            route=fourth(arm,raw,pilot)
            alpha=terminal(raw,route)
            pp,pm,selected=fresh.forward56(measured,engine,templates,coef)
            pred=alpha[:,None]*pp+(1-alpha[:,None])*pm
            torch.cuda.synchronize()
            ms=(time.perf_counter_ns()-start)/1e6
            if row>=0: clock[row,arm]=ms
            np.testing.assert_array_equal(route,masks[arm])
            np.testing.assert_array_equal(selected,actions)
            exp=independent[route.astype(int),np.arange(4)]
            np.testing.assert_allclose(alpha,exp,atol=2e-12,rtol=0)
            expected=exp[:,None]*plus+(1-exp[:,None])*minus
            np.testing.assert_allclose(pred,expected,atol=2e-9,rtol=0)
            maxerr[arm]=max(maxerr[arm],abs(pred-expected).max())
    for j in range(len(host)):
        for arm in rng.permutation(3):
            start=time.perf_counter_ns()
            fourth(arm,raw,pilot)
            host[j,arm]=(time.perf_counter_ns()-start)/1e6
    prev.r.v38.atomic_npz(ROOT/'timing56'/f'panel_{i:02d}.npz',ms=clock,host=host,
        maxerr=maxerr,routes=np.stack(masks),switched=switch,scenario=scenario,pilot_ref=pilot[0],pilot_imp=pilot[1],orders=orders)
    print('catalog',i,np.quantile(clock,.9,axis=0),flush=True)
    return clock,host,maxerr,np.stack(masks),switch


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    (ROOT/'timing56').mkdir(exist_ok=True)
    before=prev.snapshot()
    t0=time.time()
    with torch.inference_mode():
        fresh.install()
        engine,templates=prev.engine50()
        coef,_=prev.coefficients()
        rows=[measure(i,engine,templates,coef) for i in range(N)]
    after=prev.snapshot()
    times,host,errors,routes,outer=[np.stack(x) for x in zip(*rows)]
    p90=np.quantile(times,.9,axis=(0,1))
    ids=np.random.default_rng(560937).integers(N,size=(16000,N))
    boot=np.quantile(times[ids],.9,axis=(1,2))
    intervals=np.quantile(boot[:,0,None]-boot[:,1:],[.0125,.9875],axis=0).T
    report=dict(methods=METHODS,p90_ms=p90.tolist(),delta_ms=(p90[0]-p90[1:]).tolist(),
        simultaneous95_ci=intervals.tolist(),host_route_only_p90_ms=np.quantile(host,.9,axis=(0,1)).tolist(),
        per_arm=N*6,catalogs=N,paid_pilots=K,timing_sessions=N*4,switch_to_fixed_count=int(outer.sum()),
        fourth_I_count=routes.sum(axis=(0,2)).tolist(),max_replay_error=errors.max(0).tolist(),
        gpu=torch.cuda.get_device_name(),node=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'),
        torch=str(torch.__version__),cuda=torch.version.cuda,start=t0,end=time.time(),before=before,after=after)
    prev.r.v38.atomic_json(ROOT/'timing56.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
