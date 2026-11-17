"""Balanced crossover from observed pilot signs to synchronized physical forecasts."""
from pathlib import Path
import json
import os
import socket
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'fresh_surface_v55'))
from bench55 import serial_alpha, terminal, prev
from policy54 import select54
from policy57 import choose, MIXTURE, SOURCE_GAPS
import generate57 as fresh

METHODS=('MAP_4x2','MAP_8x1','risk_4x2','always_gate','always_fixed_I')
# Only fixtures and post-timing verification receive the physical group or endpoint costs.
N=24

def fixture(catalog):
    rng=np.random.default_rng([571406,catalog])
    surface=((np.arange(4)+catalog)%4)//2
    regime=(np.arange(2)+catalog)%2
    with np.load(ROOT/'pilots'/f'panel_{catalog:02d}.npz') as bank:
        repeats=rng.integers(bank['pilot_ref'].shape[3],size=2)
        keys=(np.arange(2),regime,slice(None),repeats)
        reference=bank['pilot_ref'][keys]
        impact=bank['pilot_imp'][keys]
    assert reference.shape==impact.shape==(2,2,8)
    d=fresh.raw57(catalog)
    env=np.arange(4)
    pre=rng.integers(5,size=4)
    post=np.where(rng.random(4)<.4375,4-pre,pre)
    e=np.array([rng.choice(3,p=MIXTURE[v]) for v in regime[surface]])
    rep=rng.integers(256,size=4)
    raw=(d['ow'][pre,env,rep],d['of'][pre,env,rep],d['nw'][post,env,rep],
         d['nf'][post,env,rep],d['imp'][e,post,env,rep],None)
    return raw,(reference,impact,surface),np.stack((pre,post,e,rep,surface,regime[surface],repeats[surface]))

def outer(arm,ref,imp):
    if arm==4: return np.ones(len(ref),bool)
    if arm==3: return np.zeros(len(ref),bool)
    index=0 if arm==1 else 1
    return choose(ref[:,index],imp[:,index],design='8x1' if arm==1 else '4x2',
                  rule='risk' if arm==2 else 'map')

def route(arm,raw,pilot):
    r=outer(arm,*pilot[:2])[pilot[2]]
    remaining=~r
    if remaining.any():
        r[remaining]=select54(raw[2][remaining,:3],raw[3][remaining,:3],mode='robust')
    return r

def independent_outer(ref,imp,arm):
    if arm in (1,3,4):return np.full(len(ref),arm==4,dtype=bool)
    rates=np.array([0.,.25,.5]); choices=[]
    for r,p in zip(ref[:,1],imp[:,1]):
        likelihood=np.ones(2)
        for group in range(4):
            sg=slice(2*group,2*group+2)
            valid=(r[sg]!=0)&(p[sg]!=0)
            n=valid.sum(); x=((r[sg]!=p[sg])&valid).sum()
            likelihood*=MIXTURE@(rates**x*(1-rates)**(n-x))
        # Singletons confer exactly zero information; eliminate floating-point ties.
        cutoff=1 if arm==0 else abs(SOURCE_GAPS[1]/SOURCE_GAPS[0])
        choices.append(likelihood[0]>cutoff*likelihood[1]+1e-14)
    return np.array(choices)

def crossover(catalog):
    rng=np.random.default_rng([570318,catalog//2])
    names=rng.permutation(5)
    basic=np.array([0,1,4,2,3])
    rows=names[(basic[None,:]+np.arange(5)[:,None])%5]
    rows=rows[:,::-1] if catalog%2 else rows
    return rows[rng.permutation(5)]

def measure(i,engine,templates,coef):
    raw,pilot,params=fixture(i)
    observation=fresh.measured57(fresh.pop57(i),i)
    with np.load(ROOT/'forecasts'/f'panel_{i:02d}.npz') as bank:
        plus,minus,actions=[bank[key] for key in ('pplus','pminus','actions')]
    endpoints=serial_alpha(raw)
    assert plus.shape==minus.shape==actions.shape==(4,128)
    nominal=select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
    outer_check=np.stack([independent_outer(*pilot[:2],arm) for arm in range(len(METHODS))])
    masks=outer_check[:,pilot[2]] | nominal[None,:]
    # Unselected fourth readouts cannot affect the common terminal model.
    mixed=np.array([True,False,True,False]); bad=[None if x is None else x.copy() for x in raw]
    bad[3][mixed,3]=np.nan;bad[4][~mixed]=99
    np.testing.assert_allclose(terminal(bad,mixed),endpoints[mixed.astype(int),np.arange(4)],rtol=0,atol=2e-12)
    clock=np.empty((5,5)); host=np.empty((51,5)); maxerr=np.zeros(5)
    rng=np.random.default_rng([572194,i])
    orders=crossover(i)
    for row,order in enumerate([rng.permutation(5),*orders],-1):
        for arm in order:
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            mask=route(arm,raw,pilot)
            alpha=terminal(raw,mask)
            pp,pm,act=fresh.forward57(observation,engine,templates,coef)
            prediction=alpha[:,None]*pp+(1-alpha[:,None])*pm
            torch.cuda.synchronize()
            elapsed=(time.perf_counter_ns()-start)/1e6
            if row>=0:clock[row,arm]=elapsed
            np.testing.assert_array_equal(mask,masks[arm])
            np.testing.assert_array_equal(act,actions)
            expect=endpoints[mask.astype(int),np.arange(4)]
            np.testing.assert_allclose(alpha,expect,rtol=0,atol=2e-12)
            expected=expect[:,None]*plus+(1-expect[:,None])*minus
            err=np.max(abs(prediction-expected));assert err<=2e-9
            maxerr[arm]=max(maxerr[arm],err)
    for j in range(51):
        for arm in rng.permutation(5):
            start=time.perf_counter_ns();route(arm,raw,pilot)
            host[j,arm]=(time.perf_counter_ns()-start)/1e6
    prev.r.v38.atomic_npz(ROOT/'timing57'/f'panel_{i:02d}.npz',ms=clock,host=host,maxerr=maxerr,
                          scenario=params,pilot_ref=pilot[0],pilot_imp=pilot[1],routes=masks,outers=outer_check,orders=orders)
    print('catalog',i,np.quantile(clock,.9,axis=0),flush=True)
    return clock,host,maxerr,masks,outer_check

def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    (ROOT/'timing57').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    before=prev.snapshot();t0=time.time()
    with torch.inference_mode():
        fresh.install();engine,templates=prev.engine50();coef,_=prev.coefficients()
        rows=[measure(i,engine,templates,coef) for i in range(N)]
    after=prev.snapshot()
    ms,host,err,masks,outers=[np.stack(items) for items in zip(*rows)]
    q=np.quantile(ms,.9,axis=(0,1))
    boot=np.random.default_rng(570141).integers(N,size=(16000,N))
    b=np.quantile(ms[boot],.9,axis=(1,2))
    tail=.025/4
    ci=np.quantile(b[:,[0]]-b[:,1:],[tail,1-tail],axis=0).T
    result=dict(methods=METHODS,p90_ms=q.tolist(),delta_ms=(q[0]-q[1:]).tolist(),
                ci_simultaneous95=ci.tolist(),host_p90_ms=np.quantile(host,.9,axis=(0,1)).tolist(),
                catalogs=N,per_arm=N*5,sessions=N*4,material_pilots=N*2,
                outer_fixed_count=outers.sum((0,2)).tolist(),fourth_impact_count=masks.sum((0,2)).tolist(),
                max_replay_error=err.max(axis=0).tolist(),node=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'),
                gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
                before=before,after=after,start=t0,end=time.time())
    prev.r.v38.atomic_json(ROOT/'timing57.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
