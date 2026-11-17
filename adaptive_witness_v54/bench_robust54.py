"""Paired replay for the broad acquisition-model gate with the nominal terminal decoder."""
from pathlib import Path
import json
import os
import time

import numpy as np
import torch

import bench_bins54 as trained
from bench54 import ROOT, prev, terminal
from policy54 import select54

METHODS = ('robust_EVSI', 'sign_bins', 'fixed_4R', 'fixed_3R_I')


def route54(method, rule, raw):
    if method == 'robust_EVSI':
        return select54(raw[2][:,:3], raw[3][:,:3], mode='robust')
    return trained.choose(method,rule,raw)


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    splits=json.loads((ROOT/'sign_bins54.json').read_text())['splits']
    ms=np.empty((12,4,4)); orders=np.empty_like(ms,dtype=int)
    routes=np.empty((12,4,4),dtype=bool); maxerr=np.zeros(4); host=np.empty((12,51))
    with torch.inference_mode():
        engine,tm=prev.engine50(); signed,unsigned=prev.coefficients(); model=prev.load_model()
        for k in range(12):
            xx,bought,disp,h,raw,selected,exp,_,a53,*_=prev.fixture(k,model)
            (rule,)=[np.asarray(s['impact_rule']) for s in splits if k in s['test_catalogs']]
            rng=np.random.default_rng(546722+k)
            sq=(np.array([0,1,3,2])[None,:]+np.arange(4)[:,None])%4
            orders[k]=rng.permutation(4)[sq[rng.permutation(4)]]
            for j,row in enumerate([rng.permutation(4),*orders[k]],start=-1):
                for m in row:
                    torch.cuda.synchronize()
                    t=time.perf_counter_ns()
                    route=route54(METHODS[m],rule,raw)
                    alpha=terminal(raw,route)
                    obs=prev.r.infer.actions43(xx,engine)
                    paid=prev.r.infer.paid_sensors(bought,obs)
                    p0=prev.r.infer.decode43(engine,tm[h],xx,obs,paid)[...,0]
                    pred=prev.routed_predictions(p0,disp,(obs==prev.r.r.Q).astype(np.intp),signed[1,h],unsigned[h],np.zeros(4,dtype=np.int8),alpha)
                    torch.cuda.synchronize()
                    if j>=0: ms[k,j,m]=(time.perf_counter_ns()-t)*1e-6
                    expected=np.where(route[:,None],exp[0],exp[5])
                    np.testing.assert_array_equal(obs,selected)
                    np.testing.assert_allclose(alpha,np.where(route,a53[0],a53[5]),rtol=0,atol=2e-12)
                    np.testing.assert_allclose(pred,expected,rtol=0,atol=2e-9)
                    maxerr[m]=max(maxerr[m],abs(pred-expected).max())
                    routes[k,m]=route
            for j in range(51):
                t=time.perf_counter_ns()
                route54('robust_EVSI',rule,raw)
                host[k,j]=(time.perf_counter_ns()-t)*1e-6
            print('catalog',k,flush=True)
    p=np.quantile(ms,.9,axis=(0,1)); ix=np.random.default_rng(547122).integers(12,size=(16000,12))
    b=np.quantile(ms[ix],.9,axis=(1,2)); ci=np.quantile(b[:,[0]]-b[:,1:],[.025/3,1-.025/3],axis=0).T
    data=dict(methods=METHODS,p90_ms=p.tolist(),delta_ms=(p[0]-p[1:]).tolist(),ci_simultaneous95=ci.tolist(),
              host_robust_gate_p90_ms=np.quantile(host,.9),fixture_impact_rate=routes.mean(axis=(0,2)).tolist(),
              samples_per_method=48,max_replay_error=maxerr.tolist(),job=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name())
    np.savez_compressed(ROOT/'robust_timing54.npz',ms=ms,method_order=orders,routes=routes,host_ms=host)
    (ROOT/'robust_timing54.json').write_text(json.dumps(data,indent=2))
    print(data)
if __name__=='__main__':
    main()
