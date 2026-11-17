"""Separate balanced crossover of the out-of-fold sign-bin policy and deployed controls."""
import json
import os
import time

import numpy as np
import torch

from bench54 import ROOT, fourth, prev, terminal
from sign_bins54 import bins

METHODS = ('sign_bins','adaptive_EVSI','fixed_4R','fixed_3R_I')


def choose(method, frozen_rule, raw):
    w,f=raw[2][:,:3],raw[3][:,:3]
    if method=='sign_bins':
        return frozen_rule[bins(w,f)]
    return fourth('adaptive' if method=='adaptive_EVSI' else method,w,f)


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    splits=json.loads((ROOT/'sign_bins54.json').read_text())['splits']
    torch.set_num_threads(1)
    elapsed=np.empty((12,4,4)); orders=np.empty((12,4,4),dtype=int)
    routes=np.empty((12,4,4),dtype=bool); err=np.zeros(4)
    host=np.empty((12,51,4))
    with torch.inference_mode():
        engine,templates=prev.engine50()
        signed,unsigned=prev.coefficients()
        model=prev.load_model()
        for k in range(12):
            xx,bought,disp,h,raw,selected,exp,_,alpha,*_=prev.fixture(k, model)
            # Deployment receives one immutable six-entry table, already trained on other catalogs.
            rules=[v['impact_rule'] for v in splits if k in v['test_catalogs']]
            assert len(rules)==1
            rule=np.asarray(rules[0],dtype=bool)
            rng=np.random.default_rng(544018+k)
            order=(np.array([0,1,3,2])[None,:]+np.arange(4)[:,None])%4
            orders[k]=rng.permutation(4)[order[rng.permutation(4)]]
            tr=np.zeros((4,4),dtype=int)
            np.add.at(tr,(orders[k,:,:-1],orders[k,:,1:]),1)
            np.testing.assert_array_equal(tr,1-np.eye(4,dtype=int))
            expected=[]
            for i,m in enumerate(METHODS):
                route=choose(m,rule,raw); routes[k,i]=route
                expected.append(np.where(route[:,None],exp[0],exp[5]))
            for j,seq in enumerate([rng.permutation(4),*orders[k]],start=-1):
                for i in seq:
                    torch.cuda.synchronize()
                    begin=time.perf_counter_ns()
                    choice=choose(METHODS[i],rule,raw)
                    a=terminal(raw,choice)
                    obs=prev.r.infer.actions43(xx,engine)
                    paid=prev.r.infer.paid_sensors(bought,obs)
                    p0=prev.r.infer.decode43(engine,templates[h],xx,obs,paid)[...,0]
                    arms=(obs==prev.r.r.Q).astype(np.intp)
                    pred=prev.routed_predictions(p0,disp,arms,signed[1,h],unsigned[h],np.zeros(4,dtype=np.int8),a)
                    torch.cuda.synchronize()
                    if j>=0: elapsed[k,j,i]=(time.perf_counter_ns()-begin)*1e-6
                    np.testing.assert_array_equal(choice,routes[k,i])
                    np.testing.assert_array_equal(obs,selected)
                    np.testing.assert_allclose(a,np.where(choice,alpha[0],alpha[5]),atol=2e-12,rtol=0)
                    np.testing.assert_allclose(pred,expected[i],atol=2e-9,rtol=0)
                    err[i]=max(err[i],abs(pred-expected[i]).max())
            for j in range(51):
                for i in rng.permutation(4):
                    start=time.perf_counter_ns()
                    choose(METHODS[i],rule,raw)
                    host[k,j,i]=(time.perf_counter_ns()-start)*1e-6
            print('catalog',k,flush=True)
    p90=np.quantile(elapsed,.9,axis=(0,1))
    idx=np.random.default_rng(544123).integers(12,size=(16000,12))
    boot=np.quantile(elapsed[idx],.9,axis=(1,2))
    ci=np.quantile(boot[:,[0]]-boot[:,1:],[.025/3,1-.025/3],axis=0).T
    data=dict(methods=METHODS,p90_ms=p90.tolist(),delta_ms=(p90[0]-p90[1:]).tolist(),ci_simultaneous95=ci.tolist(),
              policy_host_gate_p90=np.quantile(host,.9,axis=(0,1)).tolist(),fixture_probability=routes.mean(axis=(0,2)).tolist(),
              maximum_error=err.tolist(),samples_per_method=48,job=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name())
    np.savez_compressed(ROOT/'bins_timing54.npz',ms=elapsed,orders=orders,routes=routes,gate_ms=host)
    (ROOT/'bins_timing54.json').write_text(json.dumps(data,indent=2))
    print(data)
if __name__=='__main__':
    main()
