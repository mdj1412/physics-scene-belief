"""Time measured a one-sensor reset and a simultaneous independent camera through 512 fresh GPU forecasts."""
import os, json, socket, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'fresh_surface_v55'))
from bench55 import serial_alpha, terminal, prev
from policy54 import select54
import route59 as router
import generate59 as fresh

METHODS=('map_r1c1','map_r0c1','map_r1c0','map_r0c0','gate','fixed')
FLAGS=((1,1),(0,1),(1,0),(0,0))
MIXTURE=np.array([[.42,0,.58],[0,.84,.16]])

def fixture(i):
    rng=np.random.default_rng([5904713,i])
    material=np.arange(2)
    regime=(material+i)%2
    cause=(i//2+material)%2
    surface=((np.arange(4)+i)%4)//2
    # Both candidate policies see the same 8 specimens, the same material,
    # and the same randomized target even when its reset is sham.
    with np.load(ROOT/'pilot_audit'/f'panel_{i:02d}.npz') as bank:
        packed=bank['signs']
        paid=np.stack([packed[:,cause[j],regime[j],j,0] for j in material])
        target=bank['reset_id'][:,0].astype(int)
    assert paid.shape==(2,2,4,2,3,2) and target.shape==(2,4)
    paid=paid.reshape(2,2,8,3,2)
    ref,imp=paid[...,0],paid[...,1]
    with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as meta:
        ranges=meta['codes']
        codes=np.stack([ranges[:,:,cause[j],regime[j],j,0] for j in material])
    for reset,camera in FLAGS:
        np.testing.assert_array_equal(router.encode(ref[:,reset],imp[:,reset],target,camera=bool(camera)),codes[:,reset,camera])
    d=fresh.raw(i);env=np.arange(4)
    pre=rng.integers(5,size=4)
    post=np.where(rng.random(4)<.4375,4-pre,pre)
    eps=np.array([rng.choice(3,p=MIXTURE[j]) for j in regime[surface]])
    rep=rng.integers(256,size=4)
    raw=(d['ow'][pre,env,rep],d['of'][pre,env,rep],d['nw'][post,env,rep],
         d['nf'][post,env,rep],d['imp'][eps,post,env,rep],None)
    return raw,(ref,imp,target,surface),codes,np.stack((pre,post,eps,rep,surface,regime[surface],cause[surface]))

def acquire(method,raw,pilot):
    ref,imp,target,surface=pilot
    if method=='gate':
        choice=np.zeros(2,dtype=bool)
    elif method=='fixed':
        choice=np.ones(2,dtype=bool)
    else:
        reset,camera=FLAGS[METHODS.index(method)]
        paid=router.encode(ref[:,reset],imp[:,reset],target,camera=bool(camera))
        choice=router.route(paid,method)
    mask=choice[surface]
    run_gate=~mask
    if run_gate.any():mask[run_gate]=select54(raw[2][run_gate,:3],raw[3][run_gate,:3],mode='robust')
    return mask

def order(i):
    rng=np.random.default_rng([5908191,i//2])
    names=rng.permutation(6);rows=rng.permutation(6)
    return names[(np.array([0,1,5,2,4,3])+rows[3*(i%2):3*(i%2+1),None])%6]

def measure(i,engine,templates,coef):
    raw,pilot,codes,params=fixture(i)
    baselines=serial_alpha(raw)
    nominal=select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
    outer=np.stack([router.route(codes[:,reset,camera],m) for m,(reset,camera) in zip(METHODS,FLAGS)] + [np.zeros(2,bool),np.ones(2,bool)])
    masks=outer[:,pilot[3]] | nominal
    with np.load(ROOT/'forecasts'/f'panel_{i:02d}.npz') as f:
        plus,minus,actions=[f[k] for k in ('pplus','pminus','actions')]
    obs=fresh.observe(i);times=np.empty((3,6));errs=np.zeros(6)
    rng=np.random.default_rng([5902137,i]);orders=order(i)
    for j,seq in enumerate([rng.permutation(6),*orders],-1):
        for k in seq:
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            mask=acquire(METHODS[k],raw,pilot)
            alpha=terminal(raw,mask)
            pp,pm,act=fresh.v55.forward55(obs,engine,templates,coef)
            pred=alpha[:,None]*pp+(1-alpha[:,None])*pm
            torch.cuda.synchronize()
            ms=(time.perf_counter_ns()-start)/1e6
            if j>=0:times[j,k]=ms
            np.testing.assert_array_equal(mask,masks[k])
            np.testing.assert_array_equal(act,actions)
            expected=baselines[mask.astype(int),np.arange(4)]
            np.testing.assert_allclose(alpha,expected,rtol=0,atol=2e-12)
            error=np.abs(pred-(expected[:,None]*plus+(1-expected[:,None])*minus)).max()
            assert error<2e-9
            errs[k]=max(errs[k],error)
    path=ROOT/'timing59'/f'panel_{i:02d}.npz'
    path.parent.mkdir(exist_ok=True)
    prev.r.v38.atomic_npz(path,ms=times,maximum_error=errs,params=params,reference=pilot[0],impact=pilot[1],reset_id=pilot[2],
                         codes=codes,outer=outer,masks=masks,orders=orders)
    print(i,np.round(np.quantile(times,.90,axis=0),2),flush=True)
    return times,errs,outer,masks,orders

def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    before=prev.snapshot()
    with torch.inference_mode():
        fresh.previous.install();engine,templates=prev.engine50();coef=prev.coefficients()[0]
        rows=[measure(i,engine,templates,coef) for i in range(24)]
    ms,err,outer,masks,orders=[np.stack(a) for a in zip(*rows)]
    order_matrix=np.array([np.bincount(x,minlength=6) for x in orders.reshape(-1,6).T])
    assert np.all(order_matrix==12)
    adjacency=np.zeros((6,6),int)
    for row in orders.reshape(-1,6):np.add.at(adjacency,(row[:-1],row[1:]),1)
    assert np.all(adjacency[~np.eye(6,dtype=bool)]==12)
    q=np.quantile(ms,.9,axis=(0,1))
    draws=np.random.default_rng(5908273).integers(24,size=(16000,24))
    boot=np.quantile(ms[draws],.9,axis=(1,2))
    comparisons=[(0,1),(2,3),(0,2),(1,3)]
    delta=np.stack([boot[:,i]-boot[:,j] for i,j in comparisons],axis=-1)
    result=dict(methods=METHODS,p90_ms=q.tolist(),
           contrast_labels=[f'{METHODS[i]} minus {METHODS[j]}' for i,j in comparisons],
           delta_ms=[q[i]-q[j] for i,j in comparisons],
           simultaneous95=np.quantile(delta,[.025/4,1-.025/4],axis=0).T.tolist(),
           pointwise95=np.quantile(delta,[.025,.975],axis=0).T.tolist(),
           catalog_count=24,batch_objects=512,measured_per_arm=72,pilot_count=48,session_count=96,
           outer_fixed=outer.sum((0,2)).tolist(),bought_impact=masks.sum((0,2)).tolist(),max_error=err.max(0).tolist(),
           position=order_matrix.tolist(),adjacency=adjacency.tolist(),
           before=before,after=prev.snapshot(),gpu=torch.cuda.get_device_name(),node=socket.gethostname(),
           job=os.getenv('SLURM_JOB_ID'),cuda=torch.version.cuda,torch=str(torch.__version__))
    prev.r.v38.atomic_json(ROOT/'timing59.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
