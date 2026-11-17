"""Time measured dual-instrument pilot outcomes through 512 fresh GPU forecasts."""
import os, json, socket, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'fresh_surface_v55'))
from bench55 import serial_alpha, terminal, prev
from policy54 import select54
import route58 as router
import generate58 as fresh

METHODS=('dual4_map','dual8_map','dual4_risk','single4_naive','gate','fixed')
MIXTURE=np.array([[.42,0,.58],[0,.84,.16]])

def fixture(i):
    rng=np.random.default_rng([5804713,i])
    material=np.arange(2)
    regime=(material+i)%2
    rho=(i//2+material)%2
    latch=(i//4+material)%2
    surface=((np.arange(4)+i)%4)//2
    with np.load(ROOT/'pilot_audit'/f'panel_{i:02d}.npz') as bank:
        packed=bank['raw_signed']
        signs=np.stack([packed[:,rho[j],latch[j],regime[j],j,0] for j in material]).reshape(2,2,8)
    active=(signs&16)!=0
    def unpack(bits):
        return np.where(active[...,None],1-2*((signs[...,None]>>np.array(bits))&1),0).astype('i1')
    ref,imp=unpack([0,1]),unpack([2,3])
    with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as meta:
        code=np.stack([meta['codes'][:,rho[j],latch[j],regime[j],j,0] for j in material])
    np.testing.assert_array_equal(router.encode(ref,imp),code)
    assert ref.shape==imp.shape==(2,2,8,2)
    d=fresh.raw(i);env=np.arange(4)
    pre=rng.integers(5,size=4)
    post=np.where(rng.random(4)<.4375,4-pre,pre)
    eps=np.array([rng.choice(3,p=MIXTURE[j]) for j in regime[surface]])
    rep=rng.integers(256,size=4)
    raw=(d['ow'][pre,env,rep],d['of'][pre,env,rep],d['nw'][post,env,rep],
         d['nf'][post,env,rep],d['imp'][eps,post,env,rep],None)
    return raw,(ref,imp,surface),code,np.stack((pre,post,eps,rep,surface,regime[surface],rho[surface],latch[surface]))

def acquire(method,raw,pilot):
    ref,imp,surface=pilot
    if method=='gate': choice=np.zeros(2,dtype=bool)
    elif method=='fixed':choice=np.ones(2,dtype=bool)
    else:
        k=int(method!='dual8_map')
        if method=='single4_naive':
            # The single-sensor baseline has no access to sensor B.
            ref1,imp1=ref[:,k,:,0],imp[:,k,:,0]
            sym=np.where((ref1!=0)&(imp1!=0),1+(ref1!=imp1).astype('u1'),0)
            paid=5*sym[:,::2]+sym[:,1::2]
        else:paid=router.encode(ref[:,k],imp[:,k])
        choice=router.route(paid,method)
    mask=choice[surface]
    run_gate=~mask
    if run_gate.any():mask[run_gate]=select54(raw[2][run_gate,:3],raw[3][run_gate,:3],mode='robust')
    return mask

def order(i):
    rng=np.random.default_rng([5808191,i//2])
    names=rng.permutation(6);rows=rng.permutation(6)
    return names[(np.array([0,1,5,2,4,3])+rows[3*(i%2):3*(i%2+1),None])%6]

def measure(i,engine,templates,coef):
    raw,pilot,codes,params=fixture(i)
    baselines=serial_alpha(raw)
    nominal=select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
    outer=np.stack([router.route(codes[:,int(m!='dual8_map')],m) for m in METHODS])
    masks=outer[:,pilot[2]] | nominal
    with np.load(ROOT/'forecasts'/f'panel_{i:02d}.npz') as f:
        plus,minus,actions=[f[k] for k in ('pplus','pminus','actions')]
    obs=fresh.observe(i);times=np.empty((3,6));errs=np.zeros(6)
    rng=np.random.default_rng([5802137,i]);orders=order(i)
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
    path=ROOT/'timing58'/f'panel_{i:02d}.npz'
    path.parent.mkdir(exist_ok=True)
    prev.r.v38.atomic_npz(path,ms=times,maximum_error=errs,params=params,reference=pilot[0],impact=pilot[1],
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
    draws=np.random.default_rng(5808273).integers(24,size=(16000,24))
    boot=np.quantile(ms[draws],.9,axis=(1,2))
    deltas=boot[:,[0]]-boot[:,1:]
    result=dict(methods=METHODS,p90_ms=q.tolist(),delta_map_vs_others_ms=(q[0]-q[1:]).tolist(),
           simultaneous95=np.quantile(deltas,[.005,.995],axis=0).T.tolist(),
           map_vs_8x1_pointwise95=np.quantile(deltas[:,0],[.025,.975]).tolist(),
           catalog_count=24,batch_objects=512,measured_per_arm=72,pilot_count=48,session_count=96,
           outer_fixed=outer.sum((0,2)).tolist(),bought_impact=masks.sum((0,2)).tolist(),max_error=err.max(0).tolist(),
           position=order_matrix.tolist(),adjacency=adjacency.tolist(),
           before=before,after=prev.snapshot(),gpu=torch.cuda.get_device_name(),node=socket.gethostname(),
           job=os.getenv('SLURM_JOB_ID'),cuda=torch.version.cuda,torch=str(torch.__version__))
    prev.r.v38.atomic_json(ROOT/'timing58.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
