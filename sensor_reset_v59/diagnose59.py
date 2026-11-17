"""Audit randomization, observable double mismatch, single-pilot attribution and shared-camera sensitivity."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pilot59 as sim
import route59 as router

ROOT=Path(__file__).resolve().parent
PAIRS={'selected':(0,0),'bystander':(1,1),'cross_electronic':(0,1),
       'camera':(2,2),'camera_to_selected':(2,0),'selected_to_camera':(0,2)}
def joint_counts(bits,valid,k,l,include):
    mask=valid[...,0,k]&valid[...,1,l]&include
    a,b=bits[...,0,k],bits[...,1,l]
    return np.array([mask.sum(),(mask&a).sum(),(mask&b).sum(),(mask&a&b).sum()],dtype=np.int64)

def file_stats(i):
    path=ROOT/'pilot_audit'/f'panel_{i:02d}.npz'
    obsfile=ROOT/'pilots'/path.name
    with np.load(path) as f:
        raw,truth,ids,swapped=[f[k] for k in ('signs','truth_signs','reset_id','swap')]
    with np.load(obsfile) as f: codes=f['codes']
    act=(truth[...,0]!=0)
    t=(truth[...,0]!=truth[...,1])
    per=np.empty((2,2,2,len(PAIRS),4))
    order=np.zeros((2,2,2,2,4))
    hit=np.empty((2,2,2,2,2))  # reset,camera,cause,regime, use primary/shared camera -> select physical
    trigger=np.zeros((2,2))
    for r,k,z in np.ndindex(2,2,2):
        paid=raw[r,k,z]
        inc=paid[...,0]!=0
        fl=paid[...,0]!=paid[...,1]
        target=ids[...,None]
        bits=np.stack((np.where(target==0,fl[...,0],fl[...,1]),np.where(target==0,fl[...,1],fl[...,0]),fl[...,2]),axis=-1)
        for j,(key,(a,b)) in enumerate(PAIRS.items()):
            per[r,k,z,j]=joint_counts(bits,inc,a,b,True)
        if r:
            # electronic bystander must be unaffected by the selective reset for every chronological object.
            prev=raw[0,k,z]
            a=(prev[...,0]!=prev[...,1])
            prev_b=np.where(target==0,a[...,1],a[...,0])
            assert np.array_equal(bits[...,1],prev_b)
            prev_t=np.where(target==0,a[...,0],a[...,1])
            changed=(bits[...,0]!=prev_t)&inc[...,0]
            assert not changed[...,0].any()
            trigger[k,z]=changed[...,1].sum()/inc[...,1,0].sum()
        for o in (0,1):
            order[r,k,z,o]=joint_counts(bits,inc,2,2,swapped==bool(o))
        altered=paid.copy()
        # Share electronic upstream with camera; retain camera's own local read noise and missing pattern.
        u=t[0,z]
        altered[...,2,1] *= np.where(t[k,z]!=u,-1,1).astype('i1')
        for camera in (0,1):
            ref,imp=paid[...,0].reshape(2,sim.N,8,3),paid[...,1].reshape(2,sim.N,8,3)
            np.testing.assert_array_equal(router.encode(ref,imp,ids,camera),codes[r,camera,k,z])
            source_blind=sim.encode(altered,ids,camera)
            if k:
                np.testing.assert_array_equal(source_blind,codes[r,camera,0,z])
            for s,x in enumerate((codes[r,camera,k,z],source_blind)):
                ll=router.log_odds(x,r,camera)[1]
                hit[r,camera,k,z,s]=(ll>1e-12).mean()
    return per,order,hit,trigger,np.array([ids.sum(),swapped.sum(),ids.size])

def pct(summary):
    counts=summary[...,0]
    pa=summary[...,1]/counts
    pb=summary[...,2]/counts
    p11=summary[...,3]/counts
    return p11,p11-pa*pb

def interval(x):
    return [float(x[0]),*np.quantile(x[1:],[.025,.975]).tolist()]

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    per,order,hit,trigger,rand=(np.asarray(x) for x in zip(*(file_stats(i) for i in range(24))))
    rng=np.random.default_rng(59022974)
    wt=np.concatenate([np.ones((1,24)),rng.multinomial(24,np.full(24,1/24),size=5000)])
    def agg(groups): return np.einsum('bc,c...->b...',wt,groups)
    cov_counts=agg(per)
    p11,cov=pct(cov_counts)
    order_joint=pct(agg(order))[0]
    pos=agg(hit)/24
    acc=.5*(pos[:,:,:,0]+1-pos[:,:,:,1])
    rows={}
    for r,k,z in np.ndindex(2,2,2):
        cell={}
        for j,key in enumerate(PAIRS):
            cell[key]=dict(double=interval(p11[:,r,k,z,j]),cov=interval(cov[:,r,k,z,j]),
                           n=int(cov_counts[0,r,k,z,j,0]))
        delta=order_joint[:,r,k,z,1]-order_joint[:,r,k,z,0]
        cell['order_camera_double_delta']=interval(delta)
        rows[f'{r}{k}{z}']=cell
    mechanism={}
    for r,c in np.ndindex(2,2):
        s=pos[:,r,c,:,:,0]
        a=acc[:,r,c,: ,0].mean(axis=-1)
        mechanism[f'{r}{c}']=dict(acc=interval(a),prob_vote_physical_C_P=[s[0,1].mean(),s[0,0].mean()],
            by_regime=np.mean(acc[:1,r,c,: ,0],axis=0).tolist(),paired_reset_effect=interval((acc[:,1,c,:,0]-acc[:,0,c,:,0]).mean(-1)),
            paired_camera_effect=interval((acc[:,r,1,:,0]-acc[:,r,0,:,0]).mean(-1)),
            accuracy_shared_camera_levels=[np.average((1-frac)*acc[0,r,c,:,0]+frac*acc[0,r,c,:,1]) for frac in (0,.25,.5,1.)])
    delta={}
    for k,z in np.ndindex(2,2):
        delta[f'{k}{z}']={n:interval(cov[:,1,k,z,j]-cov[:,0,k,z,j]) for j,n in enumerate(PAIRS)}
    cov_est={}
    for k,z in np.ndindex(2,2):
        j=list(PAIRS).index('camera')
        cov_est[f'{k}{z}']=interval(cov[:,0,k,z,j]/(1-2*router.CAMERA_ERROR)**2)
    rand=rand.sum(0)
    audit=dict(rows=rows,mechanism=mechanism,reset_delta_cov=delta,cam_est_true_cov=cov_est,reset_changed_active_second=trigger.mean(0).tolist(),
               selected1=rand[0]/rand[2],swap=rand[1]/rand[2],randomized_groups=int(rand[2]),n_episode_cell=24*2*sim.N,
               arch_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in ('pilots','pilot_audit') for p in sorted((ROOT/folder).glob('*.npz'))})
    (ROOT/'diagnostics59.json').write_text(json.dumps(audit,indent=2,default=lambda x:x.tolist())+'\n')
if __name__=='__main__':main()
