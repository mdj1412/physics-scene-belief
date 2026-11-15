"""CPU replay with paired clusters across material orientation, calibration schedule and policies."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'compute_matched_stopping_v32'))
from stats32 import allocations,predict,simultaneous,clean
METHODS=('broad','fixed4','uncertainty')
SCHEDULES=('same','alternating')
STAGES=np.array([2,4,8])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')
def differ(r):
    schedule=r[...,1,:]-r[...,0,:]
    within=np.stack([r[...,j,0]-r[...,j,k] for j in range(2) for k in [1,2]],axis=-1)
    return np.concatenate([schedule,within],axis=-1)
def paired_boot(blocks,draw_idx):
    return blocks[np.arange(4)[None,:,None],draw_idx].mean((1,2))
def paired_timing(times,draw_idx):
    q=np.quantile(times.reshape(-1,2,3),.90,axis=0)
    samples=[]
    for chunk in np.array_split(draw_idx,25):
        selected=times[np.arange(4)[None,:,None],chunk]
        samples.extend(np.quantile(selected.reshape(len(chunk),-1,2,3),.9,axis=1))
    samples=np.asarray(samples)
    intervals=lambda x:np.moveaxis(np.quantile(x,[.025,.975],axis=0),0,-1)
    return dict(p90=q,ci=intervals(samples),alternating_to_same=q[1]/q[0],
            schedule_ratio_ci=intervals(samples[:,1]/samples[:,0]),vs_fixed=q/q[:,1,None],
            vs_fixed_ci=intervals(samples/samples[:,:,1,None]),samples_per_method=48)
def read_source():
    dirs=[BASE/('compute_matched_stopping_v32'+suffix)/'results_train' for suffix in ('','_rep2','_rep3')]
    locks=[read(p/'model_lock.json') for p in dirs]
    training_ids=set()
    for d,lk in zip(dirs,locks):
        for name in ('broad','repeated'):
            for block in lk['training'][name]['blocks']:
                f=d/name/f"block_{block['block']:03d}"
                with np.load(f/'data.npz') as z:
                    for k in ('prefix_seed','deployment_seeds'):
                        training_ids.update(z[k].ravel().tolist())
                training_ids.update(read(f/'meta.json')['bank_pair'])
    return locks,training_ids
def main():
    root=ROOT/'results_policy';cfg=read(ROOT/'policy_design34.json')
    comp=read(root/'completion.json');assert comp['blocks']==128 and comp['independent_clusters']==32
    locks,training_ids=read_source();laws=cfg['laws']
    arr=np.empty((4,8,2,3,2,3));storerisk=np.empty((4,8,2,2,8,3))
    full=np.empty((4,2,2,3,2,3));warm=np.empty_like(full)
    switch_cost=np.empty((4,8,2,2))
    observed=[]
    maxerr=0;target=set()
    for fam in range(4):
        for b in range(8):
            across={}
            for orient in (0,1):
                law=laws[2*fam+orient]
                for sched,name in enumerate(SCHEDULES):
                    f=root/law/f'block_{b:03d}'/name
                    m=read(f/'meta.json')
                    assert m['fingerprint']==comp['fingerprint'] and m['law']==law and m['schedule']==name
                    assert m['start']==(-1,1)[b%2]
                    assert all(m[k+'_sha256']==sha(f/(k+suffix)) for k,suffix in [('metrics','.npz'),('choices','.npz'),('allocation','.json')])
                    ids=m['bank_pair']+m['calibration_seeds']+m['deployment_seeds']
                    if orient==0 and sched==0:
                        assert target.isdisjoint(ids) and training_ids.isdisjoint(ids);target.update(ids)
                    with np.load(f/'metrics.npz') as z,np.load(f/'choices.npz') as q:
                        assert list(z['methods'])==list(METHODS)
                        v=q['gaps']; ix=np.argsort(-v,axis=-1,kind='stable')[...,:13]
                        mask=np.zeros(v.shape,dtype=bool);np.put_along_axis(mask,ix,True,axis=-1)
                        assert np.all(mask.sum(-1)==13)
                        prob=z['probabilities'];labels=z['labels']
                        assert np.isfinite(prob).all() and prob.min()>=-1e-12 and prob.max()<=1+1e-12
                        loss=((prob-labels[:,:,None,None])**2).mean(-1)
                        risk=np.where(mask,loss[:,None,:,1],loss[:,None,:,0]).mean(-1)
                        storerisk[fam,b,orient,sched]=risk
                        maxerr=max(float(np.max(abs(risk-z['oracle_stage_risk']))),maxerr)
                        np.testing.assert_allclose(risk,z['oracle_stage_risk'],atol=1e-12,rtol=0)
                        expect=np.full(8,(-1,1)[b%2]);expect[1::2]*=(-1 if sched else 1)
                        np.testing.assert_array_equal(q['paid_direction'],np.tile(expect,(8,1)))
                        if sched==0: rx=q['paid_y'].copy();lab=labels.copy();prob0=prob.copy()
                        else:
                            np.testing.assert_array_equal(rx[:,::2],q['paid_y'][:,::2])
                            np.testing.assert_array_equal(labels,lab);np.testing.assert_array_equal(prob,prob0)
                        cleanx=q['paid_x'];cleany=q['paid_y'];d=q['paid_direction']
                        for si,k in enumerate(STAGES):
                            feat=q['features'][:,si] if si<2 else q['stage8_features']
                            tang=-d[:,:k]*cleany[:,:k,1];norm=cleany[:,:k,0];xx=cleanx[:,:k,-2]
                            cov=((tang-tang.mean(1,keepdims=True))*(xx-xx.mean(1,keepdims=True))).mean(1)
                            stats=np.array([tang.mean(1),tang.std(1),norm.mean(1),norm.std(1),cov]).T
                            np.testing.assert_allclose(feat[:,10:15],stats,atol=1e-12,rtol=0)
                        factual=q['factual_pool_sha256'];ids0=q['paid_ids']
                        assert all(len(set(row))==8 for row in ids0)
                        if not across:across=dict(factual=factual.copy(),ids=ids0.copy(),alpha=z['alpha'].copy(),seeds=ids)
                        else:
                            np.testing.assert_array_equal(across['factual'],factual);np.testing.assert_array_equal(across['ids'],ids0)
                            np.testing.assert_array_equal(across['alpha'],z['alpha']);assert across['seeds']==ids
                        feat=q['features']
                        for si,lk in enumerate(locks):
                            sc=predict(lk['models']['broad'],feat)
                            karr=np.stack([allocations(sc),np.full(8,4),allocations(feat[:,:,3])],axis=1);st=np.searchsorted(STAGES,karr)
                            assert np.all(karr.sum(0)==32)
                            chosen=mask[np.arange(8)[:,None],st].transpose(0,2,1)
                            bn=risk[np.arange(8)[:,None],st]
                            arr[fam,b,orient,si,sched]=bn.mean(0)
                            if si==0:
                                np.testing.assert_allclose(q['scores'],sc,atol=1e-12,rtol=0)
                                np.testing.assert_array_equal(karr,z['counts']);np.testing.assert_array_equal(chosen,z['actions']==3)
                                np.testing.assert_allclose(bn,z['brier'],atol=1e-12,rtol=0)
                                if sched==0:
                                    reference=(chosen.copy(),karr.copy())
                                else:
                                    switch_cost[fam,b,orient]=[(chosen[:,:,0]!=reference[0][:,:,0]).mean(),(karr[:,0]!=reference[1][:,0]).mean()]
                        if b<2:
                            full[fam,b,orient,:,sched]=z['full_ms'];warm[fam,b,orient,:,sched]=z['warm_ms']
                            assert np.all((z['warm_ms']>0)&(z['full_ms']>=z['warm_ms']))
                        else:assert z['full_ms'].shape==(0,3)
                    observed.append(m['metrics_sha256'])
    rng=np.random.default_rng(3402719)
    idx=2*rng.integers(4,size=(20000,4,8))+np.arange(8)[None,None,:]%2
    averaged=arr.mean(2)
    boot=paired_boot(averaged,idx)
    point=arr.mean((0,1,2));means=point[0]
    d=differ(means);ci,crit=simultaneous(d,differ(boot[:,0]))
    timing_idx=rng.integers(2,size=(10000,4,2))
    desc=['alternating minus same: '+x for x in METHODS]+[f'{sch}: broad minus {m}' for sch in SCHEDULES for m in METHODS[1:]]
    res=dict(methods=METHODS,schedules=SCHEDULES,mean=means,individual_seed=point,delta=d,contrasts=desc,
        simultaneous95=ci,critical=crit,seed_mean_delta=differ(point.mean(0)),seed_mean_ci=simultaneous(differ(point.mean(0)),differ(boot.mean(1)))[0],
        fixed_source_models=3,training_seconds=[lk['training']['broad']['labels_plus_fit_s'] for lk in locks],
        by_family=arr[:,:,:,0].mean((1,2)),by_orientation=arr[:,:,:,0].mean((0,1)),by_start=np.stack([arr[:,::2,:,0].mean((0,1,2)),arr[:,1::2,:,0].mean((0,1,2))]),
        source_model_sha256=comp['model_sha256'],n_independent_clusters=32,n_unique_deployment_objects=32768,
        total_signed_schedule_records=128,unique_target_seed_count=len(target),all_brier_arrays_sha256=observed,max_replay_error=maxerr,
        broad_fractions_motion_changed={'deployment_observation':switch_cost.mean((0,1,2))[0],'purchased_count':switch_cost.mean((0,1,2))[1]},
        full=paired_timing(full,timing_idx),warm=paired_timing(warm,timing_idx),
        timing_scope='8 paired timed clusters, pooled 48 per method and schedule, timing intervals approximate unstratified within family since one cluster per starting sign.',
        device=comp['gpu'],allocation=comp['job'])
    write(ROOT/'primary_stats34.json',res)
    print('primary',means,'contrasts',d,ci.tolist(),'p90',res['full']['p90'],flush=True)
if __name__=='__main__':main()
