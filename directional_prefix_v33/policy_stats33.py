"""Full policy replay, block-preserving inference, direction replication and checked GPU latency."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent; B=ROOT.parent; EV=ROOT/'results_eval'
read=lambda f:json.loads(f.read_text())
hashf=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
sys.path.insert(0,str(B/'compute_matched_stopping_v32'))
from stats32 import allocations,predict,simultaneous,risk_differences,clean,PAIRS
def boot(values,ids):
    # values: family × cluster × ..., retain orientation, seed and method dimensions intact
    return values[np.arange(4)[None,:,None],ids].mean((1,2))
def main():
    cfg=read(ROOT/'design_eval.json');laws=cfg['laws'];nb=cfg['blocks_per_law'];assert nb==8
    nam=['broad','repeated','fixed4','uncertainty']
    d=[B/('compute_matched_stopping_v32'+s)/'results_train'/'model_lock.json' for s in ('','_rep2','_rep3')];locks=[read(f) for f in d];seal=read(EV/'completion.json');assert seal['complete_blocks']==64
    raw=np.empty((4,8,2,3,4));full=np.empty((4,2,2,3,4));warm=full.copy();change=np.empty((4,8))
    swaps=np.empty((4,8,2,3)); sel=np.empty((4,8,2,3)); layer=np.empty((4,8,2,3,2));maxerror=0
    training_seeds=set()
    for lock,f in zip(locks,d):
        for arm,rec in lock['training'].items():
            for block in rec['blocks']:
                x=f.parent/arm/f"block_{block['block']:03d}"
                z=np.load(x/'data.npz')
                training_seeds.update(read(x/'meta.json')['bank_pair'])
                for k in ('prefix_seed','deployment_seeds','future_seeds'):
                    if k in z:training_seeds.update(z[k].flatten().tolist())
    used=set()
    for c in range(4):
        for i in range(nb):
            paired=[]
            for o in range(2):
                law=laws[2*c+o];p=EV/law/f'block_{i:03d}';m=read(p/'meta.json')
                assert m['model_sha256']==hashf(d[0]) and m['fingerprint']==seal['fingerprint']
                for file in ['choices.npz','metrics.npz','allocation.json']:
                    assert m[file.split('.')[0]+'_sha256']==hashf(p/file)
                with np.load(p/'metrics.npz') as z,np.load(p/'choices.npz') as q:
                    assert list(z['methods'])==nam
                    feat,unc,counts=z['features'],z['uncertainty'],z['counts']
                    gaps=q['gaps'];g=np.argsort(-gaps,axis=-1,kind='stable')[...,:13]
                    a=np.zeros(gaps.shape,dtype=bool);np.put_along_axis(a,g,True,axis=-1)
                    y,prob=z['labels'],z['probabilities']
                    assert np.isfinite(prob).all() and ((prob>=-1e-10)&(prob<=1+1e-10)).all()
                    losses=((prob-y[:,:,None,None])**2).mean(-1)
                    risks=np.where(a,losses[:,None,:,1],losses[:,None,:,0]).mean(-1)
                    maxerror=max(maxerror,float(np.max(abs(risks-z['oracle_stage_risk']))))
                    np.testing.assert_allclose(risks,z['oracle_stage_risk'],atol=1e-12,rtol=0)
                    assert np.all(a.sum(-1)==13)
                    for s,lk in enumerate(locks):
                        score=[predict(lk['models'][k],feat) for k in nam[:2]]
                        ac=np.array([allocations(sc) for sc in score]+[np.full(8,4),allocations(unc)]).T
                        assert np.all(ac.sum(0)==32)
                        st=np.searchsorted([2,4,8],ac)
                        risk=risks[np.arange(8)[:,None],st];raw[c,i,o,s]=risk.mean(0)
                        alloc_diff=ac[:,1]!=ac[:,0];swaps[c,i,o,s]=alloc_diff.mean()
                        actions=a[np.arange(8)[:,None],st].transpose(0,2,1)
                        sel[c,i,o,s]=(actions[:,:,1]!=actions[:,:,0]).mean()
                        layers=np.stack([((ac[:,1]>=4).astype(float)-(ac[:,0]>=4))*(risks[:,1]-risks[:,0]),
                                          ((ac[:,1]==8).astype(float)-(ac[:,0]==8))*(risks[:,2]-risks[:,1])],axis=-1).mean(0)
                        np.testing.assert_allclose(layers.sum(),risk[:,1].mean()-risk[:,0].mean(),atol=2e-15)
                        layer[c,i,o,s]=layers
                        if s==0:
                            for k in ['features','counts','scores','uncertainty','actions']:np.testing.assert_array_equal(q[k],z[k])
                            np.testing.assert_allclose(np.array(score).transpose(1,2,0),z['scores'],atol=1e-12)
                            np.testing.assert_array_equal(ac,counts);np.testing.assert_array_equal(actions,z['actions']==3)
                            np.testing.assert_allclose(risk,z['brier'],atol=1e-12,rtol=0)
                    if i<2:
                        full[c,i,o]=z['full_ms'];warm[c,i,o]=z['warm_ms']
                        assert (full[c,i,o]>=warm[c,i,o]).all() and (warm[c,i,o]>0).all()
                    else:assert z['full_ms'].shape==(0,4)
                    ids=m['bank_pair']+m['calibration_seeds']+m['deployment_seeds']
                    paired.append((ids,y.copy(),z['nominal_labels'].copy(),z['alpha'].copy()))
            x,y=paired
            assert x[0]==y[0] and not used.intersection(x[0]) and not training_seeds.intersection(x[0])
            used.update(x[0]);np.testing.assert_array_equal(x[2],y[2]);np.testing.assert_array_equal(x[3],y[3])
            change[c,i]=(x[1]!=y[1]).mean()
    rng=np.random.default_rng(3300401);num=20000;ids=rng.integers(8,size=(num,4,8))
    res=boot(raw,ids);byseed=raw.mean((0,1,2));means=byseed.mean(0);contr=risk_differences(means)
    crossedid=rng.integers(3,size=(num,3));bydraw=res.mean(1);cross=bydraw[np.arange(num)[:,None],crossedid].mean(1)
    family=lambda est,draw:simultaneous(est,draw)[0]
    deltas=res[:,:,:,1]-res[:,:,:,0]
    direction=(raw[...,1]-raw[...,0]).mean((0,1,3));eta=np.r_[direction,direction[0]-direction[1]]
    signs=deltas.mean(2);sg=np.column_stack((signs,signs[:,0]-signs[:,1]))
    ids2=rng.integers(2,size=(10000,4,2))
    def timing(t):
        def q90(sample):return np.quantile(sample,.90,axis=-2)
        q=q90(t.reshape(-1,4));bootq=[]
        for sub in np.array_split(ids2,50):bootq.append(q90(t[np.arange(4)[None,:,None],sub].reshape(len(sub),-1,4)))
        b=np.concatenate(bootq)
        return dict(q=q,ci=np.quantile(b,[.025,.975],axis=0).T,vsfixed=q/q[2],vsfixed_ci=np.quantile(b/b[:,2,None],[.025,.975],axis=0).T)
    out=dict(mean=means,byseed=byseed,delta=contr,ci=family(contr,risk_differences(bydraw.mean(1))),crossed_ci=family(contr,risk_differences(cross)),
        orientation=eta,orientation_ci=family(eta,sg),by_condition=raw.mean((1,3)),seed_by_sign=(raw[...,1]-raw[...,0]).mean((0,1)).T,
        allocations_changed=swaps.mean(),observations_changed=sel.mean(),stage_contributions=layer.mean((0,1,3)),
        mirror_label_change=change.mean(),by_family_label_change=change.mean(1),full=timing(full),warm=timing(warm),brier_audit_error=maxerror,
        n_unique_target_seeds=len(used),source_seed_records=len(training_seeds),training=[{k:[l['training'][k]['n_environments'],l['training'][k]['labels_plus_fit_s']] for k in nam[:2]} for l in locks],
        hashes=[hashf(f) for f in d],job=seal['job'],gpu=seal['gpu'],CI_draws=num)
    (ROOT/'policy_stats33.json').write_text(json.dumps(clean(out),indent=2)+'\n')
    print('Audited',len(used),'distinct evaluation streams;',means,contr,eta)
if __name__=='__main__':main()
