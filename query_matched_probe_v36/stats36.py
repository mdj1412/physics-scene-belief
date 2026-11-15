"""Paired held-out comparisons and partial identification for the fourth probe."""
from pathlib import Path
import json
import hashlib
import numpy as np
from diagnose36 import response, sensor_limits, CANDIDATES

ROOT=Path(__file__).resolve().parent
CELLS=['sinusoid_atan','sinusoid_algebraic','quartic_atan','quartic_algebraic']
REGIMES=['strong','weak','null','global_null']
METHODS=['reference_original','reference_v35','matched_original','matched_always','matched_gate3','matched_gate4']
CONTRASTS=[(5,0),(5,1),(5,4),(5,3),(5,2),(4,3),(2,0)]
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(value):
    if isinstance(value,(np.ndarray,np.integer,np.floating,np.bool_)):return value.tolist()
    raise TypeError(type(value).__name__)
def resampling(n,seed=361025,iterations=12000):
    rng=np.random.default_rng(seed)
    ids=2*rng.integers(n//2,size=(iterations,4,n//2,2))+np.array([0,1])
    return ids.reshape(iterations,4,n)
def draw(x,ids):
    return x[np.arange(4)[None,:,None],ids].mean(axis=(1,2))
def interval(draws):
    return np.quantile(draws,[.025,.975],axis=0).T
def simultaneous(point,reps):
    se=reps.std(0,ddof=1)
    maxdev=np.max(abs(reps-point)/np.where(se>1e-14,se,1.),axis=tuple(range(1,reps.ndim)))
    width=np.quantile(maxdev,.95)*se
    return np.stack((point-width,point+width),axis=-1)
def choose_scope(x,scope,axis):
    if scope=='local_equal':return np.take(x,[0,1,2],axis=axis).mean(axis)
    if scope=='four_equal':return x.mean(axis)
    return np.take(x,REGIMES.index(scope),axis=axis)

def summaries(risks,extras,limits):
    # Arrays are paired by cell, seed block, regime. Within each regime, mirrors
    # and environments are averaged equally. No regime labels enter inference.
    ids=resampling(risks.shape[1])
    data={}
    for name in ['local_equal',*REGIMES,'four_equal']:
        item=choose_scope(risks,name,2)
        ctr=np.stack([item[...,a]-item[...,b] for a,b in CONTRASTS],-1)
        p=ctr.mean((0,1)); reps=draw(ctr,ids)
        extra=choose_scope(extras,name,2)
        q=extra.mean((0,1))
        # extra columns: G3,G4,G3*d,G4*d,d,... etc. Difference vs random
        # with same average acceptance is Cov(G,d), not Gd alone.
        re=draw(extra,ids)
        cov=q[3]-q[1]*q[4]
        rc=re[:,3]-re[:,1]*re[:,4]
        data[name]=dict(loss=item.mean((0,1)),delta=p,ci_simultaneous=simultaneous(p,reps),
                        ci_pointwise=interval(reps),extras=q,ci_extras=interval(re),
                        covariance=cov,covariance_ci=np.quantile(rc,[.025,.975]),
                        cells=item.mean(1).tolist())
    true_apply=extras[:,:,0,1]; false_apply=extras[:,:,1:3,1].mean(2)
    bal=.5*(1+true_apply-false_apply)
    ba=dict(value=bal.mean(),ci=np.quantile(draw(bal,ids),[.025,.975]))
    sweep={}
    for k in (0,1,4):
        d=risks[...,5]-risks[...,k]
        ds=d[:,:,0]; dw=d[:,:,1:3].mean(2)
        p_strong=float(ds.mean()); p_weak=float(dw.mean())
        if p_strong*p_weak<0: cross=float(-p_weak/(p_strong-p_weak))
        else:cross=None
        curve=[]
        for freq in (0.,.25,.5,.75,1.):
            mixed=freq*ds+(1-freq)*dw
            curve.append([freq,float(mixed.mean()),*np.quantile(draw(mixed,ids),[.025,.975])])
        sweep[METHODS[k]]=dict(strong_proportion_crossover=cross,curve=curve)
    pair_order=['strong_vs_null','strong_vs_weak','weak_vs_null']
    sensor=[]
    for i,(h,u) in enumerate(CANDIDATES):
        item=limits[:,:,i]
        sensor.append(dict(height=h,speed=u,pairs=pair_order,
                    metrics=['privileged_bayes_accuracy','identical_mean_fraction','both_stick_fraction'],
                    values=item.mean((0,1)),accuracy_ci=interval(draw(item[...,0],ids))))
    return dict(scopes=data,balanced_accuracy_strong_vs_weak_null=ba,sensor=sensor,
                prevalence_sweep=sweep, methods=METHODS,contrasts=[[METHODS[a],METHODS[b]] for a,b in CONTRASTS])

def shift(p,z,beta):
    w=np.exp(beta*z[:,None,None])
    return p*w/(1-p+p*w)
def inspect(f,obj,surface,sign,reg):
    far,near=[f[k].astype(float) for k in ('p_far','p_near')]
    y=f['labels'].astype(bool); yy=y[...,None]
    assert far.shape==near.shape==(8,128,32)
    for p in (far,near):
        assert np.isfinite(p).all() and np.all((p>=-1.e-11)&(p<=1+1.e-11))
    beta=float(f['beta'])
    assert beta==.8289760825845313
    z=f['z']; g3=f['accept3'].astype(bool);g4=f['accept4'].astype(bool)
    assert np.array_equal(g3,f['score3']>0) and np.array_equal(g4,f['score4']>0)
    p1=shift(near,z,beta)
    pr=[far,shift(far,f['z_far'],beta),near,p1,np.where(g3[:,None,None],p1,near),np.where(g4[:,None,None],p1,near)]
    env=np.stack([np.mean((p-yy)**2,axis=(1,2)) for p in pr],axis=-1)
    np.testing.assert_allclose(env.T,f['risk'],atol=1e-12,rtol=0.)
    assert np.all((f['near_actions']==3).sum(axis=1)==13) and np.all((f['far_actions']==3).sum(axis=1)==13)
    assert np.all(np.sort(f['ids'],axis=1)[:,1:]>np.sort(f['ids'],axis=1)[:,:-1])
    np.testing.assert_array_equal(f['far_y'][:,:3],f['near_y'][:,:3])
    amp=dict(strong=1.,weak=.3,null=0.,global_null=0.)[reg]
    true_y=response(f['theta'],obj,surface,sign,amp,.025,.425)[1]
    np.testing.assert_array_equal(true_y,y)
    # At 0.023, the local-null and globally null laws coincide.
    calculated,st=response(f['theta4'],obj,surface,sign,amp,.023,.5)
    np.testing.assert_allclose(calculated,f['matched_clean'],atol=2.e-10,rtol=0)
    np.testing.assert_array_equal(st,f['matched_stick'])
    d=env[:,3]-env[:,2]
    world_change=((p1-yy)**2-(near-yy)**2).mean(-1)
    cross=.5*((world_change[:,0::2].mean(1)<0)*world_change[:,1::2].mean(1)+
              (world_change[:,1::2].mean(1)<0)*world_change[:,0::2].mean(1))
    values=[g3,g4,g3*d,g4*d,d,g3!=g4,np.mean(f['near_actions']!=f['far_actions'],axis=1),
            (g4==(d<0)),d<0,np.minimum(d,0),cross,f['score3'],f['score4']]
    soft_losses=[]
    for weight in (.325,f['near_strength3'],f['near_strength4']):
        soft=shift(near,z*weight,beta)
        soft_losses.append(float(np.mean((soft-yy)**2)))
    return env.mean(0),np.stack(values,axis=-1).mean(0),np.array(soft_losses)
def time_summary(rows):
    # Method indices are paired within replicate, block and cell.
    raw=np.stack([np.stack([rows[(c,b)] for b in range(2)]) for c in range(4)])
    assert raw.shape==(4,2,3,6)
    q=np.quantile(raw,.9,axis=(0,1,2))
    baseline=[0,1,2,3,4]
    ratio=q[-1]/q[baseline]
    from itertools import product
    rng=np.random.default_rng(360221)
    counts=rng.binomial(2,.5,size=(12000,4))
    vals={}
    for seq in product(range(3),repeat=4):
        chosen=np.stack([raw[c,np.repeat([0,1],[n,2-n])] for c,n in enumerate(seq)])
        p=np.quantile(chosen,.9,axis=(0,1,2));vals[seq]=np.log(p[-1]/p[baseline])
    reps=np.stack([vals[tuple(t)] for t in counts])
    return dict(p90_ms=q,gate4_ratio=ratio,ci=np.exp(simultaneous(np.log(ratio),reps)),
                clusters=8,call_count_per_method=24)
def auc_summary(saved,ids):
    # Kernel over pairs of seed blocks, so the same bootstrap multiplicities
    # apply to both classes and to the correlated scores from both readers.
    kernel_results={}
    ix=ids+8*np.arange(4)[None,:,None]
    flat=ix.reshape(len(ix),-1)
    weights=np.stack([(flat==i).sum(1)/32 for i in range(32)],axis=1)
    for tag,x,ys in [('strong_vs_low',0,(1,2)),('strong_vs_weak',0,(1,)),('strong_vs_null',0,(2,)),('weak_vs_null',1,(2,))]:
        columns=[]; draws=[]
        for j in range(2):
            pos=saved[:,:,x,:,:,j].reshape(32,-1)
            neg=np.take(saved[...,j],ys,axis=2).reshape(32,-1)
            diff=pos[:,None,:,None]-neg[None,:,None,:]
            kernel=((diff>0)+.5*(diff==0)).mean((2,3))
            columns.append(float(kernel.mean()))
            draws.append(np.sum((weights@kernel)*weights,axis=1))
        kernel_results[tag]=dict(auc=columns,near_ci95=np.quantile(draws[1],[.025,.975]),
                              lift=columns[1]-columns[0],lift_ci95=np.quantile(draws[1]-draws[0],[.025,.975]))
    return kernel_results

def process():
    data=ROOT/'results'
    cfg=json.loads((ROOT/'design36.json').read_text())
    assert (data/'completion.json').exists()
    risks=np.zeros((4,8,4,6));extra=np.zeros((4,8,4,13))
    strength=np.empty((4,8,3,2,8,2))
    shrunk=np.zeros((4,8,4,3))
    lim=np.zeros((4,8,len(CANDIDATES),3,3));times={}
    seedset=set();verified=0
    common=('far_x','far_y','dirs','far_actions','p_far','z','accept3','score3','theta4','theta')
    for c,cell in enumerate(CELLS):
        obj,sur=cell.split('_')
        for b in range(8):
            folder=data/cell/f'block_{b:03d}'
            meta=json.loads((folder/'meta.json').read_text())
            seeds=meta['calibration_seeds']+meta['deployment_seeds']+meta['pair']
            assert len(set(seeds))==18 and seedset.isdisjoint(seeds)
            seedset.update(seeds)
            for sgn in (1,-1):
                means=[]
                for rid,reg in enumerate(REGIMES):
                    if rid==3 and sgn==-1:continue
                    name=reg+('_plus' if sgn==1 else '_minus')+'.npz'
                    p=folder/name
                    assert sha(p)==meta['sha256'][name]
                    verified+=1
                    with np.load(p,allow_pickle=False) as f:
                        assert tuple(f['methods'])==tuple(cfg['methods'])
                        expected_start=-1 if b%2==0 else 1
                        np.testing.assert_array_equal(f['dirs'],np.tile(expected_start*np.array([1,-1,1,-1]),(8,1)))
                        vals,xt,soft=inspect(f,obj,sur,sgn,reg)
                        den=1 if rid==3 else 2
                        risks[c,b,rid]+=vals/den;extra[c,b,rid]+=xt/den;shrunk[c,b,rid]+=soft/den
                        if rid<3:
                            strength[c,b,rid,int(sgn<0)]=np.stack([f['near_strength3'],f['near_strength4']],axis=-1)
                        if rid==0:
                            cache={k:f[k] for k in common}
                            noise=f['near_y'][:,3]-f['matched_clean']
                        elif rid<3:
                            for k in common:np.testing.assert_array_equal(f[k],cache[k])
                            np.testing.assert_allclose(f['near_y'][:,3]-f['matched_clean'],noise,atol=2.e-12,rtol=0)
                        if rid<3:means.append(f['matched_clean'])
                        ms=f['full_ms']
                        if ms.size:
                            assert (c,b) not in times
                            assert np.array_equal(np.sort(f['timing_order'],axis=1),np.tile(np.arange(6),(3,1)))
                            times[c,b]=ms
                arr=sensor_limits(cache['theta4'],np.stack(means),obj,sur,sgn)
                lim[c,b]+=arr.mean(-2)/2
    results=summaries(risks,extra,lim)
    softstats={};idx=resampling(8)
    for name in ['local_equal',*REGIMES,'four_equal']:
        v=choose_scope(shrunk,name,2); rr=choose_scope(risks,name,2)
        contrasts=np.stack([rr[...,5]-v[...,0],v[...,2]-v[...,0],v[...,2]-v[...,1]],axis=-1)
        point=contrasts.mean((0,1))
        softstats[name]=dict(brier=v.mean((0,1)),contrasts=point,ci=simultaneous(point,draw(contrasts,idx)))
    results.update(auc=auc_summary(strength,idx),shrinkage=softstats,timing=time_summary(times),extras=['apply3','apply4','applied3_delta','applied4_delta','always_delta',
        'switch','deployment_action_changed','accept_matches_measured_benefit','beneficial_fraction','in_sample_oracle_delta',
        'cross_object_oracle_delta','predicted_gain3','predicted_gain4'],verified_archives=verified,distinct_seeds=len(seedset))
    (ROOT/'stats36.json').write_text(json.dumps(results,indent=2,default=finite,allow_nan=False)+'\n')
if __name__=='__main__':process()
