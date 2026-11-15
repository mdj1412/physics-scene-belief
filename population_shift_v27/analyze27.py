"""Paired estimates on the fixed environment panel with three crossed replicate/bank blocks."""
from pathlib import Path
import json
import hashlib
import os
import numpy as np

ROOT=Path(__file__).resolve().parent
RESULT=ROOT/'results'
CFG=json.loads((ROOT/'design.json').read_text())
names=['estimated','frozen','k256','known','query','neutral']
native=np.array([0,1,1,2,1,3])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_case(e):
    d=RESULT/e['id']
    m=json.loads((d/'metrics.json').read_text())
    assert m['environment']==e and sha(d/'metrics.npz')==m['npz_sha256'] and sha(d/'choice_lock.npz')==m['choices_sha256']
    with np.load(d/'metrics.npz') as f:
        a=f['action']; y=f['label']; probs=f['probabilities']; cost=f['cost']; losses=f['loss']; times=f['times_ms']
        assert list(f['methods'])==names and set(a.ravel())<={1,3}
        assert np.allclose(((probs-y[:,None,None,None])**2).mean(-1),cost,atol=1e-13)
        assert np.allclose(losses,cost[np.arange(len(y))[:,None],native[None,:],(a==3).astype(int)],atol=1e-13)
        assert all(a[:,4]==1)
        known=cost[np.arange(len(y))[:,None],2,(a==3).astype(int)]
    setup=m['setup_ms']
    es=np.array([setup['engines'][k] for k in ('estimated','source','source','known','source','neutral')])
    es[0]+=setup['detector_fit']
    return dict(d=m,loss=np.concatenate([losses,known],axis=-1),a=a,cost=cost,t=times,am=times+es[None,None,:]/len(y),setup=es)
def metrics(entities):
    loss=np.stack([c['loss'] for c in entities]);act=np.stack([c['a'] for c in entities]);setup=np.stack([c['setup'] for c in entities])
    times=np.stack([c['t'] for c in entities]);am=np.stack([c['am'] for c in entities])
    alphas=np.array([c['d']['environment']['alpha'] for c in entities])
    est=np.array([c['d']['estimate']['alpha'] for c in entities]); ci=np.array([c['d']['estimate']['ci95'] for c in entities])
    fit=[c['d']['setup_ms']['detector_fit'] for c in entities]
    logs=np.array([c['d']['estimate']['log_bayes_factor_vs_source'] for c in entities])
    return dict(native=loss.mean((0,1))[:6].tolist(),common=loss.mean((0,1))[6:].tolist(),p90=np.quantile(times.reshape(-1,6),.9,axis=0).tolist(),
        p50=np.median(times.reshape(-1,6),axis=0).tolist(),amortized_p90=np.quantile(am.reshape(-1,6),.9,axis=0).tolist(),
        overhead_ms=setup.mean(0).tolist(),fit_ms=float(np.mean(fit)),alpha_mean=est.mean(0).tolist(),profile_mae=float(np.abs(est[:,0]-alphas[:,0]).mean()),
        strong_source_bf=int((logs>np.log(10.)).sum()), logbf_mean=float(logs.mean()),ci_c_mean_width=float(np.mean(ci[:,0,1]-ci[:,0,0])),
        mean_information=np.mean([c['d']['estimate']['posterior_kl_nats'] for c in entities],axis=0).tolist(),
        profile_coverage=float(np.mean((ci[:,0,0]<=alphas[:,0])&(ci[:,0,1]>=alphas[:,0]))),probe_fraction=(act==3).mean((0,1)).tolist(),
        choices_differ_from_est=(act!=act[:,:,[0]]).mean((0,1)).tolist(),
        estimated_switch_vs_source=(((act[:,:,0]!=act[:,:,2]))*1.).mean().item(),
        profile_sign_accuracy=float(np.mean(np.sign(est[:,0])==np.sign(alphas[:,0]))) if not np.any(alphas[:,0]==0) else None)
def contrasts(v):
    return np.concatenate([v[...,0:1]-v[...,1:6],v[...,6:7]-v[...,7:12]],axis=-1)
def bootstrap(x,draws=10000):
    # R independent replicate blocks; each selected block has its own independent resampled test worlds.
    g,r,n,d=x.shape
    rng=np.random.default_rng(27590121);out=np.empty((draws,g,d))
    for s in range(0,draws,100):
        h=min(100,draws-s)
        reps=rng.integers(r,size=(h,r))
        subtotal=np.zeros((h,g,d))
        for j in range(r):
            w=rng.integers(n,size=(h,g,n))
            subtotal+=x[np.arange(g)[None,:,None],reps[:,j,None,None],w].mean(2)
        out[s:s+h]=subtotal/r
    return out
def bands(draws,point):
    sd=draws.std(0,ddof=1)
    z=np.max(abs(draws-point)/np.where(sd>1.e-14,sd,1.),axis=-1)
    q=np.quantile(z,.95)
    return np.stack([point-q*sd,point+q*sd],axis=-1)
def timing_ci(t):
    # Pair methods and timing repetitions at the level of timed worlds.
    rng=np.random.default_rng(272031)
    g,r,w,n,k=t.shape
    out=[]
    for _ in range(2000):
        reps=rng.integers(r,size=r)
        idx=rng.integers(w,size=(g,r,w))
        sampled=t[np.arange(g)[:,None,None],reps[None,:,None],idx]
        p=np.quantile(sampled.reshape(-1,k),.9,axis=0)
        out.append(p[0]/p)
    return np.quantile(out,[.025,.975],axis=0).T.tolist()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    cases=[read_case(e) for e in CFG['environments']]
    assert len(set(c['d']['fingerprint'] for c in cases))==1
    families=list(dict.fromkeys(e['family'] for e in CFG['environments']))
    targets=[e['family'] for e in CFG['environments'] if e['rep']==0 and e['target']]
    groups={f:sorted((c for c in cases if c['d']['environment']['family']==f),key=lambda c:c['d']['environment']['rep']) for f in families}
    target=[c for f in targets for c in groups[f]]
    x=np.array([[c['loss'] for c in groups[f]] for f in targets])
    all_con=contrasts(x)
    samples=bootstrap(all_con);obs=all_con.mean((1,2))
    pooled=metrics(target);pt=obs.mean(0)
    pooled['deltas']=pt.tolist();pooled['sim95']=bands(samples.mean(1),pt).tolist()
    pooled['leave_one_block_out']=np.stack([np.delete(all_con,i,axis=1).mean((0,1,2)) for i in range(3)]).tolist()
    pooled['time_ratio_ci95']=timing_ci(np.stack([[c['t'] for c in groups[f]] for f in targets]))
    per={}
    for f in families:
        per[f]=metrics(groups[f])
        if f in targets:
            i=targets.index(f);per[f]['contrast_ci95']=np.quantile(samples[:,i],[.025,.975],axis=0).T.tolist()
    prefixes={}
    truth=np.array([c['d']['environment']['alpha'][0] for c in target])
    for n in [64,256,1024]:
        ests=[c['d']['estimate'] if n==1024 else c['d']['prefixes'][str(n)] for c in target]
        hats=np.array([v['alpha'][0] for v in ests])
        probs=np.array([v['posterior_prob_negative'][0] for v in ests])
        prefixes[n]=dict(mae=float(np.mean(abs(hats-truth))),sign_accuracy=float(np.mean((hats<0)==(truth<0))),
            sign_confident_n=int(((probs>.95)|(probs<.05)).sum()),mean_kl=float(np.mean([v['posterior_kl_nats'][0] for v in ests])))
    result=dict(names=names,target_laws=targets,pooled=pooled,by_family=per,prefixes=prefixes,
        init_ms=target[0]['d']['setup_ms']['detector_static'],device=target[0]['d']['device'],
        jobs=list(set(c['d']['job'] for c in cases)),worlds=sum(len(c['a']) for c in target),control_worlds=sum(len(c['a']) for c in cases if not c['d']['environment']['target']))
    (RESULT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Computed',result['worlds'],'worlds',pooled['native'])
if __name__=='__main__':main()
