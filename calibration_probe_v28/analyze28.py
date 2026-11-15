"""Paired estimates conditional on the fixed panel of shift laws."""
import hashlib,json,os
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
R=P/'results'
cfg=json.loads((P/'design.json').read_text())
groups=list(dict.fromkeys(e['family'] for e in cfg['environments']))
budgets=[0,1,4,16]
pol=['probe0','probe1','probe4','probe16','neutral','known']
pairs=[(j,4) for j in range(4)]+[(j,5) for j in range(4)]+[(j,0) for j in (1,2,3)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest_data(e):
    d=R/e['id']
    m=json.loads((d/'metrics.json').read_text())
    assert m['environment']==e and m['bank_pair']==cfg['bank_pairs'][e['rep']] and m['methods']==pol
    assert sha(d/'metrics.npz')==m['npz_sha256'] and sha(d/'choice_lock.npz')==m['choices_sha256']
    with np.load(d/'metrics.npz') as f, np.load(d/'choice_lock.npz') as lock:
        a=f['action'];w=len(a);label=f['label'];c=f['cost'];v=f['probabilities'];time=f['times_ms'];x=f['loss'];z=f['common_known']
        assert np.array_equal(a,lock['action']) and np.isin(a,(1,3)).all()
        assert np.max(abs(((v-label[:,None,None,None])**2).mean(-1)-c))<1.e-12
        assert np.allclose(x,c[np.arange(w)[:,None],np.arange(6)[None,:],(a==3).astype(int)])
        assert np.allclose(z,c[np.arange(w)[:,None],5,(a==3).astype(int)])
        assert len(set(lock['probe_order']))==16 and len(lock['x_cal'])==1024
    est=m['estimates'];tim=m['setup_ms']
    nn=np.array([est[str(b)]['posterior_prob_negative'][1] for b in budgets])
    assert nn[0]==.5 and est['0']['alpha'][1]==0
    fit=np.array([tim['detector_fit'][str(b)] for b in budgets]+[0.,0.])
    setup=np.array([tim['engines'][k] for k in pol])+fit
    assert time.shape==(cfg['timing_worlds'],cfg['timing_repeats'],6)
    return dict(m=m,loss=np.concatenate([x,z],-1),a=a==3,nn=nn,fit=fit,setup=setup,time=time,
        alpha_hat=np.array([est[str(b)]['alpha'] for b in budgets]),sym=[est[str(b)]['predictive_tv_vs_product'] for b in budgets])
def bnds(draw,pt):
    sd=draw.std(0,ddof=1)
    mx=np.max(abs(draw-pt)/np.where(sd>1.e-14,sd,1.),axis=-1)
    k=np.quantile(mx,.95);return np.stack([pt-k*sd,pt+k*sd],-1)
def world_draws(x,seed,draws=10000):
    # A finite library of sampled means per cohort bounds memory. Each chosen
    # replicate occurrence independently draws a fresh bootstrap world-mean index.
    g,r,w,k=x.shape;rng=np.random.default_rng(seed);poolsize=4096
    library=np.empty((g,r,poolsize,k))
    for l in range(g):
        for b in range(r):
            counts=rng.multinomial(w,np.full(w,1/w),size=poolsize)
            library[l,b]=counts@x[l,b]/w
    rep=rng.integers(0,r,(draws,r));pool=rng.integers(0,poolsize,(draws,g,r))
    return library[np.arange(g)[None,:,None],rep[:,None,:],pool].mean(axis=2)
def contr(x):
    return np.stack([x[...,off+i]-x[...,off+j] for off in (0,6) for i,j in pairs],-1)
def summarize(cases):
    x=np.stack([c['loss'] for c in cases]);a=np.stack([c['a'] for c in cases])
    tt=np.stack([c['time'] for c in cases]); setup=np.stack([c['setup'] for c in cases]); fit=np.stack([c['fit'] for c in cases])
    n=np.stack([c['nn'] for c in cases]);true=np.array([c['m']['environment']['alpha'] for c in cases])
    correct=np.where(true[:,1:2]<0,n,1-n)
    hats=np.stack([c['alpha_hat'] for c in cases])
    hz=len(a[0]);paid=np.array([*budgets,0,0])/hz
    amt=tt+setup[:,None,None,:]/hz
    high=(n>.95)|(n<.05)
    return dict(native=x.mean((0,1))[:6].tolist(),common=x.mean((0,1))[6:].tolist(),probe=a.mean((0,1)).tolist(),
        total=(a.mean((0,1))+paid).tolist(),switch0=(a!=a[:,:,[0]]).mean((0,1)).tolist(),diff_known=(a!=a[:,:,[5]]).mean((0,1)).tolist(),
        p90=np.quantile(tt.reshape(-1,6),.9,axis=0).tolist(),amortized=np.quantile(amt.reshape(-1,6),.9,axis=0).tolist(),fit=fit.mean(0).tolist(),setup=setup.mean(0).tolist(),
        profile_mae=np.abs(hats[:,:,0]-true[:,:1]).mean(0).tolist(),friction_mae=np.abs(hats[:,:,1]-true[:,1:2]).mean(0).tolist(),
        mean_af=hats[:,:,1].mean(0).tolist(),confidence=high.mean(0).tolist(),tv=np.mean([c['sym'] for c in cases],axis=0).tolist(),
        signacc=(np.mean((correct>.5)+.5*(correct==.5),axis=0).tolist() if (true[:,1]!=0).all() else None),
        signprob=correct.mean(0).tolist() if (true[:,1]!=0).all() else None, wrong_conf_count=(correct<.05).sum(0).tolist() if (true[:,1]!=0).all() else None)
def timing_int(tm):
    rng=np.random.default_rng(2866901);g,r,w,t,k=tm.shape;out=[]
    for _ in range(1500):
        b=rng.integers(r,size=r)
        a=rng.integers(w,size=(g,r,w))
        v=tm[np.arange(g)[:,None,None],b[None,:,None],a]
        q=np.quantile(v.reshape(-1,k),.9,axis=0)
        out.append(q/q[4])
    return np.quantile(out,[.025,.975],axis=0).T.tolist()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    cases={e['id']:digest_data(e) for e in cfg['environments']}
    gp={f:[cases[e['id']] for e in cfg['environments'] if e['family']==f] for f in groups}
    tg=[f for f in groups if gp[f][0]['m']['environment']['target']]
    x=np.array([[v['loss'] for v in gp[f]] for f in tg]);a=np.array([[v['a'] for v in gp[f]] for f in tg],dtype=float)
    alltar=[v for f in tg for v in gp[f]]
    point=x.mean((0,1,2));delta=contr(point);b=world_draws(np.concatenate([x,a],-1),2897071)
    bs=contr(b[...,:12]).mean(1); cb=bnds(bs,delta)
    nblocks=x.shape[1];rng=np.random.default_rng(283751);blocks=rng.integers(nblocks,size=(10000,nblocks))
    score=np.array([[v['nn'] if v['m']['environment']['alpha'][1]<0 else 1-v['nn'] for v in gp[f]] for f in tg])
    success=(score>.5)+.5*(score==.5)
    sign_block=success.mean(0);sb=sign_block[blocks].mean(1)
    signbounds=bnds(sb[:,1:],success.mean((0,1))[1:])
    time=np.array([[v['time'] for v in gp[f]] for f in tg]);summary=summarize(alltar)
    # Calibration-only and complete per-event break-even pricing vs probe0 and neutral at deployment size H=128.
    cost_ex={}
    for j in [0,4]:
        cc=[]
        for i in [1,2,3]:
            gain=point[j]-point[i];inc=budgets[i]/cfg['n_test']
            extra=inc+np.mean(a[...,i]-a[...,j])
            cc.append(dict(overhead=inc,all_increment=float(extra),risk_gain=float(gain),break_even_cal=float(gain/inc),
                break_even_all=float(gain/extra) if extra!=0 else None))
        cost_ex[pol[j]]=cc
    by={f:summarize(gp[f]) for f in gp}
    allsig=[c['m']['fingerprint'] for c in cases.values()];assert len(set(allsig))==1
    result=dict(pooled=summary,laws=by,target=tg,methods=pol,pairs=pairs,deltas=delta.tolist(),bands=cb.tolist(),
        leave_one_out=[contr(np.delete(x,i,axis=1).mean((0,1,2))).tolist() for i in range(nblocks)],
        signbands=signbounds.tolist(),timing_ratio95=timing_int(time),costs=cost_ex,
        detector_static=alltar[0]['m']['setup_ms']['detector_static'],fingerprint=allsig[0],job=alltar[0]['m']['job'],gpu=alltar[0]['m']['gpu'])
    (R/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('native',summary['native'],'sign',summary['signacc'],flush=True)
if __name__=='__main__':main()
