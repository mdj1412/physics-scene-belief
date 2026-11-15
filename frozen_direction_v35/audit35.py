"""CPU-side reconstruction of the preregistered paired v35 comparison."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'bidirectional_probe_v34'
OBJECTS = ('sinusoid', 'quartic')
SURFACES = ('atan', 'algebraic')
SIGNS = ('plus', 'minus')
SCHEDULES = ('same', 'alternating')
FOLDS = ('rounded_fine', 'rounded_coarse', 'ridged_fine', 'ridged_coarse')
FEATURES = ('original', 'corrected', 'delta', 'quadratic', 'alignment', 'q_positive',
            'direction_accuracy', 'direction_brier', 'p_base', 'p_corrected',
            'stick_rate', 'saturation', 'fold1', 'fold2', 'fold3', 'wrong_cue', 'true_cue')
FIELDS = dict(zip(FEATURES, range(len(FEATURES))))
HASHES = {
    'diag_models.json': 'b65e66c0952076a3aa7447a50cd6312584eb0f2c531392ae25c3048c531a2464',
    'anchored34.json': 'f2df999e57f47198df349d18ea4dcabd6ed3fbd6f47340f31a5c4c943c89b3eb'
}


def load_frozen():
    records = {}
    for name, expected in HASHES.items():
        raw = (SOURCE/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
        records[name] = json.loads(raw)
    beta = np.asarray(records['anchored34.json']['beta'])[:, 0]
    assert beta[0] == 0.8289760825845313
    return records['diag_models.json']['models'], beta


def read_cue(x, y, d, models, schedule):
    # Reconstruct the deployment path independently of diag34_model or v35 GPU code.
    grid = np.linspace(-.010, .010, 9)
    slope = .1 * (x[..., :9] @ grid) / np.sum(grid**2)
    jx, jy = 1-y[..., 0], 1.10*d-y[..., 1]
    mu = d*(jy+slope*jx)/np.maximum(jx-slope*jy,.35)
    vx = x[..., -2]
    functions = [np.ones_like(mu), mu, mu**2, mu*vx, jx, jy*d, vx, x[..., -1],
                 slope, vx**2]
    functions.extend(np.exp(-.5*((mu-r)/.035)**2) for r in np.linspace(.06,.5,12))
    sensor = np.stack(functions, -1)
    feats = np.concatenate((sensor.mean(1), (sensor*d[..., None]).mean(1)), -1)
    cues = []
    for name in FOLDS:
        model = models[name][schedule]['4']
        logits = ((feats-model['center'])/model['scale']) @ model['beta'] + model['intercept']
        cues.append(2/(1 + np.exp(-np.clip(logits,-40,40)))-1)
    return np.stack(cues)


def corrected(p, z, beta):
    w = np.exp(beta*z[:, None, None])
    return p*w/(1-p+p*w)


def stats_dict(x):
    return {name: float(x[i]) for i, name in enumerate(FEATURES)}


def inspect(path, schedule, orient, block, frozen):
    with np.load(path, allow_pickle=False) as fp:
        x, y, d = [fp[name].astype(float) for name in ('paid_x','paid_y','paid_direction')]
        p, p1 = [fp[name].astype(float) for name in ('p0','p1')]
        truth = fp['labels'].astype(float)
        assert x.shape==(8,4,11) and y.shape==(8,4,2) and d.shape==(8,4)
        assert truth.shape==(8,128) and p.shape==p1.shape==(8,128,32)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert np.all((p>=-1e-12)&(p<=1+1e-12)) and np.all((p1>=-1e-12)&(p1<=1+1e-12))
        assert np.isin(truth,[0,1]).all()
        direction = np.full(4, 2*(block%2)-1)
        if schedule == 'alternating':
            direction[1::2] *= -1
        np.testing.assert_array_equal(d,np.tile(direction,(8,1)))
        models, beta = frozen
        cues = read_cue(x,y,d,models,schedule)
        err = np.abs(cues-fp['z']).max()
        np.testing.assert_allclose(cues,fp['z'],rtol=1e-4,atol=1e-4)
        rec = corrected(p,cues[0],beta[0])
        pred_err = np.abs(rec-p1).max()
        np.testing.assert_allclose(rec,p1,rtol=2e-5,atol=2e-5)
        q = (1+cues[0])/2
        sign = 1 if orient=='plus' else -1
        gt = int(sign>0)
        baseline = np.mean((p-truth[..., None])**2)
        def risk(a):
            return np.mean((a-truth[..., None])**2)
        delta = rec-p
        row = np.array([baseline, risk(p1), risk(p1)-baseline, np.mean(delta**2),
                        2*np.mean(delta*(p-truth[...,None])), q.mean(),
                        ((q>.5)==gt).mean(),np.mean((q-gt)**2),p.mean(),p1.mean(),
                        truth.mean(),((p==0)|(p==1)).mean(),
                        *[risk(corrected(p,cues[j],beta[j]))-baseline for j in (1,2,3)],
                        risk(corrected(p,-cues[0],beta[0]))-baseline,
                        risk(corrected(p,np.full(8,sign),beta[0]))-baseline])
        assert abs(row[2]-sum(row[3:5])) < 1e-6
        timing = fp['full_ms'].copy()
        if timing.size:
            assert timing.ndim==2 and timing.shape[1]==2 and np.all(timing>0) and np.isfinite(timing).all()
            order = fp['timing_order'].copy()
            # Check that each repeat contains all four schedule × method cells.
            assert order.shape==(len(timing),4) and np.all(np.sort(order, axis=-1)==np.arange(4))
        else:
            order=np.zeros((0,4),dtype=int)
        actions = fp['actions'].copy()
        assert actions.shape==truth.shape
        return (row, timing, order, truth, x, y, actions, np.array([err,pred_err]),q)


def maxband(point, draws):
    se=draws.std(0,ddof=1)
    radius=np.quantile(np.max(abs((draws-point)/np.where(se>1e-12,se,1.)),axis=-1),.95)
    return np.stack((point-radius*se,point+radius*se),-1)


def summarize(samples):
    # sample dimensions family × block × orientation × schedule × field.
    risk = samples[...,FIELDS['delta']]
    rng = np.random.default_rng(350417)
    frames = rng.integers(4,size=(12000,4,4,2))*2+np.arange(2)
    idx = frames.reshape(-1,4,8)
    delta = risk.mean((0,1,2))
    draws = risk[np.arange(4)[None,:,None],idx].mean((1,2,3))
    bands = maxband(delta,draws)
    pool_ci = np.quantile(draws.mean(1),[.025,.975])
    result = dict(pooled=dict(overall=stats_dict(samples.mean((0,1,2,3))),
                  ci95=pool_ci, improved=bool(pool_ci[1]<0),practically_replicated=bool(pool_ci[1]<-.003)))
    for s,sch in enumerate(SCHEDULES):
        sub = samples[:,:,:,s]
        groups = {}
        for i,(o,v) in enumerate(itertools.product(OBJECTS,SURFACES)):
            groups[o+'_'+v]=stats_dict(sub[i].mean((0,1)))
        for i,o in enumerate(OBJECTS):
            groups[o]=stats_dict(sub[i*2:(i+1)*2].mean((0,1,2)))
        for i,v in enumerate(SURFACES):
            groups[v]=stats_dict(sub[i::2].mean((0,1,2)))
        for j,start in enumerate(('start_negative','start_positive')):
            groups[start]=stats_dict(sub[:,j::2].mean((0,1,2)))
        for j,sgn in enumerate(SIGNS):
            groups[sgn]=stats_dict(sub[:,:,j].mean((0,1)))
        result[sch]=dict(overall=stats_dict(sub.mean((0,1,2))), groups=groups,
                         simultaneous_ci95=bands[s], practically_replicated=bool(bands[s,1]<-.003),
                         improved=bool(bands[s,1]<0),
                         per_block_range=[risk[:,:,:,s].mean(2).min(),risk[:,:,:,s].mean(2).max()])
    return result


def timing_summary(rows):
    # rows contain one paired array (orientation × schedule × repeat × method) per
    # family × timed block. Runtime observations are pooled for the percentile.
    if not rows:
        return {'timed_blocks':0}
    groups={i:[] for i in range(4)}
    orders={}
    for (i,b),(times,order) in rows.items():
        groups[i].append(times)
        for a in (0,1):
            for o in (0,1):
                for r in range(times.shape[2]):
                    pairorder=order[o,a,r]
                    first=int(np.where(pairorder==2*a)[0][0]<np.where(pairorder==2*a+1)[0][0])
                    orders.setdefault((a,first),[]).append(times[o,a,r])
    stack=np.stack([v for vv in groups.values() for v in vv])
    def calc(x):
        return np.quantile(x,.9,axis=(0,1,3))
    orig=calc(stack)
    ratio=orig[:,1]/orig[:,0]
    rng=np.random.default_rng(350418)
    draws=[]
    # Enumerate repeated resamples only when the number of unique counts is small.
    # With 2 blocks per cell, there are just 3^4 cluster histograms.
    cache={}
    for _ in range(12000):
        IDs=tuple(tuple(sorted(rng.integers(len(g),size=len(g)))) for g in groups.values())
        if IDs not in cache:
            arr=np.stack([groups[i][j] for i,ix in enumerate(IDs) for j in ix])
            p90=calc(arr)
            cache[IDs]=np.log(p90[:,1]/p90[:,0])
        draws.append(cache[IDs])
    ci=np.exp(maxband(np.log(ratio),np.stack(draws)))
    return dict(timed_blocks=len(rows),per_cell=[len(v) for v in groups.values()],
                timed_calls_per_method_schedule=int(stack.shape[0]*stack.shape[1]*stack.shape[3]),
                p90_ms=orig,ratio=ratio,ratio_simultaneous_ci95=ci,within_2pct=(ci[:,1]<1.02),
                order_p90_ms={SCHEDULES[a]+('_original_first' if first else '_corrected_first'):
                              np.quantile(v,.9,axis=0) for (a,first),v in orders.items()},
                block_leave_one_out_ratio=[
                    (lambda z:z[:,1]/z[:,0])(calc(np.delete(stack,i,axis=0))) for i in range(len(stack))])


def to_json(v):
    if isinstance(v,(np.ndarray,np.bool_,np.integer,np.floating)):
        return v.tolist()
    raise TypeError(v)


def neutral_control(root, frozen):
    # Orientation is undefined on the isotropic law. Only deployment loss is scored.
    result=np.empty((2,8,2,4))
    for o,obj in enumerate(OBJECTS):
        for b in range(8):
            for a,sch in enumerate(SCHEDULES):
                path=root/f'{obj}_isotropic_plus'/f'block_{b:03d}'/(sch+'.npz')
                row,_,_,_,_,_,_,_,q=inspect(path,sch,'plus',b,frozen)
                result[o,b,a]=row[0],row[1],row[2],np.mean(abs(2*q-1))
    rng=np.random.default_rng(350419)
    idx=(rng.integers(4,size=(12000,2,4,2))*2+np.arange(2)).reshape(-1,2,8)
    delta=result[...,2]
    center=delta.mean((0,1))
    draws=delta[np.arange(2)[None,:,None],idx].mean((1,2))
    return dict(schedules=SCHEDULES,methods=['original','corrected'],mean_brier=result.mean((0,1))[...,:2],
                delta=center,simultaneous_ci95=maxband(center,draws),
                pooled_mean_brier=result.mean((0,1,2))[:2],pooled_delta=center.mean(),
                pooled_ci95=np.quantile(draws.mean(1),[.025,.975]),
                mean_absolute_cue=result[...,3].mean((0,1)),distinct_seed_blocks=16)


def run(root, output):
    frozen = load_frozen()
    data=np.empty((4,8,2,2,len(FEATURES)))
    times={}
    errors=[]
    calibration={s:[] for s in SCHEDULES}
    action_hist={}
    for f,(obj,surf) in enumerate(itertools.product(OBJECTS,SURFACES)):
        for b in range(8):
            ts=[];ords=[];any_timed=False
            for k,sgn in enumerate(SIGNS):
                pairs=[]
                for a,sch in enumerate(SCHEDULES):
                    path=root/f'{obj}_{surf}_{sgn}'/f'block_{b:03d}'/(sch+'.npz')
                    row,timing,order,tr,x,y,actions,err,q=inspect(path,sch,sgn,b,frozen)
                    data[f,b,k,a]=row
                    pairs.append((tr,x,y))
                    calibration[sch].extend(zip(q,[1-k]*len(q)))
                    errors.append(err)
                    keys,counts=np.unique(actions,return_counts=True)
                    for key,count in zip(keys,counts):
                        action_hist[str(int(key))]=action_hist.get(str(int(key)),0)+int(count)
                    if timing.size:
                        any_timed=True
                    ts.append(timing);ords.append(order)
                for j in (0,1):
                    np.testing.assert_array_equal(pairs[0][j],pairs[1][j])
                np.testing.assert_array_equal(pairs[0][2][:,::2],pairs[1][2][:,::2])
            if any_timed:
                m=len(ts[0])
                assert all(len(t)==m and m>0 for t in ts)
                times[(f,b)]=np.stack(ts).reshape(2,2,m,2),np.stack(ords).reshape(2,2,m,4)
    direction={}
    for sch,val in calibration.items():
        qv,tv=np.array(val).T
        bins=np.digitize(qv,[.2,.4,.6,.8])
        counts=np.bincount(bins,minlength=5)
        direction[sch]=dict(count=counts,q_mean=[qv[bins==i].mean() if n else None for i,n in enumerate(counts)],
                            frequency=[tv[bins==i].mean() if n else None for i,n in enumerate(counts)])
    output.write_text(json.dumps(dict(source_hash=HASHES,errors_cue_prob_max=np.max(errors,0),
       distinct_seed_blocks=32,action_hist=action_hist,direction_reliability=direction,
       risks=summarize(data),isotropic_control=neutral_control(root,frozen),timing=timing_summary(times)),indent=2,allow_nan=False,default=to_json)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'audit35.json')
    args=parser.parse_args()
    run(args.results,args.output)
