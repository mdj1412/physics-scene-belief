"""Matched-panel inference for the preregistered v40 population and paired controls."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
FAMILIES = ('shelf', 'ridge')
CONTOURS = ('base', 'hollow', 'bulge')
LAWS = ('saturating', 'crossover')
CELLS = ['_'.join(c) for c in itertools.product(FAMILIES, CONTOURS, LAWS)]
BETA = .8289760825845313
METHODS = ('fixed', 'optimizer')
rng = np.random.default_rng(402504)
SAMPLES = rng.integers(12, size=(16000, 12))


def interval(data, comparisons=1):
    data = np.asarray(data)
    n = len(data)
    mean = data.mean(0)
    radius = t.ppf(1-.025/comparisons, n-1) * data.std(0, ddof=1) / np.sqrt(n)
    return dict(mean=mean, ci=[mean-radius, mean+radius])


def archive(cell, panel, block):
    path = ROOT/'results'/cell/f'catalog_{panel:02d}'/f'block_{block:02d}.npz'
    with np.load(path) as f:
        return {k: f[k] for k in ('risk','p_base','z','weights','actions','labels',
                                    'fourth_stick','ids','first3_y','fourth_y','full_ms','alpha')}


def predictions(base, z, w):
    amp = np.exp(BETA * z[:,None,:,None,None] * w)
    base = base[:,None]
    return (amp*base) / (1-base+amp*base)


def contrast(base, cues, labels):
    fixed = predictions(base, cues['z'], .325)
    opt = predictions(base, cues['z'], cues['weights'][:,:,:,None,None])
    target = labels[...,None]
    return ((opt-fixed)*(opt+fixed-2*target)).mean()


def costs(times):
    # Two cell timing clusters per independent panel. Keep their methods and repetitions paired.
    a=np.stack([np.stack(items) for items in times])
    assert a.shape==(12,2,3,2) and np.all(a>0)
    sample=np.quantile(a[SAMPLES],.9,axis=(1,2,3))
    p90=np.quantile(a,.9,axis=(0,1,2))
    return dict(p90=p90,ci=np.quantile(sample,[.025,.975],axis=0).T,
                ratio=p90[1]/p90[0],ratio_ci=np.quantile(sample[:,1]/sample[:,0],[.025,.975]))


def main():
    completion=json.loads((ROOT/'results/completion.json').read_text())
    assert completion['blocks']==288 and completion['timing_clusters']==24 and completion['cells']==CELLS
    losses=np.empty((12,12,2,2,3,2))  # panel, condition, block, sign, regime, method
    times=[[] for _ in range(12)]
    factors={}; laws={}
    for panel in range(12):
        for b in range(2):
            for family in FAMILIES:
                base_by_law={}
                for law in LAWS:
                    base=archive(f'{family}_base_{law}',panel,b)
                    base_by_law[law]=base
                    for variant in CONTOURS:
                        key=f'{family}_{variant}_{law}'
                        obs=base if variant=='base' else archive(key,panel,b)
                        idx=CELLS.index(key)
                        losses[panel,idx,b]=obs['risk'].mean(-1)
                        if obs['full_ms'].size:
                            times[panel].append(obs['full_ms'])
                        if variant=='base':
                            continue
                        for name in ('alpha','actions','labels','z','weights','ids','first3_y','fourth_y','fourth_stick'):
                            np.testing.assert_array_equal(base[name],obs[name],err_msg=key+' '+name)
                        probe=obs['actions']==3
                        assert np.all(probe.sum(-1)==13)
                        np.testing.assert_array_equal(base['p_base'][probe], obs['p_base'][probe])
                        # z,w are identical for all contours. Both off-diagonal pairings
                        # collapse to their corresponding base-probability row.
                        d00=contrast(base['p_base'],base,base['labels'])
                        d10=contrast(obs['p_base'],base,obs['labels'])
                        d01=contrast(base['p_base'],obs,base['labels'])
                        d11=contrast(obs['p_base'],obs,obs['labels'])
                        old=base['risk'].mean((0,1,3)); new=obs['risk'].mean((0,1,3))
                        np.testing.assert_allclose([d00,d11],[old[1]-old[0],new[1]-new[0]],atol=2e-14)
                        dp=np.abs(obs['p_base'][~probe]-base['p_base'][~probe]).mean()
                        factors.setdefault(key,[]).append([(d10-d00+d11-d01)/2,(d01-d00+d11-d10)/2,dp])
                sat,cross=base_by_law.values()
                laws.setdefault(family,[]).append([np.mean(sat['labels']!=cross['labels']),
                    sat['fourth_stick'].mean(),cross['fourth_stick'].mean(),
                    sat['labels'].mean(),cross['labels'].mean()])
    risk=losses.mean((2,3,4))
    diff=risk[:,:,1]-risk[:,:,0]
    pooled=risk.mean(1)
    cells={key:{**interval(diff[:,i],12),'loss':risk[:,i].mean(0),
                 'wins':int((diff[:,i]<0).sum()),'regime_delta':(losses[:,i,...,1]-losses[:,i,...,0]).mean((0,1,2))}
           for i,key in enumerate(CELLS)}
    shape={}
    for key,arr in factors.items():
        fam,variant,law=key.split('_')
        shift=diff[:,CELLS.index(key)]-diff[:,CELLS.index(f'{fam}_base_{law}')]
        val=np.array(arr).reshape(12,2,3).mean(1)
        np.testing.assert_allclose(shift,val[:,:2].sum(1),atol=2e-14)
        shape[key]={**interval(shift,8),'fixed_and_optimizer_change':
                  (risk[:,CELLS.index(key)]-risk[:,CELLS.index(f'{fam}_base_{law}')]).mean(0),
                  'via_base_and_cue':val[:,:2].mean(0),'query_only_abs_base_change':val[:,2].mean()}
    lawcontrast=[diff[:,CELLS.index(f'{f}_{c}_crossover')]-diff[:,CELLS.index(f'{f}_{c}_saturating')]
                 for f,c in itertools.product(FAMILIES,CONTOURS)]
    out=dict(cells=cells,panel_risk=risk,pooled_loss=pooled.mean(0),pooled_delta=interval(diff.mean(1)),
             contour_change=shape,law_contrast=interval(np.mean(lawcontrast,axis=0)),
             law_by_family={f:np.array(rows).mean(0) for f,rows in laws.items()},timing=costs(times))
    (ROOT/'stats40.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,
                    default=lambda x:x.tolist(),allow_nan=False)+'\n')


if __name__=='__main__':
    main()
