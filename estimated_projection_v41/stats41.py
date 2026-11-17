"""Inference on matched panels. Rows of the paper are recomputed from the raw block scores."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
FAMILIES=('flare','roll')
HEIGHTS=('h19','h31')
VARIANTS=('base','hollow','bulge')
LAWS=('saturating','crossover')
CELLS=['_'.join(ids) for ids in itertools.product(FAMILIES,HEIGHTS,VARIANTS,LAWS)]
DECODERS=('original','oracle','estimated')
POLICIES=('fixed','optimizer','selector')
N=12
BOOT=np.random.default_rng(410871).integers(N,size=(16000,N))


def interval(values, comparisons=1):
    v=np.asarray(values)
    m=v.mean(axis=0)
    rad=v.std(axis=0,ddof=1)*t.ppf(1-.025/comparisons,len(v)-1)/np.sqrt(len(v))
    return dict(mean=m,lo=m-rad,hi=m+rad)


def timing_bounds(a):
    # panel, repeated randomized timing sweep, decoder, policy
    assert a.shape==(12,3,3,3) and np.isfinite(a).all() and np.all(a>0)
    q=np.quantile(a,.9,axis=(0,1))
    qb=np.quantile(a[BOOT],.9,axis=(1,2))
    def ratio(i,j):
        r=qb[:,:, :][:,2,2]/qb[:,i,j]
        return dict(ratio=q[2,2]/q[i,j],ci=np.quantile(r,[.025,.975]))
    return dict(p90=q,ci=np.moveaxis(np.quantile(qb,[.025,.975],axis=0),0,-1),
                selected_vs_estimated_fixed=ratio(2,0),selected_vs_original_fixed=ratio(0,0))


def block_path(cell,panel,b):
    return ROOT/'results'/cell/f'catalog_{panel:02d}'/f'block_{b:02d}.npz'


def main():
    meta=json.loads((ROOT/'results/completion.json').read_text())
    assert meta['blocks']==576
    loss=np.empty((N,24,2,3,3));rate=np.empty((N,24,2,3))
    variation=np.full((N,2,2,2,2,2,3),np.nan)
    clock={k:np.zeros((12,3,3,3)) for k in ('full_ms','cold_ms','setup_ms')}
    timed=set()
    for panel in range(N):
        for b in range(2):
            for fi,f in enumerate(FAMILIES):
                for hi,h in enumerate(HEIGHTS):
                    for li,law in enumerate(LAWS):
                        reference=None
                        for vi,var in enumerate(VARIANTS):
                            key='_'.join([f,h,var,law])
                            c=CELLS.index(key)
                            with np.load(block_path(key,panel,b)) as a:
                                assert a['risk'].shape==(2,3,3,3,8)
                                loss[panel,c,b]=a['risk'].mean((0,1,4))
                                take=a['chosen_optimizer']
                                rate[panel,c,b]=take.mean((0,1,3,4))
                                p=a['p_base']
                                branch=a['actions']==3
                                if vi==0:
                                    reference=(p,a['labels'],a['z'],a['weights'],branch)
                                else:
                                    for left,right in [(a['labels'],reference[1]),(a['z'],reference[2]),
                                                       (a['weights'],reference[3]),(branch,reference[4])]:
                                        np.testing.assert_array_equal(left,right)
                                    np.testing.assert_allclose(p[:,branch],reference[0][:,branch],atol=1e-11,rtol=0)
                                    variation[panel,fi,hi,li,b,vi-1]=abs(p[:,~branch]-reference[0][:,~branch]).mean((1,2))
                                if a['full_ms'].size:
                                    assert panel not in timed
                                    timed.add(panel)
                                    for k in clock: clock[k][panel]=a[k]
    assert len(timed)==12
    panel_loss=loss.mean((1,2)); mean=panel_loss.mean(0)
    # panel × decoder × policy
    select_fixed=panel_loss[:,:,2]-panel_loss[:,:,0]
    select_opt=panel_loss[:,:,2]-panel_loss[:,:,1]
    p=panel_loss[:,:,0]
    den=(p[:,0]-p[:,1])[BOOT].mean(1);num=(p[:,0]-p[:,2])[BOOT].mean(1)
    ratio=None
    if np.all(den>0):
        ratio=dict(value=(mean[0,0]-mean[2,0])/(mean[0,0]-mean[1,0]),ci=np.quantile(num/den,[.025,.975]))
    strata=loss.reshape((N,2,2,3,2,2,3,3)).mean((3,4,5))
    by_family_height={}
    for i,f in enumerate(FAMILIES):
        for j,h in enumerate(HEIGHTS):
            s=strata[:,i,j]
            by_family_height[f+'_'+h]=dict(brier=s.mean(0),
                selection_vs_fixed=interval(s[:,2,2]-s[:,2,0],4),
                est_vs_original=interval(s[:,2,0]-s[:,0,0],4),
                est_vs_oracle=interval(s[:,2,0]-s[:,1,0],4))
    out=dict(decoders=DECODERS,policies=POLICIES,n_panels=N,panel_loss=panel_loss,
        mean_brier=mean,selection_rate=rate.mean((0,1,2)),
        selector_minus_fixed=interval(select_fixed,2),selector_minus_optimizer=interval(select_opt,2),
        estimated_minus_original_fixed=interval(p[:,2]-p[:,0],2),estimated_minus_oracle_fixed=interval(p[:,2]-p[:,1],2),
        oracle_minus_original_fixed=interval(p[:,1]-p[:,0]),recovery=ratio,by_family_height=by_family_height,
        view_abs_change=variation.mean((0,1,2,3,4,5)),view_abs_change_by_family_height=variation.mean((0,3,4,5)),
        cell_loss=loss.mean((0,2)),cell_selection_rate=rate.mean((0,2)),
        timing=timing_bounds(clock['full_ms']),cold=timing_bounds(clock['cold_ms']),
        setup_p90=np.quantile(clock['setup_ms'],.9,axis=(0,1)))
    (ROOT/'stats41.json').write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n')


if __name__=='__main__': main()
