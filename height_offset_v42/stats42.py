"""Matched panel inference for geometry, events, decisions and synchronized latency."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
FAMILY=('flare','roll'); HEIGHT=('h19','h31')
VARIANT=('base','hollow','bulge'); LAW=('saturating','crossover')
CELLS=list(itertools.product(FAMILY,HEIGHT,VARIANT,LAW))
OFF=np.array([-1.,-.5,-.25,-.125,0.,.125,.25,.5,1.])
N=12
Z=4
NZ=OFF!=0

def ci(x,m=1):
    x=np.array(x);n=len(x);mean=x.mean(axis=0)
    w=t.ppf(1-.025/m,n-1)*x.std(axis=0,ddof=1)/np.sqrt(n)
    return dict(mean=mean,lo=mean-w,hi=mean+w)

def raw(row,k):
    return ROOT/'results'/'_'.join(row)/f'catalog_{k:02d}.npz'

def main():
    assert (ROOT/'results/completion.json').exists()
    loss=np.empty((N,2,2,3,2,9,3,4))
    events=np.empty((N,2,2,3,2,9,5))
    peak=0.
    for k,f,h,l in itertools.product(range(N),range(2),range(2),range(2)):
        for v,var in enumerate(VARIANT):
            with np.load(raw((FAMILY[f],HEIGHT[h],var,LAW[l]),k)) as a:
                y=a['labels'].astype(float)
                p=a['p_base']
                risk=a['risk'].mean(axis=(3,4,5))
                assert risk.shape==(9,3,4) and y.shape==(9,2,3,8,128)
                if v==0: base=y.copy()
                np.testing.assert_array_equal(y[Z],base[Z])
                # Per-object prediction does not depend on the hidden offset or the near-field regime.
                dp=(p[2]-p[0]).mean(-1)[:,None]
                locus=y-base
                ordinary=base-base[Z]
                mean=lambda arr: arr.mean(axis=(1,2,3,4))
                terms=np.stack([mean(abs(locus)),mean(abs(y-y[Z])),mean(-2*dp*locus),
                                mean(-2*dp*ordinary),mean(y)],axis=-1)
                diff=risk[:,2,0]-risk[:,0,0]
                err=abs(diff-diff[Z]-terms[:,2:4].sum(1)).max()
                if err>1e-11: raise ValueError('Brier identity mismatch')
                peak=max(peak,err)
                events[k,f,h,v,l]=terms
                loss[k,f,h,v,l]=risk
    q=loss.mean((1,2,3,4)); pair=q[:,:,2,3]-q[:,:,2,0]
    gap=loss[...,2,0]-loss[...,0,0]
    panel_geo=gap.mean((1,2,3,4))
    group_geometry=gap.mean((3,4))
    gr=loss.mean((3,4))
    e=events.mean((1,2,3,4)); ef=events[:,:,:,1:].mean((1,2,3,4))
    boot=np.random.default_rng(420065).integers(0,N,(16000,N))
    clock={}
    for key in ('full_ms','cold_ms','setup_ms'):
        vals=[]; names=None
        for n in range(N):
            with np.load(ROOT/'results/timing'/f'panel_{n:02d}.npz') as a:
                vals.append(a[key])
                if 'methods' in a: names=a['methods'].tolist()
        ts=np.stack(vals)
        assert ts.shape==(12,4,5) and (ts>0).all()
        qs=np.quantile(ts,.9,axis=(0,1)); bs=np.quantile(ts[boot],.9,axis=(1,2))
        ratio=bs[:,-1]/bs[:,1]
        clock[key]=dict(p90=qs,ci=np.quantile(bs,[.025,.975],axis=0).T,ratio=qs[-1]/qs[1],
                        ratio_ci=np.quantile(ratio,[.025,.975]),methods=names)
    psel=pair[:,NZ].mean(1)
    out=dict(offset_mm=OFF,panels=N,main=ci(psel),wins=np.sum(psel<0),all9=ci(pair.mean(1)),
       main_by_decoder=ci((q[:,:,:,3]-q[:,:,:,0])[:,NZ].mean(1)),
       selected_minus_fixed_opt=ci((q[:,:,2,3,None]-q[:,:,2,1:3])[:,NZ].mean(1),2),
       by_offset_selector=ci(pair,9),mean_risk_by_offset=q.mean(0),mean_risk_nonzero=q[:,NZ].mean((0,1)),
       geometry=ci(panel_geo,9),group_geometry=ci(group_geometry,36),
       group_selector=ci(gr[...,2,3]-gr[...,2,0],36),group_risk=gr.mean(0),
       law_selector=ci((loss[...,2,3]-loss[...,2,0]).mean((1,2,3))[:,:,NZ].mean(-1),2),
       group_flip=ci(events[:,:,:,1:,:,:,0].mean((3,4)),36),flip=ci(ef[:,:,0],9),self_flip=ci(e[:,:,1],9),
       events=e.mean(0),cell_geometry=gap.mean(0),cell_risk=loss.mean(0),
       variant_geometry=ci(gap.mean((1,2,4)),27),timing=clock,
       identity_max_error=peak,per_panel_nonzero_selector_minus_zero=psel,
       per_panel_loss=q,per_panel_group_geometry=group_geometry)
    (ROOT/'stats42.json').write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda v:v.tolist())+'\n')
if __name__=='__main__':main()
