"""Score independent and legacy solvers on the same held-out objects and paid predictions."""
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'blind_decoder_v43'))
import physics_check43 as old

SHAPES=('fold','ripple')
SURF=('inlay','woven')
LAWS=('rational','gaussian_rate')
HEIGHT=(19,31)
VAR=('base','hollow','bulge')
DELTA=(-.5,0.,.5)
N=12
GRID=(N,2,2,2,2,3)
HARD=(2,2,3)


def ci(array,m=1):
    a=np.asarray(array)
    mu=a.mean(axis=0)
    hw=t.ppf(1-.025/m,len(a)-1)*a.std(axis=0,ddof=1)/np.sqrt(len(a))
    return np.stack((mu,mu-hw,mu+hw),axis=-1)


def maskmean(x,mask):
    numer=np.where(mask,x,0).sum(axis=(-2,-1))
    denom=mask.sum(axis=(-2,-1))
    if np.any(denom==0):
        raise ValueError('A prespecified cell has no valid events')
    return numer/denom


def strata(a,axis):
    return np.stack((a.take(1,axis=axis),.5*(a.take(0,axis=axis)+a.take(2,axis=axis))),axis=axis)


def path(panel,i,j,k,h,v):
    return Path(f'panel_{panel:02d}')/f'{SHAPES[i]}_{SURF[j]}_{LAWS[k]}_{VAR[v]}_h{HEIGHT[h]}.npz'


def load_np(file):
    with np.load(file) as a:
        return {k:a[k] for k in a.files}


def main():
    risk=np.empty(GRID+HARD+(2,2))
    point=np.empty(GRID+(3,2,2))
    flip=np.empty(GRID+HARD)
    prevalence=np.empty_like(flip)
    point_diff=np.empty_like(flip)
    tool_flip=np.empty_like(flip)
    wide=np.empty_like(flip)
    mult=np.empty_like(flip)
    finite=np.empty_like(flip)
    sensitivity=np.empty(GRID+HARD+(2,))
    d_common=np.empty_like(flip)
    maxerr=0.
    for idx in itertools.product(*(range(v) for v in GRID)):
        panel,i,j,k,h,v=idx
        pth=path(*idx)
        with np.load(ROOT/'labels44'/pth) as a:
            y=a['labels'].astype(float); valid=a['valid'].astype(bool)
            velocity=a['slip_speed']; span=a['span_mm']; npoints=a['npoints']
        assert y.shape[:4]==(2,2,3,4) and valid.shape==y.shape
        count=y.shape[-1]
        with np.load(ROOT/'results'/pth) as raw:
            pred=raw['pred'][:,:,:,:count,:]; th=raw['theta'][:,:count]
            signs,regs=raw['signs'],raw['regimes']
            assert tuple(raw['decoders'])==('original','privileged') and tuple(raw['policies'])==('zero','temperature')
        ypoint=np.stack([np.stack([old.score(th[e],SHAPES[i],VAR[v],SURF[j],LAWS[k],signs[e],regs[e],HEIGHT[h]+off)
                                   for e in range(4)]) for off in DELTA]).astype(float)
        if v==0:
            ybase=y.copy(); validbase=valid.copy()
        prevalence[idx]=maskmean(y,valid); finite[idx]=valid.mean(axis=(-2,-1))
        flip[idx]=maskmean(y!=ybase,valid&validbase)
        point_diff[idx]=maskmean(y!=ypoint,valid)
        tool_flip[idx]=maskmean(y!=y[0,0],valid&valid[0,0])
        wide[idx]=maskmean(span,valid); mult[idx]=maskmean(npoints>=2,valid)
        common=valid.all(axis=(0,1),keepdims=True)
        for d,c in itertools.product(range(2),repeat=2):
            p=pred[d,c]
            sq=(p**2).mean(-1); dp=1-2*p.mean(-1)
            b=maskmean(sq+y*dp,valid)
            direct=maskmean(((p[None,None,None]-y[...,None])**2).mean(-1),valid)
            maxerr=max(maxerr,float(abs(b-direct).max()))
            risk[idx+(Ellipsis,d,c)]=b
            point[idx+(Ellipsis,d,c)]=(sq+ypoint*dp).mean(axis=(-2,-1))
        sq=(pred[1,0]**2-pred[0,0]**2).mean(-1); dprob=-2*(pred[1,0]-pred[0,0]).mean(-1)
        d_common[idx]=maskmean(sq+y*dprob,np.broadcast_to(common,y.shape))
        for ti,threshold in enumerate((.01,.04)):
            sensitivity[idx+(Ellipsis,ti)]=maskmean(sq+(velocity<=threshold)*dprob,valid)
    assert maxerr<1e-12
    # After averaging 2 surface, 2 laws, 2 heights, 3 contours: [panel,shape,R,L,offset,d,c].
    group=risk.mean(axis=(2,3,4,5))
    gap=group[...,1,0]-group[...,0,0]
    ds=strata(gap,4)
    primary=ds.mean(axis=(2,3))
    table=strata(group.mean(axis=1),3)
    old_gp=strata(point.mean(axis=(2,3,4,5)),2)
    old_d=old_gp[...,1,0]-old_gp[...,0,0]
    oncommon=strata(d_common.mean(axis=(1,2,3,4,5)),3)[:,:,:,1]
    effect=np.stack((oncommon[:,1].mean(1)-oncommon[:,0].mean(1),
                     oncommon[:,:,1].mean(1)-oncommon[:,:,0].mean(1),
                     (oncommon[:,1,1]-oncommon[:,1,0])-(oncommon[:,0,1]-oncommon[:,0,0])),axis=-1)
    patch=lambda arr: arr.mean(axis=(1,2,3,4,5,8))
    # Dimensions for forecast comparison: panel,shape,height,R,L,offset
    fine=risk.mean(axis=(2,3,5))
    g=fine[...,1,0]-fine[...,0,0]
    old_fine=point.mean(axis=(2,3,5)); of=old_fine[...,1,0]-old_fine[...,0,0]
    out=dict(primary=ci(primary,4),primary_panel=primary,by_tool=ci(ds,16),absolute=ci(table),
             point_primary=ci(old_d,4),point_absolute=old_gp.mean(axis=(0,1)),engine_difference=ci(primary-old_d,4),
             radius_span_interaction=ci(effect,3),relative_to_small_round_flip=ci(patch(tool_flip),4),
             old_event_disagreement=ci(patch(point_diff),4),valid_fraction=ci(patch(finite),4),
             frame_multicontact=ci(patch(mult),4),reported_span_mm=ci(patch(wide),4),prevalence=ci(patch(prevalence),4),
             threshold_geometry=ci(strata(sensitivity.mean(axis=(2,3,4,5,6,7)),2),8),
             eventflip_by_tool=ci(strata(flip[:,:,:,:,:,1:,:,:,:].mean(axis=(1,2,3,4,5)),3),8),
             label_slots=int(np.prod(GRID+HARD+(4,count))),valid_fraction_all=float(finite.mean()),
             max_replay_error=maxerr, fine_panel_d=g, same_sample_fine_point=of,
             note='D=privileged zero minus original zero. strata order nominal, shifted. Intervals conditional on tested families.')
    (ROOT/'stats44.json').write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n')


if __name__=='__main__':
    main()
