"""Exact decomposition across three experimentally manipulated contours and uncorrected scores."""
import itertools
import json
from pathlib import Path
import numpy as np
import stats41 as s

ROOT=Path(__file__).resolve().parent

def values(x):
    return {key:np.asarray(v).tolist() for key,v in s.interval(x).items()}

def main():
    metadata=json.loads((ROOT/'results/completion.json').read_text())
    beta=metadata['beta']
    raw=np.zeros((s.N,3))
    components=np.zeros((s.N,4,3))
    strata={name:np.zeros((s.N,4,3)) for name in ('flare_h19','flare_h31','roll_h19','roll_h31')}
    max_error=0.
    for panel in range(s.N):
        for f,h,law,b in itertools.product(s.FAMILIES,s.HEIGHTS,s.LAWS,range(2)):
            probs=[]; stored=[]
            for v in s.VARIANTS:
                with np.load(s.block_path('_'.join((f,h,v,law)),panel,b)) as data:
                    # decoder × sign × env × object × noise
                    base=data['p_base']
                    y=data['labels'].astype(float)
                    odds=np.exp(beta*.325*data['z'])[None,:,:,None,None]
                    pred=base*odds/(1-base+base*odds)
                    ybar=y.mean(1)[None,:,:,:,None]
                    raw[panel]+=((base-ybar)**2+ybar*(1-ybar)).mean((1,2,3,4))/48
                    if probs:
                        np.testing.assert_array_equal(y,reference)
                    else:
                        reference=y
                    probs.append(pred)
                    stored.append(data['risk'][:,:,:,0,:].mean((0,1,3)))
            probs=np.stack(probs)
            meanp=probs.mean(0)
            dispersion=probs.var(0).mean((1,2,3,4))
            mean_error=((meanp-ybar)**2+ybar*(1-ybar)).mean((1,2,3,4))
            rows=np.stack([dispersion,mean_error,stored[0],np.mean(stored[1:],axis=0)])
            err=np.abs(dispersion+mean_error-np.mean(stored,axis=0)).max()
            if err>1.e-12: raise ValueError(f'Variance identity mismatch {err}')
            max_error=max(max_error,err)
            key=f+'_'+h
            components[panel]+=rows/16
            strata[key][panel]+=rows/4
    fixed=json.loads((ROOT/'stats41.json').read_text())
    loss=np.asarray(fixed['panel_loss'])
    def effects(a):
        gain=a[:,:,2]-a[:,:,0]
        # q: mass on the two equally weighted deformed variants; affine change under fixed mixture weights
        clean=gain[:,2]; mutated=gain[:,3]
        q=float(clean.mean()/(clean.mean()-mutated.mean())) if clean.mean()*mutated.mean()<0 else None
        return dict(means=a.mean(0).tolist(),estimated_minus_original=values(gain),
                    break_even_point=q)
    out=dict(uncorrected=values(raw),uncorrected_original_minus_original_fixed=values(raw[:,0]-loss[:,0,0]),
        estimated_selected_minus_uncorrected_original=values(loss[:,2,2]-raw[:,0]),
        decoder_order=s.DECODERS, component_order=('between_contour_variance','error_of_contour_mean','base','hollow_bulge_mean'),
        pooled=effects(components),per_shape_height={k:effects(v) for k,v in strata.items()},
        max_identity_error=float(max_error))
    (ROOT/'contour_breakdown41.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
