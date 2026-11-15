"""Reconstruct per-draw signed Brier contributions across matched contours."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
TOL=1.e-12
VARIANTS=('base','hollow','bulge')
FAMILIES=('shelf','ridge')
SURFACES=('saturating','crossover')


def contributions(archive):
    h=archive['p_base'][:,None]
    z=archive['z'][:,None,:,None,None]
    w=archive['weights'][:,:,:,None,None]
    y=archive['labels'][...,None].astype(np.float64)
    def corrected(multiplier):
        shift=np.exp(.8289760825845313*z*multiplier)
        return h*shift/(1-h+h*shift)
    qc,qf=corrected(w),corrected(.325)
    delta=np.square(qc-y)-np.square(qf-y)
    expected=np.sign(z*(w-.325))*(1-2*y)
    near=np.abs(delta)<=TOL
    np.testing.assert_allclose(delta.mean(axis=(3,4)),archive['risk'][:,:,1]-archive['risk'][:,:,0],atol=2.e-15)
    counts=dict(n=delta.size,ties_exact=(delta==0).sum(),
                ties_tolerance=near.sum(),expected_sign_mismatch_above_tolerance=((~near)&(np.sign(delta)!=expected)).sum(),
                raw_expected_sign_mismatch_nonzero=((delta!=0)&(np.sign(delta)!=expected)).sum(),
                reported_base_outside_unit_interval=np.broadcast_to((h<0)|(h>1),delta.shape).sum())
    values=np.array([np.maximum(delta,0.).mean(),np.minimum(delta,0.).mean(),delta.mean()])
    return delta,counts,values


def main():
    if not (ROOT/'results/completion.json').exists():
        raise RuntimeError('Complete all 288 blocks before the sign audit')
    cells,contrasts={},{}
    for family in FAMILIES:
        for surface in SURFACES:
            per=[[] for _ in VARIANTS]
            counts=[{} for _ in VARIANTS]
            switch=[{} for _ in VARIANTS]
            for panel in range(12):
                for block in range(2):
                    reference=None
                    for vi,variant in enumerate(VARIANTS):
                        cell=f'{family}_{variant}_{surface}'
                        p=ROOT/'results'/cell/f'catalog_{panel:02d}'/f'block_{block:02d}.npz'
                        with np.load(p) as a:
                            delta,case,value=contributions(a)
                        per[vi].append(value)
                        for key,n in case.items():
                            counts[vi][key]=counts[vi].get(key,0)+int(n)
                        if vi==0:
                            reference=delta
                            continue
                        tie0=abs(reference)<=TOL
                        tie1=abs(delta)<=TOL
                        opposite=np.sign(reference)*np.sign(delta)<0
                        tally=dict(total=delta.size,opposite_sign_raw=opposite.sum(),
                                   opposite_sign_both_above_tolerance=(opposite&~tie0&~tie1).sum(),
                                   exactly_one_within_tolerance=(tie0!=tie1).sum(),
                                   both_within_tolerance=(tie0&tie1).sum())
                        for key,n in tally.items():
                            switch[vi][key]=switch[vi].get(key,0)+int(n)
            for vi,variant in enumerate(VARIANTS):
                cell=f'{family}_{variant}_{surface}'
                means=np.mean(per[vi],axis=0)
                cells[cell]=dict(positive_contribution=means[0],negative_contribution=means[1],
                                 net_delta=means[2],**counts[vi])
                if vi:
                    contrasts[cell]=switch[vi]
    result=dict(tolerance=TOL,by_cell=cells,compared_to_base=contrasts)
    if any(c['opposite_sign_both_above_tolerance'] for c in contrasts.values()):
        raise ValueError('True paired sign reversal')
    if any(c['expected_sign_mismatch_above_tolerance'] for c in cells.values()):
        raise ValueError('Sign identity failed away from numerical ties')
    (ROOT/'sign_audit40.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
