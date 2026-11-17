"""Re-score sealed predictions on preselected coarse/fine validation pairs."""
import itertools
import json
import os
from pathlib import Path
import numpy as np
from crossover68 import H,ROUTES,ENGINES,FAMS,LAWS,archive,routed,loss
from solver_labels68 import SRC

def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert (H/'LABELS_READY').is_file()
    modes=('soft_minus_full','restored_minus_dropped')
    # catalog, solver, family, law, (labels matched at standard and fine resolution), contrast
    arr=np.zeros((12,2,2,2,2,2))
    counts=np.zeros((2,2,2,5),np.int64)
    diffs={(i,j,k):[] for i,j,k in itertools.product(range(2),repeat=3)}
    old_error=0.; old_xor=0; old_n=0
    for k in range(12):
        sub=f'panel_{k:02d}'
        for ie,engine in enumerate(ENGINES):
            record=archive(H/'forecasts'/engine/(sub+'.npz'))
            for name,ix in record['ix'].items():
                fam,s,law,v,hs=name.removesuffix('.npz').split('_')
                fi=FAMS.index(fam);li=LAWS.index(law)
                with np.load(ROUTES/sub/f'{fam}_{v}_{hs}.npz') as z:
                    p=routed(record['p'][ix],record['shift'][ix],z['route'])[...,:32,:]
                with np.load(H/('labels_'+engine)/sub/name) as z:
                    ys=z['slip_speed'][...,:32].astype(float);yf=z['fine_slip_speed'].astype(float)
                    flip=z['labels'][...,:32]!=z['fine_labels']
                    within=abs(ys-.02)<=.002
                    counts[ie,fi,li]+=[ys.size,flip.sum(),within.sum(),(flip&within).sum(),((yf<=.02)!=(ys<=.02)).sum()]
                    diffs[ie,fi,li].append(abs(ys-yf).ravel())
                    for j,y in enumerate([z['labels'][...,:32],z['fine_labels']]):
                        r=loss(p,y)
                        arr[k,ie,fi,li,j]+=np.array([r[1]-r[0],r[4]-r[3]])/12
                    if engine=='prescribed':
                        with np.load(H.parent/'law_axis_v67'/'labels'/sub/name) as old:
                            yy=old['slip_speed'][1:];ll=old['labels'][1:]
                            old_error=max(old_error,float(np.max(abs(z['slip_speed']-yy))))
                            old_xor+=int(np.sum(z['labels']!=ll));old_n+=ll.size
    maxerr=np.empty((2,2,2,3))
    for key,vs in diffs.items():
        vals=np.concatenate(vs)
        maxerr[key]=[np.mean(vals),np.quantile(vals,.99),np.max(vals)]
    np.testing.assert_array_equal(counts[...,1],counts[...,4])
    assert old_error<1e-6 and old_xor==0,(old_error,old_xor)
    summary=dict(engines=ENGINES,families=FAMS,laws=LAWS,contrast=modes,
       counts_fields=['fine_pairs','label_disagreements','coarse_near_threshold','near_disagreements','float_recheck'],
       counts=counts.tolist(),absolute_slip_error_mps_mean_p99_max=maxerr.tolist(),catalog_standard_fine_contrasts=arr.tolist(),
       mean_standard_fine=arr.mean(0).tolist(),mean_fine_minus_standard=(arr[...,1,:]-arr[...,0,:]).mean(0).tolist(),
       old_prescribed_max_error=old_error,old_prescribed_label_xor=old_xor,old_prescribed_count=old_n,
       job=os.environ['SLURM_JOB_ID'])
    (H/'convergence68.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('counts:',counts.sum((1,2)).tolist(),'deltas:',summary['mean_fine_minus_standard'])
if __name__=='__main__':main()
