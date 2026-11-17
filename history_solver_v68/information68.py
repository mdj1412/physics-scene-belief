"""Audit that the manipulated law reaches the paid history without changing the optical task."""
from itertools import product
import json,os
from pathlib import Path
import numpy as np
from crossover68 import H,ENGINES,FAMS,LAWS,fn,ci

def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cols=('calibration_factual_velocity_RMS','calibration_far_velocity_RMS','calibration_fourth_velocity_RMS',
          'deployment_factual_velocity_RMS','common_paid_probe_velocity_RMS','acquisition_changed_fraction',
          'shift_law_absdiff','same_frozen_fourth_cue_changes_shift_fraction','absolute_fourth_log_odds_effect')
    vals=np.zeros((12,2,2,len(cols)))
    for k in range(12):
        sub=f'panel_{k:02d}'
        for ei,engine in enumerate(ENGINES):
            with np.load(H/'forecasts'/engine/(sub+'.npz')) as z:
                ix={n:i for i,n in enumerate(z['cells'])}
                shift=z['shift'];shift3=z['shift3'];actions=z['actions']
            for f,s,v,h in product(FAMS,('inlay','woven'),('base','hollow','bulge'),(19,31)):
                ff=FAMS.index(f)
                na,nb=[fn(f,s,l,v,h) for l in LAWS]
                a,b=ix[na],ix[nb]; aa,ab=actions[[a,b]]
                # Physics's selected query=1 (115), probe=2 (13); find the budgeted probe value from counts.
                uniq,cnt=np.unique(aa[0],return_counts=True)
                pk=uniq[np.flatnonzero(cnt==13).item()]
                assert (np.sum(aa==pk,axis=1)==13).all() and (np.sum(ab==pk,axis=1)==13).all()
                mask=(aa==pk)&(ab==pk)
                with np.load(H/'tapes'/engine/sub/na) as ta,np.load(H/'tapes'/engine/sub/nb) as tb:
                    for field in ('query','dirs'):np.testing.assert_array_equal(ta[field],tb[field])
                    for field in ('cx','xx'):np.testing.assert_array_equal(ta[field][...,:9],tb[field][...,:9])
                    differences=(ta['cx'][...,-2:]-tb['cx'][...,-2:],
                                 ta['cy'][:,:3]-tb['cy'][:,:3],ta['cy'][:,3]-tb['cy'][:,3],
                                 ta['xx'][...,-2:]-tb['xx'][...,-2:],
                                 (ta['probe']-tb['probe'])[mask])
                numbers=[np.sqrt(np.mean(d*d)) for d in differences]+[(aa!=ab).mean(),
                    abs(shift[a]-shift[b]).mean(),(abs(shift[[a,b]]-shift3[[a,b]])>1e-8).mean(),
                    abs(shift[[a,b]]-shift3[[a,b]]).mean()]
                vals[k,ei,ff]+=np.array(numbers)/12
    data=dict(engines=ENGINES,families=FAMS,metrics=cols,pointwise_ci=ci(vals),job=os.environ['SLURM_JOB_ID'],
              averages='cellwise RMS then macro average across cells and catalogs; common probe comparator uses intersection of acquisitions')
    (H/'information68.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
