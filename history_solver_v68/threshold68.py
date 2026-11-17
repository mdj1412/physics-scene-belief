"""Secondary sensitivity to a factor-of-two change in the pre-existing stick criterion."""
import json,os
import numpy as np
from crossover68 import H,FAMS,LAWS,ENGINES,ROUTES,archive,routed,ci,loss
THRESHOLDS=(.010,.020,.040)
def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    values=np.zeros((12,2,2,2,3,3))
    for k in range(12):
        sub=f'panel_{k:02d}'
        for e,engine in enumerate(ENGINES):
            record=archive(H/'forecasts'/engine/(sub+'.npz'))
            for name,idx in record['ix'].items():
                fam,_,law,var,h=name.removesuffix('.npz').split('_')
                f=FAMS.index(fam);l=LAWS.index(law)
                with np.load(ROUTES/sub/f'{fam}_{var}_{h}.npz') as z:
                    p=routed(record['p'][idx],record['shift'][idx],z['route'])
                with np.load(H/('labels_'+engine)/sub/name) as z:
                    speed=z['slip_speed']
                    assert np.array_equal(speed<=.020,z['labels'])
                for j,thr in enumerate(THRESHOLDS):
                    y=speed<=thr;r=loss(p,y)
                    values[k,e,f,l,j]+=np.array([y.mean(),r[1]-r[0],r[4]-r[3]])/12
    data=dict(engines=ENGINES,families=FAMS,laws=LAWS,thresholds_mps=THRESHOLDS,
              measures=['stick_fraction','soft_minus_full','restored_minus_dropped'],summary=ci(values),
              pooled_rescue=ci(values[...,2].mean(2)),job=os.environ['SLURM_JOB_ID'])
    (H/'threshold68.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
