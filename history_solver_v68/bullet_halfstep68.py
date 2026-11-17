"""Numerical sensitivity: all 384 paired PyBullet target draws at 10 microsecond steps."""
import os,sys,json
from pathlib import Path
import numpy as np
import bullet68 as run
from crossover68 import H,ROUTES,FAMS,LAWS,ci,routed
TARGET=H/'bullet_halfstep'
def generate(ids):
    assert (H/'bullet_history'/'forecasts_sealed.json').is_file()
    object_type=run.simulator.Surface
    def refined_surface(theta,h,f,v,dt,spacing):
        return object_type(theta,h,f,v,dt/2,spacing)
    run.simulator.Surface=refined_surface;run.H=TARGET
    run.simulator.b.connect(run.simulator.b.DIRECT)
    for k in ids:run.run_panel(k,False)
    run.simulator.b.disconnect()

def evaluate():
    risks=np.zeros((12,2,2,2,5));counts=np.zeros((12,2,2,4),int)
    for k in range(12):
        sub=f'panel_{k:02d}'
        with np.load(H/'bullet_history'/'forecasts'/(sub+'.npz')) as z:raw=z['p'];cues=z['shift'];names=z['cells'].tolist()
        for i,name in enumerate(names):
            fam,_,law,v,h=name[:-4].split('_')
            a,b=FAMS.index(fam),LAWS.index(law)
            with np.load(ROUTES/sub/f'{fam}_{v}_{h}.npz') as z: gate=z['route'][...,:8]
            p=routed(raw[i],cues[i],gate)
            both=[];valid=[]
            for directory in (H,TARGET):
                with np.load(directory/'bullet_labels'/sub/name) as z:
                    both.append(z['speed']<=.02);valid.append(z['valid'])
            mask=valid[0]&valid[1]
            counts[k,a,b]+=[mask.sum(),mask.size,((both[0]!=both[1])&mask).sum(),np.sum(valid[0]!=valid[1])]
            for j,y in enumerate(both):
                loss=(p[:,:,None]-y[None,None,...,None])**2
                risks[k,a,b,j]+=(loss.mean((0,6))*mask[None]).sum((1,2,3,4))
    risks/=counts[...,0,None,None]
    out=dict(families=FAMS,laws=LAWS,counts_valid_total_flipped_different_validity=counts.sum(0).tolist(),
             primary_versus_halfstep_soft=ci(risks[...,1]-risks[...,0]),
             primary_versus_halfstep_rescue=ci((risks[...,4]-risks[...,3]).mean(1)),
             job=os.environ['SLURM_JOB_ID'])
    (H/'bullet_halfstep68.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if sys.argv[1]=='score':evaluate()
    else:generate(list(map(int,sys.argv[1:])))
