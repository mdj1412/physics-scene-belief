"""Exploratory camera clock stress: impact clips start two milliseconds late."""
import json
import os
import numpy as np
import frame59 as f
from route59 import log_odds

ROOT=f.ROOT

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('SLURM_JOB_ID')
    f.TIMES=f.TIMES+2.  # delay the exposure sequence while the estimator assumes its original trigger
    with np.load(ROOT/'score59_cells.npz') as file:
        risk=np.einsum('cfea,ze->cfza',file['exact_by_e'],[[.42,0,.58],[0,.84,.16]])
    results=np.empty((24,3,2,3)); counts=np.zeros((3,3),dtype='i8')
    for i in range(24):
        with np.load(ROOT/'frame_panels'/f'panel_{i:02}.npz') as inp:
            nominal=inp['readout']
        with np.load(ROOT/'pilot_audit'/f'panel_{i:02}.npz') as inp:
            truth=inp['truth_signs'][:,:,:,:f.N]
        late=nominal.copy()
        for c,z,m in np.ndindex(2,2,2):
            rng=np.random.default_rng([5907331,i,m])
            signs=truth[c,z,m].ravel()
            dest=[]
            for j in range(0,len(signs),128):
                x=signs[j:j+128]
                images=f.render(x,np.arange(j,j+len(x))%2==1,rng)
                dest.append(f.read_signs(images))
            late[c,z,m,...,1]=np.concatenate(dest).reshape(f.N,4,2,2)[...,1]
        coin=np.random.default_rng([5908311,i]).random(truth.shape[2:-1])
        active=truth[...,0]!=0
        with np.load(ROOT/'pilots'/f'panel_{i:02}.npz') as inp:
            electronic=inp['codes'][:,0,:,:,:,:f.N].astype('i2')
        for k,prob in enumerate((0.,.2,1.)):
            reads=np.where((coin<prob)[None,None,...,None],late,nominal)
            visible=np.all(reads!=0,axis=-1)&active
            bad=(reads[...,0]!=reads[...,1])!=(truth[...,0]!=truth[...,1])
            counts[k]+=[active.sum(),visible.sum(),(visible&bad).sum()]
            cam=np.where(visible,1+(reads[...,0]!=reads[...,1]).astype('i2'),0)
            for r in range(2):
                a,b=np.divmod(electronic[r],13)
                x=13*(a+4*cam[...,0])+(b+4*cam[...,1])
                pol,cau=log_odds(x,r,1)
                fp,cp=pol>1e-12,cau>1e-12
                rate=fp.mean(-1).transpose(2,0,1)
                loss=risk[i,:,None,:,0]+rate*(risk[i,:,None,:,1]-risk[i,:,None,:,0])
                results[i,k,r]=[(cp[0].mean()+1-cp[1].mean())/2,
                                (fp[:,0].mean()+1-fp[:,1].mean())/2,loss.mean()]
    delta=results[:,1:,1,:]-results[:,0:1,1,:]
    boot=np.random.default_rng(5908312).integers(0,24,(5000,24))
    out=dict(late_impact_fraction=[0,.2,1],latency_ms=2,counts_active_read_wrong=counts.tolist(),
             modes=['cause_accuracy','policy_accuracy','mean_Brier'],means=results.mean(0).tolist(),
             changed_minus_nominal_reset_CI=np.quantile(delta[boot].mean(1),[.025,.975],axis=0).tolist())
    (ROOT/'latetrigger59.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    main()
