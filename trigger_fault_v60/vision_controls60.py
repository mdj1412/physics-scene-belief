"""Same-contact sign error and a miswired sync flash stress test on archived raw video."""
import json
import os
import numpy as np
from vision60 import ROOT, read_camera


def main():
    counts=[]; wiring=[]; swap_count=0; reset_count=0; quota=0
    for i in range(24):
        name=f'panel_{i:02}.npz'
        with np.load(ROOT/'vision_audit'/name) as f:
            camera=f['camera']
            truth=f['truth']
            late=f['delay_draw']<.2
            swap_count+=int(f['active_order'].sum())
            quota+=f['reset_id'].size
            reset_count+=int(f['reset_id'].sum())
        case=[]
        t=truth[...,0]!=truth[...,1]
        for d in range(3):
            good=np.all(camera[d]!=0,axis=-1)
            observed=camera[d,...,0]!=camera[d,...,1]
            common=good.all(0)&(truth[...,0]!=0)
            case.append([int(common.sum()),int(((observed[0]!=t)&common).sum()),
                         int(((observed[1]!=t)&common).sum()),int((good[1] & ~good[0]).sum()),
                         int((good[0] & ~good[1]).sum())])
        counts.append(case)
        # Independent stress: retained raw clips only, episode zero for each surface/cause/regime.
        # On the +2ms impact treatment, move the optical pulse to a later output frame by 1 or 2 ms.
        with np.load(ROOT/'fixtures'/name) as f:
            raw=f['frames'][2]
        tt=truth[:,:,:,0]
        cells=[]
        for shift in (0,1,2):
            clip=raw.copy()
            if shift:
                patch=raw[...,1,: ,13:,:]
                shifted=clip[...,1,:,13:,:]
                shifted[...,:shift,:,:]=np.flip(patch[...,-shift:,:,:],axis=-3)
                shifted[...,shift:,:,:]=patch[...,:-shift,:,:]
            measured=read_camera(clip,True)
            rr=np.all(measured!=0,axis=-1)&(tt[...,0]!=0)
            wrong=(measured[...,0]!=measured[...,1])!=(tt[...,0]!=tt[...,1])
            cells.append([int((tt[...,0]!=0).sum()),int(rr.sum()),int((rr&wrong).sum())])
        wiring.append(cells)
    x=np.array(counts)
    rng=np.random.default_rng(60098922)
    draws=rng.integers(24,size=(5000,24))
    boot=x[draws].sum(axis=1)
    delta=(boot[...,2]-boot[...,1])/boot[...,0]
    result=dict(overlap_fields=['read_by_both','wrong_nominal','wrong_sync','only_sync','only_nominal'],
                per_delay_counts=x.sum(0).tolist(),rate_difference_CI95=np.quantile(delta,[.025,.975],axis=0).tolist(),
                fixture_2ms_trigger_led_lag_ms=[0,1,2],fixture_active_read_wrong=np.sum(wiring,axis=0).tolist(),
                object_swap_fraction=swap_count/quota,sensor1_target_fraction=reset_count/quota)
    (ROOT/'vision_controls60.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    main()
