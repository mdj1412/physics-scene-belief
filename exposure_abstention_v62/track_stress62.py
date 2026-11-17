"""Exploratory raw-ROI stress control: isolated vs temporally coherent false motion."""
from pathlib import Path
import json
import os
import numpy as np
import vision62 as vision

ROOT=Path(__file__).resolve().parent
NAMES=['original','single_frame_3px_jump','coherent_track_reversal']

def move_roi(roi,pixels):
    """Subpixel, non-circular horizontal translation of the upper-marker patch."""
    x=np.arange(roi.shape[-1])-pixels[...,None]
    x=np.clip(x,0,roi.shape[-1]-1)[...,None,:]
    i=np.floor(x).astype(int)
    frac=x-i
    left=np.take_along_axis(roi,i,-1)
    right=np.take_along_axis(roi,np.minimum(i+1,roi.shape[-1]-1),-1)
    return np.rint(left*(1-frac)+right*frac).astype('u1')

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    out=np.zeros((3,4,2,3),dtype=int)
    for catalog in range(24):
        with np.load(ROOT/'fixtures'/f'panel_{catalog:02d}.npz') as f:
            fr=f['frames']; wave=f['wave']; truth=f['truth']
        impact=fr[...,1,:,:,:]
        gap,visible,_=vision.optical_features(impact)
        rng=np.random.default_rng([6208117,catalog])
        # Same draw for all exposure regimes and sign counterfactuals of one physical slot.
        event=rng.integers(8,size=(2,4,2))
        dx=rng.choice([-3,3],size=event.shape)
        spike=(np.arange(8)==event[...,None])*dx[...,None]
        flip=np.clip(-2*(gap-gap[...,:1]),-8,8)
        baseline=truth[...,0]!=truth[...,1]
        active=truth[...,0]!=0
        for case in range(3):
            film=fr if case==0 else fr.copy()
            if case:
                shifts=np.broadcast_to(spike,gap.shape) if case==1 else np.where(visible,flip,0)
                film[...,1,:,:6,:]=move_roi(impact[...,:6,:],shifts)
            for d in range(4):
                feat=vision.optical_features(film[d])
                for j,m in enumerate([1,2]):
                    s=vision.read_features(feat,wave[d],m)
                    paid=(s[...,0]!=0)&(s[...,1]!=0)&active
                    wrong=((s[...,0]!=s[...,1])!=baseline)&paid
                    out[case,d,j]+=[active.sum(),paid.sum(),wrong.sum()]
    results={n:{'total_by_reader':arr.sum(0).tolist(),'by_cell':arr.tolist()} for n,arr in zip(NAMES,out)}
    (ROOT/'track_stress62.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__': main()
