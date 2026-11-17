"""Post-hoc audit of finite LED pulse and shutter overlap on the archived impact fixtures."""
import json
import os
from pathlib import Path
import numpy as np
import vision61 as v

root = Path(__file__).resolve().parent
# Pulse duration and camera exposure duration in ms; fixed settings, not fitted to outcomes.
cases = {'native': None, 'pulse0.20_exp0.25':(.20,.25), 'pulse0.80_exp0.25':(.80,.25),
         'pulse0.80_exp0.80':(.80,.80)}

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    stats = np.zeros((len(cases),4,6))
    for i in range(24):
        with np.load(root/'fixtures'/f'panel_{i:02d}.npz') as f:
            # Optical ROI and both waveforms are shared by all four sign counterfactuals.
            frames = f['frames'][:,0,0,:,:,:,1].reshape(4,16,8,15,27)
            wave = f['wave'][:,:,:,:,1].reshape(4,16,2,128)
            times = f['true_delays'][:,:,:,:,1].reshape(4,16,2)
        features = v.optical_features(frames)
        # Preserve each fixture's actual brightness, visibility, and remaining instrument noise.
        peak = np.maximum(features[2].max(-1),0)
        seen = peak>38
        centers = times[...,1,None]-2+np.arange(8)
        for k,param in enumerate(cases.values()):
            measured = features
            if param:
                pulse,shutter = param
                start = times[...,0,None]
                duration = np.maximum(0,np.minimum(centers+.5*shutter,start+pulse)
                                      -np.maximum(centers-.5*shutter,start))
                # Estimate LED contrast over the captured portion of the shutter exposure.
                # Use a disjoint background ROI for the per-frame optical noise floor.
                background = (frames[...,13:,15:19].mean((-1,-2))-frames[...,13:,7:11].mean((-1,-2)))
                light = background + seen[...,None]*peak[...,None]*duration/shutter
                measured = (features[0],features[1],light)
            _,_,l,t = v.estimate(measured,wave,2)
            nd=np.isfinite(l)&np.isfinite(t)
            flash_count=(measured[2]>38).sum(-1)
            for d in range(4):
                mask=nd[d]
                stats[k,d]+=[16,mask.sum(),(flash_count[d]==0).sum(),(flash_count[d]>1).sum(),
                             abs(l[d,mask]-times[d,mask,0]).sum(),abs(t[d,mask]-times[d,mask,1]).sum()]
    out={}
    for name,rows in zip(cases,stats):
        out[name]=[dict(n=int(n),identified=int(g),no_pulse=int(no),multiple=int(m),
                        LED_MAE=a/g if g else None,trigger_MAE=b/g if g else None)
                   for n,g,no,m,a,b in rows]
    (root/'exposure61.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
