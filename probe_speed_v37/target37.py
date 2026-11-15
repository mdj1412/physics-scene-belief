"""Three paired contact velocities from the same fourth object and noise draw."""
import sys
from pathlib import Path
import numpy as np
PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT/'query_matched_probe_v36'))
import target36 as v36
from target36 import deployment, t, torch

def calibration(seed,alpha,law,regime,start,speeds,height):
    far,at_reference,latent=v36.calibration(seed,alpha,law,regime,start)
    x,ids,far_y,dirs=far
    th,reference_clean,_=latent
    # Keep the common observation noise exactly, including its original rounding.
    noise=(t(v36.stream(seed,99).normal(size=(4,2)))*v36.sigmas()[v36.P])[-1]
    records=[]; clean=[]; stick=[]
    for speed in speeds:
        signal,state=v36.impact(t(th[None]),height,speed,law,regime)
        yy=far_y.copy()
        yy[3]=(signal[0]+noise).cpu().numpy()
        if speed==.5:
            np.testing.assert_array_equal(yy,at_reference[2])
            np.testing.assert_array_equal(signal[0].cpu().numpy(),reference_clean)
        records.append((x,ids,yy,dirs))
        clean.append(signal[0].cpu().numpy())
        stick.append(bool(state[0]))
    return records,th,np.asarray(clean),np.asarray(stick)
