"""Calibration with a chosen signed incoming tangential speed on each distinct paid object."""
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'directional_prefix_v33'))
from target33 import (LAWS, draw_object, contact, contour, deployment_x, deployment_truth,
                      sigmas, t, torch, stream, FY, FACTUAL_INCOMING, PROBE_INCOMING, P)
SCHEDULES = ('same','alternating')
def directions(schedule, start, size=8):
    if schedule not in SCHEDULES or start not in (-1,1):
        raise ValueError('unknown motion or starting sign')
    runs = np.ones(size,dtype=np.int8)
    if schedule=='alternating':
        runs[1::2]=-1
    return start*runs

def calibration(seed, alpha, law, schedule, start):
    theta=draw_object(1024,seed,10,alpha,law)
    fv,stick=contact(theta,0,FACTUAL_INCOMING,law)
    assert bool(stick.all())
    signal=torch.cat((contour(theta,FY,law),fv),dim=-1)
    x=(signal+t(stream(seed,11).normal(size=(1024,11)))*sigmas()[0]).cpu().numpy()
    ids=stream(seed,98).permutation(1024)[:8]
    signs=directions(schedule,start)
    probe,stick=contact(theta[ids],0,t(signs)*abs(PROBE_INCOMING),law)
    assert not bool(stick.any())
    noise=stream(seed,99).normal(size=(8,2))*sigmas()[P].cpu().numpy()
    return x,ids,probe.cpu().numpy()+noise,signs
