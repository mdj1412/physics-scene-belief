"""Worst corner of an illustrative two-pulse capture envelope, not measured hardware."""
import itertools
import json
import math
import os
import socket
from pathlib import Path

OUT=Path(__file__).resolve().parent
WIDTH, WINDOW, JITTER=8.,12.,1.

def integrated_until(t, tau):
    """Optical energy accumulated by a unit rectangular pulse with exponential trailing edge."""
    if t<=0:
        return 0.
    if t<=WIDTH:
        return t
    return WIDTH-tau*math.expm1(-(t-WIDTH)/tau)

def dose(center, lamp, tau):
    origin=lamp-WIDTH/2
    return integrated_until(center+WINDOW/2-origin,tau)-integrated_until(center-WINDOW/2-origin,tau)

def leakage(delta, tau):
    largest=0.
    for a,b,c in itertools.product((-JITTER,JITTER),repeat=3):
        for first,second in ((0.,delta),(delta,0.)):
            intended=dose(first+c,first+a,tau)
            stray=dose(first+c,second+b,tau)
            largest=max(largest,stray/intended)
    return largest

def main():
    assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    vels=[1.,4.,10.] # px / ms
    out=[]
    for delta,tau in itertools.product((10.,20.,40.),(.5,2.,5.)):
        ratio=leakage(delta,tau)
        out.append(dict(separation_us=delta,decay_us=tau,largest_leakage_corner=ratio,
               max_offset_px=[(delta+2*JITTER)/1000.*v for v in vels],
               max_blur_for_pulse_window_upper_px=[WINDOW/1000.*v for v in vels]))
    assert 0<leakage(10,2) and leakage(40,2)<leakage(20,2)<leakage(10,2)
    assert leakage(20,.5)<leakage(20,2)<leakage(20,5)
    assert dose(0,0,.5)>=WIDTH
    p=.00186
    shams={'n':100,'upper_zero':1-.05**(1/100),
           'n_min_upper_below_half_planning_rate':math.floor(math.log(.05)/math.log1p(-p/2))+1}
    data=dict(slurm=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),JITTER_us=JITTER,
              pulse_us=WIDTH,window_us=WINDOW,velocity_px_ms=vels,grid=out,
              independent_zero_shams_1sided95=shams)
    (OUT/'rig_timing65.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
