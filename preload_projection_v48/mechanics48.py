"""Finite-pressure binary outcome independent of new observing instrument and optical sign."""
import argparse
import os
from itertools import product
import sys
import numpy as np
from world48 import ROOT,FAMILIES,HEIGHTS,STATES,RADII,OFFSETS,loaded_shape

sys.path.insert(0,str(ROOT.parent/'image_pressure_v45'))
import pressure45 as pressure

AXES=("family","height","signed_state","radius","offset","environment","object")


def panel_outcomes(panel,fine=False):
    dest=ROOT/("labels_fine" if fine else "labels")/f"panel_{panel:02d}.npz"
    if dest.exists():
        return
    with np.load(ROOT/"worlds"/dest.name) as pop:
        theta=pop["theta"]
        signs,regimes,surfaces,laws=(pop[k] for k in ("signs","regimes","surface","law"))
        indices=pop["object_index"]
    assert theta.shape==(4,128,6)
    th=theta.reshape(-1,6)
    speed=np.empty((4,2,3,2,3,4,128))
    nx,nt=(24,12) if fine else (12,6)
    for hi,ri,xi,si in product(range(2),range(2),range(3),range(3)):
        h=HEIGHTS[hi]; at=h+OFFSETS[xi]; s=STATES[si]
        dx,w=pressure.quadrature(RADII[ri],nx,nt)
        xx=at+dx
        for fi,family in enumerate(FAMILIES):
            if s==0 and fi:
                # Check the paired-control condition before reusing each result.
                for x in (at,xx):
                    assert np.array_equal(loaded_shape(x,h,family,s)[1],loaded_shape(x,h,FAMILIES[0],s)[1])
                speed[fi,hi,si,ri,xi]=speed[0,hi,si,ri,xi]
                continue
            gc=th[:,2]+th[:,3]*at/25+loaded_shape(at,h,family,s)[1]
            g=th[:,2,None]+th[:,3,None]*xx/25+loaded_shape(xx,h,family,s)[1]
            drive,resistance,normal,vt=pressure.impulse_geometry(g,gc,th[:,5],w)
            assert np.all(normal>0)
            for e,(sf,law) in enumerate(zip(surfaces,laws)):
                sl=slice(e*128,(e+1)*128)
                mu=pressure.material(th[sl,4],vt[sl],xx,sf,law,np.repeat(signs[e],128),np.repeat(regimes[e],128))
                speed[fi,hi,si,ri,xi,e]=pressure.residual(drive[sl],resistance[sl],mu)
    assert np.isfinite(speed).all() and speed.min()>=0
    dest.parent.mkdir(exist_ok=True)
    tmp=dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,labels=speed<=pressure.THRESHOLD,slip_speed=speed.astype(np.float32),axes=AXES,
         state=STATES,radii_mm=RADII,heights_mm=HEIGHTS,families=FAMILIES,offsets_mm=OFFSETS,
         surface=surfaces,law=laws,object_index=indices,nx=nx,nt=nt)
    os.replace(tmp,dest)
    print("completed",dest.name,"fine",fine,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--panels",nargs="+",type=int,required=True);ap.add_argument("--fine",action="store_true")
    args=ap.parse_args()
    if not os.getenv("SLURM_JOB_ID") or os.getenv("CUDA_VISIBLE_DEVICES")!="" or not (ROOT/"TARGET_GO").is_file():
        raise RuntimeError("Use a Slurm CPU worker after target release.")
    for panel in args.panels:
        panel_outcomes(panel,args.fine)
