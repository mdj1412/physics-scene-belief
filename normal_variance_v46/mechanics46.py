"""Run pressure-prescribed and independent rigid-body impacts for the same new specimens."""
import argparse
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np
from geometry46 import SHAPES, VARIANTS, shape_profile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"image_pressure_v45"))
import pressure45 as press
import bullet45 as rigid

HEIGHTS=(19,31)
SURFACES=press.SURFACES
LAWS=press.LAWS
NOBJ=128
NENV=4


def slope_mm(h, shape, variant):
    return shape_profile(h,shape,variant)[1]


def mesh_metres(theta,h_mm,x_mm,shape,variant):
    if shape is None:
        return -theta[:,2,None]*x_mm*.001
    rel = (shape_profile(h_mm+x_mm,shape,variant)[0] -
           shape_profile(h_mm,shape,variant)[0])
    return -.001*(theta[:,2,None]*x_mm +
                  theta[:,3,None]*(2*h_mm*x_mm+x_mm*x_mm)/50.+rel)


def configure():
    press.ROOT=ROOT
    press.slope=slope_mm
    press.SHAPES=SHAPES
    press.output_folder=lambda high:ROOT/("labels46_fine" if high else "labels46")
    rigid.base.geom_profile=mesh_metres


def rigid_group(panel,shape,var,h0):
    folder=ROOT/"labels46_bullet"/f"panel_{panel:02d}"
    paths={(sf,law):folder/f"{shape}_{sf}_{law}_{var}_h{h0}.npz"
           for sf,law in product(SURFACES,LAWS)}
    if all(p.exists() for p in paths.values()):
        return
    with np.load(ROOT/"results"/"worlds"/f"panel_{panel:02d}.npz") as manifest:
        th=manifest["theta"]
        signs=np.repeat(manifest["signs"],8)
        regimes=np.repeat(manifest["regimes"],8)
    assert th.shape==(4,128,6)
    centers=h0+np.array(press.OFFSETS)
    data={key:{} for key in paths}
    for start in range(0,NOBJ,8):
        t=th[:,start:start+8].reshape(-1,6)
        for o,h in enumerate(centers):
            surface=rigid.base.Surface(t,h,shape,var,rigid.base.DT,rigid.base.GRID_MM)
            g=t[:,2]+t[:,3]*h/25.+slope_mm(h,shape,var)
            for sf,law in paths:
                mu=rigid.base.material_mu(t,h,np.zeros_like(g),g,sf,law,signs,regimes)
                obs=rigid.simulate(surface,g,mu,t[:,5])
                current=data[sf,law]
                for k,v in obs.items():
                    if k not in current:
                        current[k]=np.empty((len(centers),NENV,NOBJ),dtype=v.dtype)
                    current[k][o,:,start:start+8]=v.reshape(4,8)
    for key,path in paths.items():
        d=data[key]
        d["labels"]=(d["slip_speed"]<=press.THRESHOLD)&d["valid"]
        d["desired_centers_mm"]=centers
        d["first_contact_x_mm"]=centers[:,None,None]+d["first_site_delta_mm"]
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,**d)
        os.replace(tmp,path)


def calibrate():
    x=np.linspace(-10.,45.,2201)
    step=2.e-5
    rows=[]
    for shape,var in product(SHAPES,VARIANTS):
        f,g=shape_profile(x,shape,var)
        fd=(shape_profile(x+step,shape,var)[0]-shape_profile(x-step,shape,var)[0])/(2*step)
        zero=shape_profile(0,shape,var)
        assert np.allclose(zero,0,atol=2.e-15) and np.max(abs(fd-g)) < 5e-8
        h=19.
        t=np.array([[0,0,.17,-.1,.2,.5]])
        numerical=(mesh_metres(t,h,x+step,shape,var)-mesh_metres(t,h,x-step,shape,var))*(-1000/(2*step))
        truth=t[:,2,None]+t[:,3,None]*(x+h)/25.+slope_mm(h+x,shape,var)
        assert abs(numerical-truth).max()<5e-8
        rows.append(dict(shape=shape,variant=var,max_central_derivative_error=float(np.abs(fd-g).max()),
                         mesh_max_gradient_error=float(abs(numerical-truth).max()),origin=[float(q) for q in zero],
                         max_profile_abs_mm=float(np.abs(f).max()), max_abs_slope=float(np.abs(g).max())))
    press.neutral()
    neutral=(ROOT/"pressure_neutral45.json")
    neutral.rename(ROOT/"pressure_neutral46.json")
    # Neutral mode writes only into ROOT. Direct control uses an analytic plane.
    rigid.ROOT=ROOT
    rigid.b.connect(rigid.b.DIRECT)
    rigid.control()
    (ROOT/"bullet_neutral45.json").rename(ROOT/"bullet_neutral46.json")
    rigid.b.disconnect()
    (ROOT/"geometry_checks46.json").write_text(json.dumps(rows,indent=2)+"\n")


def run(mode):
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Requires an allocated compute node")
    if not (ROOT/"MECHANICS_GO").is_file():
        raise RuntimeError("Waiting for evaluator release")
    if mode=="bullet":
        rigid.b.connect(rigid.b.DIRECT)
    tasks=ROOT/"physics_tasks46"/mode
    tasks.mkdir(parents=True,exist_ok=True)
    for panel,sh,var,h in product(range(12),SHAPES,VARIANTS,HEIGHTS):
        manifest=ROOT/"results"/"worlds"/f"panel_{panel:02d}.npz"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        stem=f"{panel:02d}_{sh}_{var}_{h}"
        done=tasks/(stem+".done")
        if done.exists():
            continue
        with (tasks/(stem+".lock")).open("a") as fp:
            try:
                fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if done.exists():
                continue
            if mode=="bullet":
                rigid_group(panel,sh,var,h)
            else:
                press.run_group(panel,sh,var,h,mode=="fine")
            done.write_text("complete\n")
        print(stem,mode,flush=True)
    if mode=="bullet":
        rigid.b.disconnect()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("check","primary","fine","bullet"))
    args=parser.parse_args()
    configure()
    if args.mode=="check":calibrate()
    else:run(args.mode)
