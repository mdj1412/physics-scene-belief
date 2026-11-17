"""PyBullet rigid impact on an extruded contour. Units mm, ms, kg (mm/ms = m/s)."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pybullet as pb

ROOT = Path(__file__).resolve().parent
V48 = ROOT.parent / "preload_projection_v48"
sys.path.insert(0, str(V48))
from world48 import FAMILIES, HEIGHTS, OFFSETS, RADII, STATES, loaded_shape

THRESHOLD = .020
REGIME = {"strong": 1., "weak": .30, "null": 0., "global_null": 0.}
AXES = ("family","height","signed_state","radius","offset","environment","object")


class Impact:
    def __init__(self, fine=False):
        self.client = pb.connect(pb.DIRECT)
        self.dt = .0125 if fine else .025
        self.step = .025 if fine else .05
        pb.setGravity(0,0,0)
        pb.setTimeStep(self.dt)
        pb.setPhysicsEngineParameter(numSolverIterations=80 if fine else 40,
                                      restitutionVelocityThreshold=0, contactBreakingThreshold=.002,
                                      contactERP=0, deterministicOverlappingPairs=1)
        self.spheres = {}
        for radius in RADII:
            shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
            self.spheres[radius] = pb.createMultiBody(1., shape, basePosition=[100.,0.,20.])
            pb.changeDynamics(self.spheres[radius],-1,localInertiaDiagonal=[0.,0.,0.],
                              linearDamping=0.,angularDamping=0.,rollingFriction=0.,spinningFriction=0.)
        self.meshes = {}
        self.active = None

    def terrain(self, family, height, state, curv):
        # The rigid facet mesh lives in an object-local frame. Both input speed
        # components are rotated by the base tilt for each specimen.
        key = family,height,state,curv
        if self.active == key:
            return self.meshes[key]
        if self.active is not None:
            pb.resetBasePositionAndOrientation(self.meshes[self.active][0],[200.,0.,0.],[0,0,0,1])
        if key not in self.meshes:
            x = np.linspace(-2.5,2.5,round(5/self.step)+1)
            z = np.zeros_like(x) if family == "flat" else -loaded_shape(x+height,height,family,state)[0]
            z -= z[len(x)//2]
            z -= .5*curv*x*x/25
            vertices = [[float(xx),y,float(zz)] for xx,zz in zip(x,z) for y in (-5.,5.)]
            indices = [v for i in range(len(x)-1) for v in (2*i,2*i+2,2*i+1,2*i+2,2*i+3,2*i+1)]
            shape = pb.createCollisionShape(pb.GEOM_MESH, vertices=vertices, indices=indices,
                                            flags=pb.GEOM_FORCE_CONCAVE_TRIMESH | pb.GEOM_CONCAVE_INTERNAL_EDGE)
            body = pb.createMultiBody(0,shape,basePosition=[0.,0.,0.])
            pb.changeDynamics(body,-1,lateralFriction=1., restitution=1., collisionMargin=0.)
            self.meshes[key] = body,x,z
        body,x,z = self.meshes[key]
        pb.resetBasePositionAndOrientation(body,[0.,0.,0.],[0,0,0,1])
        self.active = key
        return body,x,z

    def site(self, terrain, offset, radius):
        body,x,z = terrain
        i = np.searchsorted(x,offset,side="right")-1
        # Give the rigid sphere a closest point at the nominal surface site.
        slope = (z[i+1]-z[i-1])/(x[i+1]-x[i-1])
        norm = math.hypot(1.,slope)
        return offset-(radius+.003)*slope/norm, np.interp(offset,x,z)+(radius+.003)/norm

    def trial(self, body, start, radius, vx, vz, mu, e):
        tip = self.spheres[radius]
        # The collision detector sees a positive separation between consecutive
        # independent trials so persistent manifolds cannot update the next trial.
        pb.resetBasePositionAndOrientation(tip,[100.,30.*radius,20.],[0,0,0,1])
        pb.resetBaseVelocity(tip,[0,0,0],[0,0,0])
        pb.performCollisionDetection()
        pb.resetBasePositionAndOrientation(tip,[start[0],0.,start[1]],[0,0,0,1])
        pb.resetBaseVelocity(tip,[vx,0.,vz],[0,0,0])
        pb.changeDynamics(tip,-1,lateralFriction=mu,restitution=e)
        first = None
        for _ in range(12):
            pb.stepSimulation()
            hits = pb.getContactPoints(tip,body)
            loaded = [c for c in hits if c[9]>1.e-9]
            if loaded:
                normal = max(loaded,key=lambda c:c[9])[7]
                first = normal if first is None else first
                break
        if first is None:
            raise RuntimeError(f"missed impact {start} {vx} {vz}")
        # Confirm any persistent contact through the end of the short impact.
        for _ in range(2):
            pb.stepSimulation()
        v,_ = pb.getBaseVelocity(tip)
        pb.resetBasePositionAndOrientation(tip,[100.,30.*radius,20.],[0,0,0,1])
        pb.resetBaseVelocity(tip,[0,0,0],[0,0,0])
        return abs(v[0]*first[2]-v[2]*first[0]), v[0]*first[0]+v[2]*first[2]

    def close(self):
        pb.disconnect(self.client)


def velocity(base):
    norm=np.hypot(1.,base)
    return (.425+base)/norm, (.425*base-1.)/norm


def material(theta, height, speed, surface, law, sign, regime):
    # Two isotropic rate terms and a small signed material anisotropy are
    # evaluated at impact onset and held fixed for this rigid impact.
    v=abs(speed)
    if law == "rational":
        rate=1.+.095*np.exp(-v/.37)
    else:
        rate=.96+.105*np.tanh(v/.45)
    spatial = 1.+.07*np.cos(height*.17) if surface == "inlay" else 1.+.05*np.sin(height*.28)
    return theta*rate*spatial*(1.+.045*sign*REGIME[regime]*np.tanh(speed/.34))


def analytic_check():
    out={}
    for fine in (False,True):
        sim=Impact(fine)
        terrain=sim.terrain("flat",19,0,0.)
        errs=[]; nerrs=[]; hysteresis=[]
        for radius in RADII:
            site=sim.site(terrain,0.,radius)
            for mu in (0.,.10,.24,.45):
                for tilt in (-.26,.0,.19):
                    for e in (.55,.82):
                        u,w=velocity(tilt)
                        got,outward=sim.trial(terrain[0],site,radius,u,w,mu,e)
                        repeat,_=sim.trial(terrain[0],site,radius,u,w,mu,e)
                        errs.append(abs(got-max(abs(u)-mu*(1+e)*abs(w),0.)))
                        nerrs.append(abs(outward+e*w))
                        hysteresis.append(abs(repeat-got))
        out[str(fine)]=dict(max_tangent_error=max(errs),max_restitution_error=max(nerrs),
                            max_repeat_difference=max(hysteresis))
        sim.close()
    print(json.dumps(out,indent=2))
    return out


def panel(panel_index, fine=False):
    name=f"panel_{panel_index:02d}.npz"
    dest=ROOT/("labels49_fine" if fine else "labels49")/name
    dest.parent.mkdir(exist_ok=True)
    with np.load(V48/"worlds"/name) as w:
        theta=w["theta"]; surface=w["surface"]; law=w["law"]
        sign=w["signs"]; regime=w["regimes"]; idx=w["object_index"]
    # Fixed refinement subset, counterbalanced across all 12 catalog supports.
    objects=np.arange((panel_index*7)%16,128,16) if fine else np.arange(128)
    theta=theta[:,objects]; ids=idx[:,objects]
    speeds=np.empty((4,2,3,2,3,4,len(objects)),dtype=np.float32)
    normal=np.empty_like(speeds)
    sim=Impact(fine)
    curv=np.round(theta[...,3]/.05)*.05
    for fi, family in enumerate(FAMILIES):
        for hi,h in enumerate(HEIGHTS):
            for j,state in enumerate(STATES):
                if fi>0 and state==0:
                    # Local unloaded shape is identical across all four matched families.
                    ref=loaded_shape(np.linspace(h-2.5,h+2.5,51),h,FAMILIES[0],state)[0]
                    assert np.array_equal(ref,loaded_shape(np.linspace(h-2.5,h+2.5,51),h,family,state)[0])
                    speeds[fi,hi,j]=speeds[0,hi,j]; normal[fi,hi,j]=normal[0,hi,j]
                    continue
                for k in np.unique(curv):
                    terrain=sim.terrain(family,h,state,float(k))
                    envs,ids2=np.nonzero(curv==k)
                    for ri,radius in enumerate(RADII):
                        for oi,off in enumerate(OFFSETS):
                            site=sim.site(terrain,off,radius)
                            u,w=velocity(theta[...,2]+theta[...,3]*h/25)
                            for env,i in zip(envs,ids2):
                                mu=material(theta[env,i,4],h+off,u[env,i],surface[env],law[env],sign[env],regime[env])
                                speed,vn=sim.trial(terrain[0],site,radius,u[env,i],w[env,i],mu,theta[env,i,5])
                                speeds[fi,hi,j,ri,oi,env,i]=speed
                                normal[fi,hi,j,ri,oi,env,i]=vn
    sim.close()
    assert np.isfinite(speeds).all() and speeds.min()>=0
    tmp=dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,labels=speeds<=THRESHOLD,slip_speed=speeds,separation_speed=normal,object_slots=objects,
                        object_index=ids,axes=AXES,state=STATES,radii_mm=RADII,heights_mm=HEIGHTS,
                        families=FAMILIES,offsets_mm=OFFSETS,bullet_api=pb.getAPIVersion(),dt_ms=.0125 if fine else .025,
                        edge_mm=.025 if fine else .05)
    os.replace(tmp,dest)
    print(f"finished {name} fine={fine}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--panels",type=int,nargs="+")
    parser.add_argument("--fine",action="store_true")
    parser.add_argument("--flat",action="store_true")
    args=parser.parse_args()
    if args.flat:
        analytic_check()
    elif args.panels:
        if not os.getenv("SLURM_JOB_ID") or not (ROOT/"solver49.sha256").exists():
            raise RuntimeError("generate target labels only on a Slurm worker after freeze")
        for k in args.panels:
            panel(k,args.fine)
