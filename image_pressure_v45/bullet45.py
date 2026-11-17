"""Held-out transfer to a small rigid sphere in Bullet; observe actual onset coordinates."""
import argparse
import itertools
import json
from pathlib import Path
import os
import sys
import numpy as np
import pybullet as b

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"contact_patch_v44"))
import solver44 as base
from physics_check43 import slope

R=.5
GAP=.030


def simulate(surface, gloc, mu, rest, velocity=base.INCOMING):
    # The desired support is x=0 in the local mesh. Extrapolate launch backward from
    # a tangent sphere center along the incoming trajectory to a vertical gap GAP.
    norm=np.hypot(1.,gloc)
    cx=(R*gloc/norm-velocity*GAP/(1.-velocity*gloc))*.001
    cz=(R/norm+GAP/(1.-velocity*gloc))*.001
    shoe=surface.shoe(R,0.)
    ids=[]
    for x,z,lane,material,e in zip(cx,cz,surface.lanes,mu,rest):
        i=b.createMultiBody(baseMass=1.,baseCollisionShapeIndex=shoe,
            basePosition=[x*base.SCALE,lane,z*base.SCALE])
        b.changeDynamics(i,-1,lateralFriction=float(material),restitution=float(e),
            spinningFriction=0.,rollingFriction=0.,linearDamping=0.,angularDamping=0.,localInertiaDiagonal=[0.,0.,0.])
        b.resetBaseVelocity(i,[velocity*base.SCALE,0.,-base.SCALE])
        ids.append(i)
    idx={bid:i for i,bid in enumerate(ids)}
    n=len(ids)
    normal=np.zeros((n,2))
    seen=np.zeros(n,bool); first=np.full(n,np.nan); end=np.zeros(n,bool)
    for step in range(round(base.HORIZON/base.DT)):
        b.stepSimulation()
        moment=np.zeros((n,4))
        for c in b.getContactPoints():
            f=c[9]
            if f<=1e-8:continue
            if c[1] in idx:
                i=idx[c[1]]; x=c[6][0]; nx,nz=c[7][0],c[7][2]
            elif c[2] in idx:
                i=idx[c[2]]; x=c[5][0]; nx,nz=-c[7][0],-c[7][2]
            else:
                continue
            if nz<=.2:continue
            moment[i]+=[f,nx*f,nz*f,x*f]
        end=moment[:,0]>0
        onset=end&~seen
        first[onset]=1000.*moment[onset,3]/moment[onset,0]/base.SCALE
        seen|=end
        normal[end]=moment[end,1:3]
    vel=np.asarray([b.getBaseVelocity(i)[0] for i in ids])[:,[0,2]]/base.SCALE
    for i in ids:b.removeBody(i)
    den=np.linalg.norm(normal,axis=1)
    slip=abs(vel[:,0]*normal[:,1]-vel[:,1]*normal[:,0])/np.where(seen,den,1.)
    return dict(slip_speed=np.where(seen,slip,np.nan), first_site_delta_mm=first, valid=seen,loaded_end=end)


def simulate_group(panel,shape,var,height):
    paths={(sf,law):ROOT/"labels45_bullet"/f"panel_{panel:02d}"/f"{shape}_{sf}_{law}_{var}_h{height}.npz"
           for sf,law in itertools.product(base.SURFACES,base.LAWS)}
    if all(p.exists() for p in paths.values()):return
    with np.load(ROOT/"results"/"worlds"/f"panel_{panel:02d}.npz") as f:
        th=f["theta"][:,:8].reshape(-1,6);signs=np.repeat(f["signs"],8);regs=np.repeat(f["regimes"],8)
    fields={key:{} for key in paths}
    for d,off in enumerate(base.OFFSETS):
        h=height+off
        surface=base.Surface(th,h,shape,var,base.DT,base.GRID_MM)
        g=th[:,2]+th[:,3]*(h/25.)+slope(h,shape,var)
        for sf,law in paths:
            mu=base.material_mu(th,h,np.zeros_like(g),g,sf,law,signs,regs)
            values=simulate(surface,g,mu,th[:,5])
            for k,v in values.items():
                fields[sf,law].setdefault(k,[]).append(v.reshape(4,8))
    for key,path in paths.items():
        a={k:np.stack(v) for k,v in fields[key].items()}
        a["labels"]=(a["slip_speed"]<=base.THRESHOLD)&a["valid"]
        a["desired_centers_mm"]=height+np.array(base.OFFSETS)
        a["first_contact_x_mm"] = a["desired_centers_mm"][:,None,None] + a["first_site_delta_mm"]
        path.parent.mkdir(exist_ok=True,parents=True)
        tmp=path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,**a)
        os.replace(tmp,path)


def control():
    th=np.zeros((6,6));th[:,2]=np.tile([0.,-.2,.2],2)
    th[:,5]=.75
    mu=np.repeat([0.,1.],3)
    surface=base.Surface(th,0,None,"base",base.DT,base.GRID_MM)
    rows=simulate(surface,th[:,2],mu,th[:,5])
    assert rows["valid"].all() and rows["slip_speed"][0]>.420 and rows["slip_speed"][3]<.005
    assert np.max(abs(rows["first_site_delta_mm"]))<.055
    out={k:v.tolist() for k,v in rows.items()}
    out["slope"]=th[:,2].tolist();out["mu"]=mu.tolist();out["api"]=b.getAPIVersion()
    (ROOT/"bullet_neutral45.json").write_text(json.dumps(out,indent=2)+"\n")
    print(out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--neutral",action="store_true");p.add_argument("--panels",type=int,nargs="+")
    args=p.parse_args()
    b.connect(b.DIRECT)
    if args.neutral:control()
    else:
        assert (ROOT/"BULLET_GO").exists()
        for panel in args.panels:
            for s,v,h in itertools.product(base.SHAPES,base.VARIANTS,(19,31)):
                simulate_group(panel,s,v,h)
            print("Bullet completed",panel,flush=True)
    b.disconnect()
