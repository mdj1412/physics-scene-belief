"""Independent Bullet rigid-body impact on resolved extruded contour meshes.

Lengths supplied in metres. A uniform similarity scaling of 50 improves the
small-geometry conditioning of the collision detector; speeds scale likewise.
World provenance is restricted to contour position and material coefficients.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as b

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'blind_decoder_v43'))
from physics_check43 import displacement, friction

SCALE = 50.
RADII = (.5, 2.)
SPANS = (0., 2.)
OFFSETS = (-.5, 0., .5)
SURFACES = ('inlay', 'woven')
LAWS = ('rational', 'gaussian_rate')
SHAPES = ('fold','ripple')
VARIANTS = ('base','hollow','bulge')
NOBJ = 8
DT = 2e-5
HORIZON = 4e-4
GRID_MM = .10
THRESHOLD = .02
INCOMING = .425
FIELDS = ('slip_speed','valid','span_mm','npoints','loaded_end','support_mm','approach_speed')


def set_engine(dt):
    b.resetSimulation()
    b.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=80,
            restitutionVelocityThreshold=0., contactBreakingThreshold=.005,
            contactSlop=.00001, deterministicOverlappingPairs=1)
    b.setGravity(0,0,0)


def geom_profile(theta, h_mm, x_mm, shape, variant):
    """Mesh vertical ordinate in metres, zeroed at the commanded location."""
    y = (h_mm+x_mm)*.001
    h = h_mm*.001
    if shape is None:
        return -theta[:,2,None] * x_mm[None,:] * .001
    deform = np.array([displacement(h_mm+v,shape,variant) for v in x_mm])
    d0 = displacement(h_mm,shape,variant)
    return -(theta[:,2,None]*(y-h) + theta[:,3,None]*(y*y-h*h)/.05
             + (deform-d0)*.001)


class Surface:
    def __init__(self, theta, h_mm, shape, variant, dt, spacing):
        self.dt=dt
        set_engine(dt)
        self.theta=theta
        self.h=h_mm
        self.x=np.linspace(-5.,5.,1+round(10/spacing))
        self.z=geom_profile(theta,h_mm,self.x,shape,variant)
        self.shoes={}
        self.ground=[]
        self.lanes=np.arange(len(theta))*.56
        inds=[]
        for k in range(len(self.x)-1):
            a=2*k
            inds.extend((a,a+2,a+1,a+1,a+2,a+3))
        verts=np.zeros((len(self.x)*2,3))
        verts[:,0]=np.repeat(self.x*.001*SCALE,2)
        verts[:,1]=np.tile([-.14,.14],len(self.x))
        for z,lane in zip(self.z,self.lanes):
            verts[:,2]=np.repeat(z*SCALE,2)
            mesh=b.createCollisionShape(b.GEOM_MESH,vertices=verts.tolist(),indices=inds,
                    flags=b.GEOM_FORCE_CONCAVE_TRIMESH | b.GEOM_CONCAVE_INTERNAL_EDGE)
            ground=b.createMultiBody(baseMass=0.,baseCollisionShapeIndex=mesh,basePosition=[0.,lane,0.])
            b.changeDynamics(ground,-1,lateralFriction=1.,restitution=1.,collisionMargin=0.)
            self.ground.append(ground)

    def shoe(self, radius, span):
        if (radius,span) not in self.shoes:
            r=radius*.001*SCALE
            if span:
                geom=b.createCollisionShape(b.GEOM_CAPSULE,radius=r,height=span*.001*SCALE,
                        collisionFrameOrientation=b.getQuaternionFromEuler([0,math.pi/2,0]))
            else:
                geom=b.createCollisionShape(b.GEOM_SPHERE,radius=r)
            self.shoes[radius,span]=geom
        return self.shoes[radius,span]

    def support(self,radius,span):
        # Max over an upsampled piecewise-linear triangulation of the actual mesh.
        x=np.linspace(-span/2-radius,span/2+radius,round((span+2*radius)/.01)+1)
        underside=np.sqrt(np.maximum(radius**2-np.maximum(np.abs(x)-span/2,0.)**2,0.))*.001
        center=[]; site=[]; slope=[]
        for z in self.z:
            scores=np.interp(x,self.x,z)+underside
            peak=scores.max()
            u=np.mean(x[scores>=peak-1.e-11])
            k=np.searchsorted(self.x,u,side='right').clip(1,len(self.x)-1)
            center.append(peak)
            site.append(u)
            slope.append(-(z[k]-z[k-1])/(.001*(self.x[k]-self.x[k-1])))
        return np.asarray(center),np.asarray(site),np.asarray(slope)

    def simulate(self,radius,span,mu,rest,velocity=INCOMING):
        # Pre-contact gap is 0.03 mm and is fixed in physical coordinates across dt.
        center,site,g=self.support(radius,span)
        geom=self.shoe(radius,span)
        ids=[]
        for i,(z,material,e) in enumerate(zip(center,mu,rest)):
            bid=b.createMultiBody(baseMass=1.,baseCollisionShapeIndex=geom,
                    basePosition=[0.,self.lanes[i],(z+.00003)*SCALE])
            b.changeDynamics(bid,-1,lateralFriction=float(material),restitution=float(e),
                    spinningFriction=0.,rollingFriction=0.,linearDamping=0.,
                    angularDamping=0.,localInertiaDiagonal=[0.,0.,0.])
            b.resetBaseVelocity(bid,[velocity*SCALE,0.,-SCALE])
            ids.append(bid)
        body_to_row={v:i for i,v in enumerate(ids)}
        n=len(ids)
        normal=np.zeros((n,2)); points=np.zeros(n,np.uint8); width=np.zeros(n)
        seen=np.zeros(n,bool); current=np.zeros(n,bool)
        for _ in range(round(HORIZON/self.dt)):
            b.stepSimulation()
            loaded={}
            for c in b.getContactPoints():
                if c[9] <= 1.e-8: continue
                if c[1] in body_to_row:
                    idx=body_to_row[c[1]]; locus=c[6][0]; nx,nz=c[7][0],c[7][2]
                elif c[2] in body_to_row:
                    idx=body_to_row[c[2]]; locus=c[5][0]; nx,nz=-c[7][0],-c[7][2]
                else: continue
                if nz <= .2: continue
                loaded.setdefault(idx,[]).append((locus,nx,nz,c[9]))
            current[:]=False
            for i,rows in loaded.items():
                current[i]=seen[i]=True
                # Last loaded contact normal, weighted by normal impulse.
                normal[i]=[sum(q[1]*q[3] for q in rows),sum(q[2]*q[3] for q in rows)]
                points[i]=max(points[i],len(rows))
                xx=[q[0] for q in rows]
                width[i]=max(width[i],(max(xx)-min(xx))*1000/SCALE)
        vel=np.asarray([b.getBaseVelocity(i)[0] for i in ids])[:,[0,2]]/SCALE
        for item in ids:
            b.removeBody(item)
        denom=np.linalg.norm(normal,axis=1)
        vt=(vel[:,0]*normal[:,1]-vel[:,1]*normal[:,0])/np.where(seen,denom,1.)
        slip=np.where(seen,abs(vt),np.nan)
        return dict(valid=seen,slip_speed=slip,npoints=points,span_mm=width,loaded_end=current,
                    support_mm=site,approach_speed=(velocity+g)/np.sqrt(1+g*g))


def material_mu(theta, height, site, g, sf, law, signs, regimes):
    vt=(INCOMING+g)/np.sqrt(1+g*g)
    return np.array([friction(th[4],v,(height+s)*.001,sf,law,sign,regime)[0]
            for th,v,s,sign,regime in zip(theta,vt,site,signs,regimes)])


def shape_run(panel,shape,variant,h,sensitivity):
    manifest=ROOT/'results'/'worlds'/f'panel_{panel:02d}.npz'
    with np.load(manifest) as a:
        if sensitivity=='primary':
            theta=a['theta'][:,:NOBJ].reshape(-1,6)
            signs=np.repeat(a['signs'],NOBJ); regimes=np.repeat(a['regimes'],NOBJ)
        else:
            theta=a['theta'][0,:2]; signs=np.repeat(a['signs'][0],2); regimes=np.repeat(a['regimes'][0],2)
    dt=DT/2 if sensitivity=='halfdt' else DT
    spacing=GRID_MM/2 if sensitivity=='halfmesh' else GRID_MM
    rows={}
    for i,offset in enumerate(OFFSETS):
        obj=Surface(theta,h+offset,shape,variant,dt,spacing)
        for ir,r in enumerate(RADII):
            for iw,w in enumerate(SPANS):
                _,site,g=obj.support(r,w)
                for sf,law in itertools.product(SURFACES,LAWS):
                    mu=material_mu(theta,h+offset,site,g,sf,law,signs,regimes)
                    result=obj.simulate(r,w,mu,theta[:,5])
                    key=(sf,law)
                    if key not in rows:
                        rows[key]={k:np.empty((2,2,3,len(theta)),dtype=v.dtype) for k,v in result.items()}
                    for k,v in result.items():
                        rows[key][k][ir,iw,i]=v
    outdir=ROOT/('labels44' if sensitivity=='primary' else 'labels44_'+sensitivity)/f'panel_{panel:02d}'
    outdir.mkdir(parents=True,exist_ok=True)
    for (sf,law),fields in rows.items():
        if sensitivity=='primary':
            fields={k:v.reshape(2,2,3,4,NOBJ) for k,v in fields.items()}
        fields['labels']=(fields['slip_speed']<=THRESHOLD)&fields['valid']
        path=outdir/f'{shape}_{sf}_{law}_{variant}_h{h}.npz'
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**fields)
        os.replace(tmp,path)


def neutral_controls():
    b.connect(b.DIRECT)
    data=[]
    start=time.perf_counter()
    for dt in (DT,DT/2):
        th=np.zeros((4,6));th[:,5]=.75
        ground=Surface(th,0,None,'base',dt,GRID_MM)
        for span in SPANS:
            for radius in RADII:
                row=ground.simulate(radius,span,np.array([0.,.02,1.,1.]),th[:,5])
                print('fixture',dt,radius,span,row['slip_speed'],flush=True)
                assert row['valid'].all(),row
                assert row['slip_speed'][0]>.39 and row['slip_speed'][2]<.02,row
                data.append(dict(dt=dt,radius_mm=radius,span_mm=span,
                     slip=row['slip_speed'].tolist(),footprint_mm=row['span_mm'].tolist(),
                     loaded=row['loaded_end'].tolist(),npoints=row['npoints'].tolist()))
    result=dict(engine=b.getAPIVersion(),build_path=b.__file__,wall_seconds=time.perf_counter()-start,flat=data)
    (ROOT/'neutral_controls44.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
    b.disconnect()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control',action='store_true')
    parser.add_argument('--shard',type=int)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--sensitivity',choices=('primary','halfdt','halfmesh'),default='primary')
    args=parser.parse_args()
    if args.control:
        neutral_controls()
        return
    if not (ROOT/'SOLVER_GO').exists():
        raise RuntimeError('Scoring requires parent freeze approval')
    b.connect(b.DIRECT)
    for j,(panel,shape,variant,h) in enumerate(itertools.product(range(12),SHAPES,VARIANTS,(19,31))):
        if j%args.workers!=args.shard: continue
        d=ROOT/('labels44' if args.sensitivity=='primary' else 'labels44_'+args.sensitivity)/f'panel_{panel:02d}'
        if all((d/f'{shape}_{sf}_{law}_{variant}_h{h}.npz').exists() for sf,law in itertools.product(SURFACES,LAWS)):
            continue
        shape_run(panel,shape,variant,h,args.sensitivity)
        print('done',args.sensitivity,panel,shape,variant,h,flush=True)
    b.disconnect()


if __name__=='__main__':
    main()
