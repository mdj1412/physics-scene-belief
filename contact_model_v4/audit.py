import json,time,platform
from pathlib import Path
import numpy as np
import torch
import pybullet as p
from engine import *
from inference import *
from sensor import observe

def physics_audit():
    rng=np.random.default_rng(863);p.connect(p.DIRECT);rows=[]
    for family in FAMILIES:
        for i in range(12):
            theta=BOUNDS[:,0]+rng.random(4)*(BOUNDS[:,1]-BOUNDS[:,0]);theta[2]=0 if i<6 else .3
            b=[0.,.001,.003][i%3];shift=float(rng.choice(SHIFTS));th=tensor(theta[None,:])
            exact=simulate(th,tensor([b]),family,shift)['velocity'][0].cpu().numpy()
            centers,half=boxes(th,family);centers=centers.cpu().numpy()[0];half=half.cpu().numpy()[0]
            for dt in [1/7680,1/15360]:
                p.resetSimulation();p.setGravity(0,0,0);p.setTimeStep(dt)
                p.setPhysicsEngineParameter(numSolverIterations=100,restitutionVelocityThreshold=0,
                    contactERP=0,contactSlop=0,deterministicOverlappingPairs=1)
                sphere=p.createCollisionShape(p.GEOM_SPHERE,radius=R)
                ball=p.createMultiBody(baseMass=1,baseCollisionShapeIndex=sphere,basePosition=[-.18,shift,0])
                p.changeDynamics(ball,-1,restitution=float(theta[1]),lateralFriction=float(theta[2]),linearDamping=0,angularDamping=0,rollingFriction=0,spinningFriction=0,contactProcessingThreshold=0)
                p.resetBaseVelocity(ball,linearVelocity=[1,0,0])
                for c,h in zip(centers,half):
                    shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[*h,.04]);body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=[*c,0])
                    p.changeDynamics(body,-1,restitution=1,lateralFriction=1,collisionMargin=b,contactProcessingThreshold=0)
                for _ in range(round(.8/dt)):p.stepSimulation()
                velocity=np.array(p.getBaseVelocity(ball)[0][:2])
                rows.append(dict(family=family,i=i,theta=theta.tolist(),rounding=b,shift=shift,dt=dt,exact=exact.tolist(),bullet=velocity.tolist(),error=float(np.linalg.norm(velocity-exact))))
    p.disconnect();return rows

def posterior_audit(out):
    results=[]
    # Predetermined validation cases only, one per family/regime. Test untouched.
    for case in [0,1,2,24,25,26,48,49,50]:
        path=Path('results/main')/f'case_{case:03d}.json'
        if not path.exists():continue
        d=json.loads(path.read_text());family=d['family'];obs=d['observations']
        data=[('factual',obs['factual'],1.)]
        for n in [768,3072]:
            for seed in [91,92]:
                atoms,stat=smc(family,data,n=n,seed=seed,moves=12)
                pred=predict(atoms,family);gain,eig=acquisition(atoms,family,pred,samples=1024,seed=381)
                means=torch.einsum('qnd,n->qd',pred['velocity'],atoms[2]).cpu().numpy()
                results.append(dict(case=case,n=n,seed=seed,mean=means.tolist(),choice=gain.argmax(1).tolist(),gains=gain.tolist(),cost=stat))
    return results

if __name__=='__main__':
    import sys
    out=Path('results/audit');out.mkdir(parents=True,exist_ok=True)
    if '--posterior-only' not in sys.argv:
        rows=physics_audit();(out/'cross_engine_matched.json').write_text(json.dumps(rows,indent=2));print('physics complete',len(rows),flush=True)
    results=posterior_audit(out);(out/'posterior_convergence.json').write_text(json.dumps(results,indent=2));print('posterior complete',len(results),flush=True)
