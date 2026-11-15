"""Separate Bullet box-margin effects from timestep effects."""
import sys,json,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import physics
import pybullet as p
import numpy as np
from core import impact

out=Path('results/audit');d=np.load(out/'physics.npz')
g,e=d['geometry'],d['restitution'];world=physics.World();physics.DT=1/30720
start=time.perf_counter();records=[];values=[]
for gi,ei in zip(g,e):
    world.reset(float(gi),float(ei),wall_x=.2,start_x=-.15,speed=.6)
    margin_default=p.getDynamicsInfo(world.wall,-1)[-1]
    p.changeDynamics(world.wall,-1,collisionMargin=0)
    for _ in range(round(.85/physics.DT)):p.stepSimulation()
    values.append(np.array(p.getBaseVelocity(world.ball)[0][:2])/.6)
values=np.array(values);exact=np.array(impact(g,e)).T[:,:2]
# Bullet represents box cores with an inward margin and rounds the corners.
margin=float(margin_default)
u=np.clip((abs(g)-(.03-margin))/(.02+margin),0,1);A=1-u*u
rounded=np.c_[1-(1+e)*A,-(1+e)*np.sqrt(A)*u*np.sign(g)]
def errors(a,b):return dict(rmse=float(np.sqrt(np.mean((a-b)**2))),max_error=float(abs(a-b).max()))
result=dict(host=platform.node(),seconds=time.perf_counter()-start,simulations=world.calls,dt=physics.DT,
    default_box_margin=margin,zero_margin_vs_sharp=errors(values,exact),
    default_margin_vs_sharp=errors(d['pybullet'][-1,:,:2],exact),
    default_margin_vs_rounded=errors(d['pybullet'][-1,:,:2],rounded))
(out/'margin.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'margin.npz',zero_margin=values,sharp=exact,rounded=rounded)
print(json.dumps(result,indent=2));p.disconnect()
