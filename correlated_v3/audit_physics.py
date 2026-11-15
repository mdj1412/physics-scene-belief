"""Independent PyBullet verification; never used as the policy's hidden input."""
import sys,time,json,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import physics
import pybullet as p
import numpy as np
from core import impact

out=Path('results/audit');out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20261105)
# Non-grid-aligned impacts and offsets spanning head-on, grazing and miss.
gs=np.r_[rng.uniform(.032,.0498,48),rng.uniform(-.08,.08,48)]
es=rng.uniform(.1,.9,len(gs))
analytic=np.array(impact(gs,es)).T
world=physics.World();values=[];start=time.perf_counter()
for dt in [1/960,1/3840,1/7680,1/30720]:
    physics.DT=dt;vv=[]
    for g,e in zip(gs,es):
        world.reset(float(g),float(e),wall_x=.2,start_x=-.15,speed=.6)
        hit=False
        for i in range(round(.85/dt)):
            p.stepSimulation()
            hit|=any(c[9]>1e-7 for c in p.getContactPoints(world.ball,world.wall))
        v=np.array(p.getBaseVelocity(world.ball)[0][:2])/.6
        vv.append([*v,float(hit)])
    values.append(vv)
    print('verified dt',dt,flush=True)
values=np.array(values)
metrics=[]
for i,dt in enumerate([1/960,1/3840,1/7680,1/30720]):
    diff=values[i,:,:2]-analytic[:,:2]
    metrics.append(dict(dt=dt,velocity_rmse=float(np.sqrt(np.mean(diff**2))),
        vx_rmse=float(np.sqrt(np.mean(diff[:,0]**2))),max_velocity_error=float(abs(diff).max()),
        factual_vx_rmse=float(np.sqrt(np.mean(diff[:48,0]**2))),
        contact_disagreement=float(np.mean(values[i,:,2]!=analytic[:,2]))))
np.savez_compressed(out/'physics.npz',geometry=gs,restitution=es,analytic=analytic,pybullet=values)
(out/'physics.json').write_text(json.dumps(dict(host=platform.node(),seconds=time.perf_counter()-start,simulations=world.calls,metrics=metrics),indent=2))
p.disconnect()
