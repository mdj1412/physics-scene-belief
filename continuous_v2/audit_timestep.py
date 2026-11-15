"""Audit discretization of grazing impacts without modifying the main dataset."""
import json
import sys
from pathlib import Path
import numpy as np
import pybullet as p
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from physics import World


def main():
    world=World()
    offsets=np.linspace(.025,.055,121)
    es=np.array([.2,.5,.8]);rates=[960,3840,7680]
    all_v=[];all_c=[]
    for rate in rates:
        vv=[];cc=[]
        for y in offsets:
            for e in es:
                world.reset(float(y),float(e),wall_x=.2,start_x=-.15,speed=.6)
                p.setTimeStep(1/rate)
                contact=False
                for _ in range(round(.8*rate)):
                    p.stepSimulation()
                    contact |= any(c[9]>1e-7 for c in p.getContactPoints(world.ball,world.wall))
                vv.append(np.array(p.getBaseVelocity(world.ball)[0][:2])/.6)
                cc.append(int(contact))
        all_v.append(vv);all_c.append(cc)
        print('Completed timestep',rate,flush=True)
    v=np.array(all_v);c=np.array(all_c)
    result=dict(case_count=len(offsets)*len(es),rates=rates,scope='grazing-contact diagnostic, not a rerun of policy metrics',comparisons=[])
    for i,j in [(0,1),(1,2),(0,2)]:
        err=v[i]-v[j]
        result['comparisons'].append(dict(coarse_rate=rates[i],fine_rate=rates[j],
            vector_component_rmse=float(np.sqrt(np.mean(err**2))),max_abs_error=float(abs(err).max()),
            contact_disagreement=float(np.mean(c[i]!=c[j]))))
    out=Path('results/timestep_audit');out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'curves.npz',offset=offsets,restitution=es,velocity=v,contact=c,rates=rates)
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    p.disconnect()


if __name__=='__main__':main()
