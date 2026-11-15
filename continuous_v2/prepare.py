"""Direct PyBullet truth, shared continuous-response lookup, RGB sensor tables."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from physics import World, centroid, DT
import numpy as np
import pybullet as p

SEED=20261104
FAMILIES=128
WORLDS=32
QUERIES=12


def truncated(rng, mean, sd, low, high, n):
    result=[]
    while len(result)<n:
        x=rng.normal(mean,sd,n*3)
        result.extend(x[(x>=low)&(x<=high)].tolist())
    return np.array(result[:n])


def spec(f):
    rng=np.random.default_rng([SEED,f,10])
    center=float(rng.uniform(-.025,.025))
    sy=float(rng.uniform(.003,.012))
    em=float(rng.uniform(.3,.7))
    es=float(rng.uniform(.08,.23))
    # Numeric intervention values are stratified continuous draws, not type labels.
    residual=-.09+(np.arange(QUERIES)+rng.random(QUERIES))*.18/QUERIES
    return dict(family=f,split='dev' if f<32 else 'validation' if f<64 else 'test',
        geometry_mean=center,geometry_sd=sy,restitution_mean=em,restitution_sd=es,
        delta=(residual-center).tolist(),residual_mean=residual.tolist(),
        sigma_view_px=float(rng.uniform(.5,2)),sigma_probe_px=float(rng.uniform(.5,2)),
        query_type='translate_wall_y',lambda_mse=.005)


def simulate(world,offset,e):
    world.reset(float(offset),float(e),wall_x=.2,start_x=-.15,speed=.6)
    contact=False
    first=-1
    for i in range(round(1.2/DT)):
        p.stepSimulation()
        if any(c[9]>1e-7 for c in p.getContactPoints(world.ball,world.wall)):
            contact=True
            if first<0:first=i
    v=np.array(p.getBaseVelocity(world.ball)[0][:2])/.6
    return np.r_[v,float(contact),first*DT]


def table(shard,out):
    out.mkdir(parents=True,exist_ok=True)
    offsets=np.unique(np.round(np.r_[np.linspace(0,.16,321),np.linspace(.025,.055,601)],8))
    es=np.linspace(.1,.9,33)
    flat=np.arange(len(offsets)*len(es))[shard::4]
    world=World(); t=time.perf_counter(); values=[]
    for j,idx in enumerate(flat):
        values.append(simulate(world,offsets[idx//len(es)],es[idx%len(es)]))
        if j%1500==0: print(f'table shard={shard} {j}/{len(flat)}',flush=True)
    np.savez_compressed(out/f'table_{shard}.npz',offsets=offsets,es=es,indices=flat,values=values)
    (out/f'table_{shard}.json').write_text(json.dumps(dict(host=platform.node(),simulations=world.calls,
        seconds=time.perf_counter()-t,dt=DT,shard=shard,force_threshold=1e-7),indent=2))
    p.disconnect()


def sensor_tables(out):
    world=World(); t=time.perf_counter()
    gs=np.linspace(-.1,.1,4001); es=np.linspace(.1,.9,513)
    gm=[]; em=[]
    for g in gs:
        world.reset(float(g),.5,start_x=-.15)
        gm.append(centroid(world.image(top=True),'blue')[1])
    for e in es:
        frames,_=world.probe(float(e))
        em.append(centroid(frames[1],'red')[0]-centroid(frames[0],'red')[0])
    np.savez_compressed(out/'sensor_tables.npz',g=gs,gmu=gm,e=es,emu=em)
    print('sensor tables seconds',time.perf_counter()-t,flush=True)
    p.disconnect()


def dataset(shard,out):
    out.mkdir(parents=True,exist_ok=True)
    world=World(); started=time.perf_counter()
    for f in range(shard,FAMILIES,4):
        cfg=spec(f); rng=np.random.default_rng([SEED,f,20])
        g=truncated(rng,cfg['geometry_mean'],cfg['geometry_sd'],
                    cfg['geometry_mean']-4*cfg['geometry_sd'],cfg['geometry_mean']+4*cfg['geometry_sd'],WORLDS)
        e=truncated(rng,cfg['restitution_mean'],cfg['restitution_sd'],.1,.9,WORLDS)
        observations=[]; outcomes=[]; base=None; maxdiff=0
        for gi,ei in zip(g,e):
            world.reset(float(gi),float(ei),start_x=-.15)
            frontal=world.image()
            if base is None:base=frontal
            maxdiff=max(maxdiff,int(np.abs(frontal.astype(int)-base.astype(int)).max()))
            top=world.image(top=True)
            frames,_=world.probe(float(ei))
            observations.append([centroid(top,'blue')[1],
                centroid(frames[1],'red')[0]-centroid(frames[0],'red')[0],centroid(frontal,'blue')[1]])
            outcomes.append([simulate(world,float(gi+d),float(ei)) for d in cfg['delta']])
        assert maxdiff==0
        np.savez_compressed(out/f'family_{f:03d}.npz',geometry=g,restitution=e,
                            observation_mu=observations,truth=np.array(outcomes),
                            factual_rgb=base)
        cfg['factual_max_rgb_difference']=maxdiff
        (out/f'family_{f:03d}.json').write_text(json.dumps(cfg,indent=2))
        print(f'data shard={shard} family={f}',flush=True)
    (out/f'data_runtime_{shard}.json').write_text(json.dumps(dict(host=platform.node(),
        simulations=world.calls,renders=world.renders,seconds=time.perf_counter()-started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    p.disconnect()


def merge(out):
    parts=[np.load(out/f'table_{s}.npz') for s in range(4)]
    offsets=parts[0]['offsets']; es=parts[0]['es']
    a=np.full((len(offsets)*len(es),4),np.nan)
    for part in parts:a[part['indices']]=part['values']
    assert np.isfinite(a).all()
    a=a.reshape(len(offsets),len(es),4)
    np.savez_compressed(out/'physics_table.npz',offsets=offsets,es=es,values=a)
    print('Merged',a.shape,flush=True)
    sensor_tables(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['table','data','merge'])
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--out',type=Path,default=Path('data'))
    args=parser.parse_args()
    if args.stage=='merge':merge(args.out)
    elif args.stage=='table':table(args.shard,args.out)
    else:dataset(args.shard,args.out)
