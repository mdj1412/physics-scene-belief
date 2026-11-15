import argparse,json,os,time,platform
from pathlib import Path
import numpy as np
import torch
from engine import BOUNDS,FAMILIES,SHIFTS,ACTIONS,tensor,batch
from belief import Belief,parameter_summary,predict,metrics
from geometry import render,extract,layout
from dynamic_sensor import observe

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def geometry_score(parameters,theta,b,family):
    def envelope(g,d,rounding):
        points=[]
        for x,y,hx,hy in layout(g,d,family):
            for lo in [0,np.pi/2,np.pi,3*np.pi/2]:
                angles=np.linspace(lo,lo+np.pi/2,9);normal=np.stack([np.cos(angles),np.sin(angles)],1)
                signs=np.sign([np.cos(lo+np.pi/4),np.sin(lo+np.pi/4)])
                center=np.array([x,y])+signs*(np.array([hx,hy])-rounding)
                points.extend(center+(.02+rounding)*normal)
        return np.array(points)
    est=parameters['mean'];a=envelope(est[0],est[3],est[4]);t=envelope(theta[0],theta[3],b)
    return dict(rounding_abs_error_mm=abs(est[4]-b)*1e3,
        boundary_rmse_mm=float(np.sqrt(np.mean(np.sum((a-t)**2,1)))*1e3),
        rounding_coverage90=bool(parameters['b_interval'][0]<=b<=parameters['b_interval'][1]))

def run_case(case,out):
    family=FAMILIES[case//24];unit=case%24;rng=np.random.default_rng(2026110800+case)
    theta=BOUNDS[:,0]+rng.random(4)*(BOUNDS[:,1]-BOUNDS[:,0]);b=rng.uniform(0,.006)
    obs={a:observe(theta,b,family,a,rng) for a in ['factual','left','right','center']}
    images=out/'images'/f'{case:03d}';images.mkdir(parents=True,exist_ok=True)
    view={}
    for quality in ['high','low']:
        im=render(theta[0],theta[3],b,family,quality,np.random.default_rng(202611088000+case))
        view[quality]=extract(im,family,quality);im.save(images/f'{quality}.png')
    sim=batch(tensor(theta[None,:]),tensor([b]),family,SHIFTS);truth={k:v[:,0].cpu().numpy() for k,v in sim.items()}
    assert not truth['overflow'].any()
    meta=dict(case=case,measurement_model='finite_frame_50ms',family=family,split='validation' if unit<8 else 'test',theta=theta,rounding=b,observations=obs,view=view,truth=truth)
    belief=Belief(family);initial,cost=belief.smc([('factual',obs['factual'])],n=4096,seed=150100+case)
    pred=predict(initial,family);records=[dict(name='no_observation',metrics=metrics(initial,pred,truth),parameters=parameter_summary(initial),cost=cost)]
    # These observations and likelihoods are identical across geometry variants.
    for action in ['left','right','center']:
        atoms,cost=belief.smc([('factual',obs['factual']),(action,obs[action])],seed=160100+case)
        records.append(dict(name='dynamic',action=action,metrics=metrics(atoms,predict(atoms,family),truth),parameters=parameter_summary(atoms),cost=cost))
    policies={}
    for quality in ['high','low']:
        policies[quality]={}
        for mode in ['bbox','boundary']:
            belief=Belief(family,quality,mode)
            gains,eigs=belief.acquisition(initial,pred,seed=170100+case)
            policies[quality][mode]=dict(gains=gains,eigs=eigs)
            atoms,cost=belief.smc([('factual',obs['factual']),('view',view[quality])],seed=160100+case)
            records.append(dict(name='view',quality=quality,mode=mode,action='view',metrics=metrics(atoms,predict(atoms,family),truth),parameters=parameter_summary(atoms),cost=cost))
    for record in records:record['geometry']=geometry_score(record['parameters'],theta,b,family)
    meta['records']=records;meta['policies']=policies
    np.savez_compressed(out/f'factual_{case:03d}.npz',theta=initial[0].cpu().numpy(),rounding=initial[1].cpu().numpy(),weight=initial[2].cpu().numpy())
    temporary=out/f'.case_{case:03d}.json.tmp'
    temporary.write_text(json.dumps(plain(meta),allow_nan=False));temporary.replace(out/f'case_{case:03d}.json')
    print('case',case,family,'seconds',round(sum(r['cost']['seconds'] for r in records),2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--limit',type=int,default=72);args=p.parse_args()
    out=Path('results/main');out.mkdir(parents=True,exist_ok=True)
    runtime=dict(host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),start=time.time(),args=vars(args))
    (out/f'runtime_{args.worker}.json').write_text(json.dumps(runtime))
    for case in range(args.worker,args.limit,2):
        if not (out/f'case_{case:03d}.json').exists():run_case(case,out)
    runtime.update(end=time.time(),peak_memory=torch.cuda.max_memory_allocated());(out/f'runtime_{args.worker}.json').write_text(json.dumps(runtime,indent=2))
