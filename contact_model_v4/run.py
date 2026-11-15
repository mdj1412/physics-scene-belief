import argparse,json,time,platform,os
from pathlib import Path
import numpy as np
import torch
from engine import *
from inference import *
from sensor import observe

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def run_case(case,n,out):
    family=FAMILIES[case//24];unit=case%24;split='validation' if unit<8 else 'test'
    rng=np.random.default_rng(2026110500+case)
    theta=BOUNDS[:,0]+rng.random(4)*(BOUNDS[:,1]-BOUNDS[:,0])
    regime=['sharp','rounded','outside'][unit%3];b={'sharp':0.,'rounded':.003,'outside':.005}[regime]
    obs={a:observe(theta,b,family,a,rng,out/'images'/str(case) if unit==0 else None) for a in ['factual']+ACTIONS}
    sim=batch(tensor(theta[None,:]),tensor([b]),family,SHIFTS)
    truth={k:v[:,0].cpu().numpy() for k,v in sim.items()}
    assert not truth['overflow'].any()
    meta=dict(case=case,unit=unit,family=family,split=split,regime=regime,theta=theta,rounding=b,observations=obs,truth=truth)
    records=[];costs=[];seed=50100+case*100
    def record(name,atoms,pred,action='all',**kw):
        v=metrics(atoms,pred,truth)
        records.append(dict(name=name,action=action,metrics=v,**kw))
    data=[('factual',obs['factual'],1.)]
    atoms,stat=smc(family,data,n=n,seed=seed);costs.append(dict(name='broad_factual',**stat))
    pred=predict(atoms,family);gains,eigs=acquisition(atoms,family,pred,samples=256,seed=seed+1)
    meta['gains']=gains;meta['eigs']=eigs
    record('broad',atoms,pred,action='stop')
    for action in ACTIONS:record('broad',reweight(atoms,family,action,obs[action]),pred,action=action)
    # Factual-only point diagnostic: maximum likelihood particle under the
    # uniform prior. This is joint MAP, not conditional plug-in geometry.
    tt,mm,ww=atoms
    ll=-.5*((observation_means(tt,mm,family,'factual')[:,0]-obs['factual'][0])/.025)**2
    best=int(ll.argmax());map_atoms=(tt[best:best+1],mm[best:best+1],tensor([1.]))
    map_pred={k:v[:,best:best+1] for k,v in pred.items()};record('joint_MAP',map_atoms,map_pred,action='stop')
    # Same strong posterior, same support size; upstream cost is shared, not free.
    for size in [64,128]:
        for method in ['systematic','branch']:
            for replicate in range(3):
                small,sp=compress(atoms,pred,size,method,seed+10+replicate)
                gg,_=acquisition(small,family,sp,samples=256,seed=seed+1)
                for action in ACTIONS:
                    record(f'{method}_{size}',reweight(small,family,action,obs[action]),sp,action=action,replicate=replicate,choice=gg.argmax(1),gains=gg)
    # Closed-loop diagnostic: factual + top view, then independently measured center probe.
    initdata=data+[('view',obs['view'],1.)]
    sharp,stat=smc(family,initdata,model='sharp',n=n,seed=seed+2);costs.append(dict(name='sharp_initial',**stat))
    sharp_pred=predict(sharp,family);record('sharp_initial',sharp,sharp_pred,action='factual+view')
    mu=observation_means(sharp[0],sharp[1],family,'center');w=sharp[2];mean=w@mu
    delta=mu-mean;cov=delta.T@(delta*w[:,None])+torch.diag(tensor(NOISE['center']**2))
    residual=tensor(obs['center'])-mean
    meta['check_score']=float(residual@torch.linalg.solve(cov,residual))
    # Pre-probe discrepancy score uses independent null calibration worlds.
    updated=initdata+[('center',obs['center'],1.)]
    for label,model,scale in [('sharp_refit','sharp',1.),('broad_refit','broad',1.),('noise_inflation','sharp',2.)]:
        dd=[(a,y,scale) for a,y,_ in updated]
        aa,stat=smc(family,dd,model=model,n=n,seed=seed+3);costs.append(dict(name=label,**stat))
        record(label,aa,predict(aa,family),action='factual+view+center')
    meta['records']=records;meta['costs']=costs
    (out/f'case_{case:03d}.json').write_text(json.dumps(plain(meta),allow_nan=False))
    np.savez_compressed(out/f'posterior_{case:03d}.npz',theta=atoms[0].cpu().numpy(),rounding=atoms[1].cpu().numpy(),weight=atoms[2].cpu().numpy())
    print(json.dumps(dict(case=case,family=family,regime=regime,seconds=sum(x['seconds'] for x in costs),calls=sum(x['calls'] for x in costs),check=meta['check_score'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--n',type=int,default=768);p.add_argument('--limit',type=int,default=72);p.add_argument('--out',default='results/main');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    runtime=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),args=vars(args),start=time.time())
    (out/f'runtime_{args.worker}.json').write_text(json.dumps(runtime,indent=2))
    for case in range(args.worker,args.limit,args.workers):
        if (out/f'case_{case:03d}.json').exists():continue
        run_case(case,args.n,out)
    runtime.update(end=time.time(),peak_memory=torch.cuda.max_memory_allocated());(out/f'runtime_{args.worker}.json').write_text(json.dumps(runtime,indent=2))
