"""New-world experiment. Acquisition paths never open private truth/measurements."""
import hashlib,json,os,socket,time
from pathlib import Path
import numpy as np
import torch
from engine import tensor,decode,batch,SHIFTS,BOUNDS,FAMILIES,SIM_COUNTS
from bridge import sample,physics_likelihood,summaries
from belief import Belief

from config import N_WORLDS,REPEATS,CONDITIONS,AUDIT,RASTER,COSTS
ROOT=Path('results')
class Model(Belief):
    def __init__(self,family,precision='native'):
        super().__init__(family,'pixel_only','boundary');self.scale=.25 if precision=='precise' else 1.
    def mean_cov(self,theta,b,action):
        mu,cov=super().mean_cov(theta,b,action)
        return mu,cov*self.scale**2 if action=='view' else cov

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(path)
def seed_for(*parts):return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8],16)
def meta():return dict(job=os.environ.get('SLURM_JOB_ID'),hostname=socket.gethostname(),pid=os.getpid(),time=time.time())
def public(case):return json.loads((ROOT/'worlds'/f'case_{case:03d}'/'public.json').read_text())

def factual_bank(family,yf,n,seed):
    model=Model(family)
    def like(x,ids):
        flat=x.reshape(-1,5);mu,cov=model.mean_cov(decode(flat[:,:4]),flat[:,4]*.006,'factual')
        ll=-.5*((mu[:,0]-float(yf[0]))/.025)**2
        return ll.reshape(x.shape[:2]),torch.zeros(x.shape[:2],device=x.device,dtype=x.dtype)
    x,z,s=sample(like,1,n,'full',seed,moves=18)
    return x[0],s

def make_world(case):
    folder=ROOT/'worlds'/f'case_{case:03d}';folder.mkdir(parents=True,exist_ok=True)
    family=FAMILIES[case//16];rng=np.random.default_rng(202611110700+case)
    true_x=tensor(rng.random((1,5)));theta=decode(true_x[:,:4]);b=true_x[:,4]*.006
    model=Model(family);mf,_=model.mean_cov(theta,b,'factual');yf=(mf.cpu().numpy()[0]+rng.normal(0,.025,1)).tolist()
    # Separate files make the public/private data boundary inspectable.
    truth=batch(theta,b,family,SHIFTS);assert not bool(truth['overflow'].any())
    private=folder/'private';private.mkdir(exist_ok=True)
    save_json(private/'truth.json',dict(theta=theta[0].cpu().tolist(),rounding=float(b[0]),velocity=truth['velocity'][:,0].cpu().tolist(),contact=(truth['code'][:,0]>0).cpu().tolist()))
    noise=tensor(np.random.default_rng(750000+case).normal(size=(REPEATS,3)));actual={}
    for precision,action in CONDITIONS:
        m,c=Model(family,precision).mean_cov(theta,b,action)
        actual[f'{precision}_{action}']=(m+noise[:,:m.shape[1]]@torch.linalg.cholesky(c).T).cpu().numpy()
    np.savez_compressed(private/'gaussian.npz',**actual)
    if case in RASTER:
        from geometry import render,extract
        from dynamic_sensor import observe
        ys={f'native_{a}':[] for a in ['view','left','right','center']}
        tt=theta[0].cpu().numpy();bb=float(b[0])
        for rep in range(8):
            rr=np.random.default_rng(760000+100*case+rep)
            im=render(tt[0],tt[3],bb,family,'pixel_only',rr);ys['native_view'].append(extract(im,family,'pixel_only'))
            if case in [0,16,32] and rep==0:im.save(private/'raster_view_example.png')
            for a in ['left','right','center']:ys['native_'+a].append(observe(tt,bb,family,a,rr))
        np.savez_compressed(private/'raster.npz',**{k:np.array(v) for k,v in ys.items()})
    # Independent posterior approximations for proposal and outer generator.
    x,s=factual_bank(family,yf,16384,710000+case)
    order=torch.as_tensor(np.random.default_rng(730000+case).permutation(len(x)),device='cuda');x=x[order]
    pred=batch(decode(x[:,:4]),x[:,4]*.006,family,SHIFTS);assert not bool(pred['overflow'].any())
    bank={'x':x.cpu().numpy(),'z':pred['velocity'].permute(1,0,2).cpu().numpy(),'contact':(pred['code']>0).T.cpu().numpy()}
    for action in ['view','left','right','center']:
        mu,_=model.mean_cov(decode(x[:,:4]),x[:,4]*.006,action);bank['mu_'+action]=mu.cpu().numpy()
    np.savez_compressed(folder/'bank.npz',**bank);save_json(folder/'bank_stat.json',s)
    gen,gs=factual_bank(family,yf,8192,720000+case);current=summaries(gen[None,:,:],family)['mean'][0]
    np.savez_compressed(folder/'generator.npz',x=gen.cpu().numpy(),current=current);save_json(folder/'generator_stat.json',gs)
    rr=np.random.default_rng(740000+case);draws=rr.integers(len(gen),size=512);noise=tensor(rr.normal(size=(512,3)));selected=gen[torch.as_tensor(draws,device='cuda')];observations={}
    for precision,action in CONDITIONS:
        mu,cov=Model(family,precision).mean_cov(decode(selected[:,:4]),selected[:,4]*.006,action)
        observations[f'{precision}_{action}']=(mu+noise[:,:mu.shape[1]]@torch.linalg.cholesky(cov).T).cpu().numpy()
    np.savez_compressed(folder/'outer.npz',**observations)
    save_json(folder/'public.json',dict(case=case,family=family,factual=yf,world_seed=202611110700+case,current=current.tolist()))
    save_json(folder/'complete.json',dict(case=case,family=family,**meta()));print(json.dumps(dict(phase='world',case=case,**meta())),flush=True)

def snis(case,folder,observations,conditions):
    """Only public bank, query predictions and supplied observations are read."""
    d=public(case);bank=np.load(ROOT/'worlds'/f'case_{case:03d}'/'bank.npz');z=tensor(bank['z']);contact=tensor(bank['contact']);current=tensor(d['current'])
    for precision,action in conditions:
        stem=f'{precision}_{action}';y=tensor(observations[stem]);mu=tensor(bank['mu_'+action]);_,cov=Model(d['family'],precision).mean_cov(decode(tensor(bank['x'][:1,:4])),tensor(bank['x'][:1,4])*.006,action)
        for n in [4096,16384]:
            out=folder/f'{stem}_snis{n}.npz'
            if out.exists():continue
            t0=time.perf_counter();delta=y[:,None,:]-mu[None,:n,:]
            weights=torch.softmax(-.5*torch.einsum('hnd,dk,hnk->hn',delta,torch.linalg.inv(cov),delta),1)
            mean=torch.einsum('hn,nqd->hqd',weights,z[:n]);cp=weights@contact[:n]
            torch.cuda.synchronize()
            np.savez_compressed(out,mean=mean.cpu().numpy(),gain=((mean-current)**2).sum(-1).cpu().numpy()/2,
                ess=(1/(weights*weights).sum(1)).cpu().numpy(),contact=cp.cpu().numpy(),seconds=time.perf_counter()-t0)

def run_group(cases,phase,h,n,moves,reps=2,method='bridge'):
    out=ROOT/phase;out.mkdir(exist_ok=True);ds=[]
    for case in cases:
        d=public(case);folder=out/f'case_{case:03d}';folder.mkdir(exist_ok=True)
        world=ROOT/'worlds'/f'case_{case:03d}'
        is_acq=phase.startswith('acq') or phase=='full_audit'
        # No private files are opened on the acquisition path.
        source=world/'outer.npz' if is_acq else world/'private'/('raster.npz' if phase=='raster_eval' else 'gaussian.npz')
        with np.load(source) as ys:observations={k:ys[k][:h].copy() for k in ys.files}
        conditions=[c for c in CONDITIONS if f'{c[0]}_{c[1]}' in observations]
        initial=None
        if method=='bridge':
            with np.load(world/'bank.npz') as bank:initial=tensor(bank['x'][:4096])
        ds.append(dict(**d,folder=folder,ys=observations,initial=initial))
        if phase!='full_audit' and phase!='decoder_audit':snis(case,folder,observations,conditions)
    assert len({d['family'] for d in ds})==1
    for precision,action in conditions:
        cond=f'{precision}_{action}';model=Model(ds[0]['family'],precision)
        for rep in range(reps):
            pending=[d for d in ds if not (d['folder']/f'{cond}_{method}_{rep}.json').exists()]
            if not pending:continue
            group=[d['case'] for d in pending];seed=seed_for(phase,group,cond,method,rep,h,n,moves)
            yy=tensor(np.concatenate([d['ys'][cond] for d in pending]));yf=tensor([d['factual'] for d in pending]).repeat_interleave(h,0)
            initial=torch.cat([d['initial'][None,:,:].expand(h,-1,-1) for d in pending],0) if method=='bridge' else None
            x,z,stat=sample(physics_likelihood(model,yf,action,yy),len(yy),n,method,seed,initial=initial,moves=moves)
            final_beta=np.zeros(len(yy))
            for entry in stat['history']:final_beta[entry['populations']]=entry['beta']
            for i,d in enumerate(pending):
                sl=slice(i*h,(i+1)*h);summ=summaries(x[sl],d['family']);stem=d['folder']/f'{cond}_{method}_{rep}'
                np.savez_compressed(stem.with_suffix('.npz'),**summ,logz=z[sl].cpu().numpy(),current=np.array(d['current']))
                ss={k:v for k,v in stat.items() if k!='history'}
                for key in ['stages','acceptance','one_step_ess','min_stage_ess','max_stage_weight','surviving_initial_ancestors','likelihood_candidates']:ss[key]=stat[key][sl]
                ss.update(case=d['case'],h=h,precision=precision,action=action,rep=rep,batch_cases=group,batch_h=len(yy),batch_seconds=stat['seconds'],
                    final_beta=final_beta[sl].tolist(),seconds=stat['seconds']/len(pending),batch_trajectory_rollouts=stat['trajectory_rollouts'],trajectory_rollouts=stat['trajectory_rollouts']/len(pending),**meta())
                save_json(stem.with_suffix('.json'),ss)
            print(json.dumps(dict(phase=phase,cases=group,condition=cond,method=method,rep=rep,seconds=stat['seconds'])),flush=True)
    for d in ds:save_json(d['folder']/'complete.json',dict(case=d['case'],h=h,n=n,moves=moves,reps=reps,method=method,**meta()))
