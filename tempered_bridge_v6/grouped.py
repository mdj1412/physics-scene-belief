"""Execution-only batching across worlds of one family; same targets/budgets."""
import hashlib,json,os,socket,time
from pathlib import Path
import numpy as np
import torch
from bridge import normalize,sample,physics_likelihood,summaries
from belief import Belief
from engine import tensor
from run import ROOT,CONDITIONS,save_json

def prepare(case,h):
    d=json.loads((ROOT/'main'/f'case_{case:03d}.json').read_text())
    family=d['family'];yf=d['observations']['factual'];split=d['split'];del d
    bank=np.load(ROOT/'main'/f'factual_{case:03d}.npz');theta=tensor(bank['theta']);b=tensor(bank['rounding'])
    x0=normalize(theta,b);current=summaries(x0[None,:,:],family)['mean'][0]
    rng=np.random.default_rng(85000+case);draws=rng.integers(len(theta),size=1024);noise=rng.normal(size=(1024,3))
    return dict(case=case,family=family,yf=yf,split=split,theta=theta,b=b,x0=x0,current=current,
        ids=torch.as_tensor(draws[:h],device='cuda'),noise=tensor(noise[:h]))

def run_group(cases,out,h=128,n=1024,moves=12):
    start=time.time();out=Path(out);ds=[prepare(c,h) for c in cases]
    assert len({d['family'] for d in ds})==1
    for d in ds:
        d['folder']=out/f"case_{d['case']:03d}";d['folder'].mkdir(exist_ok=True)
    for quality,action in CONDITIONS:
        model=Belief(ds[0]['family'],quality=quality)
        for d in ds:
            mu,cov=model.mean_cov(d['theta'],d['b'],action)
            d['yy']=mu[d['ids']]+d['noise'][:,:mu.shape[1]]@torch.linalg.cholesky(cov).T
            obs=d['folder']/f'{quality}_{action}_observation.npy'
            if obs.exists():assert np.array_equal(np.load(obs),d['yy'].cpu().numpy())
            else:np.save(obs,d['yy'].cpu().numpy())
        for method in ['bridge','full']:
            for rep in [0,1]:
                pending=[d for d in ds if not (d['folder']/f'{quality}_{action}_{method}_{rep}.json').exists()]
                if not pending:continue
                group=[d['case'] for d in pending]
                identity=json.dumps([str(out),group,quality,action,method,rep,h,n,moves])
                batch_id=hashlib.sha256(identity.encode()).hexdigest()[:16];seed=int(batch_id[:8],16)
                yy=torch.cat([d['yy'] for d in pending]);yf=tensor([d['yf'][:1] for d in pending]).repeat_interleave(h,0)
                initial=torch.cat([d['x0'][None,:,:].expand(h,-1,-1) for d in pending],0) if method=='bridge' else None
                x,z,s=sample(physics_likelihood(model,yf,action,yy),len(yy),n,method,seed,initial=initial,moves=moves)
                # Bound memory in query forward prediction, independently of SMC grouping.
                for i,d in enumerate(pending):
                    sl=slice(i*h,(i+1)*h);summary=summaries(x[sl],d['family'])
                    stem=d['folder']/f'{quality}_{action}_{method}_{rep}'
                    np.savez_compressed(stem.with_suffix('.npz'),**summary,logz=z[sl].cpu().numpy(),current=d['current'])
                    stat={k:v for k,v in s.items() if k!='history'}
                    for key in ['stages','acceptance','one_step_ess','min_stage_ess','max_stage_weight','surviving_initial_ancestors','likelihood_candidates']:stat[key]=s[key][sl]
                    stat.update(h=h,seconds=s['seconds']/len(pending),trajectory_rollouts=s['trajectory_rollouts']/len(pending),
                        history=[dict(populations=[p-i*h for p in hist['populations'] if i*h<=p<(i+1)*h],
                            beta=[v for p,v in zip(hist['populations'],hist['beta']) if i*h<=p<(i+1)*h],
                            ess=[v for p,v in zip(hist['populations'],hist['ess']) if i*h<=p<(i+1)*h]) for hist in s['history']],
                        batch_id=batch_id,batch_cases=group,batch_h=len(yy),batch_seconds=s['seconds'],batch_trajectory_rollouts=s['trajectory_rollouts'],
                        cost_attribution='equal share of batch totals; per-world timing/rollouts not separately measured',
                        case=d['case'],family=d['family'],split=d['split'],quality=quality,action=action,rep=rep,
                        hostname=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),completed=time.time(),execution='cross_world_batch_v1')
                    save_json(stem.with_suffix('.json'),stat)
                print(json.dumps(dict(cases=group,condition=f'{quality}_{action}',method=method,rep=rep,seconds=s['seconds'],batch_h=len(yy))),flush=True)
    for d in ds:
        save_json(d['folder']/'complete.json',dict(case=d['case'],family=d['family'],split=d['split'],h=h,n=n,moves=moves,reps=2,
            methods=['bridge','full'],elapsed=time.time()-start,hostname=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),execution='cross_world_batch_v1',batch_cases=cases))
