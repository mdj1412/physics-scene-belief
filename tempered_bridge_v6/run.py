"""Resumable batched nested acquisition; no truth or outcomes in selection."""
import argparse,json,os,socket,time,hashlib
from pathlib import Path
import numpy as np
import torch
from belief import Belief
from bridge import sample,normalize,physics_likelihood,summaries
from engine import tensor,decode

ROOT=Path('../boundary_belief_v5/results')
CONDITIONS=[('high','view'),('pixel_only','view'),('high','left'),('high','right'),('high','center')]

def save_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)

def run_case(case,out,h=64,n=1024,moves=12,methods=('bridge','full'),reps=2,seed_offset=0):
    started=time.time();folder=out/f'case_{case:03d}';folder.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'main'/f'case_{case:03d}.json').read_text())
    # Deliberately only copy fields available before observation selection.
    family=data['family'];yf=data['observations']['factual'];split=data['split'];del data
    bank=np.load(ROOT/'main'/f'factual_{case:03d}.npz')
    theta=tensor(bank['theta']);b=tensor(bank['rounding']);x0=normalize(theta,b)
    current=summaries(x0[None,:,:],family)['mean'][0]
    # NumPy RNG gives prefix-identical outer samples irrespective of h.
    rng=np.random.default_rng(85000+case+seed_offset)
    draws=rng.integers(len(theta),size=1024);noise=rng.normal(size=(1024,3))
    ids=torch.as_tensor(draws[:h],device='cuda');noise=tensor(noise[:h])
    for cond,(quality,action) in enumerate(CONDITIONS):
        model=Belief(family,quality=quality);mu,cov=model.mean_cov(theta,b,action)
        yy=mu[ids]+noise[:,:mu.shape[1]]@torch.linalg.cholesky(cov).T
        obsfile=folder/f'{quality}_{action}_observation.npy'
        if obsfile.exists():assert np.array_equal(np.load(obsfile),yy.cpu().numpy())
        else:np.save(obsfile,yy.cpu().numpy())
        for method in methods:
            for rep in range(reps):
                stem=folder/f'{quality}_{action}_{method}_{rep}'
                if stem.with_suffix('.json').exists():continue
                seed=1000000+case*10000+cond*1000+(0 if method=='bridge' else 500)+rep+seed_offset
                x,z,stat=sample(physics_likelihood(model,yf,action,yy),h,n,method,seed,initial=x0,moves=moves)
                summary=summaries(x,family)
                np.savez_compressed(stem.with_suffix('.npz'),**summary,logz=z.cpu().numpy(),current=current)
                stat.update(case=case,family=family,split=split,quality=quality,action=action,rep=rep,
                    hostname=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),completed=time.time())
                save_json(stem.with_suffix('.json'),stat)
                print(json.dumps(dict(case=case,condition=f'{quality}_{action}',method=method,rep=rep,seconds=stat['seconds'],stages_mean=float(np.mean(stat['stages'])))),flush=True)
    save_json(folder/'complete.json',dict(case=case,family=family,split=split,h=h,n=n,moves=moves,reps=reps,
        methods=list(methods),elapsed=time.time()-started,hostname=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/main');p.add_argument('--h',type=int,default=64)
    p.add_argument('--n',type=int,default=1024);p.add_argument('--moves',type=int,default=12)
    p.add_argument('--cases',default=','.join(map(str,range(72))));p.add_argument('--methods',default='bridge,full')
    p.add_argument('--reps',type=int,default=2);p.add_argument('--seed-offset',type=int,default=0)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for case in map(int,a.cases.split(',')):
        if (out/f'case_{case:03d}'/'complete.json').exists():continue
        lock=out/f'claim_{case:03d}'
        try:lock.mkdir()
        except FileExistsError:continue
        save_json(lock/'worker.json',dict(pid=os.getpid(),host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID')))
        run_case(case,out,a.h,a.n,a.moves,tuple(a.methods.split(',')),a.reps,a.seed_offset)
    print('WORKER_COMPLETE',flush=True)

if __name__=='__main__':main()
