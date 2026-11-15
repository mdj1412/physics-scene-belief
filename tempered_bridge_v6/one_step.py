"""Matched-observation finite-bank SNIS control; no tempering or rejuvenation."""
import json,time,os,socket,platform
from pathlib import Path
import numpy as np
import torch
from belief import Belief
from engine import tensor,batch,SHIFTS,SIM_COUNTS
from run import ROOT,CONDITIONS,save_json

out=Path('results/one_step');out.mkdir(exist_ok=True)
save_json(Path('results/runtime.json'),dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
    device=torch.cuda.get_device_name(0),host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID')))
for case in range(72):
    marker=out/f'case_{case:03d}.json'
    if marker.exists():continue
    start=time.perf_counter();before=SIM_COUNTS['trajectories']
    d=json.loads((ROOT/'main'/f'case_{case:03d}.json').read_text());family=d['family'];del d
    bank=np.load(ROOT/'main'/f'factual_{case:03d}.npz');theta=tensor(bank['theta']);b=tensor(bank['rounding'])
    pred=batch(theta,b,family,SHIFTS);assert not bool(pred['overflow'].any())
    z=pred['velocity'];contact=(pred['code']>0).double();current=z.mean(1)
    rng=np.random.default_rng(85000+case);draws=rng.integers(len(theta),size=1024);noise=tensor(rng.normal(size=(1024,3))[:512]);ids=torch.as_tensor(draws[:512],device='cuda')
    means=[];esses=[];gains=[];contacts=[]
    for quality,action in CONDITIONS:
        model=Belief(family,quality=quality);mu,cov=model.mean_cov(theta,b,action)
        yy=mu[ids]+noise[:,:mu.shape[1]]@torch.linalg.cholesky(cov).T
        delta=yy[:,None,:]-mu[None,:,:];ll=-.5*torch.einsum('hnd,dk,hnk->hn',delta,torch.linalg.inv(cov),delta)
        w=torch.softmax(ll,1);m=torch.einsum('hn,qnd->hqd',w,z)
        means.append(m.cpu().numpy());gains.append(((m-current)**2).sum(2).cpu().numpy());esses.append((1/(w*w).sum(1)).cpu().numpy());contacts.append(torch.einsum('hn,qn->hq',w,contact).cpu().numpy())
    np.savez_compressed(out/f'case_{case:03d}.npz',mean=np.stack(means),gain=np.stack(gains),ess=np.stack(esses),contact=np.stack(contacts),current=current.cpu().numpy())
    torch.cuda.synchronize();stat=dict(case=case,family=family,n=4096,outer=512,seconds=time.perf_counter()-start,trajectory_rollouts=SIM_COUNTS['trajectories']-before,
        hostname=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),conditions=CONDITIONS,source='same factual bank as hypothetical generator; self-inclusion possible')
    save_json(marker,stat);print(json.dumps(stat),flush=True)

failure=[]
for case in [66,68]:
    old=json.loads((ROOT/'conditional_failure_audit'/f'case_{case:03d}.json').read_text());bank=np.load(ROOT/'main'/f'factual_{case:03d}.npz')
    theta=tensor(bank['theta']);b=tensor(bank['rounding']);z=batch(theta,b,'deflection',SHIFTS)['velocity'];model=Belief('deflection')
    mu,cov=model.mean_cov(theta,b,'view');yy=tensor([r['observation'] for r in old['rows']]);delta=yy[:,None,:]-mu[None,:,:]
    w=torch.softmax(-.5*torch.einsum('hnd,dk,hnk->hn',delta,torch.linalg.inv(cov),delta),1)
    means=torch.einsum('hn,qnd->hqd',w,z).cpu().numpy();ess=(1/(w*w).sum(1)).cpu().numpy()
    for i,row in enumerate(old['rows']):
        ref=np.mean([r['mean'] for r in row['references']],axis=0)
        failure.append(dict(case=case,hypothesis=row['hypothesis'],ess=float(ess[i]),rmse=float(np.sqrt(np.mean((means[i]-ref)**2))),mean=means[i].tolist()))
save_json(out/'failure.json',failure)
print('ONE_STEP_COMPLETE',flush=True)
