import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from belief import Forward,BeliefModel

METHODS=[
    ('joint_evsi','dense','evsi'),
    ('plugin_refit_evsi','dense','plugin'),
    ('plugin_explore25','dense','explore'),
    ('joint_variance_rule','dense','variance'),
    ('joint_parameter_eig','dense','eig'),
    ('grid33_evsi','grid33','evsi'),
    ('quadrature256_evsi','quadrature16','evsi'),
    ('particles512_evsi','particles512','evsi'),
    ('plugin_static_vp','dense','plugin_vp'),
    ('static_vv','dense','vv'),('static_vp','dense','vp'),
    ('static_pv','dense','pv'),('static_pp','dense','pp'),
    ('static_vs','dense','vs'),('static_ps','dense','ps'),('static_ss','dense','ss'),
    ('joint_random','dense','random'),
]


def policy_action(model,p,qindex,policy,step,random):
    if policy in ['evsi','plugin','explore','eig']:
        belief=model.collapse(p) if policy in ['plugin','explore'] else p
        gains=model.gains(belief,qindex,'eig' if policy=='eig' else 'evsi')
        threshold=1e-9 if policy=='eig' else .005
        action=torch.where(gains.max(-1).values>threshold+1e-10,gains.argmax(-1),-1)
        if policy=='explore' and step==0: action=torch.where(random<.25,0,action)
        return action
    if policy=='variance':return model.variance_rule(p,qindex)
    if policy=='random':return (random*3).long().clamp_max(2)
    code='vp' if policy=='plugin_vp' else policy
    return torch.full((len(p),),{'v':0,'p':1,'s':-1}[code[step]],device=model.device,dtype=torch.int64)


def run(shard,out,limit=128):
    assert torch.cuda.is_available(),'CUDA required; do not silently move this extension onto login CPU'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_default_dtype(torch.float64)
    out.mkdir(parents=True,exist_ok=True)
    forward=Forward()
    rows=[];details=[];references=[];timings={name:0. for name,_,_ in METHODS}
    started=time.perf_counter()
    for f in range(shard,limit,4):
        cfg=json.loads(Path(f'data/family_{f:03d}.json').read_text())
        data=np.load(f'data/family_{f:03d}.npz')
        Q=len(cfg['delta']);W=len(data['geometry']);R=2;N=W*R;B=Q*N
        rng=np.random.default_rng([20261104,f,333])
        base_noise=rng.standard_normal((N,3,2))
        random=np.tile(rng.random((N,2)),(Q,1))
        noise=np.tile(base_noise,(Q,1,1))
        mu=np.tile(np.repeat(data['observation_mu'],R,axis=0),(Q,1))
        truth=np.repeat(data['truth'].transpose(1,0,2),R,axis=1).reshape(B,4)
        qindex=torch.arange(Q,device='cuda').repeat_interleave(N)
        t=lambda x:torch.as_tensor(x,dtype=torch.float64,device='cuda')
        true_v=t(truth[:,:2]);true_c=t(truth[:,2]);noise=t(noise);mu=t(mu);random=t(random)
        models={kind:BeliefModel(cfg,forward,kind) for kind in ['dense','grid33','quadrature16','particles512']}
        dense=models['dense']
        refgain=dense.gains(dense.prior[None,:].expand(Q,-1),torch.arange(Q,device='cuda'))
        refaction=torch.where(refgain.max(-1).values>.005+1e-10,refgain.argmax(-1),-1)
        refscore=torch.cat([refgain-.005,torch.zeros((Q,1),device='cuda')],-1)
        bestscore=refscore.max(-1).values
        for q in range(Q):
            references.append(dict(family=f,split=cfg['split'],q=q,delta=cfg['delta'][q],
                residual_mean=cfg['residual_mean'][q],geometry_sd=cfg['geometry_sd'],
                view_gain=float(refgain[q,0].cpu()),probe_gain=float(refgain[q,1].cpu()),
                repeat_gain=float(refgain[q,2].cpu()),reference_action=int(refaction[q].cpu())))
        for mi,(name,kind,policy) in enumerate(METHODS):
            model=models[kind]
            torch.cuda.synchronize();t0=time.perf_counter()
            posterior=model.prior[None,:].expand(B,-1).clone()
            cost=torch.zeros(B,device='cuda');active=torch.ones(B,device='cuda',dtype=torch.bool)
            visits=torch.zeros((B,3),device='cuda',dtype=torch.int64)
            history=torch.full((B,2),-2,device='cuda',dtype=torch.int64)
            for budget in range(3):
                prediction=model.collapse(posterior) if policy in ['plugin','explore','plugin_vp'] else posterior
                metrics=model.metrics(prediction,qindex,true_v,true_c)
                metrics['cost']=cost
                metrics['penalized_mse']=metrics['mse']+.005*cost
                for a,key in [(0,'view'),(1,'probe'),(2,'repeat'),(-1,'stop')]:
                    metrics['first_'+key]=(history[:,0]==a).double()
                    metrics['second_'+key]=(history[:,1]==a).double()
                if budget:
                    chosen=history[:,0].clone();chosen[chosen<0]=3
                    metrics['design_regret']=bestscore[qindex]-refscore[qindex,chosen]
                    metrics['action_agreement']=(history[:,0]==refaction[qindex]).double()
                else:
                    metrics['design_regret']=torch.zeros_like(cost)
                    metrics['action_agreement']=torch.zeros_like(cost)
                means={k:v.reshape(Q,N).mean(-1).cpu().numpy() for k,v in metrics.items()}
                for q in range(Q):
                    row=dict(family=f,split=cfg['split'],q=q,method=name,budget=budget,
                        delta=cfg['delta'][q],residual_mean=cfg['residual_mean'][q],
                        geometry_sd=cfg['geometry_sd'],sigma_view_px=cfg['sigma_view_px'],
                        sigma_probe_px=cfg['sigma_probe_px'],atoms=len(model.prior),worlds=W,noise_repeats=R)
                    row.update({k:float(v[q]) for k,v in means.items()});rows.append(row)
                if f>=64 and budget>0 and name in ['joint_evsi','plugin_refit_evsi','grid33_evsi']:
                    details.append(np.c_[np.full(B,f),np.repeat(np.arange(Q),N),
                        np.tile(np.repeat(np.arange(W),R),Q),np.tile(np.arange(R),Q*W),
                        np.full(B,budget),np.full(B,mi),truth[:,0],truth[:,2],
                        metrics['prediction'].cpu().numpy(),metrics['contact_probability'].cpu().numpy(),
                        history.cpu().numpy(),cost.cpu().numpy()])
                if budget==2:break
                action=policy_action(model,posterior,qindex,policy,budget,random[:,budget])
                action[~active]=-1
                history[:,budget]=action;active &= action>=0
                observation=torch.zeros(B,device='cuda')
                for a in range(3):
                    ii=torch.where(action==a)[0]
                    if len(ii):
                        observation[ii]=mu[ii,a]+model.sd[a]*noise[ii,a,visits[ii,a]]
                        visits[ii,a]+=1;cost[ii]+=1
                posterior=model.update(posterior,action,observation)
            torch.cuda.synchronize();timings[name]+=time.perf_counter()-t0
        del models
        print(f'GPU shard={shard} family={f} complete',flush=True)
    for name,records in [('family_metrics',rows),('design_reference',references)]:
        with (out/f'{name}_{shard}.csv').open('w') as handle:
            w=csv.DictWriter(handle,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    if details:
        np.savez_compressed(out/f'test_details_{shard}.npz',values=np.concatenate(details),
            columns=['family','q','world','repeat','budget','method_index','truth_vx','truth_contact',
                     'prediction','contact_probability','first_action','second_action','cost'])
    props=torch.cuda.get_device_properties(0)
    manifest=dict(host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=str(props),
        torch=torch.__version__,cuda=torch.version.cuda,elapsed_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        timing_seconds=timings,methods=METHODS,integration_points=257,shard=shard,
        code_hashes={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['prepare.py','belief.py','run.py']})
    (out/f'runtime_{shard}.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--out',type=Path,default=Path('results/main'))
    parser.add_argument('--limit',type=int,default=128)
    args=parser.parse_args()
    with torch.inference_mode():run(args.shard,args.out,args.limit)
