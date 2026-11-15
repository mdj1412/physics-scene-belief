import argparse,csv,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
import torch
from core import *
from design import Design

def case_spec(f):
    rng=np.random.default_rng([SEED,300,f])
    return dict(family=f,split='dev' if f<16 else 'validation' if f<32 else 'test',
        g=float(rng.uniform(GLO,GHI)),e=float(rng.uniform(ELO,EHI)),sign=int(rng.choice([-1,1])),
        factual_noise=float(rng.normal()),observation_noise=rng.normal(size=2).tolist(),
        # Queries chosen without hidden parameter; all are translate_wall_y.
        deltas=(np.array([-.08,-.05,-.025,-.008,.004,.012,.025,.05])+rng.uniform(-.002,.002,8)).tolist(),
        sd=[float(rng.uniform(.001,.004)),float(rng.uniform(.04,.15))])

def run(args):
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];diagnostics=[];checks=[];timings=[];started=time.perf_counter()
    for f in range(args.shard,args.limit,4):
        spec=case_spec(f)
        for signed in [False,True]:
            gt=spec['g']*(spec['sign'] if signed else 1);et=spec['e']
            truth,_,contact=impact(gt+np.array(spec['deltas']),et)
            observed=np.array([gt,et])+np.array(spec['sd'])*spec['observation_noise']
            for ni,sigma in enumerate(NOISES):
                y=float(impact(gt,et)[0]+sigma*spec['factual_noise'])
                case=dict(family=f,split=spec['split'],signed=int(signed),sigma=sigma,y=y)
                t0=time.perf_counter()
                refatoms,rcost=conditional_quad(y,sigma,signed)
                reference=Design(refatoms,spec['deltas'],spec['sd'])
                refgain,refaction=reference.gains();refeval=reference.evaluate(observed,truth,contact)
                scores=np.c_[refgain-.0005,np.zeros(8)]
                # Independent, higher-order reference + integration audit on dev.
                if spec['split']=='dev':
                    finer,_=conditional_quad(y,sigma,signed,512,32)
                    fine=Design(finer,spec['deltas'],spec['sd'],513)
                    fg,fa=fine.gains()
                    checks.append(dict(**case,gain_max_error=float(abs(fg-refgain).max()),
                        action_agreement=float(np.mean(fa==refaction)),
                        mean_max_error=float(abs((fine.z@fine.p).cpu().numpy()-(reference.z@reference.p).cpu().numpy()).max())))
                    del fine
                makers=[('reference',-1,lambda:(refatoms,rcost)),
                    ('grid17x17',-1,lambda:fixed_grid(y,sigma,signed,17,17)),
                    ('grid33x17',-1,lambda:fixed_grid(y,sigma,signed,33,17)),
                    ('grid65x33',-1,lambda:fixed_grid(y,sigma,signed,65,33)),
                    ('conditional_quad512',-1,lambda:conditional_quad(y,sigma,signed,16 if signed else 32,16)),
                    ('factorized_marginals',-1,lambda:factorized(refatoms))]
                for rep in range(3):
                    for name,func in [('prior512',lambda r:prior_particles(y,sigma,signed,r,512)),
                        ('prior2048',lambda r:prior_particles(y,sigma,signed,r,2048)),
                        ('posterior512',lambda r:posterior_particles(y,sigma,signed,r,512)),
                        ('smc512',lambda r:smc(y,sigma,signed,r,512))]:
                        rng=np.random.default_rng([SEED,f,ni,int(signed),rep,471])
                        makers.append((name,rep,lambda func=func,rng=rng:func(rng)))
                for name,rep,make in makers:
                    torch.cuda.synchronize();t1=time.perf_counter()
                    atoms,cost=make()
                    model=reference if name=='reference' else Design(atoms,spec['deltas'],spec['sd'])
                    gains,actions=model.gains() if name!='reference' else (refgain,refaction)
                    evaluations=model.evaluate(observed,truth,contact) if name!='reference' else refeval
                    torch.cuda.synchronize()
                    timings.append(dict(**case,method=name,seed=rep,seconds=time.perf_counter()-t1,atoms=len(atoms[0]),**cost))
                    diagnostics.append(dict(**case,method=name,seed=rep,**diagnostic(atoms,y,sigma)))
                    for q in range(8):
                        a=int(actions[q])
                        row=dict(**case,method=name,seed=rep,q=q,delta=spec['deltas'][q],action=a,
                            reference_action=int(refaction[q]),agreement=float(a==refaction[q]),
                            regret=float(scores[q].max()-scores[q,a]),gain_error=float(abs(gains[q]-refgain[q]).max()),
                            prediction_error_vs_ref=float(abs(evaluations[a]['prediction'][q]-refeval[a]['prediction'][q])),
                            cost=float(a<2),atoms=len(atoms[0]))
                        for key,value in evaluations[a].items():row[key]=float(value[q])
                        row['penalized_mse']=row['mse']+.0005*row['cost']
                        for a2,tag in [(0,'view'),(1,'probe'),(2,'factual')]:
                            for key in ['mse','crps','brier','coverage90','prediction']:
                                row[tag+'_'+key]=float(evaluations[a2][key][q])
                        rows.append(row)
                    if f in [0,32,64] and ni in [0,3] and name in ['reference','grid33x17','prior512','smc512','conditional_quad512','factorized_marginals'] and rep in [-1,0]:
                        np.savez_compressed(out/f'posterior_f{f}_s{int(signed)}_n{ni}_{name}.npz',g=atoms[0],e=atoms[1],w=atoms[2],y=y,sigma=sigma,true_g=gt,true_e=et)
                    if name!='reference':del model
                del reference
                print(f'shard={args.shard} family={f} signed={int(signed)} sigma={sigma} seconds={time.perf_counter()-t0:.2f}',flush=True)
        # Checkpoint complete family; results remain reviewable if allocation ends.
        for name,records in [('metrics',rows),('diagnostics',diagnostics),('timings',timings),('reference_checks',checks)]:
            if records:
                keys=list(dict.fromkeys(k for row in records for k in row))
                with (out/f'{name}_{args.shard}.csv').open('w') as h:
                    wr=csv.DictWriter(h,fieldnames=keys);wr.writeheader();wr.writerows(records)
    props=torch.cuda.get_device_properties(0)
    manifest=dict(host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),gpu=str(props),torch=torch.__version__,cuda=torch.version.cuda,
        seconds=time.perf_counter()-started,peak_memory=torch.cuda.max_memory_allocated(),shard=args.shard,limit=args.limit,
        seed=SEED,noise_levels=NOISES,methods=[m[0] for m in makers],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')})
    (out/f'runtime_{args.shard}.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--limit',type=int,default=96);p.add_argument('--out',default='results/main')
    args=p.parse_args()
    with torch.inference_mode():run(args)
