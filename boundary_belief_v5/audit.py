import json,argparse,time
from pathlib import Path
import numpy as np
import torch
from belief import Belief,predict,metrics,parameter_summary

parser=argparse.ArgumentParser();parser.add_argument('--cases',type=int,nargs='+',default=[1,25,49]);parser.add_argument('--wait-inputs',action='store_true');args=parser.parse_args()
out=Path('results/audit');out.mkdir(parents=True,exist_ok=True)
for case in args.cases:
    dest=out/f'case_{case:03d}.json'
    if dest.exists():continue
    while args.wait_inputs and not (Path('results/pixel_only')/dest.name).exists():time.sleep(2)
    d=json.loads((Path('results/main')/dest.name).read_text());family=d['family'];data=[('factual',d['observations']['factual'])];records=[]
    truth={k:np.array(v) for k,v in d['truth'].items()}
    for seed in [250100+case,350100+case]:
        atoms,cost=Belief(family).smc(data,n=8192,seed=seed,moves=12);pred=predict(atoms,family);policies={}
        for quality in ['high','low']:
            policies[quality]={}
            for mode in ['bbox','boundary']:
                g,e=Belief(family,quality,mode).acquisition(atoms,pred,samples=1024,seed=seed+1)
                policies[quality][mode]=dict(gains=g.tolist(),eigs=e.tolist())
        records.append(dict(name='acquisition',seed=seed,cost=cost,policies=policies))
    for quality in ['high','low','pixel_only']:
        view=d['view'][quality] if quality!='pixel_only' else json.loads((Path('results/pixel_only')/dest.name).read_text())['view']
        for seed in [260100+case,360100+case]:
            atoms,cost=Belief(family,quality,'boundary').smc(data+[('view',view)],n=3072,seed=seed,moves=18)
            mm=metrics(atoms,predict(atoms,family),truth)
            records.append(dict(name='view_refit',quality=quality,seed=seed,cost=cost,parameters=parameter_summary(atoms),metrics={k:v.tolist() for k,v in mm.items()}))
    dest.write_text(json.dumps(dict(case=case,family=family,records=records),indent=2));print('audited',case,flush=True)
