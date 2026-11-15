"""Refit after observations to separate compression/importance error from policy.

All candidate outcomes are evaluated offline; acquisition choices are frozen
from factual information only. Known proxy is explicitly privileged.
"""
import argparse,json
from pathlib import Path
import numpy as np
from engine import *
from inference import *
from run import plain

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);args=p.parse_args()
    out=Path('results/refined');out.mkdir(parents=True,exist_ok=True)
    for case in range(args.worker,72,2):
        dest=out/f'case_{case:03d}.json'
        if dest.exists():continue
        path=Path('results/main')/dest.name
        assert path.exists(),path
        d=json.loads(path.read_text());f=d['family'];obs=d['observations'];truth={k:np.array(v) for k,v in d['truth'].items()};results=[]
        for action in ACTIONS:
            data=[('factual',obs['factual'],1.),(action,obs[action],1.)]
            atoms,stat=smc(f,data,n=1536,seed=80900+case,moves=12)
            results.append(dict(name='broad_refit_action',action=action,metrics=metrics(atoms,predict(atoms,f),truth),cost=stat))
        data=[(a,obs[a],1.) for a in ['factual','view','center']]
        MODELS['known_proxy']=[d['rounding']]
        for model in ['broad','known_proxy']:
            atoms,stat=smc(f,data,model=model,n=1536,seed=90900+case,moves=12)
            results.append(dict(name=model,action='factual+view+center',metrics=metrics(atoms,predict(atoms,f),truth),cost=stat))
        dest.write_text(json.dumps(plain(dict(case=case,family=f,records=results)),allow_nan=False));print('refined',case,flush=True)
