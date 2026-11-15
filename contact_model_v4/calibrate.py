"""Independent null-model predictive-check calibration; no test outcomes used."""
import json
from pathlib import Path
import numpy as np
import torch
from engine import *
from inference import *
from sensor import observe

if __name__=='__main__':
    out=Path('results/calibration');out.mkdir(parents=True,exist_ok=True);rows=[]
    for family_i,family in enumerate(FAMILIES):
        for i in range(64):
            path=out/f'{family}_{i:02d}.json'
            if path.exists():continue
            rng=np.random.default_rng(881000+family_i*1000+i)
            theta=BOUNDS[:,0]+rng.random(4)*(BOUNDS[:,1]-BOUNDS[:,0])
            obs={a:observe(theta,0.,family,a,rng) for a in ['factual','view','center']}
            atoms,stat=smc(family,[(a,obs[a],1.) for a in ['factual','view']],model='sharp',n=768,seed=91000+family_i*1000+i)
            mu=observation_means(atoms[0],atoms[1],family,'center');w=atoms[2];mean=w@mu;delta=mu-mean
            cov=delta.T@(delta*w[:,None])+torch.diag(tensor(NOISE['center']**2));res=tensor(obs['center'])-mean
            row=dict(family=family,i=i,score=float(res@torch.linalg.solve(cov,res)),cost=stat)
            path.write_text(json.dumps(row));print(family,i,row['score'],flush=True)
