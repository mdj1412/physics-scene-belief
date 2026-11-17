"""Post-design sensitivity. Only the impact direction of purchased pilot pairs is noisy."""
from pathlib import Path
from math import comb
import json
import os
import numpy as np
from switch56 import W,CROSSING,PRIMARY,estimate

ROOT=Path(__file__).resolve().parent
K=8
NOISE=(0.,.05,.10)

def bernoulli_sum(trials,probability):
    return np.array([comb(trials,s)*probability**s*(1-probability)**(trials-s) for s in range(trials+1)])

def reconsider(noise):
    table=np.zeros((K+1,K+1))
    for n in range(1,K+1):
        for x in range(n+1):
            flips=np.convolve(bernoulli_sum(x,1-noise),bernoulli_sum(n-x,noise))
            switch=(np.arange(n+1)/n-.125)/.375 > CROSSING
            table[n,x]=flips[switch].sum()
    return table

def main():
    assert os.getenv('SLURM_JOB_ID')
    data=[]
    for i in range(24):
        with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as p:
            n,x,*_=estimate(p['pilot_ref'][...,:K],p['pilot_imp'][...,:K])
        with np.load(ROOT/'scored56'/f'panel_{i:02d}.npz') as sc:
            risk=sc['exact_cells'][...,0:2].mean(axis=1)
        surface=((np.arange(4)+i)%4)//2
        cond=np.stack([risk[:,surface==m].mean(axis=1) for m in (0,1)])
        eweight=np.column_stack(((1-W)/2,(1-W)/2,W))
        risk=np.einsum('meb,we->mwb',cond,eweight)
        baseline=risk[...,0]
        difference=risk[...,1]-baseline
        rows=[]
        for u in NOISE:
            p=reconsider(u)[n,x].mean(-1)
            delta=p*difference
            rows.append(np.stack([baseline+delta,delta,delta-difference,p],-1).mean(0))
        data.append(rows)
    data=np.stack(data)
    primary=data[:,:,list(PRIMARY),:].mean(2)
    samples=np.random.default_rng(56044825).integers(24,size=(5000,24))
    result={}
    for j,u in enumerate(NOISE):
        row=primary[:,j]
        boot=row[samples].mean(axis=1)
        result[str(u)]=dict(primary=row.mean(0).tolist(),
               by_w=data[:,j,:,3].mean(0).tolist(),
               diff_mean_pointwise95=np.quantile(boot[:,1:3],[.025,.975],axis=0).tolist())
    (ROOT/'pilot_noise56.json').write_text(json.dumps(result,indent=2))
    print({key:val['primary'] for key,val in result.items()})

if __name__=='__main__':main()
