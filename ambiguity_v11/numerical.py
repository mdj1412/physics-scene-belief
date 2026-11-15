"""Separate deterministic decoding convergence from acquisition Monte Carlo."""
import os,json,time
from pathlib import Path
import numpy as np
import torch
from model import Bank
from run import R,write
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
reports=[]
for n in [81,101]:
    b=Bank(n);records=[];out=R/'numerical'/str(n);out.mkdir(parents=True,exist_ok=True)
    for i in range(32):
        p=json.loads((R/'test/public'/f'{i:04d}.json').read_text());w=b.posterior(p);wf=b.factorize(w);truth=np.load(R/'test/private'/f'{i:04d}.npz')
        old=np.load(R/'test/evaluation_61'/f'{i:04d}.npz');pred={k:[] for k in ['joint','plugin','mixture','factorized']}
        for a in range(4):
            ww=w if a==0 else b.update(w,a,truth['obs'+str(a)])
            ff=wf if a==0 else b.update(wf,a,truth['obs'+str(a)])
            m,pv,mix=b.decode(ww,True);fm,_=b.decode(ff)
            for k,v in [('joint',m),('plugin',pv),('mixture',mix),('factorized',fm)]:pred[k].append(v.expand(32,-1).cpu().numpy())
        vals={k+'_prediction':np.stack(v) for k,v in pred.items()};diff={}
        for k in pred:
            vals[k+'_loss']=.5*((vals[k+'_prediction']-truth['target'])**2).sum(-1)
            delta=abs(vals[k+'_prediction']-old[k+'_prediction'])
            entry=dict(max_prediction=float(delta.max()),mean_prediction=float(delta.mean()),max_loss=float(abs(vals[k+'_loss']-old[k+'_loss']).max()),mean_loss_change=float((vals[k+'_loss']-old[k+'_loss']).mean()))
            if n==101:
                lower=np.load(R/'numerical/81'/f'{i:04d}.npz');entry['max_prediction_81_101']=float(abs(vals[k+'_prediction']-lower[k+'_prediction']).max())
                entry['mean_loss_change_81_101']=float((vals[k+'_loss']-lower[k+'_loss']).mean())
            diff[k]=entry
        np.savez_compressed(out/f'{i:04d}.npz',**vals);records.append(dict(world=i,errors=diff));write(R/'numerical'/f'audit_{n}.json',records)
        if i%8==0:print('numerical',n,i,flush=True)
    del b;torch.cuda.empty_cache()
print('Numerical audit complete',flush=True)
