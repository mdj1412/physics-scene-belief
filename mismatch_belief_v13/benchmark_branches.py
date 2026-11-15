"""Supplement: rare triggered branches absent from the original fixed-case timing."""
import os,sys,time,gc
import numpy as np
import torch
import physics
CPU='--cpu' in sys.argv
if CPU:physics.DEV='cpu'
from runtime import Engine
from common import *
def sync():
    if not CPU:torch.cuda.synchronize()
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    x=public('test');obs=observations('test');flags=np.load(R/'expansion_flags.npz');pred=np.load(R/'decision_predictions.npz');out={}
    for name in ['primary_gate','probe_gate']:
        rng=np.random.default_rng(20261114);f=flags[name].reshape(256,6,16);m=Engine(name);events=[]
        for j in range(6):
            for flag in [False,True]:
                candidates=np.argwhere(f[:,j]==flag)
                chosen=rng.choice(len(candidates),size=min(4,len(candidates)),replace=False)
                for k in chosen:
                    world,noise=candidates[k];events.append((int(world)*6+j,int(noise),flag,j))
        rows=[]
        for rep in range(-1,3):
            for case,noise,flag,j in (events[:2] if rep<0 else events):
                sync();start=time.perf_counter();ctx=m.start(x[case:case+1]);a=ctx['action'];p,actual=m.finish(ctx,obs[a][case:case+1,noise] if a else None);sync();seconds=time.perf_counter()-start
                assert actual==flag and abs(p.cpu().numpy()[0]-pred[name][case,noise]).max()<1e-8
                if rep>=0:rows.append(dict(case=case,noise=noise,condition=j,flag=flag,rep=rep,seconds=seconds))
        conditional={}
        for j in range(6):
            mu={str(flag):float(np.mean([r['seconds'] for r in rows if r['condition']==j and r['flag']==flag])) for flag in [False,True]}
            rate=float(f[:,j].mean());conditional[str(j)]=dict(rate=rate,conditional_mean_s=mu,weighted_mean_s=(1-rate)*mu['False']+rate*mu['True'])
        out[name]=dict(rows=rows,by_condition=conditional,stratum_median_s={str(flag):float(np.median([r['seconds'] for r in rows if r['flag']==flag])) for flag in [False,True]},weighted_mean_s=float(np.mean([v['weighted_mean_s'] for v in conditional.values()])))
        del m;gc.collect()
    write(R/('branches_cpu.json' if CPU else 'branches_gpu.json'),out)
    print({k:{a:b for a,b in v.items() if a!='rows'} for k,v in out.items()})
if __name__=='__main__':main()
