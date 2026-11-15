import os,sys,time,gc
from pathlib import Path
import numpy as np
import torch
import physics,common
CPU='--cpu' in sys.argv
if CPU:physics.DEV='cpu'
common.R=Path('results_actionwise')
from common import *
from runtime import Engine
from detection import predictive_score,base_predict

class ActionEngine(Engine):
    def __init__(self,actionwise):super().__init__('primary_gate');self.actionwise=actionwise
    def finish(self,ctx,obs):
        a=ctx['action'];flag=ctx['pre'];s=ctx['state']
        if not flag and a:
            threshold=self.thresholds['post_actionwise']['primary'][str(a)] if self.actionwise else self.thresholds['post']['primary']
            flag=bool(predictive_score(self.method,s,a,obs).item()>threshold)
        if flag:
            s=self.expansion.fit(ctx['x']);s=self.expansion.update(s,a,obs) if a else s;p=self.expansion.predict(s)
        else:
            s=self.method.update(s,a,obs) if a else s;p=base_predict(self.method,s)
        return p,flag
def sync():
    if not CPU:torch.cuda.synchronize()
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    x=public('test');obs=observations('test');pred=np.load(R/'decision_predictions.npz');flags=np.load(R/'flags.npz');out={}
    for nm,aw in [('original_gate',False),('actionwise_gate',True)]:
        m=ActionEngine(aw);f=flags[nm].reshape(256,6,16);rng=np.random.default_rng(20261114)
        events=[(int(i),0,'fixed') for i in np.linspace(0,len(x)-1,48,dtype=int)]
        for j in range(6):
            for flag in [False,True]:
                candidates=np.argwhere(f[:,j]==flag)
                for k in rng.choice(len(candidates),size=min(4,len(candidates)),replace=False):
                    world,noise=candidates[k];events.append((int(world)*6+j,int(noise),'stratum'))
        rows=[]
        for rep in range(-1,3):
            for i,n,kind in (events[:2] if rep<0 else events):
                sync();start=time.perf_counter();ctx=m.start(x[i:i+1]);a=ctx['action'];p,flag=m.finish(ctx,obs[a][i:i+1,n] if a else None);sync();seconds=time.perf_counter()-start
                assert flag==flags[nm][i,n] and abs(p.cpu().numpy()[0]-pred[nm][i,n]).max()<1e-8
                if rep>=0:rows.append(dict(case=i,noise=n,kind=kind,flag=flag,rep=rep,seconds=seconds))
        weighted=[]
        for j in range(6):
            rate=float(f[:,j].mean());mu={b:float(np.mean([r['seconds'] for r in rows if r['kind']=='stratum' and r['case']%6==j and r['flag']==b])) for b in [False,True]}
            weighted.append((1-rate)*mu[False]+rate*mu[True])
        out[nm]=dict(rows=rows,fixed_median_s=float(np.median([r['seconds'] for r in rows if r['kind']=='fixed'])),stratum_median_s={str(b):float(np.median([r['seconds'] for r in rows if r['kind']=='stratum' and r['flag']==b])) for b in [False,True]},weighted_mean_s=float(np.mean(weighted)))
        print(nm,out[nm]['weighted_mean_s'],flush=True);del m;gc.collect()
    write(R/('timing_cpu.json' if CPU else 'timing_gpu.json'),out)
if __name__=='__main__':main()
