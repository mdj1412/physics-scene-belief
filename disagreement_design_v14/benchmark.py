import os,sys,time,gc
import numpy as np
import torch
import physics
CPU='--cpu' in sys.argv
if CPU:physics.DEV='cpu'
from runtime import Engine,METHODS
from common import *
def sync():
    if not CPU:torch.cuda.synchronize()
def timed(fn):
    sync();start=time.perf_counter();result=fn();sync();return result,time.perf_counter()-start
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    x=public('test');observations=obs('test');pred=np.load(R/'decision_predictions.npz');act=np.load(R/'decision_actions.npz')
    ids=np.linspace(0,len(x)-1,48,dtype=int);out={}
    for nm in METHODS:
        engine,setup=timed(lambda:Engine(nm));rows=[];err=0.
        for rep in range(-1,3):
            for i in (ids[:2] if rep<0 else ids):
                (state,a),first=timed(lambda:engine.start(x[i:i+1]));y=observations[a][i:i+1,0] if a else None
                p,last=timed(lambda:engine.finish(state,a,y));difference=float(abs(p.cpu().numpy()[0]-pred[nm][i,0]).max());err=max(err,difference)
                assert a==act[nm][i] and difference<1e-8,(nm,i,a,act[nm][i],difference)
                if rep>=0:rows.append(dict(case=int(i),rep=rep,action=a,fit_select_s=first,update_predict_s=last,total_s=first+last))
        times=[r['total_s'] for r in rows];out[nm]=dict(setup_s=setup,state_bytes=engine.state_bytes(),cache_bytes=engine.cache_bytes(),rows=rows,median_s=float(np.median(times)),p90_s=float(np.quantile(times,.9)),max_prediction_difference=err)
        write(R/('timing_cpu.json' if CPU else 'timing_gpu.json'),out);print('timing',CPU,nm,out[nm]['median_s'],flush=True)
        del engine;gc.collect()
        if not CPU:torch.cuda.empty_cache()
    write(R/('timing_cpu_done.json' if CPU else 'timing_gpu_done.json'),dict(time=time.time(),source=sha('benchmark.py'),runtime_source=sha('runtime.py'),cases=ids.tolist(),repeats=3,cpu=CPU))
if __name__=='__main__':main()
