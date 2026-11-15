"""Run with --cpu for planned same-code CPU timing; before importing inference."""
import sys,os,time,gc
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
    x=public('test');ob=observations('test');stored=np.load(R/'decision_predictions.npz');acts=np.load(R/'decision_actions.npz');flags=np.load(R/'expansion_flags.npz')
    indices=np.linspace(0,len(x)-1,48,dtype=int);results={}
    for name in METHODS:
        m,setup=timed(lambda:Engine(name));rows=[];err=0.
        for rep in range(-1,3):
            for i in (indices[:2] if rep<0 else indices):
                ctx,first=timed(lambda:m.start(x[i:i+1]));a=ctx['action'];y=ob[a][i:i+1,0] if a else None
                (p,flag),last=timed(lambda:m.finish(ctx,y))
                error=float(abs(p.cpu().numpy()[0]-stored[name][i,0]).max());err=max(err,error)
                assert a==acts[name][i],(name,i,a,acts[name][i]);assert error<1e-8,(name,i,error)
                if name in flags.files:assert flag==flags[name][i,0] or name.startswith('expanded_')
                if rep>=0:rows.append(dict(case=int(i),repeat=rep,action=a,expanded=flag or name.startswith('expanded_'),fit_detect_select_s=first,update_expand_predict_s=last,total_s=first+last))
        totals=[r['total_s'] for r in rows]
        results[name]=dict(setup_s=setup,cache_bytes=m.cache_bytes(),base_state_bytes=m.method.state_bytes(),rows=rows,median_s=float(np.median(totals)),p90_s=float(np.quantile(totals,.9)),max_prediction_diff=err,mean_expand_fraction=float(np.mean([r['expanded'] for r in rows])))
        write(R/('timing_cpu.json' if CPU else 'timing_gpu.json'),results);print('timing',CPU,name,results[name]['median_s'],flush=True)
        del m;gc.collect()
        if not CPU:torch.cuda.empty_cache()
    write(R/('timing_cpu_protocol.json' if CPU else 'timing_gpu_protocol.json'),dict(time=time.time(),cpu=CPU,threads=2,cases=indices.tolist(),repeats=3,dtype='float64',source=sha('benchmark.py'),runtime_source=sha('runtime.py')))
if __name__=='__main__':main()
