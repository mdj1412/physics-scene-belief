"""Same-device serial latency, synchronized stages, setup reported separately."""
import os,time,json,gc
from pathlib import Path
import numpy as np
import torch
from physics import *
from compact import Compact
from bank import Bank
from common import R,write,public,private,tree_choice,sha

def predict(method,state):
    if isinstance(method,Compact):
        _,th=method.points(state);return torch.einsum('bk,q,bkqd->bd',state.logp.exp(),method.weights,target(th))
    return method.predict(state)
def clocked(fn):
    torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize();return out,time.perf_counter()-start
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    assert (R/'baselines_done.json').exists();x=public('test');truth=private('test');selected=json.loads((R/'development.json').read_text())['selected']
    constructors={f'compact_g{o}_t{k}':(lambda o=o,k=k:Compact(o,k)) for o in [3,5] for k in [1,4]}
    constructors[selected['name']+'_diag']=lambda:Compact(selected['order'],selected['steps'],True)
    constructors.update(qmc4096=lambda:Bank('qmc',4096),qmc16384=lambda:Bank('qmc',16384),qmc65536=lambda:Bank('qmc',65536),reference61=lambda:Bank('quad',61))
    indices=[int(i)*3+(j%3) for j,i in enumerate(np.linspace(0,255,16,dtype=int))]
    records={}
    for nm,build in constructors.items():
        method,setup=clocked(build);tree_options=[False,True] if nm in ['qmc16384','reference61'] else [False]
        for use_tree in tree_options:
            label=nm+('_tree' if use_tree else '');rows=[]
            for rep in range(-1,3):
                for i in (indices[:2] if rep<0 else indices):
                    case=x[i:i+1];before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    state,tfit=clocked(lambda:method.fit(case))
                    action,tsel=clocked(lambda:int(tree_choice(case)[0]) if use_tree else int(method.choose(state).item()))
                    def finish():
                        state2=state if action==0 else method.update(state,action,truth[f'obs{action}'][i:i+1,0,:])
                        return predict(method,state2)
                    output,tupdate=clocked(finish)
                    if rep>=0:rows.append(dict(case=i,repeat=rep,action=action,fit_s=tfit,select_s=tsel,update_predict_s=tupdate,total_s=tfit+tsel+tupdate,temporary_peak_bytes=max(0,torch.cuda.max_memory_allocated()-before)))
                    assert torch.isfinite(output).all()
                    del state,output
            records[label]=dict(setup_s=setup,cache_bytes=method.cache_bytes(),state_bytes=method.state_bytes(),rows=rows,
                                median_total_s=float(np.median([r['total_s'] for r in rows])),mean_total_s=float(np.mean([r['total_s'] for r in rows])),
                                median_stage_s={k:float(np.median([r[k] for r in rows])) for k in ['fit_s','select_s','update_predict_s']})
            write(R/'benchmark.json',records);print('timing',label,records[label]['median_total_s'],flush=True)
        del method;gc.collect();torch.cuda.empty_cache()
    write(R/'benchmark_protocol.json',dict(device=torch.cuda.get_device_name(),dtype='float64',batch=1,case_indices=indices,repetitions=3,explicit_cuda_sync=True,source_sha256=sha('benchmark.py'),completed=time.time()))
if __name__=='__main__':main()
