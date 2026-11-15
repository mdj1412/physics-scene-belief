import os,time,json,gc
from pathlib import Path
import numpy as np
import torch
import physics
physics.DEV='cpu'
from compact import Compact
from bank import Bank
from benchmark import predict
from common import R,public,private,write

assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
x=public('test');truth=private('test');sel=json.loads((R/'development.json').read_text())['selected']
indices=json.loads((R/'benchmark_protocol.json').read_text())['case_indices']
constructors={sel['name']:lambda:Compact(sel['order'],sel['steps']),
              'compact_g3_t1':lambda:Compact(3,1),'qmc4096':lambda:Bank('qmc',4096),
              'qmc16384':lambda:Bank('qmc',16384),'qmc65536':lambda:Bank('qmc',65536),'reference61':lambda:Bank('quad',61)}
records={}
for nm,build in constructors.items():
    start=time.perf_counter();method=build();setup=time.perf_counter()-start;rows=[];maxdiff=0.
    original=np.load(R/'evaluations'/f'{nm}.npz')['predictive']
    for rep in range(-1,3):
        for i in (indices[:2] if rep<0 else indices):
            start=time.perf_counter();s=method.fit(x[i:i+1]);tfit=time.perf_counter()-start
            start=time.perf_counter();a=int(method.choose(s).item());tselect=time.perf_counter()-start
            start=time.perf_counter();s2=s if a==0 else method.update(s,a,truth[f'obs{a}'][i:i+1,0,:]);pr=predict(method,s2);tupdate=time.perf_counter()-start
            maxdiff=max(maxdiff,float(abs(pr.detach().numpy()[0]-original[i,a,0]).max()))
            if rep>=0:rows.append(dict(case=i,repeat=rep,action=a,fit_s=tfit,select_s=tselect,update_predict_s=tupdate,total_s=tfit+tselect+tupdate))
            del s,s2,pr
    records[nm]=dict(setup_s=setup,rows=rows,median_total_s=float(np.median([r['total_s'] for r in rows])),
                    max_prediction_difference_from_gpu=maxdiff,cache_bytes=method.cache_bytes(),state_bytes=method.state_bytes())
    write(R/'benchmark_cpu.json',records);print(nm,records[nm]['median_total_s'],maxdiff,flush=True)
    del method;gc.collect()
write(R/'benchmark_cpu_protocol.json',dict(cpu_threads=2,dtype='float64',cases=indices,repeats=3,post_primary_supplement=True,completed=time.time()))
