import os,sys,time
os.environ['BELIEF_DEVICE']='cpu'
# Supplemental CPU timing: record, rather than hide, cross-device action differences.
import torch
import numpy as np
from model import *
from runtime import Runtime
from mixture import payload
from common import *
METHODS=['region32','global32','region64','region_cf32','region32_compact','native','nested128','nested512','qmc64_native','pair','query','probe']
def sync():
    if DEV=='cuda':torch.cuda.synchronize()
def run():
    require();torch.set_num_threads(2);pub=np.load(R/'data/test/public.npz');priv=np.load(R/'data/test/private.npz');pol=np.load(R/'policies/matched_nominal.npz');starts=np.cumsum([0]+SIZES);out={}
    for name in METHODS:
        sync();st=time.perf_counter();rt=Runtime(name);sync();setup=time.perf_counter()-st
        ev=np.load(R/f'evaluations/matched_nominal_{rt.bank.N}.npz')['pred'];cv=np.load(R/'evaluations/matched_nominal_compact.npz')['pred'];times=[];errs=[];peaks=[];payloads=[];mismatches=[]
        for i in range(3):
            state=rt.begin(t(pub['factual'][i,0]),t(pub['context'][i,0]),TASKS[0]);rt.finish(state,t(priv[f'obs{state[0]}'][i,0,0]));sync()
        for rep in range(3):
            for i in range(20):
                for k,task in enumerate(TASKS):
                    x=t(pub['factual'][i,0]);ctx=t(pub['context'][i,0]);ys={a:t(priv[f'obs{a}'][i,0,0]) for a in [1,2,3]}
                    if DEV=='cuda' and rep==0:base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    sync();st=time.perf_counter();state=rt.begin(x,ctx,task);p=rt.finish(state,ys[state[0]]);sync();times.append(time.perf_counter()-st)
                    if DEV=='cuda' and rep==0:peaks.append(torch.cuda.max_memory_allocated()-base)
                    expected=1 if name=='query' else (3 if name=='probe' else int(pol['region32' if name=='region32_compact' else name][i,k]))
                    if state[0]!=expected and rep==0:
                        record=dict(world=i,task=task,cpu_action=state[0],gpu_action=expected)
                        if rt.comp is not None and name!='region_cf32':
                            from mixture import risks
                            record['cpu_risks']=risks(rt.comp,rt.eps).tolist()
                            record['gpu_risks']=pol[name+'_risk'][i,:,k].tolist()
                        mismatches.append(record)
                    target=cv[i,0,starts[k]:starts[k+1]] if name=='region32_compact' else ev[i,state[0],0,starts[k]:starts[k+1]];errs.append(float(np.max(np.abs(p.cpu().numpy()-target))))
                    if rt.comp is not None:payloads.append(payload(rt.comp))
        assert max(errs)<1e-9,(name,max(errs))
        out[name]=dict(median_s=float(np.median(times)),p90_s=float(np.quantile(times,.9)),task_medians_s={task:float(np.median(np.array(times).reshape(3,20,5)[:,:,k])) for k,task in enumerate(TASKS)},n=len(times),cross_device_action_mismatches=mismatches,max_prediction_error=max(errs),setup_s=setup,bank_cache_bytes=rt.cache_bytes(),compact_payload_max_bytes=max(payloads) if payloads else None,temporary_peak_bytes=max(peaks) if peaks else None)
        print('Timing',DEV,name,out[name]['median_s'],flush=True);del rt
        if DEV=='cuda':torch.cuda.empty_cache()
    dev='gpu' if DEV=='cuda' else 'cpu';write(R/f'timing_{dev}.json',out);write(R/f'timing_{dev}_done.json',dict(time=time.time(),source=sha(__file__),runtime=sha('runtime.py'),mixture=sha('mixture.py')))
if __name__=='__main__':run()
