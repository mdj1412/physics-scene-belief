import os,sys,time
if '--cpu' in sys.argv:os.environ['BELIEF_DEVICE']='cpu'
from model import *
from common import *
from runtime_fast import Runtime
from run_policy import ETASKS,MIX,NEST
METHODS=MIX+['native','qmc64_native']+list(NEST)+['query','probe']
def sync():
    if DEV=='cuda':torch.cuda.synchronize()
def run():
    require();torch.set_num_threads(2);pub=np.load(R/'data/test/public.npz');priv=np.load(R/'data/test/private.npz');pol=np.load(R/'policies/matched_nominal.npz');out={}
    for name in METHODS:
        sync();st=time.perf_counter();rt=Runtime(name);sync();setup=time.perf_counter()-st;ev=np.load(R/f'evaluations/matched_nominal_{rt.bank.N}.npz')['pred'];times=[];errs=[];diff=[];peaks=[];cache=[];work=[]
        for i in range(3):
            state=rt.begin(t(pub['factual'][i,0]),t(pub['context'][i,0]),ETASKS[0]);rt.finish(state,t(priv[f'obs{state[0]}'][i,0,0]));sync()
        for rep in range(3):
            for i in range(20):
                for k,task in enumerate(ETASKS):
                    x=t(pub['factual'][i,0]);ctx=t(pub['context'][i,0]);ys={a:t(priv[f'obs{a}'][i,0,0]) for a in [1,2,3]}
                    rt.engine=None
                    if DEV=='cuda' and rep==0:base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    sync();st=time.perf_counter();state=rt.begin(x,ctx,task);p=rt.finish(state,ys[state[0]]);sync();times.append(time.perf_counter()-st)
                    if DEV=='cuda' and rep==0:peaks.append(torch.cuda.max_memory_allocated()-base)
                    expected=1 if name=='query' else (3 if name=='probe' else int(pol[name][i,k]))
                    if state[0]!=expected:
                        if DEV=='cuda':raise AssertionError((name,i,k,state[0],expected))
                        if rep==0:diff.append(dict(world=i,task=task,cpu=state[0],gpu=expected))
                    errs.append(abs(p.item()-ev[i,state[0],0,k]))
                    if rt.engine is not None:cache.append(rt.engine.bytes());work.append(rt.engine.work)
        assert max(errs)<1e-9,(name,max(errs))
        out[name]=dict(median_s=float(np.median(times)),p90_s=float(np.quantile(times,.9)),task_medians_s={task:float(np.median(np.array(times).reshape(3,20,2)[:,:,k])) for k,task in enumerate(ETASKS)},n=len(times),max_prediction_error=max(errs),action_mismatches=diff,setup_s=setup,bank_plan_bytes=rt.cache_bytes(),acquisition_cache_max_bytes=max(cache) if cache else 0,temporary_peak_bytes=max(peaks) if peaks else None,mean_component_observation_evaluations=float(np.mean(work)) if work else None)
        print('Timing',DEV,name,out[name]['median_s'],flush=True);del rt
        if DEV=='cuda':torch.cuda.empty_cache()
    dev='gpu' if DEV=='cuda' else 'cpu';write(R/f'timing_fast_{dev}.json',out);write(R/f'timing_fast_{dev}_done.json',dict(time=time.time(),source=sha(__file__),runtime=sha('runtime_fast.py'),budget=sha('budget_fast.py')))
if __name__=='__main__':run()
