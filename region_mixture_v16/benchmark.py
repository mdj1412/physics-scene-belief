import os,sys,time
if '--cpu' in sys.argv:os.environ['BELIEF_DEVICE']='cpu'
import numpy as np
import torch
from model import *
from acquisition import *
from common import *

METHODS=['native','pair','probe','query','nested128','nested512','qmc64_native','qmc64_pair']
def sync():
    if DEV=='cuda':torch.cuda.synchronize()
class Runtime:
    def __init__(self,name):
        self.name=name;self.bank=Bank(65536,2026111203) if name.startswith('qmc64') else Bank()
        self.outer=Bank(seed=2026111202) if name.startswith('nested') else None
        self.M=int(name[6:]) if self.outer else 0;self.plan=plan(self.M) if self.outer else None
    def begin(self,x,ctx,task):
        b=self.bank;w=b.fit(x);z=targets(b.theta,ctx,task=task);sizes=[z.shape[-1]]
        if self.name in ['probe','query']:a=3 if self.name=='probe' else 1
        elif self.outer:
            ow=self.outer.fit(x);oz=targets(self.outer.theta,ctx,task=task);r=nested(b,w,z,self.outer,ow,oz,sizes,M=self.M,preplan=self.plan);a=int(choose_risk(r['risk'])[0])
        elif self.name.endswith('pair'):a=int(choose(pair(b,w,z,x,sizes)[0])[0])
        else:a=int(choose(lmmse(b,w,z,sizes))[0])
        return a,w,z
    def finish(self,state,y):
        a,w,z=state;return self.bank.update(w,a,y)@z
def run():
    require();torch.set_num_threads(2);pub=np.load(R/'data/test/public.npz');priv=np.load(R/'data/test/private.npz');pol=np.load(R/'policies/matched_nominal.npz')
    device='cpu' if DEV=='cpu' else 'gpu';out={};starts=np.cumsum([0]+SIZES);ids=list(range(20));cases=[(i,k) for i in ids for k in range(5)]
    for name in METHODS:
        sync();st=time.perf_counter();rt=Runtime(name);sync();setup=time.perf_counter()-st
        ev=np.load(R/f'evaluations/matched_nominal_{rt.bank.N}.npz')['pred'];samples=[];errors=[];peaks=[]
        for warm in range(3):
            state=rt.begin(t(pub['factual'][warm,0]),t(pub['context'][warm,0]),TASKS[0]);rt.finish(state,t(priv[f'obs{state[0]}'][warm,0,0]));sync()
        for rep in range(3):
            for i,k in cases:
                x=t(pub['factual'][i,0]);ctx=t(pub['context'][i,0]);obs={a:t(priv[f'obs{a}'][i,0,0]) for a in [1,2,3]}
                if DEV=='cuda' and rep==0 and k==0:
                    base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                sync();st=time.perf_counter();state=rt.begin(x,ctx,TASKS[k]);pred=rt.finish(state,obs[state[0]]);sync();samples.append(time.perf_counter()-st)
                if DEV=='cuda' and rep==0 and k==0:peaks.append(torch.cuda.max_memory_allocated()-base)
                a=state[0];expected=3 if name=='probe' else (1 if name=='query' else int(pol[name][i,k]));assert a==expected,(name,i,k,a,expected)
                errors.append(float(np.max(np.abs(pred.cpu().numpy()-ev[i,a,0,starts[k]:starts[k+1]]))))
        out[name]=dict(median_s=float(np.median(samples)),p90_s=float(np.quantile(samples,.9)),setup_s=setup,max_prediction_error=max(errors),state_bytes=rt.bank.N*8*(2 if rt.outer else 1),cache_bytes=rt.bank.cache_bytes()+(rt.outer.cache_bytes() if rt.outer else 0),temporary_peak_bytes=max(peaks) if peaks else None,n=len(samples),task_medians_s={TASKS[k]:float(np.median(np.array(samples).reshape(3,20,5)[:,:,k])) for k in range(5)})
        assert max(errors)<1e-9,(name,max(errors));print('Timing',device,name,out[name]['median_s'],flush=True)
        del rt
        if DEV=='cuda':torch.cuda.empty_cache()
    write(R/f'timing_{device}.json',out);write(R/f'timing_{device}_done.json',dict(time=time.time(),source=sha(__file__)))
if __name__=='__main__':run()
