"""Post-primary implementation audit: identical acquisition objectives, efficient contractions.
No policies or main result arrays are changed. Remove unused variance computation in direct-risk runtime.
"""
import os,sys,time
if '--cpu' in sys.argv:os.environ['BELIEF_DEVICE']='cpu'
import numpy as np
import torch
from model import *
from acquisition import choose,choose_risk,pair,plan
from common import *
from benchmark import Runtime,sync

def mv(A,Z):return torch.stack([A@Z[:,j] for j in range(Z.shape[1])],-1)
class FastRuntime(Runtime):
    def __init__(self,name):
        super().__init__(name);self.features={}
        if name.endswith('native'):
            for a in [1,2,3]:
                y=self.bank.white[a];self.features[a]=torch.cat([y,(y[:,:,None]*y[:,None,:]).flatten(1)],-1)
    def begin(self,x,ctx,task):
        b=self.bank;w=b.fit(x);z=targets(b.theta,ctx,task=task);self.last_scores=None
        if self.name in ['query','probe']:a=1 if self.name=='query' else 3
        elif self.name.endswith('pair'):a=int(choose(pair(b,w,z,x,[z.shape[-1]])[0])[0])
        elif self.name.endswith('native'):
            zm=w@z;scores=[t(0.)]
            for action in [1,2,3]:
                y=b.white[action];D=y.shape[1];mom=w@self.features[action];ym=mom[:D]
                S=mom[D:].reshape(D,D)-ym[:,None]*ym[None,:]+torch.eye(D,device=DEV,dtype=DT)
                C=torch.stack([(w*z[:,j])@y for j in range(z.shape[1])])-zm[:,None]*ym[None,:]
                scores.append(.5*(C*torch.linalg.solve(S,C.T).T).sum())
            self.last_scores=torch.stack(scores);a=int(self.last_scores[1:].argmax())+1
        else:
            o=self.outer;ow=o.fit(x);oz=targets(o.theta,ctx,task=task);u,eps=self.plan;ix=torch.searchsorted(ow.cumsum(0),u.contiguous()).clamp_max(o.N-1)
            truth=oz[ix];risks=[.5*((truth-w@z)**2).sum(-1).mean()];lp=w.clamp_min(1e-300).log()
            for action in [1,2,3]:
                sums=[]
                for start in range(0,self.M,128):
                    idx=ix[start:start+128];y=o.mu[action][idx]+eps[start:start+128,:o.mu[action].shape[-1]]*noise(action)
                    weights=torch.softmax(lp+b.ll(action,y),-1);pred=mv(weights,z)
                    sums.append(.5*((pred-truth[start:start+128])**2).sum(-1))
                risks.append(torch.cat(sums).mean())
            self.last_scores=torch.stack(risks);a=int(self.last_scores[1:].argmin())+1
        return a,w,z

def run():
    require();torch.set_num_threads(2);pub=np.load(R/'data/test/public.npz');priv=np.load(R/'data/test/private.npz');pol=np.load(R/'policies/matched_nominal.npz');starts=np.cumsum([0]+SIZES)
    names=['native','pair','nested128','nested512','qmc64_native','qmc64_pair','query','probe'];out={}
    for name in names:
        sync();st=time.perf_counter();rt=FastRuntime(name);sync();setup=time.perf_counter()-st
        ev=np.load(R/f'evaluations/matched_nominal_{rt.bank.N}.npz')['pred'];samples=[];errs=[];scoreerrs=[];peaks=[]
        for i in range(3):
            state=rt.begin(t(pub['factual'][i,0]),t(pub['context'][i,0]),TASKS[0]);rt.finish(state,t(priv[f'obs{state[0]}'][i,0,0]));sync()
        for rep in range(3):
            for i in range(20):
                for k,task in enumerate(TASKS):
                    x=t(pub['factual'][i,0]);ctx=t(pub['context'][i,0]);obs={a:t(priv[f'obs{a}'][i,0,0]) for a in [1,2,3]}
                    if DEV=='cuda' and rep==0 and k==0:base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    sync();st=time.perf_counter();state=rt.begin(x,ctx,task);pred=rt.finish(state,obs[state[0]]);sync();samples.append(time.perf_counter()-st)
                    if DEV=='cuda' and rep==0 and k==0:peaks.append(torch.cuda.max_memory_allocated()-base)
                    expected=1 if name=='query' else (3 if name=='probe' else int(pol[name][i,k]));assert state[0]==expected,(name,i,k,state[0],expected)
                    errs.append(float(np.max(np.abs(pred.cpu().numpy()-ev[i,expected,0,starts[k]:starts[k+1]]))))
                    if rt.last_scores is not None and name in ['native','nested128','nested512']:
                        # Main nested128 uses first128 conditional samples but retains the512-sample
                        # no-observation diagnostic. Only actions1..3 are eligible, so validate these.
                        key='native_scores' if name=='native' else name+'_risk';scoreerrs.append(float(np.max(np.abs(rt.last_scores.cpu().numpy()[1:]-pol[key][i,1:,k]))))
        assert max(errs)<1e-9
        if scoreerrs:assert max(scoreerrs)<1e-9,(name,max(scoreerrs))
        out[name]=dict(median_s=float(np.median(samples)),p90_s=float(np.quantile(samples,.9)),setup_s=setup,n=len(samples),max_prediction_error=max(errs),max_acquisition_error=max(scoreerrs) if scoreerrs else None,state_bytes=rt.bank.N*8*(2 if rt.outer else 1),cache_bytes=rt.bank.cache_bytes()+(rt.outer.cache_bytes() if rt.outer else 0)+sum(v.numel()*v.element_size() for v in rt.features.values()),temporary_peak_bytes=max(peaks) if peaks else None,task_medians_s={task:float(np.median(np.array(samples).reshape(3,20,5)[:,:,k])) for k,task in enumerate(TASKS)})
        print('Optimized',DEV,name,out[name]['median_s'],flush=True);del rt
        if DEV=='cuda':torch.cuda.empty_cache()
    device='gpu' if DEV=='cuda' else 'cpu';write(R/f'timing_optimized_{device}.json',out);write(R/f'timing_optimized_{device}_done.json',dict(source=sha(__file__),time=time.time()))
if __name__=='__main__':run()
