import os,sys,time
import numpy as np
import torch
from physics import t,noise
from flexible import Expanded,measure,response
from acquisition import pair_select,pair_from_theta
from common import *
def generate(split,N,seed,test=False):
    from worlds import dataset
    x,h=dataset(N,seed,test);p=R/'data'/split;p.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p/'public.npz',**x);np.savez_compressed(p/'private.npz',**h)
def validate():
    cases={}
    for kind in ['contact','geometry']:
        if kind=='contact':
            b=np.log(.4/.6);z=np.log(.37/.5)
            th=t([[[0,2,0,0,.6,b+z,-1],[0,2,0,0,.6,b-z,1]]])
        else:th=t([[[0,1,0,.1,.6,0,0],[1,1,0,.1,.6,0,0]]])
        factual=measure(th,0);x=factual.mean(1).cpu().numpy();score,d=pair_from_theta(th,x)
        assert int(score.argmax(1))==(3 if kind=='contact' else 1)
        if kind=='contact':assert abs(factual[:,0]-factual[:,1]).max()<1e-12
        cases[kind]=dict(scores=score.cpu().tolist(),factual_prediction_difference=float(abs(factual[:,0]-factual[:,1]).max()),query_predictions=response(th).cpu().tolist(),expected_action=3 if kind=='contact' else 1)
    m=Expanded();state=m.fit(public('pilot'));pred=m.predict(state)
    assert torch.isfinite(pred).all() and torch.linalg.eigvalsh(state.cov).min()>0
    assert abs(state.logp.exp().sum(1)-1).max()<1e-12
    cases['pilot']=dict(min_cov_eigenvalue=float(torch.linalg.eigvalsh(state.cov).min()),no_bank_import='bank' not in sys.modules)
    write(R/'validation.json',cases)
def evaluate(method,x,observations,batch=16):
    pred=[]
    for i in range(0,len(x),batch):
        end=min(i+batch,len(x));state=method.fit(x[i:end]);pp=[]
        for a in range(4):
            if a==0:st=state
            else:
                rep=state.repeat(16) if hasattr(state,'logp') else state.repeat_interleave(16,0)
                st=method.update(rep,a,observations[a][i:end].reshape(-1,len(noise(a))))
            p=method.predict(st);pp.append((p[:,None,:].expand(-1,16,-1) if a==0 else p.reshape(end-i,16,2)).cpu().numpy())
        pred.append(np.stack(pp,1))
    out=np.concatenate(pred);assert np.isfinite(out).all();return out
def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    write(R/'environment.json',dict(host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,python=sys.version))
    generate('pilot',16,3000101);validate();snapshot('source_compact.json');generate('test',256,3100101,True)
    x=public('test');m=Expanded();(R/'policies').mkdir(exist_ok=True);(R/'evaluations').mkdir(exist_ok=True)
    modes={'native':[],'pair':[],'unfiltered':[],'agnostic':[]};diags={k:[] for k in modes if k!='native'}
    for i in range(0,len(x),16):
        xx=x[i:i+16];st=m.fit(xx);modes['native'].append(m.gains(st).cpu().numpy())
        for k in diags:
            g,d=pair_select(m,st,xx,k);modes[k].append(g.cpu().numpy());diags[k].append({a:b.cpu().numpy() for a,b in d.items()})
        if i%256==0:print('compact policy',i,flush=True)
    for k,parts in modes.items():
        g=np.concatenate(parts);out=dict(actions=g.argmax(1),gains=g)
        if k in diags:out.update({a:np.concatenate([d[a] for d in diags[k]]) for a in diags[k][0]})
        np.savez_compressed(R/'policies'/f'compact_{k}.npz',**out);freeze('compact_'+k)
    write(R/'compact_policy_freeze.json',dict(time=time.time(),no_bank_import='bank' not in sys.modules))
    pred=evaluate(m,x,obs('test'));np.savez_compressed(R/'evaluations/compact.npz',predictive=pred)
    write(R/'compact_done.json',dict(time=time.time(),no_bank_import='bank' not in sys.modules));print('COMPACT DONE',flush=True)
if __name__=='__main__':main()
