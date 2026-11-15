import os,sys,time
import numpy as np
import torch
from physics import *
from compact import Compact
from flexible import Expanded
from detection import pre_score,predictive_score,base_predict,upper_threshold
from common import *

def generate(split,N,seed,test=False):
    from worlds import dataset
    p,h=dataset(N,seed,test);d=R/'data'/split;d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/'public.npz',**p);np.savez_compressed(d/'private.npz',**h)

def factual_scores(c,x):
    return np.concatenate([pre_score(c,x[i:i+32]).cpu().numpy() for i in range(0,len(x),32)])

def calibrate():
    c=Compact(5,4);x=public('calpre');threshold=upper_threshold(factual_scores(c,x))
    x=public('calpost');ob=observations('calpost');scores={k:[] for k in ['primary','native','probe']};pre=factual_scores(c,x)>threshold
    native=[];selected=[]
    for start in range(0,len(x),32):
        end=min(start+32,len(x));state=c.fit(x[start:end]);a=c.choose(state).cpu().numpy();native.extend(a)
        allscores=np.zeros((end-start,4))
        for action in [1,2,3]:allscores[:,action]=predictive_score(c,state,action,ob[action][start:end,0]).cpu().numpy()
        choices=dict(native=a,probe=np.full(len(a),3),primary=np.where(pre[start:end],3,a));selected.extend(choices['primary'])
        for k,aa in choices.items():scores[k].extend(allscores[np.arange(len(a)),aa])
    thresholds=dict(pre=threshold,post={k:upper_threshold(v) for k,v in scores.items()},alpha=.025,calpre=512,calpost=512,time=time.time())
    write(R/'thresholds.json',thresholds);np.savez_compressed(R/'calibration_scores.npz',preflag=pre,native_actions=native,primary_actions=selected,**scores)
    print('thresholds',thresholds,flush=True)

def validate():
    c=Compact(5,4);e=Expanded();x=public('calpre')[:8]
    results={}
    for nm,m in [('base',c),('expanded',e)]:
        s=m.fit(x);p=base_predict(m,s) if nm=='base' else m.predict(s)
        ev=float(torch.linalg.eigvalsh(s.cov).min());prob=float(abs(s.logp.exp().sum(1)-1).max())
        assert ev>0 and prob<1e-12 and torch.isfinite(p).all()
        results[nm]=dict(min_cov_eigenvalue=ev,prob_sum_error=prob,state_bytes=m.state_bytes())
    # Linear Gaussian moment/tempering check on the expanded update implementation.
    import flexible
    original=flexible.measure
    A=t([[1,.2,-.1,.1,.3],[.1,.2,.7,-.4,.2]])
    def linear(th,action):
        pars=th[...,2:];lo=t([-.4,-.1,.35,-4,-2]);width=t([.8,.2,.55,6,4])
        u=torch.special.ndtri(((pars-lo)/width).clamp(1e-15,1-1e-15));return u@A.T
    flexible.measure=linear
    y=t([[.3,-.2]]);prior=e.prior(1);state=e.update(prior,3,y)
    cov=torch.linalg.inv(torch.eye(5,device=DEV,dtype=DT)+A.T@A/.05**2);mu=cov@A.T@y[0]/.05**2
    err=max(float(abs(state.mean-mu).max()),float(abs(state.cov-cov).max()));assert err<1e-10
    flexible.measure=original;results['linear_closed_form_max_error']=err
    write(R/'validation.json',results)

def test():
    assert 'bank' not in sys.modules
    x=public('test');thresholds=read(R/'thresholds.json');pre=factual_scores(Compact(5,4),x)
    (R/'policies').mkdir(exist_ok=True);(R/'evaluations').mkdir(exist_ok=True)
    for name,m in [('base',Compact(5,4)),('expanded',Expanded())]:
        aa=[];gg=[]
        for i in range(0,len(x),32):
            s=m.fit(x[i:i+32]);g=m.gains(s).cpu().numpy();gg.append(g);aa.extend(g.argmax(1))
        np.savez_compressed(R/'policies'/f'{name}.npz',actions=aa,gains=np.concatenate(gg),pre_score=pre)
        freeze(name)
    write(R/'compact_policy_freeze.json',dict(time=time.time(),no_bank_import='bank' not in sys.modules))
    obs=observations('test')
    for name,m in [('base',Compact(5,4)),('expanded',Expanded())]:
        predictions=[];scores=[]
        for i in range(0,len(x),16):
            end=min(i+16,len(x));s=m.fit(x[i:end]);pp=[];ss=[]
            for a in [0,1,2,3]:
                st=s if a==0 else m.update(s.repeat(16),a,obs[a][i:end].reshape(-1,len(noise(a))))
                p=base_predict(m,st) if name=='base' else m.predict(st)
                pp.append((p[:,None,:].expand(-1,16,-1) if a==0 else p.reshape(end-i,16,2)).cpu().numpy())
                if name=='base':
                    score=torch.zeros((end-i,16),device=DEV) if a==0 else predictive_score(m,s.repeat(16),a,obs[a][i:end].reshape(-1,len(noise(a)))).reshape(end-i,16)
                    ss.append(score.cpu().numpy())
            predictions.append(np.stack(pp,1))
            if ss:scores.append(np.stack(ss,1))
            if i%256==0:print(name,i,flush=True)
        p=np.concatenate(predictions);assert np.isfinite(p).all();out=dict(predictive=p)
        if scores:out['post_score']=np.concatenate(scores)
        np.savez_compressed(R/'evaluations'/f'{name}.npz',**out)
    write(R/'compact_done.json',dict(time=time.time(),no_bank_import='bank' not in sys.modules))

def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    write(R/'environment.json',dict(host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,python=sys.version))
    generate('calpre',512,2100101);generate('calpost',512,2200101);validate();calibrate()
    snapshot('source_freeze.json');generate('test',256,2300101,True);test()
if __name__=='__main__':main()
