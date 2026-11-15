"""Compact phase: no import of bank or any teacher/reference posterior."""
import os,sys,time,json
from pathlib import Path
import numpy as np
import torch
from physics import *
from compact import Compact,State
from common import R,write,public,private,freeze,check,source_snapshot

CONFIGS=[(3,1),(3,4),(5,1),(5,4)]
def name(order,steps,diagonal=False):return f'compact_g{order}_t{steps}'+('_diag' if diagonal else '')
def prepare():
    for split,N,seed in [('pilot',16,1200101),('dev',64,1300101),('test',256,1400101)]:
        x=[];hidden=[]
        for i in range(N):
            p,h=make_world(seed+137*i,split=='test');x.append(p);hidden.append(h)
        path=R/'data'/split;path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'public.npz',factual=np.array(x))
        np.savez_compressed(path/'private.npz',**{k:np.array([h[k] for h in hidden]) for k in hidden[0]})
    source_snapshot('source_prepare.json')
    write(R/'environment.json',dict(time=time.time(),host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],torch=torch.__version__,numpy=np.__version__,python=sys.version,gpu=torch.cuda.get_device_name()))

def fit_all(method,x,batch=64):
    states=[];gains=[]
    for start in range(0,len(x),batch):
        s=method.fit(x[start:start+batch]);states.append(s);gains.append(method.gains(s).cpu().numpy())
    return State(torch.cat([s.mean for s in states]),torch.cat([s.cov for s in states]),torch.cat([s.logp for s in states])),np.concatenate(gains)

def evaluate(method,states,hidden,batch=32):
    results={k:[] for k in ['predictive','mixture','hard']}
    for start in range(0,len(states.mean),batch):
        end=min(start+batch,len(states.mean));s=states.slice(start,end);this={k:[] for k in results}
        for a in range(4):
            updated=s if a==0 else method.update(s.repeat(16),a,hidden[f'obs{a}'][start:end].reshape(-1,len(noise(a))))
            pred=method.predict(updated)
            for k,v in pred.items():
                ar=v[:,None,:].expand(-1,16,-1) if a==0 else v.reshape(end-start,16,2)
                this[k].append(ar.cpu().numpy())
        for k in results:results[k].append(np.stack(this[k],1))
    return {k:np.concatenate(v) for k,v in results.items()}

def validation():
    p=public('pilot');truth=private('pilot');c=Compact(5,4);s=c.fit(p);pr=c.predict(s)
    covmin=float(torch.linalg.eigvalsh(s.cov).min());prob_error=float(abs(s.logp.exp().sum(1)-1).max())
    assert covmin>0 and prob_error<1e-12 and all(torch.isfinite(v).all() for v in pr.values())
    theta=t(truth['theta']);vin=(1,.9);v=velocity(theta,L,vin);vv=velocity(theta,L,vin,viscous=.65)
    energy=max(float(((v*v).sum(1)-1-.9**2).max()),float(((vv*vv).sum(1)-1-.9**2).max()))
    assert energy<=1e-12
    # A compact update has exactly 52 floats per world; no factual dataset in state.
    assert sum(z.numel() for z in [s.mean,s.cov,s.logp])==len(p)*52
    write(R/'validation.json',dict(covariance_min_eigenvalue=covmin,model_probability_sum_error=prob_error,max_energy_excess=energy,state_scalars_per_world=52,no_bank_import='bank' not in sys.modules))

def develop():
    assert 'bank' not in sys.modules
    p=public('dev');truth=private('dev');records=[]
    for order,steps in CONFIGS:
        started=time.time();c=Compact(order,steps);s,g=fit_all(c,p);a=g.argmax(1);pred=evaluate(c,s,truth)
        loss=.5*((pred['predictive']-truth['target'][:,None,None,:])**2).sum(-1)
        value=float(loss[np.arange(len(a)),a].mean());record=dict(name=name(order,steps),order=order,steps=steps,loss=value,seconds=time.time()-started,action_counts=np.bincount(a,minlength=4).tolist())
        records.append(record);print('dev',record,flush=True)
    chosen=min(records,key=lambda x:x['loss']);write(R/'development.json',dict(records=records,selected=chosen));return chosen

def test_compact(selected):
    assert 'bank' not in sys.modules
    p=public('test');states={};methods={}
    configs=[(o,k,False) for o,k in CONFIGS]+[(selected['order'],selected['steps'],True)]
    (R/'policies').mkdir(exist_ok=True);(R/'evaluations').mkdir(exist_ok=True)
    # ALL compact policies are frozen before opening any private test outcome.
    for order,steps,diag in configs:
        nm=name(order,steps,diag);c=Compact(order,steps,diag);s,g=fit_all(c,p)
        np.savez_compressed(R/'policies'/f'{nm}.npz',actions=g.argmax(1),gains=g,mean=s.mean.cpu().numpy(),cov=s.cov.cpu().numpy(),logp=s.logp.cpu().numpy())
        freeze(nm);states[nm]=s;methods[nm]=c
    write(R/'compact_policy_freeze_done.json',dict(time=time.time(),no_bank_import='bank' not in sys.modules,methods=list(methods)))
    truth=private('test')
    for nm,c in methods.items():
        check(nm);pred=evaluate(c,states[nm],truth)
        np.savez_compressed(R/'evaluations'/f'{nm}.npz',**pred)
        print('compact evaluated',nm,flush=True)
    write(R/'compact_stage_done.json',dict(time=time.time(),no_bank_import='bank' not in sys.modules,methods=list(methods)))

def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2);prepare();validation();selected=develop();source_snapshot('source_main.json');test_compact(selected)
if __name__=='__main__':main()
