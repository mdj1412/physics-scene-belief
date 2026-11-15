"""Paired intervention schedules and full observed stopping/acquisition/decoding latency."""
from pathlib import Path
import hashlib, json, os, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'directional_prefix_v33'))
import evaluate33 as v33
from target34 import calibration,SCHEDULES,LAWS,deployment_x,deployment_truth
from detector34 import ActionConditionedDetector
from score30 import features,predict
from contact30 import Engine,Q,P,torch
from runner29 import PosteriorPredictiveEngine,atomic_json,atomic_npz,jsonable,digest
from runner28 import potential_costs
from batch31 import batch_gap

METHODS=('broad','fixed4','uncertainty')
STAGES=(2,4,8)
OUT=ROOT/'results_policy'
LOCK=BASE/'compute_matched_stopping_v32/results_train/model_lock.json'

class History:
    def __init__(self,detector,record):
        self.detector=detector
        self.x,self.ids,self.y,self.d=record
        self.ll=detector.component_loglikelihood(self.x)
        self.fits={}
    def fit(self,k):
        if k not in self.fits:
            ids=self.ids[:k]
            joint=self.detector.signed_joint(self.x[ids],self.y[:k],self.d[:k])
            self.fits[k]=self.detector.posterior_from_components(self.ll,ids,joint)
        return self.fits[k]
    def feature(self,k):
        y=self.y[:k].copy(); y[:,1]*=-self.d[:k]
        return features(self.fit(k),self.x,self.ids[:k],y)

def allocate(histories,method,model,head=None):
    if method=='fixed4':
        return np.full(8,4,dtype=int)
    active=np.arange(8);counts=np.full(8,2,dtype=int)
    for stage,k in enumerate((2,4)):
        scores=[]
        for e in active:
            h=histories[e]
            if method=='broad':
                v=predict(model[stage],h.feature(k))
            elif method=='uncertainty':
                v=h.fit(k)['sd'][1]
            elif method=='signed':
                v=head(h,k)
            else:
                raise ValueError(method)
            scores.append(float(v))
        active=np.sort(active[np.argsort(-np.array(scores),kind='stable')[:len(active)//2]])
        counts[active]=2*k
    assert counts.sum()==32
    return counts

def full_call(det,rec,method,model,banks,decs,xx,sensor,head):
    torch.cuda.synchronize(); tic=time.perf_counter()
    hs=[History(det,r) for r in rec]; counts=allocate(hs,method,model,head)
    engines=[PosteriorPredictiveEngine(banks,h.fit(int(k))) for h,k in zip(hs,counts)]
    torch.cuda.synchronize(); t1=time.perf_counter()
    chosen=[];predicted=[]
    for eng,dec,x,y in zip(engines,decs,xx,sensor):
        a=v33.decisions(batch_gap(eng,x));p=v33.decode_selected(dec,x,a,y)
        chosen.append(a);predicted.append(p)
    torch.cuda.synchronize(); end=time.perf_counter()
    return counts,np.array(chosen),np.array(predicted),1000*(end-tic),1000*(end-t1)

def paired_timing(det,recs,methods,model,pair,decs,xx,truth,ref,seed,heads):
    timing=np.empty((2,3,len(methods),2))
    sensor=[{a:obs[a][:,0] for a in (Q,P)} for _,_,obs in truth]
    rng=np.random.default_rng(seed)
    for rep in (-1,0,1,2):
        for ix in rng.permutation(2*len(methods)):
            sc,m=divmod(ix,len(methods))
            counts,chosen,pr,full,warm=full_call(det,recs[sc],methods[m],model,pair,decs,xx,sensor,heads[sc])
            expect=ref[sc]
            np.testing.assert_array_equal(counts,expect['counts'][:,m])
            np.testing.assert_array_equal(chosen,expect['actions'][:,:,m])
            expected=np.take_along_axis(expect['probabilities'][...,0],(chosen==P)[:,:,None].astype(int),axis=2)[:,:,0]
            np.testing.assert_allclose(pr,expected,atol=1e-9,rtol=0)
            if rep>=0: timing[sc,rep,m]=[full,warm]
    return timing

def setup(cfg,i,b):
    base=901_000_000+(i//2)*1_000_000+b*10_000
    alpha=np.array(cfg['alphas'])[np.random.default_rng(base+401).permutation(8)]
    pair=[base+301,base+302];law=LAWS[i];start=(-1,1)[b%2]
    recs=[[calibration(base+100+e,al,law,sc,start) for e,al in enumerate(alpha)] for sc in SCHEDULES]
    for u,v in zip(*recs):
        for j in (0,1):np.testing.assert_array_equal(u[j],v[j])
        np.testing.assert_array_equal(u[2][::2],v[2][::2])
    xx=[deployment_x(base+200+e,al,law) for e,al in enumerate(alpha)]
    return base,alpha,pair,recs,xx

def run_cell(det,model,fp,cfg,i,b):
    law=LAWS[i];folders=[OUT/law/f'block_{b:03d}'/n for n in SCHEDULES]
    for folder in folders:folder.mkdir(parents=True,exist_ok=True)
    if all((f/'meta.json').exists() for f in folders):
        for f in folders:
            info=json.loads((f/'meta.json').read_text())
            assert info['fingerprint']==fp and all(info[k+'_sha256']==digest(f/(k+'.npz')) for k in ('choices','metrics'))
        return
    start=time.perf_counter();base,alpha,pair,recs,xx=setup(cfg,i,b)
    references=[];metadata=[]
    for schedule,records,folder in zip(SCHEDULES,recs,folders):
        hs=[History(det,r) for r in records]
        counts=np.stack([allocate(hs,m,model) for m in METHODS],axis=1)
        atomic_json(folder/'allocation.json',jsonable(dict(counts=counts,methods=METHODS,model_sha256=digest(LOCK))))
        feat=np.array([[h.feature(k) for k in STAGES] for h in hs])
        uncertainty=feat[:,:2,3]
        scores=np.stack([predict(model[s],feat[:,s]) for s in range(2)],axis=1)
        gaps=np.array([[batch_gap(PosteriorPredictiveEngine(pair,h.fit(k)),x) for k in STAGES] for h,x in zip(hs,xx)])
        all_actions=v33.decisions(gaps)
        selected=np.stack([all_actions[np.arange(8),np.searchsorted(STAGES,col)] for col in counts.T],axis=-1)
        assert np.all((selected==P).sum(axis=1)==13)
        fact_hash=[hashlib.sha256(r[0].tobytes()).hexdigest() for r in records]
        atomic_npz(folder/'choices.npz',counts=counts,methods=METHODS,actions=selected,gaps=gaps,features=feat[:,:2],
                   stage8_features=feat[:,2],uncertainty=uncertainty,scores=scores,alpha=alpha,paid_ids=[r[1] for r in records],
                   paid_x=[r[0][r[1]] for r in records],paid_y=[r[2] for r in records],paid_direction=[r[3] for r in records],
                   factual_pool_sha256=fact_hash)
        references.append(dict(counts=counts,actions=selected,all_actions=all_actions,features=feat[:,:2],scores=scores,uncertainty=uncertainty))
        metadata.append(dict(choices_sha256=digest(folder/'choices.npz'),allocation_sha256=digest(folder/'allocation.json')))
    truth=[deployment_truth(base+200+e,al,law) for e,al in enumerate(alpha)]
    decs=[Engine(pair,*al) for al in alpha]
    costs=[];probs=[]
    for x,dec,(label,_,obs) in zip(xx,decs,truth):
        c,p=potential_costs(x,label,obs,[dec]);costs.append(c[:,0]);probs.append(p[:,0])
    cost=np.array(costs);prob=np.array(probs)
    for r in references:r['probabilities']=prob
    times=np.empty((2,0,len(METHODS),2))
    if b<cfg['timed_blocks_per_cell']:
        times=paired_timing(det,recs,METHODS,model,pair,decs,xx,truth,references,base+501+i%2,[None,None])
    for idx,(schedule,folder,r,md) in enumerate(zip(SCHEDULES,folders,references,metadata)):
        risk=np.take_along_axis(cost,(r['actions']==P).astype(int),axis=2).mean(1)
        stage_risk=np.take_along_axis(cost,(r['all_actions'].transpose(0,2,1)==P).astype(int),axis=2).mean(1)
        atomic_npz(folder/'metrics.npz',methods=METHODS,counts=r['counts'],actions=r['actions'],labels=[q[0] for q in truth],
             nominal_labels=[q[1] for q in truth],probabilities=prob,brier=risk,oracle_stage_risk=stage_risk,
             scores=r['scores'],uncertainty=r['uncertainty'],features=r['features'],full_ms=times[idx,:,:,0],warm_ms=times[idx,:,:,1],alpha=alpha)
        md.update(methods=METHODS,schedule=schedule,start=(-1,1)[b%2],law=law,block=b,model_sha256=digest(LOCK),fingerprint=fp,
            metrics_sha256=digest(folder/'metrics.npz'),bank_pair=pair,calibration_seeds=[base+100+e for e in range(8)],
            deployment_seeds=[base+200+e for e in range(8)],gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'])
        atomic_json(folder/'meta.json',jsonable(md))
    print('finished',law,b,round(time.perf_counter()-start,2),flush=True)

def main():
    cfg=json.loads((ROOT/'policy_design34.json').read_text())
    assert torch.cuda.device_count()==1 and torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(2)
    inputs=[Path(__file__),ROOT/'target34.py',ROOT/'detector34.py',ROOT/'policy_design34.json',LOCK,
        BASE/'directional_prefix_v33/target33.py',BASE/'directional_prefix_v33/evaluate33.py',
        BASE/'probe_gain_reliability_v31/batch31.py',BASE/'calibration_probe_v28/estimator28.py',BASE/'learned_probe_value_v30/score30.py']
    fp='|'.join(digest(p) for p in inputs)
    model=json.loads(LOCK.read_text())['models']['broad']
    with torch.inference_mode():
        dt=ActionConditionedDetector()
        for b in range(cfg['blocks_per_law']):
            for i,law in enumerate(LAWS):run_cell(dt,model,fp,cfg,i,b)
        atomic_json(OUT/'completion.json',dict(blocks=8*cfg['blocks_per_law']*2,independent_clusters=4*cfg['blocks_per_law'],
                      fingerprint=fp,model_sha256=digest(LOCK),gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
