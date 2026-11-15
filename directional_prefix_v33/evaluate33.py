"""Sealed held-out evaluation and complete, uncached policy timing for v33, with paired orientations and independent shape/surface drawing blocks."""
from pathlib import Path
import json
import os
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_eval'
sys.path.insert(0,str(ROOT.parent/'probe_gain_reliability_v31'))
from batch31 import batch_gap
from target33 import LAWS, calibration, deployment_x, deployment_truth
from contact30 import Engine, Q, P, t, torch
from runner29 import (ProbeShiftEstimator, PosteriorPredictiveEngine, atomic_json, atomic_npz,
                      digest, jsonable)
from runner28 import potential_costs
from score30 import features, predict, NAMES

METHODS=('broad','repeated','fixed4','uncertainty')
STAGES=(2,4,8)


class History:
    def __init__(self,detector,records):
        self.detector=detector
        self.x,self.ids,self.y=records
        self.factual_likelihood=detector.component_loglikelihood(self.x)
        self.estimates={}

    def fit(self,k):
        if k not in self.estimates:
            ids,y=self.ids[:k],self.y[:k]
            joint=self.detector.joint_component_loglikelihood(self.x[ids],y)
            self.estimates[k]=self.detector.posterior_from_components(self.factual_likelihood,ids,joint)
        return self.estimates[k]

    def feature(self,k):
        return features(self.fit(k),self.x,self.ids[:k],self.y[:k])


def allocate(histories,method,models):
    if method=='fixed4':
        return np.full(8,4,dtype=int)
    counts=np.full(8,2,dtype=int)
    active=np.arange(8)
    for stage,k in enumerate((2,4)):
        if method=='uncertainty':
            values=[histories[e].fit(k)['sd'][1] for e in active]
        else:
            values=[float(predict(models[method][stage],histories[e].feature(k))) for e in active]
        active=np.sort(active[np.argsort(-np.asarray(values),kind='stable')[:len(active)//2]])
        counts[active]=2*k
    assert sorted(counts.tolist())==[2,2,2,2,4,4,8,8]
    return counts


def decisions(gaps):
    order=np.argsort(-gaps,axis=-1,kind='stable')
    chosen=np.full(gaps.shape,Q,dtype=np.int8)
    np.put_along_axis(chosen,order[...,:13],P,axis=-1)
    return chosen


def decode_selected(decoder,x,chosen,available):
    probs=np.empty(len(x))
    for action in (Q,P):
        ix=np.flatnonzero(chosen==action)
        prior=decoder.inner.fit(t(x[ix]))
        posterior=decoder.inner.update(prior,action,t(available[action][ix]))
        probs[ix]=(posterior @ decoder.inner.label[:,0]).cpu().numpy()/np.sqrt(2.)
    return probs


def complete_call(detector,records,method,models,pair,decoders,xx,available):
    torch.cuda.synchronize()
    start=time.perf_counter()
    histories=[History(detector,record) for record in records]
    counts=allocate(histories,method,models)
    selectors=[PosteriorPredictiveEngine(pair,h.fit(int(k))) for h,k in zip(histories,counts)]
    torch.cuda.synchronize()
    warm_start=time.perf_counter()
    chosen=[];probs=[]
    for selector,decoder,x,ys in zip(selectors,decoders,xx,available):
        a=decisions(batch_gap(selector,x))
        chosen.append(a)
        probs.append(decode_selected(decoder,x,a,ys))
    torch.cuda.synchronize()
    end=time.perf_counter()
    return counts,np.array(chosen),np.array(probs),1000*(end-start),1000*(end-warm_start)


def measure(detector,records,models,pair,decoders,xx,available,counts,actions,probs,seed):
    full=np.zeros((3,len(METHODS)))
    warm=np.zeros_like(full)
    expected=np.take_along_axis(probs[:,:,:,0],(actions==P).astype(int),axis=2)
    rng=np.random.default_rng(seed)
    for rep in range(-1,3):
        for m in rng.permutation(len(METHODS)):
            k,a,pr,total_ms,deploy_ms=complete_call(detector,records,METHODS[m],models,pair,decoders,xx,available)
            np.testing.assert_array_equal(k,counts[:,m])
            np.testing.assert_array_equal(a,actions[:,:,m])
            np.testing.assert_allclose(pr,expected[:,:,m],atol=1.e-9,rtol=0)
            if rep>=0:
                full[rep,m]=total_ms;warm[rep,m]=deploy_ms
    return full,warm


def block_eval(detector,models,model_hash,fingerprint,cfg,law,index,b):
    folder=OUT/law/f'block_{b:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    meta_path=folder/'meta.json'
    if meta_path.exists():
        md=json.loads(meta_path.read_text())
        assert md['fingerprint']==fingerprint and md['model_sha256']==model_hash
        for stem,ext in [('choices','npz'),('metrics','npz'),('allocation','json')]:
            assert digest(folder/f'{stem}.{ext}')==md[stem+'_sha256']
        return
    start=time.perf_counter()
    base=510000000+(index//2)*1000000+b*10000
    alpha=np.asarray(cfg['alphas'])[np.random.default_rng(base+401).permutation(8)]
    pair=[base+301,base+302]
    records=[calibration(base+100+e,al,law) for e,al in enumerate(alpha)]
    xx=[deployment_x(base+200+e,al,law) for e,al in enumerate(alpha)]
    histories=[History(detector,record) for record in records]
    counts=np.stack([allocate(histories,method,models) for method in METHODS],axis=1)
    assert np.all(counts.sum(0)==32)
    alloc=folder/'allocation.json'
    atomic_json(alloc,jsonable(dict(methods=METHODS,counts=counts,model_sha256=model_hash)))
    alloc_hash=digest(alloc)
    # Auxiliary policy scores on all prefixes are computed only after the sequential path is fixed.
    feats=np.asarray([[h.feature(k) for k in STAGES[:2]] for h in histories])
    scores=np.stack([[predict(models[m][s],feats[:,s]) for m in METHODS[:2]] for s in range(2)]).transpose(2,0,1)
    uncertainty=np.array([[h.fit(k)['sd'][1] for k in STAGES[:2]] for h in histories])
    gaps=np.array([[batch_gap(PosteriorPredictiveEngine(pair,h.fit(k)),x) for k in STAGES]
                   for h,x in zip(histories,xx)])
    all_actions=decisions(gaps)
    selected=np.stack([all_actions[np.arange(8),np.searchsorted(STAGES,column)] for column in counts.T],axis=-1)
    assert np.all((selected==P).sum(1)==13)
    choice_path=folder/'choices.npz'
    atomic_npz(choice_path,gaps=gaps,counts=counts,actions=selected,features=feats,scores=scores,
               uncertainty=uncertainty,alpha=alpha)
    choice_hash=digest(choice_path)
    truths=[deployment_truth(base+200+e,al,law) for e,al in enumerate(alpha)]
    decoders=[Engine(pair,*al) for al in alpha]
    prob=[];cost=[]
    for x,decoder,(label,_,obs) in zip(xx,decoders,truths):
        c,p=potential_costs(x,label,obs,[decoder])
        prob.append(p[:,0]);cost.append(c[:,0])
    prob=np.array(prob);cost=np.array(cost)
    risk=np.take_along_axis(cost,(selected==P).astype(int),axis=2).mean(1)
    stage_risk=np.take_along_axis(cost,(all_actions.transpose(0,2,1)==P).astype(int),axis=2).mean(1)
    full=warm=np.empty((0,4))
    if b<cfg['timed_blocks_per_cell']:
        sensor=[{a:obs[a][:,0] for a in (Q,P)} for _,_,obs in truths]
        full,warm=measure(detector,records,models,pair,decoders,xx,sensor,counts,selected,prob,base+501+index%2)
    assert digest(alloc)==alloc_hash and digest(choice_path)==choice_hash
    out=folder/'metrics.npz'
    atomic_npz(out,methods=METHODS,counts=counts,actions=selected,labels=[z for z,_,_ in truths],
        nominal_labels=[z for _,z,_ in truths],probabilities=prob,brier=risk,oracle_stage_risk=stage_risk,
        scores=scores,uncertainty=uncertainty,features=feats,full_ms=full,warm_ms=warm,alpha=alpha)
    atomic_json(meta_path,jsonable(dict(law=law,block=b,model_sha256=model_hash,fingerprint=fingerprint,
        metrics_sha256=digest(out),choices_sha256=choice_hash,allocation_sha256=alloc_hash,
        bank_pair=pair,calibration_seeds=[base+100+i for i in range(8)],deployment_seeds=[base+200+i for i in range(8)],
        gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],elapsed_s=time.perf_counter()-start)))
    print('complete',law,b,round(time.perf_counter()-start,2),flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1 and torch.cuda.is_available()
    torch.set_num_threads(2)
    model_file=ROOT.parent/'compute_matched_stopping_v32/results_train/model_lock.json'
    lock=json.loads(model_file.read_text())
    cfg=json.loads((ROOT/'design_eval.json').read_text())
    assert list(METHODS)==cfg['methods'] and list(LAWS)==cfg['laws']
    files=[Path(__file__),ROOT/'target33.py',ROOT/'design_eval.json',model_file,
           ROOT.parent/'probe_gain_reliability_v31/batch31.py',ROOT.parent/'learned_probe_value_v30/contact30.py',
           ROOT.parent/'learned_probe_value_v30/score30.py']
    frozen='|'.join(digest(p) for p in files)
    h=digest(model_file)
    with torch.inference_mode():
        detector=ProbeShiftEstimator(seed=3003913,bank_size=8192)
        for b in range(cfg['blocks_per_law']):
            for i,law in enumerate(LAWS):
                block_eval(detector,lock['models'],h,frozen,cfg,law,i,b)
        assert h==digest(model_file)
        atomic_json(OUT/'completion.json',dict(complete_blocks=len(LAWS)*cfg['blocks_per_law'],model_sha256=h,
            fingerprint=frozen,job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name()))

if __name__=='__main__':
    main()
