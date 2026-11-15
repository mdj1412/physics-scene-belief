"""Frozen supervised gain allocation on new contact-law cohorts under an exact budget."""
import json
import os
import time

import numpy as np

from contact30 import (LAWS, V29, calibration_data, evaluation_data, factual, matched_decoder,
                       Q, P, Engine, t, torch)
from score30 import ROOT, features, predict, quota_actions, train
from runner29 import (ProbeShiftEstimator, PosteriorPredictiveEngine, atomic_json, atomic_npz,
                      digest, jsonable, timed_call)
from runner28 import potential_costs
from quota_latency29 import risk_gap, quota_prediction, stream_prediction

METHODS = ("learned","fixed4","uncertainty")
PREFIX = (2,4,8)
OUT = ROOT/"results"


class Calibration:
    def __init__(self, detector, seed, alpha, law):
        self.x, self.order, self.y = calibration_data(seed, alpha, law)
        self.ll, self.base_time = timed_call(detector.component_loglikelihood, self.x)
        self.detector = detector
        self.est, self.fit_time = {}, {}

    def fit(self,k):
        if k not in self.est:
            ids = self.order[:k]
            def calculate():
                joint=self.detector.joint_component_loglikelihood(self.x[ids], self.y[:k])
                return self.detector.posterior_from_components(self.ll,ids,joint)
            self.est[k],self.fit_time[k]=timed_call(calculate)
        return self.est[k]

    def score(self,k,model):
        return float(predict(model,features(self.fit(k), self.x, self.order[:k], self.y[:k])))


def allocations(views,models):
    learned = np.full((8,3),2)
    learned[:,1]=4
    score_time=np.zeros((8,3))
    for m in (0,2):
        active=np.arange(8)
        for stage,k in enumerate((2,4)):
            values=[]
            for e in active:
                view=views[e];est=view.fit(k)
                before=time.perf_counter()
                score=view.score(k,models[stage]) if m==0 else est['sd'][1]
                values.append(score)
                score_time[e,m]+=(time.perf_counter()-before)*1000
            before=time.perf_counter()
            keep=active[np.argsort(-np.array(values),kind='stable')[:len(active)//2]]
            score_time[active,m]+=(time.perf_counter()-before)*1000/len(active)
            learned[keep,m]=k*2
            active=np.sort(keep)
    assert np.all(learned.sum(0)==32)
    return learned,score_time


def timing(selectors,decoder,x,observations,stream_actions,batch_actions,probabilities,seed):
    rng=np.random.default_rng(seed)
    n_methods=len(selectors)
    yy={a:observations[a][:,0] for a in (Q,P)}
    def check(observed,chosen,worlds,method):
        act,pred=observed
        target=chosen[worlds,method]
        ref=probabilities[worlds,(target==P).astype(int),0]
        assert np.array_equal(act,target) and np.max(np.abs(pred-ref))<1e-8
    worlds=np.arange(len(x))
    batch=np.zeros((3,n_methods))
    for rep in range(-1,3):
        for m in rng.permutation(n_methods):
            measurement,ms=timed_call(quota_prediction,selectors[m],decoder,x,yy,13)
            check(measurement,batch_actions,worlds,m)
            if rep>=0:batch[rep,m]=ms
    idx=rng.choice(len(x),12,replace=False)
    stream=np.zeros((12,2,n_methods))
    for rep in range(-1,2):
        for j in (range(3) if rep<0 else rng.permutation(12)):
            i=idx[j]
            for m in rng.permutation(n_methods):
                measurement,ms=timed_call(stream_prediction,selectors[m],decoder,x[i],yy[Q][i],yy[P][i])
                check(measurement,stream_actions,i,m)
                if rep>=0:stream[j,rep,m]=ms
    return stream,batch,idx


def run(detector,trained,law,law_id,block,design,fingerprint):
    path=OUT/law/f'block_{block:03d}';path.mkdir(parents=True,exist_ok=True)
    meta=path/'metrics.json'
    if meta.exists():
        r=json.loads(meta.read_text());assert r['source_fingerprint']==fingerprint
        for nm,k in [('metrics.npz','metrics_sha256'),('allocation_lock.json','allocation_sha256'),('choices.npz','choices_sha256')]:
            assert digest(path/nm)==r[k]
        return
    rng=np.random.default_rng(np.random.SeedSequence([30091249,law_id,block]))
    alphas=np.asarray(design['alphas'])[rng.permutation(8)]
    seeds=rng.integers(1000,2**30,size=11)
    pair=seeds[-3:-1].tolist()
    env_timed=int(rng.integers(8)) if block<8 else -1
    views=[Calibration(detector,int(seed),alpha,law) for seed,alpha in zip(seeds,alphas)]
    counts,score_time=allocations(views,trained['models'])
    allocation_file=path/'allocation_lock.json'
    atomic_json(allocation_file,jsonable(dict(counts=counts,methods=METHODS,model=digest(OUT/'model_lock.json'),
        public_early_checks=[[v.fit(2)['alpha'],v.fit(2)['sd']] for v in views])))
    allocation_hash=digest(allocation_file)
    # Only now may the evaluator obtain stopped prefixes and build counterfactual ranking tables.
    scores=np.array([[v.score(k,m) for k,m in zip(PREFIX,trained['models'])] for v in views])
    gaps=[];engine_ms=[];bank_cache=[];xx=[]
    for e,v in enumerate(views):
        row=factual(128,int(seeds[e]),20,alphas[e],law);xx.append(row)
        engines=[];spent=[];columns=[]
        for k in PREFIX:
            en,ms=timed_call(PosteriorPredictiveEngine,pair,v.fit(k))
            engines.append(en);spent.append(ms)
            columns.append([risk_gap(en,x) for x in row])
        gaps.append(np.array(columns).T);engine_ms.append(spent)
        if e==env_timed:bank_cache=engines
    gap=np.stack(gaps)
    prefix_index=np.searchsorted(PREFIX,counts)
    all_quota=quota_actions(gap);all_stream=np.where(gap>1e-12,P,Q).astype(np.int8)
    ix=prefix_index[:,None,:]
    quota=np.take_along_axis(all_quota,ix,axis=2)
    streaming=np.take_along_axis(all_stream,ix,axis=2)
    assert np.all((quota==P).sum(1)==13)
    choice_file=path/'choices.npz'
    atomic_npz(choice_file,gap=gap,quota_action=quota,stream_action=streaming,counts=counts,scores=scores,
               heldout_x=xx,cal_x=[v.x for v in views],order=[v.order for v in views],probe_y=[v.y for v in views])
    choice_hash=digest(choice_file)
    costs=[];labs=[];source_labs=[];deltas=[];rawpred=[]
    timing_info={}
    for e,v in enumerate(views):
        z,z0,obs,shifts=evaluation_data(128,int(seeds[e]),alphas[e],law)
        nominal=Engine(pair,*alphas[e]); matched=matched_decoder(pair,alphas[e],law)
        raw,probs=potential_costs(xx[e],z,obs,[nominal,matched])
        costs.append(raw);labs.append(z);source_labs.append(z0);deltas.append(shifts);rawpred.append(probs)
        if e==env_timed:
            selectors=[bank_cache[j] for j in prefix_index[e]]
            st,bt,ids=timing(selectors,nominal,xx[e],obs,streaming[e],quota[e],probs[:,0],int(seeds[-1]))
            timing_info=dict(stream_ms=st,batch_ms=bt,timing_worlds=ids)
    assert digest(allocation_file)==allocation_hash and digest(choice_file)==choice_hash
    stage_fit=np.zeros((8,3))
    for e,v in enumerate(views):
        for m,k in enumerate(counts[e]):
            prefix=[4] if m==1 else [a for a in PREFIX if a<=k]
            stage_fit[e,m]=v.base_time+sum(v.fit_time[a] for a in prefix)
    build=np.take_along_axis(np.asarray(engine_ms),prefix_index,axis=1)
    data=path/'metrics.npz'
    atomic_npz(data,methods=METHODS,counts=counts,gap=gap,cost=costs,quota_action=quota,
        stream_action=streaming,labels=labs,source_labels=source_labs,scores=scores,probabilities=rawpred,
        uncertainty=[[v.fit(k)['sd'][1] for k in (2,4)] for v in views],
        factual_shift_norm=np.array(deltas)[:,0],probe_shift_norm=np.array(deltas)[:,1],alpha=alphas,
        env_timed=env_timed,setup_ms=stage_fit+score_time+build,fit_ms=stage_fit,
        score_ms=score_time,engine_ms=build,**timing_info)
    atomic_json(meta,jsonable(dict(block=block,law=law,source_fingerprint=fingerprint,counts=counts,
        model_sha256=digest(OUT/'model_lock.json'),allocation_sha256=allocation_hash,choices_sha256=choice_hash,
        metrics_sha256=digest(data),gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],
        host=os.environ.get('SLURMD_NODENAME'),alphas=alphas,seeds=seeds,bank_pair=pair,timed_env=env_timed)))
    print('completed',law,block,flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1 and torch.cuda.is_available()
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    design=json.loads((ROOT/'design.json').read_text())
    record=OUT/'model_lock.json'
    dependencies=[ROOT/p for p in ['score30.py','contact30.py','runner30.py','design.json']]+[
        V29/'runner29.py',V29/'quota_latency29.py',V29.parent/'calibration_probe_v28'/'estimator28.py',
        V29.parent/'calibration_probe_v28'/'runner28.py',V29.parent/'population_shift_v27'/'runner27.py',
        V29.parent/'within_calibration_v25'/'physics25.py',V29.parent/'proxy_shift_v26'/'physics26.py',
        V29.parent/'factual_tail_v23'/'physics23.py']
    fingerprint='|'.join(digest(f) for f in dependencies)
    with torch.inference_mode():
        if record.exists():
            trained=json.loads(record.read_text())
            if trained.get('source_fingerprint')!=fingerprint:
                raise RuntimeError('Frozen model belongs to another source version')
        else:
            trained=train()
            trained['source_fingerprint']=fingerprint
            atomic_json(record,jsonable(trained))
        model_hash=digest(record)
        detector,ms=timed_call(ProbeShiftEstimator,seed=3003913,bank_size=8192)
        print('Frozen model',model_hash,'estimator static ms',ms,flush=True)
        # Run fresh independent laws round-robin, without updating the frozen scorer.
        for block in range(design['blocks_per_law']):
            for law_id,law in enumerate(LAWS):
                run(detector,trained,law,law_id,block,design,fingerprint)
        assert model_hash==digest(record)
        atomic_json(OUT/'completion.json',dict(model_sha256=model_hash,detector_static_ms=ms,
            complete_blocks=2*design['blocks_per_law'],split_static=design['blocks_per_law'],
            speed_weak=design['blocks_per_law'],gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID']))

if __name__=='__main__':
    main()
