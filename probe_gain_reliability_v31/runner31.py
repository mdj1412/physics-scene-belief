"""Cross independent paid continuations with independent deployment batches at frozen prefixes."""
import json
import os
import time
from pathlib import Path

import numpy as np
from batch31 import batch_gap, compare_reference
from contact30 import (calibration_data, factual, evaluation_data, means, sigmas, draw_theta,
                       stream, t, torch, Engine, V29)
from runner29 import ProbeShiftEstimator, PosteriorPredictiveEngine, atomic_json, atomic_npz, digest, jsonable
from runner28 import potential_costs
from score30 import features, predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
V30=ROOT.parent/'learned_probe_value_v30'
F,B,W=4,4,128
STAGES=(2,4)


def continuation(k,seed,alpha,law):
    theta=draw_theta(k,seed,60,alpha)
    mu=means(theta,law)
    x=mu[0]+t(stream(seed,61).normal(size=(k,11)))*sigmas()[0]
    y=mu[3]+t(stream(seed,62).normal(size=(k,2)))*sigmas()[3]
    return x.cpu().numpy(),y.cpu().numpy()


def actions(gaps):
    decisions=np.ones_like(gaps,dtype=np.int8)
    ix=np.argsort(-gaps,axis=-1,kind='stable')[...,:13]
    np.put_along_axis(decisions,ix,3,axis=-1)
    assert ((decisions==3).sum(-1)==13).all()
    return decisions


def run_block(detector,law,li,block,design,models,model_hash,fingerprint):
    folder=OUT/law/f'block_{block:03d}';folder.mkdir(parents=True,exist_ok=True)
    record=folder/'metrics.json';paths=[folder/n for n in ('prefix.npz','choices.npz','metrics.npz')]
    if record.exists():
        meta=json.loads(record.read_text())
        assert meta['fingerprint']==fingerprint
        assert all(digest(p)==meta[k+'_sha256'] for p,k in zip(paths,('prefix','choices','metrics')))
        return
    start=time.perf_counter()
    base = 310000000 + li*1000000 + block*10000
    pair=[base+8901,base+8902]
    alphas=np.array(design['alpha_panel'])[np.random.default_rng(base+8903).permutation(8)]
    pool_x=[];prefix_ids=[];prefix_y=[];scores=[];uncs=[];all_est=[];x_only=[];branches=[];base_seeds=[]
    for e,alpha in enumerate(alphas):
        home=base+e*1000
        base_seeds.append(home+1)
        x,ids,y=calibration_data(home+1,alpha,law)
        ids,y=ids[:4],y[:4]
        pool_x.append(x);prefix_ids.append(ids);prefix_y.append(y)
        ll=detector.component_loglikelihood(x)
        joint=detector.joint_component_loglikelihood(x[ids],y)
        ests=[];roots=[];sd=[];ext=[];xests=[]
        for s,k in enumerate(STAGES):
            before=detector.posterior_from_components(ll,ids[:k],joint[:k])
            roots.append(predict(models[s],features(before,x,ids[:k],y[:k])))
            sd.append(before['sd'][1])
            estimates=[before];pairs=[];controls=[]
            for j in range(F):
                cseed=home+100+s*20+j
                xx,yy=continuation(k,cseed,alpha,law)
                lx=detector.component_loglikelihood(xx)
                ly=detector.joint_component_loglikelihood(xx,yy)
                # Added records only contribute a paired likelihood, replacing their single
                # factual factor. The original 1024 records are identical in all branches.
                newid=np.r_[ids[:k],len(x)+np.arange(k)]
                est=detector.posterior_from_components(np.vstack([ll,lx]),newid,np.vstack([joint[:k],ly]))
                estimates.append(est)
                if block==0:
                    controls.append(detector.posterior_from_components(np.vstack([ll,lx]),ids[:k],joint[:k]))
                pairs.append(np.concatenate([xx,yy],axis=1))
            ext.append(np.stack(pairs));ests.append(estimates);xests.append(controls)
        all_est.append(ests);x_only.append(xests);scores.append(roots);uncs.append(sd);branches.append(ext)
    future_seeds=np.array([[[base+e*1000+100+s*20+j for j in range(F)] for e in range(8)] for s in range(2)])
    batch_seeds=np.array([[base+e*1000+300+j for j in range(B)] for e in range(8)])
    atomic_npz(paths[0],factual_pool=pool_x,paid_ids=prefix_ids,paid_y=prefix_y,pool_seeds=base_seeds,
               future_seeds=future_seeds,batch_seeds=batch_seeds,bank_pair=pair,alpha=alphas,
               additional_2=np.stack([a[0] for a in branches]),additional_4=np.stack([a[1] for a in branches]),
               predictive_masses=np.array([[[v['variant_probabilities'] for v in group] for group in env]
                                            for env in all_est]),score=np.array(scores).T,uncertainty=np.array(uncs).T)
    prefix_hash=digest(paths[0]);factual_rows=[];gaps=np.empty((2,8,1+F,B,W));checks=[]
    control_gaps=np.empty((2,8,F,B,W)) if block==0 else None
    for e,alpha in enumerate(alphas):
        xx=np.stack([factual(W,int(seed),20,alpha,law) for seed in batch_seeds[e]])
        factual_rows.append(xx)
        for s in range(2):
            for i,est in enumerate(all_est[e][s]):
                engine=PosteriorPredictiveEngine(pair,est)
                if e==0 and block==0 and i in (0,1):
                    checks.append(compare_reference(engine,xx[0]))
                gaps[s,e,i]=batch_gap(engine,xx.reshape(-1,11)).reshape(B,W)
            for j,est in enumerate(x_only[e][s]):
                engine=PosteriorPredictiveEngine(pair,est)
                control_gaps[s,e,j]=batch_gap(engine,xx.reshape(-1,11)).reshape(B,W)
    choice=actions(gaps)
    controls={} if block else dict(factual_only_action=actions(control_gaps),factual_only_gap=control_gaps)
    atomic_npz(paths[1],gap=gaps,action_before=choice[:,:,0],action_after=choice[:,:,1:],
               deployment_x=factual_rows,alpha=alphas,**controls)
    choice_hash=digest(paths[1]);labels=[];prob=[];costs=[]
    for e,alpha in enumerate(alphas):
        dec=Engine(pair,*alpha)
        cc=[];pp=[];labs=[]
        for b,seed in enumerate(batch_seeds[e]):
            z,_,obs,_=evaluation_data(W,int(seed),alpha,law)
            cost,p=potential_costs(factual_rows[e][b],z,obs,[dec])
            cc.append(cost[:,0]);pp.append(p[:,0]);labs.append(z)
        costs.append(cc);prob.append(pp);labels.append(labs)
    cost=np.array(costs)
    risks=np.where(choice==3,cost[None,:,None,:,:,1],cost[None,:,None,:,:,0]).mean(-1)
    before=risks[:,:,0];after=risks[:,:,1:];gain=before[:,:,None]-after
    assert digest(paths[0])==prefix_hash and digest(paths[1])==choice_hash
    decomposition={}
    if block==0:
        only=np.where(controls['factual_only_action']==3,cost[None,:,None,:,:,1],cost[None,:,None,:,:,0]).mean(-1)
        decomposition=dict(factual_only_brier=only,gain_factual_only=before[:,:,None]-only,gain_probe_only=only-after)
    atomic_npz(paths[2],gain=gain,before=before,after=after,alpha=alphas,score=np.array(scores).T,
               uncertainty=np.array(uncs).T,cost=cost,probabilities=prob,labels=labels,**decomposition)
    payload=dict(fingerprint=fingerprint,law=law,block=block,prefix_sha256=prefix_hash,
        choices_sha256=choice_hash,metrics_sha256=digest(paths[2]),model_sha256=model_hash,
        alpha=alphas,bank_pair=pair,pool_seeds=base_seeds,future_seeds=future_seeds,batch_seeds=batch_seeds,
        reference_max_errors=checks,elapsed_s=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),
        job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename)
    atomic_json(record,jsonable(payload))
    print('completed',law,block,round(payload['elapsed_s'],2),'seconds',flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1 and torch.cuda.is_available()
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    design=json.loads((ROOT/'design.json').read_text())
    model_file=V30/'results/model_lock.json';model_hash=digest(model_file)
    models=json.loads(model_file.read_text())['models']
    dependencies=[ROOT/'runner31.py',ROOT/'batch31.py',ROOT/'design.json',model_file,
        V30/'contact30.py',V30/'score30.py',V29/'runner29.py',V29/'quota_latency29.py',
        V29.parent/'calibration_probe_v28'/'runner28.py',V29.parent/'calibration_probe_v28'/'estimator28.py',
        V29.parent/'tail_recovery_v22'/'policy22.py',V29.parent/'factual_tail_v23'/'physics23.py']
    fingerprint='|'.join(digest(p) for p in dependencies)
    with torch.inference_mode():
        detector=ProbeShiftEstimator(seed=3003913,bank_size=8192)
        print('v31 start, frozen configuration',digest(ROOT/'design.json'),'GPU',torch.cuda.get_device_name(),flush=True)
        for block in range(design['blocks_per_law']):
            for li,law in enumerate(design['laws']):
                run_block(detector,law,li,block,design,models,model_hash,fingerprint)
        atomic_json(OUT/'completion.json',dict(blocks=design['blocks_per_law']*len(design['laws']),
            fingerprint=fingerprint,model_sha256=model_hash,gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
