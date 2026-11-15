"""V30-matched secondary panel: buy probe responses on the already seen finite factual pool."""
import json
import os
import time
import numpy as np
from runner31 import (ROOT, OUT, STAGES, F, B, W, actions, batch_gap, draw_theta, means,
                      stream, sigmas, torch, t, atomic_json, atomic_npz, digest, jsonable,
                      ProbeShiftEstimator, PosteriorPredictiveEngine)

DEST=ROOT/'results_pool'


def run_block(detector,law,li,block,fingerprint):
    source=OUT/law/f'block_{block:03d}'
    folder=DEST/law/source.name
    folder.mkdir(exist_ok=True,parents=True)
    record=folder/'metrics.json'
    if record.exists():
        md=json.loads(record.read_text())
        assert md['fingerprint']==fingerprint and md['metrics_sha256']==digest(folder/'metrics.npz')
        assert md['choices_sha256']==digest(folder/'choices.npz')
        return
    start=time.perf_counter()
    parent=json.loads((source/'metrics.json').read_text())
    for k in ('choices','prefix','metrics'):
        assert digest(source/(k+'.npz'))==parent[k+'_sha256']
    pair=parent['bank_pair']
    with np.load(source/'prefix.npz') as z:
        pools=z['factual_pool'];paid_ids=z['paid_ids'];paid_y=z['paid_y']
        seeds=z['pool_seeds'];alphas=z['alpha']
        scores=z['score'];unc=z['uncertainty']
    with np.load(source/'choices.npz') as z:
        factual=z['deployment_x'];before_choice=z['action_before']
    gaps=np.empty((2,8,F,B,W))
    overlaps=np.zeros((2,8),int)
    ids_per_stage=[np.empty((8,F,k),int) for k in STAGES]
    y_per_stage=[np.empty((8,F,k,2)) for k in STAGES]
    allseeds=np.empty((2,8,F),int)
    for e,(seed,alpha,pool) in enumerate(zip(seeds,alphas,pools)):
        # Only the environment generating purchased sensor responses accesses these latents.
        theta=draw_theta(len(pool),int(seed),10,alpha)
        measured=means(theta,law)
        noise=t(stream(int(seed),11).normal(size=tuple(measured[0].shape)))
        check=(measured[0]+noise*sigmas()[0]).cpu().numpy()
        assert np.allclose(check,pool,rtol=0,atol=1.e-12)
        probe_signal=measured[3].cpu().numpy()
        ll=detector.component_loglikelihood(pool)
        for s,k in enumerate(STAGES):
            pids=paid_ids[e,:k]
            py=paid_y[e,:k]
            available=np.setdiff1d(np.arange(len(pool)),pids)
            appeared=[]
            for j in range(F):
                cseed=321000000+li*1000000+block*10000+e*1000+100+s*20+j
                allseeds[s,e,j]=cseed
                ids=stream(cseed,81).choice(available,size=k,replace=False)
                y=probe_signal[ids]+stream(cseed,82).normal(size=(k,2))*sigmas()[3].cpu().numpy()
                acquired=np.r_[pids,ids]
                y_all=np.vstack([py,y])
                assert len(set(acquired))==2*k
                joint=detector.joint_component_loglikelihood(pool[acquired],y_all)
                posterior=detector.posterior_from_components(ll,acquired,joint)
                engine=PosteriorPredictiveEngine(pair,posterior)
                gaps[s,e,j]=batch_gap(engine,factual[e].reshape(-1,11)).reshape(B,W)
                ids_per_stage[s][e,j]=ids;y_per_stage[s][e,j]=y;appeared.extend(ids)
            overlaps[s,e]=len(appeared)-len(set(appeared))
    act=actions(gaps)
    choice_file=folder/'choices.npz'
    atomic_npz(choice_file,gap_after=gaps,action_before=before_choice,action_after=act,
               ids_2=ids_per_stage[0],ids_4=ids_per_stage[1],responses_2=y_per_stage[0],
               responses_4=y_per_stage[1],future_seeds=allseeds,overlap_excess=overlaps)
    choice_hash=digest(choice_file)
    # Previously locked outcomes only opened after the new branch selections are immutable.
    with np.load(source/'metrics.npz') as z:
        costs=z['cost'];before=z['before']
        assert np.allclose(before,np.where(before_choice==3,costs[None,:,:,:,1],costs[None,:,:,:,0]).mean(-1))
    after=np.where(act==3,costs[None,:,None,:,:,1],costs[None,:,None,:,:,0]).mean(-1)
    gain=before[:,:,None]-after
    assert digest(choice_file)==choice_hash
    metric_file=folder/'metrics.npz'
    atomic_npz(metric_file,gain=gain,before=before,after=after,alpha=alphas,score=scores,
               uncertainty=unc,overlap_excess=overlaps)
    md=dict(fingerprint=fingerprint,law=law,block=block,source_folder=str(source.relative_to(ROOT)),
            metrics_sha256=digest(metric_file),choices_sha256=choice_hash,prefix_sha256=parent['prefix_sha256'],
            primary_choices_sha256=parent['choices_sha256'],primary_metrics_sha256=parent['metrics_sha256'],
            alpha=alphas,future_seeds=allseeds,bank_pair=pair,model_sha256=parent['model_sha256'],
            elapsed_s=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'])
    atomic_json(record,jsonable(md))
    print('complete finite pool',law,block,round(md['elapsed_s'],2),'seconds',flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1 and torch.cuda.is_available()
    torch.set_num_threads(2)
    assert json.loads((OUT/'completion.json').read_text())['blocks']==18
    dest=ROOT/'gpu_worker/results/worker_exit.json'
    assert json.loads(dest.read_text())['code']==0
    cfg=json.loads((ROOT/'design_pool.json').read_text());DEST.mkdir(exist_ok=True)
    fingerprint='|'.join(digest(ROOT/f) for f in ['design_pool.json','runner_pool31.py','batch31.py','runner31.py','results/completion.json'])
    with torch.inference_mode():
        det=ProbeShiftEstimator(seed=3003913,bank_size=8192)
        for b in range(cfg['blocks_per_law']):
            for li,law in enumerate(cfg['laws']):run_block(det,law,li,b,fingerprint)
    atomic_json(DEST/'completion.json',dict(blocks=18,fingerprint=fingerprint,
        gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
