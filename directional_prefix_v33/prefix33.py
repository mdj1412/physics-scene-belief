"""Prospective crossed gain measurement on disjoint discovery and confirmation worlds."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "results_prefix"
sys.path.insert(0, str(PROJECT / "probe_gain_reliability_v31"))
from runner31 import (actions, batch_gap, ProbeShiftEstimator, PosteriorPredictiveEngine, Engine,
                      atomic_json, atomic_npz, digest, jsonable, potential_costs, features)
from target33 import (LAWS, calibration_pool, deployment_x, deployment_truth, stream, sigmas, torch, P)
from prefix33_stats import freeze_winner, finalize

COHORTS = (("A",610_000_000),("B",710_000_000))
STAGES = (2,4)
BATCHES = FUTURES = 2


def prefix_branches(detector, alpha, home, pair, law):
    pool_seed = home + 1
    x, clean = calibration_pool(pool_seed, alpha, law)
    ids = stream(pool_seed,98).permutation(len(x))[:4]
    sigma = sigmas()[P].cpu().numpy()
    y = clean[ids] + stream(pool_seed,99).normal(size=(8,2))[:4]*sigma
    log_x = detector.component_loglikelihood(x)
    log_paid = detector.joint_component_loglikelihood(x[ids],y)
    stage_features, branches, extra_ids, extra_y, future_seeds = [], [], [], [], []
    for s, k in enumerate(STAGES):
        paid = ids[:k]
        root = detector.posterior_from_components(log_x,paid,log_paid[:k])
        stage_features.append(features(root,x,paid,y[:k]))
        estimates=[root]
        available=np.setdiff1d(np.arange(len(x)),paid)
        indices=[]; observed=[]; seeds=[]
        for f in range(FUTURES):
            seed=home+100+20*s+f
            selected=stream(seed,81).choice(available,size=k,replace=False)
            measured=clean[selected]+stream(seed,82).normal(size=(k,2))*sigma
            new_joint=detector.joint_component_loglikelihood(x[selected],measured)
            both_ids=np.r_[paid,selected]
            assert len(np.unique(both_ids))==2*k
            estimates.append(detector.posterior_from_components(log_x,both_ids,np.vstack((log_paid[:k],new_joint))))
            indices.append(selected); observed.append(measured); seeds.append(seed)
        branches.append(estimates); extra_ids.append(indices); extra_y.append(observed); future_seeds.append(seeds)
    batch_seeds=np.arange(BATCHES)+home+300
    factual=np.stack([deployment_x(int(seed),alpha,law) for seed in batch_seeds])
    shape=(len(STAGES),FUTURES+1,BATCHES,128)
    gaps=np.empty(shape)
    for s,stage in enumerate(branches):
        for f,estimate in enumerate(stage):
            selector=PosteriorPredictiveEngine(pair,estimate)
            gaps[s,f]=batch_gap(selector,factual.reshape(-1,11)).reshape(BATCHES,128)
    return dict(prefix_seed=pool_seed,original_ids=ids,paid_y=y,paid_x=x[ids],x_hash=hashlib.sha256(x.tobytes()).hexdigest(),
                future_seed=future_seeds,batch_seed=batch_seeds,features=stage_features,
                additional_2=extra_ids[0],additional_4=extra_ids[1],response_2=extra_y[0],response_4=extra_y[1],
                x=factual,action=actions(gaps),gap=gaps)


def measure_signed_block(detector,cohort,base_seed,family,block,dir_idx,alphas,fingerprint):
    start=time.perf_counter()
    law=LAWS[2*family+dir_idx]
    folder=OUT/cohort/law/f"block_{block:03d}"
    folder.mkdir(parents=True,exist_ok=True)
    record=folder/"meta.json"; cfile=folder/"choices.npz"; mfile=folder/"metrics.npz"
    if record.exists():
        meta=json.loads(record.read_text())
        assert meta["fingerprint"]==fingerprint and meta["choices_sha256"]==digest(cfile) and meta["metrics_sha256"]==digest(mfile)
        return
    base=base_seed+family*1_000_000+block*10_000
    pair=[base+8901,base+8902]
    reordered=alphas[np.random.default_rng(base+8903).permutation(8)]
    rows=[prefix_branches(detector,al,base+e*1000,pair,law) for e,al in enumerate(reordered)]
    combined={key:np.stack([r[key] for r in rows]) for key in rows[0]}
    atomic_npz(cfile,**combined,bank_pair=pair,alpha=reordered)
    choice_hash=digest(cfile)
    costs=[]; probabilities=[]; truths=[]
    for alpha,row in zip(reordered,rows):
        engine=Engine(pair,*alpha)
        cc=[]; pp=[]; ll=[]
        for x,seed in zip(row["x"],row["batch_seed"]):
            labels,_,ys=deployment_truth(int(seed),alpha,law)
            cost,prediction=potential_costs(x,labels,ys,[engine])
            cc.append(cost[:,0]); pp.append(prediction[:,0]); ll.append(labels)
        costs.append(cc); probabilities.append(pp); truths.append(ll)
    cost=np.array(costs)
    selected=combined["action"]==P
    risks=np.where(selected,cost[:,None,None,:,:,1],cost[:,None,None,:,:,0]).mean(-1)
    gain=risks[:,:,:1]-risks[:,:,1:]
    switched=(selected[:,:,:1]!=selected[:,:,1:]).mean(-1)
    assert digest(cfile)==choice_hash and np.all(selected.sum(-1)==13)
    atomic_npz(mfile,gain=gain,switch=switched,risk=risks,cost=cost,probabilities=probabilities,labels=truths)
    torch.cuda.synchronize()
    meta=dict(fingerprint=fingerprint,law=law,cohort=cohort,block=block,base=base,bank_pair=pair,
              choices_sha256=choice_hash,metrics_sha256=digest(mfile),wall_s=time.perf_counter()-start,
              gpu=torch.cuda.get_device_name(),job=os.environ["SLURM_JOB_ID"])
    atomic_json(record,jsonable(meta))
    print(cohort,law,block,f'{meta["wall_s"]:.2f}s',flush=True)


def main():
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    config=json.loads((ROOT/"design_eval.json").read_text())
    alphas=np.asarray(config["alphas"])
    files=[ROOT/n for n in ("PREFIX_PLAN_KO.md","prefix33.py","prefix33_stats.py","target33.py","design_eval.json")]
    for mod in ("runner31","batch31","contact30","runner29","runner28","estimator28","runner27",
                "score30","physics25"):
        files.append(Path(sys.modules[mod].__file__))
    fingerprint=hashlib.sha256("|".join(digest(p) for p in files).encode()).hexdigest()
    OUT.mkdir(exist_ok=True)
    detector=ProbeShiftEstimator(seed=3003913,bank_size=8192)
    for cohort,base in COHORTS:
        for b in range(4):
            for family in range(4):
                for di in range(2):
                    measure_signed_block(detector,cohort,base,family,b,di,alphas,fingerprint)
        if cohort=="A":
            freeze_winner()
    finalize()
    atomic_json(OUT/"completion.json",dict(fingerprint=fingerprint,blocks=64,independent_clusters=32,
                gpu=torch.cuda.get_device_name(),job=os.environ["SLURM_JOB_ID"]))
if __name__=="__main__":
    with torch.inference_mode():
        main()
