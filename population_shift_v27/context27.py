"""Post-protocol context-size and known-profile diagnostics on four fixed paired laws."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from runner27 import (ROOT, V26, SOURCE, Engine, FactualShiftEstimator, Q, P, atomic_json,
                      atomic_npz, digest, evaluation_only, factual_batch, t, timed_call, torch)

FAMILIES = ("neg_neg", "neg_pos", "pos_neg", "pos_pos")
METHODS = ("large_estimated", "known_profile")
CONTEXT_COUNT = 16384


def checked_record(directory):
    record = json.loads((directory/"metrics.json").read_text())
    if digest(directory/"metrics.npz") != record["npz_sha256"]:
        raise ValueError(f"Changed main results at {directory}")
    if digest(directory/"choice_lock.npz") != record["choices_sha256"]:
        raise ValueError(f"Changed sealed main choices at {directory}")
    return record


def evaluate(config, env, estimator, static_ms, fingerprint):
    base = ROOT/"results"/env["id"]
    main = checked_record(base)
    assert main["environment"]==env
    folder = ROOT/"results"/"context27"/env["id"]
    folder.mkdir(parents=True, exist_ok=True)
    result = folder/"metrics.npz"
    info = folder/"metrics.json"
    if info.is_file():
        existing = json.loads(info.read_text())
        if (existing["fingerprint"]!=fingerprint or existing["npz_sha256"]!=digest(result)
                or main["npz_sha256"]!=existing["main_npz_sha256"]):
            raise ValueError(f"Conflicting cached result {folder}")
        return
    # New generator domain and fresh seed independent of the original (calibration/test) draws.
    seed = int(np.random.SeedSequence([env["seed"],270592011]).generate_state(1)[0])
    assert seed not in {candidate["seed"] for candidate in config["environments"]}
    alpha = tuple(env["alpha"])
    x_cal = factual_batch(CONTEXT_COUNT,seed,74,alpha)
    with np.load(base/"choice_lock.npz") as locked:
        x = locked["x_test"]
    fit, fit_ms = timed_call(estimator.fit,x_cal)
    assert fit["n_worlds"]==CONTEXT_COUNT and fit["alpha"][1]==0
    settings = (fit["alpha"], [alpha[0],0.])
    pair = tuple(config["bank_pairs"][env["bank_index"]])
    assert list(pair)==main["bank_pair"]
    engines = []
    setup = []
    for setting in settings:
        en, ms = timed_call(Engine,pair,*setting)
        engines.append(en)
        setup.append(ms)
    actions = np.empty((len(x),len(engines)),np.int8)
    for j,en in enumerate(engines):
        for i,v in enumerate(x):
            actions[i,j] = en.choose_pair(en.fit(t(v)),256)
    choicefile = folder/"choice_lock.npz"
    atomic_npz(choicefile,action=actions,alpha=settings,x_cal=x_cal)
    choicehash = digest(choicefile)
    # Outcomes are loaded only after sealing these two policies' choices.
    with np.load(base/"metrics.npz") as prior:
        oracle = list(prior["decoders"]).index("known")
        common_cost = prior["cost"][:,oracle,:]
        previous_labels, previous_cells = prior["label"], prior["cell"]
    label, cell, observations = evaluation_only(len(x),config["repeats"],env["seed"],alpha)
    assert np.array_equal(label,previous_labels) and np.array_equal(cell,previous_cells)
    unique = np.arange(len(x))[:,None]
    picked = (actions==P).astype(int)
    common_known = common_cost[unique,picked]
    cost = np.empty((len(x),len(engines),2))
    for j,en in enumerate(engines):
        inner = en.inner
        for i,v in enumerate(x):
            w = inner.fit(t(v))
            for k,a in enumerate((Q,P)):
                prob = inner.update(w,a,t(observations[a][i])) @ inner.label[:,0] / np.sqrt(2.)
                cost[i,j,k] = (prob-float(label[i])).square().mean().item()
    loss = cost[unique,np.arange(len(METHODS)),picked]
    assert np.isfinite(loss).all() and np.isfinite(common_known).all()
    assert choicehash==digest(choicefile)
    atomic_npz(result,methods=METHODS,action=actions,cost=cost,loss=loss,common_known=common_known)
    atomic_json(info,dict(environment=env,methods=METHODS,estimate=fit,alpha=settings,context_seed=seed,
        n_cal=CONTEXT_COUNT,context_lane=74,bank_pair=pair,npz_sha256=digest(result),fingerprint=fingerprint,
        choices_sha256=choicehash,main_npz_sha256=main["npz_sha256"],main_choices_sha256=main["choices_sha256"],
        setup_ms=dict(detector_static=static_ms,detector_fit=fit_ms,engines=dict(zip(METHODS,setup))),
        job=os.environ["SLURM_JOB_ID"],node=os.environ.get("SLURMD_NODENAME"),device=torch.cuda.get_device_name()))
    print(env["id"],"large context",fit["alpha"],"loss",loss.mean(0).round(5).tolist(),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default=str(ROOT/"design.json"))
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    args = parser.parse_args()
    cfg = Path(args.config).resolve()
    config = json.loads(cfg.read_text())
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() and torch.cuda.device_count()==1
    assert 0<=args.shard<args.shards
    torch.set_num_threads(2)
    envs = [env for env in config["environments"] if env["family"] in FAMILIES]
    assert len(envs)==12 and all({e["rep"] for e in envs if e["family"]==f}=={0,1,2} for f in FAMILIES)
    sources = [Path(__file__).resolve(),ROOT/"runner27.py",ROOT/"estimator27.py",cfg,V26/"physics26.py",
               SOURCE/"physics25.py"]
    fingerprint = hashlib.sha256("".join(digest(p) for p in sources).encode()).hexdigest()
    with torch.inference_mode():
        estimator,static_ms = timed_call(FactualShiftEstimator,seed=config["estimator_seed"],
                                         bank_size=config.get("estimator_bank_size",8192))
        for i,env in enumerate(envs):
            if i%args.shards==args.shard:
                evaluate(config,env,estimator,static_ms,fingerprint)

if __name__=="__main__":
    main()
