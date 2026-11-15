"""Evaluate six sealed policies on independent IID calibration and evaluation worlds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
V26 = ROOT.parent / "proxy_shift_v26"
sys.path.insert(0, str(V26))
from physics26 import (Engine, SOURCE, UNIT_DIM, change_proxy, independent_uniforms,
                       labels, observations, parameters, sigmas, t, torch, variants)
from physics25 import FACTUAL_INCOMING, FY, impact, profile
from run26 import frozen
from estimator27 import FactualShiftEstimator

METHODS = ("estimated", "frozen", "k256", "known", "query", "neutral")
DECODERS = ("estimated", "source", "known", "neutral")
NATIVE = np.array([0, 1, 1, 2, 1, 3])
Q, P = 1, 3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, values):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(values, indent=2, allow_nan=False))
    tmp.replace(path)


def atomic_npz(path, **values):
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    tmp.replace(path)


def timed_call(function, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def stream(seed, lane):
    return np.random.default_rng(np.random.SeedSequence([seed, 270000, lane]))


def draw_theta(n, seed, lane, alpha):
    # Independent width indicators. A fixed hidden quota would give the detector extra information.
    base = parameters(t(stream(seed, lane).random((n, UNIT_DIM))))
    u = independent_uniforms(seed, n, 270510 + lane)
    code = torch.as_tensor(variants(u, *alpha), device=base.device, dtype=torch.uint8)
    return change_proxy(base, code)


def factual_batch(n, seed, lane, alpha):
    theta = draw_theta(n, seed, lane, alpha)
    vel, sticking = impact(theta, 0, FACTUAL_INCOMING)
    assert sticking.all().item()
    raw = torch.cat([profile(theta, FY), vel], dim=-1)
    return (raw + t(stream(seed, lane+1).normal(size=(n,11))) * sigmas()[0]).cpu().numpy()


def evaluation_only(n, repeats, seed, alpha):
    theta = draw_theta(n, seed, 20, alpha)
    views = observations(theta)
    obs = {}
    for a in (Q, P):
        mu = views[a]
        z = t(stream(seed, 40+a).normal(size=(n,repeats,mu.shape[1])))
        obs[a] = (mu[:,None] + z * sigmas()[a]).cpu().numpy()
    return labels(theta).cpu().numpy(), theta[:,0].cpu().numpy().astype(int), obs


def seal_choices(x, engine, pred, cut):
    actions = np.ones((len(x),len(METHODS)), dtype=np.int8)
    score = np.asarray(pred.score(x))
    for i,v in enumerate(x):
        xv = t(v)
        source = engine["source"]
        state = source.fit(xv)
        actions[i,2] = source.choose_pair(state,256)
        if abs(score[i]) >= cut:
            actions[i,1] = Q if score[i] >= 0 else P
        else:
            actions[i,1] = source.choose_pair(state,1024)
        for j,name in [(0,"estimated"),(3,"known"),(5,"neutral")]:
            en = engine[name]
            actions[i,j] = actions[i,2] if en is source else en.choose_pair(en.fit(xv),256)
    return actions, score


def potential_costs(x, label, obs, engines):
    shape = (len(x),len(DECODERS),2)
    cost = np.empty(shape)
    repeats = obs[Q].shape[1]
    probabilities = np.empty((*shape,repeats))
    by_engine = {}
    for d,name in enumerate(DECODERS):
        engine = engines[name]
        if id(engine) in by_engine:
            old = by_engine[id(engine)]
            cost[:,d] = cost[:,old]
            probabilities[:,d] = probabilities[:,old]
            continue
        by_engine[id(engine)] = d
        inner = engine.inner
        for i,v in enumerate(x):
            w = inner.fit(t(v))
            for k,a in enumerate((Q,P)):
                probs = inner.update(w,a,t(obs[a][i])) @ inner.label[:,0] / np.sqrt(2.)
                probabilities[i,d,k] = probs.cpu().numpy()
            cost[i,d] = np.square(probabilities[i,d] - label[i]).mean(-1)
    return cost, probabilities


def online(method, engines, pred, cut, x, q, p):
    engine = engines[DECODERS[NATIVE[method]]]
    inner = engine.inner
    if method == 4:
        act = Q
        w = inner.fit(t(x))
    elif method == 1:
        margin = float(pred.score(x))
        if abs(margin) >= cut:
            act = Q if margin >= 0 else P
            w = inner.fit(t(x))
        else:
            state = engine.fit(t(x))
            act = engine.choose_pair(state,1024)
            w = state[0]
    else:
        state = engine.fit(t(x))
        act = engine.choose_pair(state,256)
        w = state[0]
    y = q if act == Q else p
    prob = inner.update(w,act,t(y)) @ inner.label[:,0] / np.sqrt(2.)
    return act, prob


def measure_online(x, obs, acts, probabilities, engines, pred, cut, config, seed):
    rng = stream(seed, 83)
    n = min(len(x),config["timing_worlds"])
    reps = config["timing_repeats"]
    ids = rng.choice(len(x),n,replace=False)
    times = np.empty((n,reps,len(METHODS)))
    for rep in range(-1,reps):
        slots = np.arange(min(n,3)) if rep < 0 else rng.permutation(n)
        for j in slots:
            i = ids[j]
            for k in rng.permutation(len(METHODS)):
                (act,prob),ms = timed_call(online,k,engines,pred,cut,x[i],obs[Q][i,0],obs[P][i,0])
                assert act == acts[i,k]
                expected = probabilities[i,NATIVE[k],int(act==P),0]
                assert abs(prob.item()-expected)<1e-9
                if rep >= 0:
                    times[j,rep,k] = ms
    assert np.isfinite(times).all() and (times>0).all()
    return times, ids


def evaluate(config, env, pred, cut, detector, setup_detector_ms, fingerprint):
    d = ROOT / "results" / env["id"]
    assert d.parent == ROOT/"results", "Environment identifier must be a single path component"
    d.mkdir(parents=True, exist_ok=True)
    result = d/"metrics.npz"
    json_file = d/"metrics.json"
    if json_file.exists():
        record = json.loads(json_file.read_text())
        if record["fingerprint"] != fingerprint or record["npz_sha256"] != digest(result):
            raise ValueError(f"Existing result belongs to another protocol or failed integrity check: {d}")
        return
    seed, alpha = env["seed"], tuple(env["alpha"])
    pair = tuple(config["bank_pairs"][env["bank_index"]])
    x_cal = factual_batch(config["n_cal"],seed,10,alpha)
    x_test = factual_batch(config["n_test"],seed,20,alpha)
    estimate, fit_ms = timed_call(detector.fit,x_cal)
    prefixes = {str(n):detector.fit(x_cal[:n]) for n in config.get("calibration_prefixes",[]) if n<len(x_cal)}
    settings = {"source":(1.,1.),"estimated":tuple(estimate["alpha"]),"known":alpha,"neutral":(0.,0.)}
    engine = {}
    setup = {}
    cache = {}
    for name,setting in settings.items():
        if setting not in cache:
            cache[setting] = timed_call(Engine,pair,*setting)
        engine[name], setup[name] = cache[setting]
    actions,score = seal_choices(x_test,engine,pred,cut)
    choicefile = d/"choice_lock.npz"
    atomic_npz(choicefile,action=actions,score=score,x_cal=x_cal,x_test=x_test,
               alpha_hat=estimate["alpha"],bank_pair=pair)
    choicehash = digest(choicefile)
    # Only the downstream evaluator now constructs the potential additional views and target.
    label,cell,obs = evaluation_only(len(x_test),config["repeats"],seed,alpha)
    cost,probabilities = potential_costs(x_test,label,obs,engine)
    loss = cost[np.arange(len(x_test))[:,None],NATIVE[None,:],(actions==P).astype(int)]
    assert np.isfinite(loss).all() and np.logical_and(0<=loss,loss<=1+1e-10).all()
    times,ids = measure_online(x_test,obs,actions,probabilities,engine,pred,cut,config,seed)
    assert choicehash==digest(choicefile)
    atomic_npz(result,action=actions,score=score,label=label,cell=cell,loss=loss,cost=cost,
               probabilities=probabilities,times_ms=times,timing_ids=ids,methods=METHODS,decoders=DECODERS)
    atomic_json(json_file,dict(fingerprint=fingerprint,npz_sha256=digest(result),choices_sha256=choicehash,
        environment=env,bank_pair=pair,estimate=estimate,prefixes=prefixes,
        setup_ms=dict(detector_static=setup_detector_ms,detector_fit=fit_ms,engines=setup),
        methods=list(METHODS),decoders=list(DECODERS),native_decoder_indices=NATIVE.tolist(),
        device=torch.cuda.get_device_name(),host=os.environ.get("SLURMD_NODENAME"),job=os.environ["SLURM_JOB_ID"]))
    print(env["id"],"done; estimate",estimate["alpha"],"loss",loss.mean(0).round(5).tolist(),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default=str(ROOT/"design.json"))
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    args = parser.parse_args()
    configfile = Path(args.config).resolve()
    config = json.loads(configfile.read_text())
    assert 0<=args.shard<args.shards and os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
    assert torch.cuda.device_count()==1, "Request a single Slurm GPU for each runner"
    torch.set_num_threads(2)
    oldplan = json.loads((SOURCE/"results/plan.json").read_text())
    selected = SOURCE/"results/selected.json"
    assert digest(selected)==oldplan["files"]["results/selected.json"]
    inputs = [Path(__file__).resolve(),ROOT/"estimator27.py",configfile,V26/"physics26.py",V26/"run26.py",selected]
    fingerprint = hashlib.sha256("".join(digest(p) for p in inputs).encode()).hexdigest()
    envs = config["environments"]
    assert len(set(e["id"] for e in envs))==len(envs)
    pred, cut = frozen()
    with torch.inference_mode():
        detector,init_ms = timed_call(FactualShiftEstimator, seed=config["estimator_seed"],
                                      bank_size=config.get("estimator_bank_size",8192))
        for i,env in enumerate(envs):
            if i % args.shards == args.shard:
                evaluate(config,env,pred,cut,detector,init_ms,fingerprint)

if __name__=="__main__":
    main()
