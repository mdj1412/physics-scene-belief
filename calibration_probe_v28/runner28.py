"""Compare nested paid calibration probes with neutral and environment-known K256 policies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
V27 = ROOT.parent / "population_shift_v27"
sys.path.insert(0, str(V27))
from runner27 import (P, Q, Engine, atomic_json, atomic_npz, digest, draw_theta, evaluation_only,
                      factual_batch, impact, sigmas, stream, t, timed_call, torch)
from physics25 import PROBE_INCOMING
from estimator28 import ProbeShiftEstimator, PosteriorPredictiveEngine

BUDGETS = (0, 1, 4, 16)
METHODS = ("probe0", "probe1", "probe4", "probe16", "neutral", "known")


def calibration_probe_measurements(n_cal, seed, alpha, ordered_ids):
    """Simulator environment: the selected calibration row has the same latent state as its factual X."""
    theta = draw_theta(n_cal, seed, 10, alpha)
    selected = torch.as_tensor(ordered_ids, dtype=torch.long, device=theta.device)
    signal, sticks = impact(theta[selected], 0, PROBE_INCOMING)
    assert not sticks.any().item()
    noise = t(stream(seed, 99).normal(size=tuple(signal.shape)))
    return (signal + noise * sigmas()[P]).cpu().numpy()


def select_all(x, engines):
    action = np.empty((len(x),len(engines)),dtype=np.int8)
    for i, row in enumerate(x):
        value = t(row)
        for j, engine in enumerate(engines):
            action[i,j] = engine.choose_pair(engine.fit(value),256)
    return action


def potential_costs(x, labels, obs, engines):
    """Retain the full paired potential-cost table for native and common-decoder comparisons."""
    probabilities = np.empty((len(x),len(engines),2,obs[Q].shape[1]))
    for j, engine in enumerate(engines):
        inner = engine.inner
        for i,row in enumerate(x):
            posterior = inner.fit(t(row))
            for k, action in enumerate((Q, P)):
                preds = inner.update(posterior,action,t(obs[action][i])) @ inner.label[:,0]/np.sqrt(2.)
                probabilities[i,j,k] = preds.cpu().numpy()
    return np.square(probabilities-labels[:,None,None,None]).mean(axis=-1), probabilities


def online(engine, x, observations):
    state = engine.fit(t(x))
    action = engine.choose_pair(state,256)
    prediction = engine.inner.update(state[0],action,t(observations[action])) @ engine.inner.label[:,0]/np.sqrt(2.)
    return action,prediction


def measure_online(x, obs, engines, actions, probabilities, config, seed):
    rng = stream(seed,83)
    n_worlds = min(config["timing_worlds"],len(x))
    repeats = config["timing_repeats"]
    ids = rng.choice(len(x),n_worlds,replace=False)
    times = np.empty((n_worlds,repeats,len(engines)))
    for repeat in range(-1,repeats):
        order = np.arange(min(3,n_worlds)) if repeat<0 else rng.permutation(n_worlds)
        for j in order:
            i = ids[j]
            views = {a:obs[a][i,0] for a in (Q,P)}
            for k in rng.permutation(len(engines)):
                (act,pred),latency = timed_call(online,engines[k],x[i],views)
                assert act==actions[i,k]
                assert abs(pred.item()-probabilities[i,k,int(act==P),0])<1e-9
                if repeat>=0:
                    times[j,repeat,k] = latency
    assert np.isfinite(times).all() and (times>0).all()
    return times,ids


def evaluate(cfg,env,estimator,static_ms,fingerprint):
    folder = ROOT/"results"/env["id"]
    assert folder.parent == ROOT/"results"
    folder.mkdir(parents=True,exist_ok=True)
    info,cost_file,choices_file = [folder/name for name in ("metrics.json","metrics.npz","choice_lock.npz")]
    if info.is_file():
        record = json.loads(info.read_text())
        if (record["fingerprint"]!=fingerprint or record["npz_sha256"]!=digest(cost_file)
                or record["choices_sha256"]!=digest(choices_file)):
            raise RuntimeError(f"Fingerprint or data mismatch: {folder}")
        return

    seed = env["seed"]
    alpha = tuple(env["alpha"])
    pair = tuple(cfg["bank_pairs"][env["bank_index"]])
    n_cal = cfg["n_cal"]
    ordered_ids = stream(seed,98).permutation(n_cal)[:max(BUDGETS)]
    assert len(np.unique(ordered_ids))==max(BUDGETS)
    x_cal = factual_batch(n_cal,seed,10,alpha)
    y_cal = calibration_probe_measurements(n_cal,seed,alpha,ordered_ids)
    x_test = factual_batch(cfg["n_test"],seed,20,alpha)
    engines = []
    estimates = {}
    fits = {}
    setups = {}
    # Each policy pays full cold-in-environment fit, including the shared factual likelihood.
    for budget in BUDGETS:
        estimate,fit_ms = timed_call(estimator.fit,x_cal,ordered_ids[:budget],y_cal[:budget])
        engine,setup_ms = timed_call(PosteriorPredictiveEngine,pair,estimate)
        key = str(budget)
        fits[key], estimates[key], setups["probe"+key] = fit_ms,estimate,setup_ms
        engines.append(engine)
    for name,setting in (("neutral",(0.,0.)),("known",alpha)):
        engine,setups[name] = timed_call(Engine,pair,*setting)
        engines.append(engine)
    actions = select_all(x_test,engines)
    assert np.isin(actions,(Q,P)).all()
    atomic_npz(choices_file,x_cal=x_cal,x_test=x_test,probe_order=ordered_ids,probe_y=y_cal,
               action=actions,alpha_hat=np.array([estimates[str(k)]["alpha"] for k in BUDGETS]),bank_pair=pair)
    sealed = digest(choices_file)

    # Held-out label and hypothetical views are first materialized after all six methods are sealed.
    label,cell,observations = evaluation_only(len(x_test),cfg["repeats"],seed,alpha)
    costs,predictions = potential_costs(x_test,label,observations,engines)
    assert np.isfinite(costs).all() and ((costs>=0)&(costs<=1+1e-10)).all()
    slots = (actions==P).astype(int)
    rows = np.arange(len(label))[:,None]
    native = costs[rows,np.arange(len(METHODS))[None,:],slots]
    common = costs[rows,METHODS.index("known"),slots]
    timing,ids = measure_online(x_test,observations,engines,actions,predictions,cfg,seed)
    assert digest(choices_file)==sealed
    atomic_npz(cost_file,action=actions,label=label,cell=cell,loss=native,common_known=common,
               cost=costs,probabilities=predictions,times_ms=timing,timing_ids=ids,methods=METHODS)
    atomic_json(info,dict(environment=env,estimates=estimates,bank_pair=list(pair),methods=list(METHODS),
        budgets=list(BUDGETS),calibration_probe_counts=[*BUDGETS,0,0],n_cal=n_cal,n_test=len(x_test),
        setup_ms=dict(detector_static=static_ms,detector_fit=fits,engines=setups),
        fingerprint=fingerprint,npz_sha256=digest(cost_file),choices_sha256=sealed,
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),host=os.environ.get("SLURMD_NODENAME")))
    print(env["id"],"Brier",native.mean(0).round(5).tolist(),"friction",
          [round(estimates[str(k)]["alpha"][1],4) for k in BUDGETS],flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default=str(ROOT/"design.json"))
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    args = parser.parse_args()
    config = Path(args.config).resolve()
    cfg = json.loads(config.read_text())
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() and torch.cuda.device_count()==1
    assert 0<=args.shard<args.shards
    assert cfg["n_cal"]>=max(BUDGETS) and cfg["timing_worlds"]>0 and cfg["timing_repeats"]>0
    assert len({e["id"] for e in cfg["environments"]})==len(cfg["environments"])
    torch.set_num_threads(2)
    deps = [Path(__file__).resolve(),ROOT/"estimator28.py",config,V27/"estimator27.py",V27/"runner27.py",
            ROOT.parent/"proxy_shift_v26"/"physics26.py",ROOT.parent/"within_calibration_v25"/"physics25.py"]
    fingerprint = hashlib.sha256("".join(digest(p) for p in deps).encode()).hexdigest()
    with torch.inference_mode():
        estimator,static_ms = timed_call(ProbeShiftEstimator,seed=cfg["estimator_seed"],
                                         bank_size=cfg.get("estimator_bank_size",8192))
        for i,env in enumerate(cfg["environments"]):
            if i%args.shards==args.shard:
                evaluate(cfg,env,estimator,static_ms,fingerprint)

if __name__=="__main__":
    main()
