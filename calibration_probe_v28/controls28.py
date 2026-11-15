"""Two exploratory budget-16 mechanism controls, evaluated on the original test worlds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from runner28 import (ROOT, ProbeShiftEstimator, PosteriorPredictiveEngine, atomic_json, atomic_npz,
                      digest, evaluation_only, potential_costs, select_all, stream, torch)
from estimator28 import independent_masses

NAMES = ("friction_only", "deranged_pairs")


def derangement(n, rng):
    """Uniform rejection sampling over permutations with no fixed points."""
    source = np.arange(n)
    while True:
        candidate = rng.permutation(n)
        if (candidate!=source).all():
            return candidate


def combine_marginals(original,paired):
    # Exact v27/b=0 profile marginal times the b=16 friction marginal. For a single
    # predicted world only their means determine these independent four flip masses.
    alpha = [original["alpha"][0],paired["alpha"][1]]
    result = {key:[original[key][0],paired[key][1]]
              for key in ("ci95","sd","posterior_prob_negative","posterior_kl_nats")}
    return dict(result,alpha=alpha,variant_probabilities=independent_masses(alpha).tolist(),
                covariance=0.,n_probes=16,profile_source="budget0",friction_source="budget16")


def evaluate(cfg, env, estimator, fingerprint):
    src = ROOT/"results"/env["id"]
    dest = ROOT/"results"/"controls"/env["id"]
    dest.mkdir(parents=True,exist_ok=True)
    src_choice = src/"choice_lock.npz"
    src_info = json.loads((src/"metrics.json").read_text())
    src_metrics = src/"metrics.npz"
    assert src_info["environment"]==env and digest(src_choice)==src_info["choices_sha256"]
    meta_file,choice_file,cost_file = [dest/name for name in ("metrics.json","choice_lock.npz","metrics.npz")]
    if meta_file.exists():
        cached = json.loads(meta_file.read_text())
        if (cached["fingerprint"]!=fingerprint or cached["main_npz_sha256"]!=src_info["npz_sha256"]
                or digest(choice_file)!=cached["choices_sha256"] or digest(cost_file)!=cached["npz_sha256"]):
            raise ValueError(f"Conflicting or corrupt supplementary run: {dest}")
        return

    with np.load(src_choice) as raw:
        x_cal,probe_order,y,x_test = [raw[name] for name in ("x_cal","probe_order","probe_y","x_test")]
    assert len(probe_order)==16 and len(np.unique(probe_order))==16 and y.shape==(16,2)
    shuffle = derangement(16,stream(env["seed"],281009))
    estimates = {
        "friction_only":combine_marginals(src_info["estimates"]["0"],src_info["estimates"]["16"]),
        "deranged_pairs":estimator.fit(x_cal,probe_order,y[shuffle]),
    }
    pair = cfg["bank_pairs"][env["bank_index"]]
    assert pair==src_info["bank_pair"]
    engines = [PosteriorPredictiveEngine(pair, estimates[key]) for key in NAMES]
    actions = select_all(x_test, engines)
    atomic_npz(choice_file,action=actions,permutation=shuffle,probe_order=probe_order,methods=NAMES)
    sealed = digest(choice_file)

    # Original evaluation file is read only after both new selectors have committed.
    assert digest(src_metrics)==src_info["npz_sha256"]
    with np.load(src_metrics) as prior:
        k = list(prior["methods"]).index("known")
        cost_known = prior["cost"][:,k]
        original_labels,original_cells = prior["label"],prior["cell"]
    label,cell,views = evaluation_only(len(x_test),cfg["repeats"],env["seed"],env["alpha"])
    assert np.array_equal(label,original_labels) and np.array_equal(cell,original_cells)
    cost,_ = potential_costs(x_test,label,views,engines)
    slots = (actions==3).astype(int)
    idx = np.arange(len(x_test))[:,None]
    native = cost[idx,np.arange(len(engines))[None,:],slots]
    common = cost_known[idx,slots]
    assert np.isfinite(native).all() and np.isfinite(common).all() and digest(choice_file)==sealed
    atomic_npz(cost_file,action=actions,loss=native,common_known=common,cost=cost,methods=NAMES)
    atomic_json(meta_file,dict(environment=env,methods=list(NAMES),estimates=estimates,
        bank_pair=pair,fingerprint=fingerprint,main_npz_sha256=src_info["npz_sha256"],
        main_choices_sha256=digest(src_choice),choices_sha256=sealed,npz_sha256=digest(cost_file),
        derangement=shuffle.tolist(),job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name()))
    print(env["id"],"controls loss",native.mean(0).round(5).tolist(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=str(ROOT/"design.json"))
    parser.add_argument("--shard",default=0,type=int)
    parser.add_argument("--shards",default=1,type=int)
    args=parser.parse_args()
    cfg_path=Path(args.config).resolve()
    cfg=json.loads(cfg_path.read_text())
    cases=[e for e in cfg["environments"] if e["target"] and e["rep"] < 3]
    assert len(cases)==24 and 0<=args.shard<args.shards
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.device_count()==1
    paths=[Path(__file__).resolve(),cfg_path,ROOT/"estimator28.py",ROOT/"runner28.py"]
    fingerprint=hashlib.sha256("".join(digest(p) for p in paths).encode()).hexdigest()
    torch.set_num_threads(2)
    with torch.inference_mode():
        estimator=ProbeShiftEstimator(seed=cfg["estimator_seed"],bank_size=cfg.get("estimator_bank_size",8192))
        for i,env in enumerate(cases):
            if i%args.shards==args.shard:
                evaluate(cfg,env,estimator,fingerprint)
if __name__=="__main__":
    main()
