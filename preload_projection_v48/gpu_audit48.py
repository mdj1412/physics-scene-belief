"""Verify saved decoders, sensor masks, station parity, online replay, and timing."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from route48 import load_policy, decision

ROOT=Path(__file__).resolve().parent
MODES=("fixed","preload","none","unsigned")


def main():
    assert os.getenv("SLURM_JOB_ID") and not torch.cuda.is_available()
    torch.set_num_threads(1)
    clock=[]; cells=[]; freq=[]; unique=0; max_err=0.; n=0; n_eligible=0; n_decisions=0
    mismatch={m:0 for m in MODES}; n_pairs=0
    model=load_policy("cpu")
    with torch.inference_mode():
        for i in range(12):
            path=ROOT/"forecast"/f"panel_{i:02d}.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads(path.with_suffix(".json").read_text())["sha256"]
            with np.load(path) as a:
                orig,pro,act,q=a["p_original"],a["p_projected"],a["selected"],a["query_paid"]
                assert orig.shape==pro.shape==(4,2,2,2,4,128,4)
                assert np.isfinite(orig).all() and np.isfinite(pro).all()
                eligible=np.broadcast_to((act==a["query_id"])[:,None,:,:,:,:,None],orig.shape)
                assert np.all(np.isfinite(q)==eligible[...,None])
                np.testing.assert_allclose(orig[~eligible],pro[~eligible],atol=1e-12,rtol=0)
                assert np.all((act==a["query_id"]).sum(-1)==115)
                n+=orig.size
                n_eligible+=int(eligible.sum())
                F,A,H,V,E,O,N=orig.shape
                readings=a["readings_mm"]; assert readings.shape==(F,H,3,2,E,O,N)
                heights=a["heights_mm"].tolist()
                fixed_i=heights.index(a["fixed_height_mm"])
                active={m:a["gate_"+m] for m in MODES}
                for h,height in enumerate(heights):
                    for s in range(3):
                        for mode in MODES:
                            station=fixed_i if mode=="fixed" else h
                            obs=readings[:,h,s,station][:,None,None]
                            if mode=="none": obs=np.zeros_like(obs)
                            expected=decision(q[:,:,h],obs,height,mode,model).numpy()
                            mismatch[mode]+=int((expected!=active[mode][:,:,h,:,s]).sum())
                    if h==fixed_i:
                        n_pairs+=int((active["fixed"][:,:,h]!=active["preload"][:,:,h]).sum())
                n_decisions+=active["fixed"].size
            with np.load(ROOT/"worlds"/f"panel_{i:02d}.npz") as w:
                unique+=len(np.unique(w["theta"].reshape(-1,6), axis=0))
            with np.load(ROOT/"timing"/f"panel_{i:02d}.npz") as a:
                assert a["warm_ms"].shape==(5,4) and np.all(a["warm_ms"]>0)
                assert np.all(np.sort(a["order"],axis=1)==np.arange(4))
                clock.append(a["warm_ms"]);cells.append(a["cell"].tolist());freq.append(a["route_freq"].tolist())
                methods=a["methods"].tolist();max_err=max(max_err,float(a["rerun_max_error"]))
    assert not any(mismatch.values()) and n_pairs==0
    clock=np.array(clock)
    point=np.quantile(clock,.9,axis=(0,1))
    idx=np.random.default_rng(4813997).integers(0,12,(16000,12))
    boot=np.quantile(clock[idx],.9,axis=(1,2))
    def bounds(x): return np.quantile(x,(.025,.975),axis=0).T.tolist()
    diff=point[2]-point[[0,1,3]]
    db=boot[:,2,None]-boot[:,[0,1,3]]
    result=dict(methods=methods,p90_ms=point.tolist(),p90_ci_ms=bounds(boot),new_minus=["always_original","fixed","none"],
                p90_diff_ms=diff.tolist(),p90_diff_ci_ms=bounds(db),num_boot=16000,
                n_panels=12,objects_per_batch=512,one_warm_per_panel=True,observed_reps_per_panel=5,
                batch_cells_f_a_h_v=cells,projection_rate_timing=freq,max_single_pass_probability_error=max_err,
                n_probabilities=2*n,n_distinct_prototypes=unique,n_query_draws=n_eligible,
                n_inherited_probe_draws=n-n_eligible,n_route_slots_per_method=n_decisions,
                mismatches_cpu_gpu=mismatch,mismatches_at_same_station=n_pairs,
                runs={k:json.loads((ROOT/"runs"/(k+".json")).read_text()) for k in ("even","odd","timing")})
    (ROOT/"gpu_audit48.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("methods","p90_ms","p90_ci_ms","p90_diff_ms","p90_diff_ci_ms","mismatches_cpu_gpu","n_distinct_prototypes","max_single_pass_probability_error")},indent=1))


if __name__=="__main__":
    main()
