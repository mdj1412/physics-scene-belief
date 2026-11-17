"""Validate full-probability forecasts, homogeneous information and cold-free p90 measurement."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    clock=[]; groups=[]; max_error=0.; counts={"eligible":0,"probabilities":0,"clips":0}
    finite_rows=0
    unique_support=0
    for i in range(12):
        fn=ROOT/"forecast"/f"panel_{i:02d}.npz"
        assert hashlib.sha256(fn.read_bytes()).hexdigest()==json.loads(fn.with_suffix(".json").read_text())["sha256"]
        with np.load(fn) as p:
            prob=p["p_original"];other=p["p_projected"];act=p["selected"]
            assert prob.shape==(4,2,2,2,4,128,4) and other.shape==prob.shape
            query=(act==p["query_id"])
            mask=np.broadcast_to(query[:,None,:,:,:,:,None],prob.shape)
            measured = np.isfinite(p["query_paid"])
            assert np.all(measured==mask[...,None])
            np.testing.assert_allclose(prob[~mask],other[~mask],atol=1e-12,rtol=0)
            counts["eligible"]+=int(mask.sum());counts["probabilities"]+=prob.size*2
            counts["clips"]+=p["x_mm"].size
            counts_here=query.sum(axis=-1)
            assert np.all(counts_here==115)
            finite_rows+=int(measured[...,0].sum())
        with np.load(ROOT/"worlds"/f"panel_{i:02d}.npz") as pop:
            unique_support+=len(np.unique(pop["theta"].reshape(-1,6),axis=0))
        with np.load(ROOT/"timing"/f"panel_{i:02d}.npz") as t:
            clock.append(t["warm_ms"])
            assert np.all(np.sort(t["order"],axis=1)==[0,1,2])
            max_error=max(max_error,float(t["rerun_max_error"]))
            groups.append(t["cell"].tolist())
            names=t["methods"].tolist()
    clock=np.array(clock)
    point=np.quantile(clock,.9,axis=(0,1))
    idx=np.random.default_rng(4713913).integers(0,12,(16000,12))
    boot=np.quantile(clock[idx],.9,axis=(1,2))
    def bounds(a): return np.quantile(a,(.025,.975),axis=0).T.tolist()
    out=dict(methods=names,batch=512,units="ms",warm_per_panel=1,timed_per_panel=5,n_panels=12,
             p90=point.tolist(),p90_ci=bounds(boot),p90_difference_new_vs_refs=(point[2]-point[:2]).tolist(),
             p90_difference_ci=bounds(boot[:,2,None]-boot[:,:2]),bootstrap_resamples=16000,
             maximum_replay_error=max_error,counts=counts,finite_query_rows=finite_rows,
             n_distinct_within_catalog_sum=unique_support,timing_family_height_align_variant=groups,
             run={name:json.loads((ROOT/"runs"/(name+".json")).read_text()) for name in ("even","odd","timing")})
    (ROOT/"gpu_audit47.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:out[k] for k in ("p90","p90_ci","p90_difference_new_vs_refs","p90_difference_ci","counts","maximum_replay_error","n_distinct_within_catalog_sum")},indent=2))


if __name__=="__main__":
    main()
