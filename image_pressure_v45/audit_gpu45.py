"""Verify the model archives and summarize panel-paired crossover latencies."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"

def main():
    methods=("always_original","oracle_x","visual_x","matched_oracle_x","matched_visual_x")
    max_probe_error=0.
    unique=0
    files=0
    for n in range(12):
        with np.load(OUT/"worlds"/f"panel_{n:02d}.npz") as x:
            theta=x["theta"];ids=x["object_index"]
            unique+=sum(len(np.unique(row)) for row in ids)
            assert x["support_panel"]==500+n
            assert x["seed"]==450000000+n*100000 and theta.shape==(4,128,6)
        names=list((OUT/f"panel_{n:02d}").glob("*.npz"))
        assert len(names)==48
        for p in names:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads(p.with_suffix(".json").read_text())["sha256"]
            with np.load(p) as x:
                assert set(x.files)=={"theta","xx","selected","pred","decoders","policies","query_paid","probe_paid"}
                np.testing.assert_array_equal(x["theta"],theta)
                assert x["decoders"].tolist()==["original","source_axis","privileged"]
                act=x["selected"]; pred=x["pred"][:,0]
                assert pred.shape==(3,4,128,4) and np.isfinite(pred).all()
                assert pred.min()>-1e-12 and pred.max()<1.+1e-12
                assert np.all((act==3).sum(axis=1)==13) and np.all((act==1).sum(axis=1)==115)
                for f,sensor in [("query_paid",1),("probe_paid",3)]:
                    assert np.isfinite(x[f][act==sensor]).all() and np.isnan(x[f][act!=sensor]).all()
                max_probe_error=max(max_probe_error,float(abs(pred[1:,act==3]-pred[:1,act==3]).max()))
                files+=1
    warm=[]; locate=[]; error=[]; route=[]
    for n in range(12):
        with np.load(OUT/"timing"/f"panel_{n:02d}.npz") as x:
            assert list(x["methods"])==list(methods)
            np.testing.assert_array_equal(np.sort(x["order"],axis=1),np.tile(np.arange(5),(3,1)))
            warm.append(x["warm_ms"]);locate.append(x["location_ms"]);error.append(float(x["max_error"]))
            route.append(x["projection_share"])
            assert x["image_bytes"]==512*3*32*96
    warm=np.stack(warm);locate=np.stack(locate)
    rng=np.random.default_rng(45011053);idx=rng.integers(0,12,(16000,12))
    val=np.quantile(warm,.90,axis=(0,1))
    boot=np.quantile(warm[idx],.90,axis=(1,2))
    ci=np.quantile(boot,[.025,.975],axis=0).T
    ratio=val/val[0]
    ratio_ci=np.quantile(boot/boot[:,:1],[.025,.975],axis=0).T
    summary=dict(methods=methods,device=json.loads((OUT/"run_timing.json").read_text())["device"],
                 probability_files=files,probabilities=files*3*4*128*4,prototype_draws=12*4*128,
                 unique_prototypes=int(unique),max_probe_prediction_delta=max_probe_error,
                 max_mixed_decoder_replay_error=max(error),warm_p90_ms=val.tolist(),
                 warm_p90_ci=ci.tolist(),ratio_to_always=ratio.tolist(),ratio_to_always_ci=ratio_ci.tolist(),
                 p90_location_transfer_and_lookup_ms=np.quantile(locate,.90,axis=(0,1)).tolist(),
                 timing_route_rate=np.mean(route,axis=0).tolist(),
                 paired_five_policy_repetitions_per_panel=3,bootstrap_panels=12,bootstrap_reps=len(idx))
    assert files==576 and max_probe_error<2e-9 and max(error)<2e-9
    (ROOT/"audit_gpu45.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    main()
