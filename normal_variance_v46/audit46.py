"""Independently audit probability archives, then compute panel-bootstrap p90 contrasts."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
METHODS=("always_original","legacy_x","refit_x","x_normal")

def archive():
    count=0; unique=0; err=0.; low=1.; high=0.
    for panel in range(12):
        folder=f"panel_{panel:02d}"
        with np.load(OUT/"worlds"/(folder+".npz")) as a:
            theta=a["theta"]; ids=a["object_index"]
            assert a["support_panel"]==600+panel and a["seed"]==460000000+100000*panel
            assert theta.shape==(4,128,6)
            unique+=sum(len(np.unique(x)) for x in ids)
        files=sorted((OUT/folder).glob("*.npz"))
        assert len(files)==48
        for path in files:
            assert path.name.split("_")[0] in ("terrace","dimple")
            assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads(path.with_suffix(".json").read_text())["sha256"]
            with np.load(path) as a:
                pred=a["pred"][:,0]
                assert pred.shape==(2,4,128,4) and tuple(a["decoders"])==("original","source_axis")
                np.testing.assert_array_equal(a["theta"],theta)
                act=a["selected"]
                for sensor,field,ns in ((1,"query_paid",115),(3,"probe_paid",13)):
                    assert np.all((act==sensor).sum(axis=1)==ns)
                    assert np.isfinite(a[field][act==sensor]).all() and np.isnan(a[field][act!=sensor]).all()
                assert np.isfinite(pred).all()
                err=max(err,float(abs(pred[1,act==3]-pred[0,act==3]).max()))
                low=min(low,pred.min());high=max(high,pred.max())
            count+=1
    assert err < 2e-9 and low >= -1e-12 and high<=1+1e-12
    return dict(archives=count,draws=12*512,unique_prototypes=unique,probabilities=count*512*4*2,
                p_range=[low,high],max_probe_probability_delta=err)

def summarize(folder):
    walls=[];features=[];errors=[];rates=[]
    for n in range(12):
        with np.load(folder/f"panel_{n:02d}.npz") as a:
            assert tuple(a["methods"])==METHODS and a["image_bytes"]==512*3*32*96
            np.testing.assert_array_equal(np.sort(a["order"],axis=1),np.tile(np.arange(4),(4,1)))
            walls.append(a["warm_ms"]);features.append(a["feature_ms"]);errors.append(float(a["err"]));rates.append(a["rates"])
    w=np.stack(walls); f=np.stack(features)
    p=np.quantile(w,.9,axis=(0,1))
    samples=np.random.default_rng(4601002).integers(0,12,(16000,12))
    q=np.quantile(w[samples],.9,axis=(1,2))
    ci=lambda vals:np.quantile(vals,[.025,.975],axis=0).T
    return dict(methods=METHODS,p90_ms=p.tolist(),p90_95_ci=ci(q).tolist(),p90_feature_ms=np.quantile(f,.9,axis=(0,1)).tolist(),
                p90_diff_vs_all_ms=(p[-1]-p).tolist(),diff_vs_all_95_ci=ci(q[:,-1:]-q).tolist(),
                ratio_vs_all=(p[-1]/p).tolist(),ratio_vs_all_95_ci=ci(q[:,-1:]/q).tolist(),max_replay_error=max(errors),
                panel_medians_ms=np.median(w,axis=1).tolist(),route_freq=np.mean(rates,axis=0).tolist(),reps=4)
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("archive","latency46","latency46_repeat"))
    mode=ap.parse_args().mode
    result=archive() if mode=="archive" else summarize(ROOT/mode)
    (ROOT/("audit_"+mode+".json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="panel_medians_ms"}))
