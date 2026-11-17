"""Panel-paired p90 comparison for the two identical measurement passes."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
out={}
for folder in ("results/timing","timing_repeat45/timing"):
    warm=[];loc=[];error=[];order=[]
    for n in range(12):
        with np.load(root/folder/f"panel_{n:02d}.npz") as a:
            methods=a["methods"].tolist()
            warm.append(a["warm_ms"]);loc.append(a["location_ms"]);error.append(float(a["max_error"]));order.append(a["order"])
    if folder.startswith("results"):
        initial_order=order
    else:
        np.testing.assert_array_equal(initial_order,order)
    assert max(error)<2e-9
    w=np.stack(warm)
    rng=np.random.default_rng(45011053)
    resampled=w[rng.integers(0,12,(16000,12))]
    q=np.quantile(w,.9,axis=(0,1));boot=np.quantile(resampled,.9,axis=(1,2))
    ratios=boot/boot[:,:1]
    ratios90=np.quantile(ratios,[.025,.975],axis=0).T
    ci=np.quantile(boot,[.025,.975],axis=0).T
    delta=boot[:,2]-boot[:,1]
    out[folder]=dict(methods=methods,p90_ms=q.tolist(),p90_95_ci=ci.tolist(),relative_to_original=(q/q[0]).tolist(),
                     relative_to_original_95_ci=ratios90.tolist(),visual_minus_oracle_ms=[q[2]-q[1],*np.quantile(delta,[.025,.975])],
                     visual_minus_original_ms=[q[2]-q[0],*np.quantile(boot[:,2]-boot[:,0],[.025,.975])],
                     location_p90_ms=np.quantile(np.stack(loc),.9,axis=(0,1)).tolist(),
                     per_panel_median=np.median(w,axis=1).tolist(),max_error=max(error))
(root/"timing_comparison45.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:{q:v[q] for q in ('p90_ms','p90_95_ci','visual_minus_oracle_ms','visual_minus_original_ms','location_p90_ms')} for k,v in out.items()}))
