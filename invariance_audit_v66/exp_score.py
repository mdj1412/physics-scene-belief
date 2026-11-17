"""Score committed target predictions, keeping catalog as the independent sampling unit."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t
from exp_readout import ROOT, TARGET, REPS, SEEDS, cells, digest, labels


def ci(x, multiplicity=1):
    mean = x.mean(0)
    width = x.std(0,ddof=1)*t.ppf(1-.025/multiplicity,len(x)-1)/np.sqrt(len(x))
    return np.stack((mean,mean-width,mean+width),axis=-1)


def run():
    assert os.getenv("SLURM_JOB_ID")
    commit = json.loads((ROOT/"exp_forecasts.json").read_text())
    panel_scores, fractions = [], []
    for k in range(12):
        block=f"panel_{k:02d}"
        path = ROOT/"exp_forecasts"/f"{block}.npz"
        assert digest(path) == commit[block]
        with np.load(path) as archive:
            preds=archive["pred"]
            np.testing.assert_array_equal(archive["cells"], [cell for _,_,cell in cells(TARGET)])
        risks, mask_fractions = [], []
        for idx, (_, _, cell) in enumerate(cells(TARGET)):
            with np.load(TARGET/"results"/block/cell) as data:
                frozen=data["pred"][:2,0]
                paid=np.isfinite(data["query_paid"]).all(-1)
            truth=labels(TARGET/"labels46"/block/cell)
            ps=np.concatenate([frozen, preds[idx]],axis=0).astype(float)
            loss=ps**2+(1-2*ps)*truth[...,None]
            if idx==0:
                with np.load(TARGET/"labels46"/block/cell) as data:
                    actual=data["labels"][1:]
                direct = ((ps[:,None,None]-actual[None,...,None])**2).mean((1,2))
                np.testing.assert_allclose(loss,direct,atol=4e-15)
            risks.append(np.stack([loss.mean(axis=(1,2,3)),loss[:,paid].mean(1)],axis=1))
            mask_fractions.append(float(paid.mean()))
        panel_scores.append(np.array(risks).reshape(2,24,11,2).mean(1))
        fractions.append(mask_fractions)
    arr=np.stack(panel_scores)
    # panel, family, methods (two frozen arms then 3 x 3), all/query-only
    collapsed=arr.mean(1)
    official=json.loads((TARGET/"score46.json").read_text())["absolute"]
    np.testing.assert_allclose(collapsed[:,:2,0].mean(0),[official[0][0],official[6][0]],atol=2e-12)
    methods=["legacy_full","legacy_projected",*REPS]
    means=np.concatenate([arr[:,:,:2],arr[:,:,2:].reshape(12,2,3,3,2).mean(3)],axis=2)
    pooled=means.mean(1)
    contrasts=dict(restored_minus_dropped=pooled[:,4]-pooled[:,3],full_minus_dropped=pooled[:,2]-pooled[:,3])
    result=dict(methods=methods, metrics=["all purchased actions","query only"],
        absolute=ci(pooled), primary_simultaneous={n:ci(v,2) for n,v in contrasts.items()},
        family=["terrace","dimple"],family_absolute=ci(means),
        family_restored_minus_dropped=ci(means[:,:,4]-means[:,:,3],2),
        restored_minus_full=ci(pooled[:,4]-pooled[:,2]),
        learned_minus_legacy=dict(full=ci(pooled[:,2]-pooled[:,0]),dropped=ci(pooled[:,3]-pooled[:,1])),
        paired_catalog_losses=means,seed_risks=collapsed[:,2:,:].reshape(12,3,3,2).mean(0),
        paid_query_fraction=np.mean(fractions), source_seeds=SEEDS, support_ids=[600+k for k in range(12)],
        heldout_query_rows=sum(np.array(fractions).ravel()*2048).round(), target_object_draws=6144,
        hash_forecasts_manifest=digest(ROOT/"exp_forecasts.json"), job=os.environ["SLURM_JOB_ID"])
    (ROOT/"exp_results.json").write_text(json.dumps(result,indent=2,default=lambda x:np.asarray(x).tolist())+"\n")
    print("full / dropped / restored (all)",pooled.mean(0)[:,0],flush=True)
    print("primary",result["primary_simultaneous"],flush=True)


if __name__=="__main__":
    run()
