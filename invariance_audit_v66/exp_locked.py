"""Apply previously frozen source heads to a new target; separate forecast and score stages."""
import argparse
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.stats import t
import torch
from exp_readout import ROOT, REPS, SEEDS, DEVICE, Head, coordinates, digest, inputs, labels

LOCK=ROOT/"locked"
FAMS=("wendland","dispersive")
OUT=ROOT/"exp_locked_preds"
MANIFEST=ROOT/"exp_locked_forecasts.json"
PRIMARY=(("restored_minus_dropped",4,3),("full_minus_dropped",2,3))


def cells():
    for f,s,l,v,h in itertools.product(FAMS, ("inlay","woven"), ("rational","gaussian_rate"),
                                       ("base","hollow","bulge"), (19,31)):
        yield h, f"{f}_{s}_{l}_{v}_h{h}.npz"


def intervals(a,m=1):
    mean=a.mean(0)
    width=a.std(0,ddof=1)*t.ppf(1-.025/m,len(a)-1)/np.sqrt(len(a))
    return np.stack((mean,mean-width,mean+width),axis=-1)


def audit_models():
    for line in (ROOT/"exp_locked_plan.sha256").read_text().splitlines():
        expected,rel=line.split(maxsplit=1)
        assert digest(ROOT/rel)==expected, rel


@torch.inference_mode()
def predict():
    assert torch.cuda.device_count()==1
    # No target outcome should be generated before the full forecast archive is sealed.
    assert not any((LOCK/"labels").glob("panel_*/*.npz"))
    meta=json.loads((ROOT/"exp_fit.json").read_text())
    nets=[]
    for rep,seed in itertools.product(REPS, SEEDS):
        key=f"{rep}_{seed}"
        path=ROOT/f"exp_fit_{key}.pt"
        assert digest(path)==meta["models"][key]["sha256"]
        state=torch.load(path,map_location=DEVICE,weights_only=True)
        net=Head().to(DEVICE)
        net.load_state_dict(state["weights"])
        nets.append((rep,net.eval(),state["mean"],state["scale"]))
    OUT.mkdir(exist_ok=True)
    manifest={}
    shifts=[]
    for k in range(12):
        block=f"panel_{k:02d}"
        rows=[]
        sources={}
        panel_ood=[]
        for h,cell in cells():
            path=LOCK/"results"/block/cell
            features,mask,common,_=inputs(path,h,4)
            src=torch.as_tensor(features,device=DEVICE)
            p=np.broadcast_to(common,(len(nets),*common.shape)).astype(np.float32,copy=True)
            for j,(rep,net,mu,scale) in enumerate(nets):
                z=(coordinates(src,rep)-mu)/scale
                p[j][mask]=net(z,src[:,12]).cpu().numpy()
                if j==0: # componentwise fraction among seven unrotated paid geometry channels
                    panel_ood.append([float((z[:,-7:].abs()>3.).float().mean()),
                                      float((z[:,-7:].abs()>5.).float().mean()),
                                      float(z[:,-7:].abs().max())])
            assert np.isfinite(p).all() and 0<=p.min() and p.max()<=1.
            rows.append(p)
            sources[cell]=digest(path)
        dest=OUT/f"{block}.npz"
        np.savez_compressed(dest,pred=np.stack(rows),cells=[c for h,c in cells()])
        manifest[block]=dict(sha256=digest(dest),input_sha256=sources)
        shifts.append(panel_ood)
    stats=np.array(shifts)
    MANIFEST.write_text(json.dumps(dict(by_panel=manifest,utc_unix=time.time(),job=os.environ["SLURM_JOB_ID"],
        families=FAMS,input_component_fractions_over3sd_and_5sd=stats.mean((0,1))[:2].tolist(),
        max_standardized_height_channel=float(stats[:,:,2].max()),fit_meta_sha256=digest(ROOT/"exp_fit.json"),
        protocol_manifest_sha256=digest(ROOT/"exp_locked_plan.sha256")),indent=2)+"\n")
    print("Forecast manifest",digest(MANIFEST),flush=True)


def score():
    with (ROOT/"exp_locked_seal.sha256").open() as f:
        for line in f:
            expected,rel=line.split(maxsplit=1)
            assert digest(ROOT/rel)==expected
    panelmeta=json.loads(MANIFEST.read_text())["by_panel"]
    totals=[]
    counts=[]
    for k in range(12):
        block=f"panel_{k:02d}"
        path=OUT/f"{block}.npz"
        assert digest(path)==panelmeta[block]["sha256"]
        with np.load(path) as f:
            probs=f["pred"]
            np.testing.assert_array_equal(f["cells"],[c for h,c in cells()])
        rows=[]
        for i,(_,cell) in enumerate(cells()):
            obs=LOCK/"results"/block/cell
            assert digest(obs)==panelmeta[block]["input_sha256"][cell]
            with np.load(obs) as f:
                frozen=f["pred"][:2,0]
                paid=np.isfinite(f["query_paid"]).all(-1)
            lab=LOCK/"labels"/block/cell
            y=labels(lab)
            pred=np.concatenate([frozen,probs[i]],axis=0).astype(float)
            np.testing.assert_allclose(pred[:,~paid],np.broadcast_to(pred[:1,~paid],pred[:,~paid].shape),rtol=0.,atol=4e-8)
            cost=pred*pred+(1.-2*pred)*y[...,None]
            if i==0:
                with np.load(lab) as f:
                    all_y=f["labels"][1:]
                np.testing.assert_allclose(cost,((pred[:,None,None]-all_y[None,...,None])**2).mean((1,2)),atol=2e-14)
            rows.append(np.stack((cost.mean((1,2,3)),cost[:,paid].mean(1)),axis=1))
            counts.append([int(paid.sum()),int(paid.size)])
        totals.append(np.array(rows).reshape(2,24,11,2).mean(1))
    vals=np.stack(totals)
    means=np.concatenate([vals[:,:,:2],vals[:,:,2:].reshape(12,2,3,3,2).mean(3)],axis=2)
    avg=means.mean(1)
    primary={n:intervals(avg[:,i]-avg[:,j],2) for n,i,j in PRIMARY}
    result=dict(families=FAMS,methods=["frozen_full","frozen_projected",*REPS],metrics=["all","purchased query"],
                absolute=intervals(avg),primary_simultaneous_ci=primary,practical_delta=.002,
                direction_and_practical_pass={n:[bool(v[0,2]<0),bool(v[0,2]<=-.002)] for n,v in primary.items()},
                family_absolute=intervals(means),family_restored_minus_dropped=intervals(means[:,:,4]-means[:,:,3],2),
                restored_minus_full=intervals(avg[:,4]-avg[:,2]),seed_risks=vals[:,:,2:].reshape(12,2,3,3,2).mean((0,1)),
                counts_query_total=np.sum(counts,axis=0),panel_losses=means,forecasts_sha256=digest(MANIFEST),
                source_sha256=digest(ROOT/"exp_fit.json"),job=os.environ["SLURM_JOB_ID"])
    (ROOT/"exp_locked_results.json").write_text(json.dumps(result,indent=2,default=lambda v:np.asarray(v).tolist())+"\n")
    print("all risk",avg.mean(0)[:,0],"contrasts",primary,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=("predict","score"))
    stage=ap.parse_args().stage
    assert os.getenv("SLURM_JOB_ID")
    audit_models()
    if stage=="predict":
        predict()
    else:
        score()
