"""Apply the three frozen CPU head families to the separately released contour archive."""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import moment_readout as m
from exp_score import ci

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "locked"
FAMILIES = ("wendland","dispersive")
CELLS = [(height, f"{family}_{surf}_{law}_{shape}_h{height}.npz") for family,surf,law,shape,height
         in itertools.product(FAMILIES,("inlay","woven"),("rational","gaussian_rate"),
                              ("base","hollow","bulge"),(19,31))]


@torch.inference_mode()
def forecast():
    assert not (LOCK / "labels").exists(), "A label directory already exists: abandon prospective status."
    models = json.loads((ROOT/"moment_fit.json").read_text())["models"]
    heads = []
    for rep, seed in itertools.product(m.METHODS, m.base.SEEDS):
        name = f"moment_fit_{rep}_{seed}.pt"
        assert m.base.digest(ROOT/name) == models[name]["sha256"]
        state = torch.load(ROOT/name, map_location="cpu",weights_only=True)
        net = m.base.Head().eval()
        net.load_state_dict(state["weights"])
        heads.append((rep,net,state["mean"],state["scale"]))
    folder = LOCK / "moment_forecasts"
    folder.mkdir(exist_ok=True)
    seals = {}
    for k in range(12):
        block = f"panel_{k:02d}"
        pcell = []
        for h,cell in CELLS:
            raw,paid,common,_ = m.base.inputs(LOCK/"results"/block/cell,h,4)
            raw = torch.from_numpy(raw)
            pred = np.repeat(common[None],len(heads),axis=0).astype(np.float32)
            for j,(rep,net,mean,scale) in enumerate(heads):
                pred[j][paid] = net((m.coordinates(raw,rep)-mean)/scale,raw[:,12]).numpy()
            assert np.all((pred>=0)&(pred<=1))
            pcell.append(pred)
        path = folder/f"{block}.npz"
        np.savez(path,pred=np.stack(pcell),cells=[cell for _,cell in CELLS])
        seals[block]=m.base.digest(path)
    assert not (LOCK/"labels").exists()
    m.save("moment_locked_forecasts.json",dict(frozen_heads=models, panels=seals, slurm=os.environ["SLURM_JOB_ID"],
           plan_sha256=m.base.digest(ROOT/"moment_lock_plan.md"),code_sha256=m.base.digest(Path(__file__))))


def score():
    manifest=json.loads((ROOT/"moment_locked_forecasts.json").read_text())
    catalogs=[]
    for k in range(12):
        block=f"panel_{k:02d}"
        path=LOCK/"moment_forecasts"/f"{block}.npz"
        assert m.base.digest(path)==manifest["panels"][block]
        with np.load(path) as data:
            np.testing.assert_array_equal(data["cells"],[cell for _,cell in CELLS])
            raw=data["pred"].astype(float)
        cell_scores=[]
        for pred,(_,cell) in zip(raw,CELLS):
            y=m.base.labels(LOCK/"labels"/block/cell)
            with np.load(LOCK/"results"/block/cell) as data:
                paid=np.isfinite(data["query_paid"]).all(-1)
            loss=pred**2+(1-2*pred)*y[...,None]
            cell_scores.append(np.stack([loss.mean((1,2,3)),loss[:,paid].mean(1)],axis=-1))
        catalogs.append(np.stack(cell_scores).reshape(2,24,3,3,2).mean((1,3)))
    family=np.stack(catalogs)
    all_scores=family.mean(1)
    mf=ci(all_scores[:,0]-all_scores[:,1],2)
    md=ci(all_scores[:,0]-all_scores[:,2],2)
    m.save("moment_locked_results.json",dict(methods=m.METHODS, metrics=["complete","query-only"],
           families=FAMILIES, absolute=ci(all_scores), family_absolute=ci(family), moment_minus_full=mf,
           moment_minus_dropped=md, noninferiority_margin=.002,
           noninferior_complete=bool(mf[0,2]<.002),superior_to_dropped_complete=bool(md[0,2]<0),
           family_catalog_means=family,source_seeds=m.base.SEEDS,slurm=os.environ["SLURM_JOB_ID"]))
    print(all_scores.mean(0)[:,0],mf[0],md[0],flush=True)


if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("forecast","score"))
    {"forecast":forecast,"score":score}[p.parse_args().stage]()
