"""Retrospective profile-moment baseline with paired CPU fits of its controls."""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import exp_readout as base
from exp_score import ci

ROOT = Path(__file__).resolve().parent
METHODS = ("moment", "full", "dropped")
base.DEVICE = torch.device("cpu")
x = 2. * np.arange(-3, 4)
design = np.column_stack((np.ones(7), x, .5 * x*x))
coef = np.linalg.pinv(design).T
np.testing.assert_allclose(design.T @ coef, np.eye(3), atol=3e-15)
moment_operator = torch.tensor(coef, dtype=torch.float32)


def coordinates(raw, rep):
    if rep != "moment":
        return base.coordinates(raw, rep)
    coeff = raw[:, 13:20] @ moment_operator
    return torch.cat((raw[:, :13], coeff, raw.new_zeros(len(raw), 4)), dim=1)


def save(name, value):
    path = ROOT / name
    path.write_text(json.dumps(value, indent=2, default=lambda obj: np.asarray(obj).tolist()) + "\n")


def fit():
    raw, y = base.source_data(range(8), 2)
    val, vy = base.source_data(range(8,12), 4)
    summary = {}
    for rep in METHODS:
        z = coordinates(raw, rep)
        mean = z.mean(0)
        scale = z.std(0, unbiased=False).clamp_min(1e-7)
        train = (z - mean) / scale
        valid = (coordinates(val, rep) - mean) / scale
        for seed in base.SEEDS:
            torch.manual_seed(seed)
            net = base.Head()
            optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
            generator = torch.Generator().manual_seed(seed+10)
            best = np.inf
            validation = {}
            name = f"moment_fit_{rep}_{seed}.pt"
            for epoch in range(1, 17):
                perm = torch.randperm(len(y), generator=generator)
                for group in perm.split(4096):
                    p = net(train[group], raw[group,12])
                    loss = (p-y[group]).square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                if epoch in (4,8,16):
                    with torch.no_grad():
                        val_loss = float(base.risk(net(valid, val[:,12]), vy))
                    validation[epoch] = val_loss
                    if val_loss < best:
                        best = val_loss
                        validation["best_epoch"] = epoch
                        torch.save(dict(weights=net.state_dict(), mean=mean, scale=scale), ROOT/name)
            summary[name] = dict(validation=validation, sha256=base.digest(ROOT/name))
            print(name, validation, flush=True)
    save("moment_fit.json", dict(models=summary, train=len(y), validation=len(vy),
         seed=base.SEEDS, device="cpu", threads=torch.get_num_threads(),
         slurm=os.environ["SLURM_JOB_ID"], torch=str(torch.__version__)))


@torch.inference_mode()
def forecast():
    models = json.loads((ROOT / "moment_fit.json").read_text())["models"]
    heads = []
    for rep, seed in itertools.product(METHODS, base.SEEDS):
        name = f"moment_fit_{rep}_{seed}.pt"
        path = ROOT / name
        assert base.digest(path) == models[name]["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        net = base.Head().eval()
        net.load_state_dict(state["weights"])
        heads.append((rep,net,state["mean"],state["scale"]))
    data_dir = ROOT / "moment_forecasts"
    data_dir.mkdir(exist_ok=True)
    manifest = {}
    ordered = list(base.cells(base.TARGET))
    for panel in range(12):
        block = f"panel_{panel:02d}"
        values = []
        for _, height, name in ordered:
            raw, paid, common, _ = base.inputs(base.TARGET/"results"/block/name,height,4)
            raw = torch.from_numpy(raw)
            p = np.repeat(common[None],len(heads),axis=0).astype(np.float32)
            for index, (rep,net,mean,scale) in enumerate(heads):
                p[index][paid] = net((coordinates(raw,rep)-mean)/scale,raw[:,12]).numpy()
            assert np.all((p >= 0) & (p <= 1))
            values.append(p)
        path = data_dir / f"{block}.npz"
        np.savez_compressed(path, pred=np.stack(values), cells=[name for _,_,name in ordered])
        manifest[block] = base.digest(path)
    save("moment_forecasts.json", manifest)


def score():
    manifest = json.loads((ROOT/"moment_forecasts.json").read_text())
    catalogs = []
    for panel in range(12):
        block = f"panel_{panel:02d}"
        path = ROOT / "moment_forecasts" / f"{block}.npz"
        assert base.digest(path) == manifest[block]
        cells = list(base.cells(base.TARGET))
        with np.load(path) as file:
            np.testing.assert_array_equal(file["cells"], [name for _,_,name in cells])
            preds = file["pred"].astype(float)
        risks = []
        for pred, (_,_,name) in zip(preds, cells):
            truth = base.labels(base.TARGET/"labels46"/block/name)
            with np.load(base.TARGET/"results"/block/name) as record:
                mask = np.isfinite(record["query_paid"]).all(-1)
            loss = pred**2 + (1-2*pred)*truth[...,None]
            risks.append(np.stack((loss.mean((1,2,3)), loss[:,mask].mean(1)),axis=-1))
        catalogs.append(np.stack(risks).reshape(2,24,3,3,2).mean(1))
    arr = np.stack(catalogs)
    fam = arr.mean(3)
    pooled = fam.mean(1)
    gpu = json.loads((ROOT/"exp_results.json").read_text())
    original = np.asarray(gpu["paired_catalog_losses"]).mean(1)
    controls = {}
    for i, rep in enumerate(METHODS[1:], 1):
        controls[rep] = ci(pooled[:,i]-original[:,gpu["methods"].index(rep)])
    res = dict(methods=METHODS, metrics=["complete", "query_only"], absolute=ci(pooled),
               family=["terrace","dimple"], family_absolute=ci(fam),
               moment_minus_full=ci(pooled[:,0]-pooled[:,1],2),
               moment_minus_dropped=ci(pooled[:,0]-pooled[:,2],2),
               same_rep_cpu_minus_gpu=controls, family_catalog_means=fam,
               seed_means=arr.mean((0,1)), train_seeds=base.SEEDS,
               target_catalogs=list(range(600,612)), forecast_sha256=base.digest(ROOT/"moment_forecasts.json"),
               slurm=os.environ["SLURM_JOB_ID"])
    save("moment_results.json",res)
    print(res["absolute"],res["moment_minus_full"],res["moment_minus_dropped"],flush=True)


if __name__ == "__main__":
    assert os.getenv("SLURM_JOB_ID") and os.getenv("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("fit","forecast","score"))
    stage = parser.parse_args().stage
    {"fit":fit,"forecast":forecast,"score":score}[stage]()
