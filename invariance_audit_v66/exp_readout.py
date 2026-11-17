"""Source-trained common heads; target-forecast mode never opens an outcome archive."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "image_pressure_v45"
TARGET = ROOT.parent / "normal_variance_v46"
REPS = ("full", "dropped", "restored")
SEEDS = (66101, 66102, 66103)
DEVICE = torch.device("cuda")
WIDTH = 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells(root):
    families = ("fold", "ripple") if root == SOURCE else ("terrace", "dimple")
    for f, s, l, v, h in itertools.product(families, ("inlay","woven"), ("rational","gaussian_rate"),
                                          ("base","hollow","bulge"), (19,31)):
        yield f, h, f"{f}_{s}_{l}_{v}_h{h}.npz"


def basis(height):
    u = np.array(json.loads((SOURCE/"source_axis45.json").read_text())[str(height)]["axis"])
    # A Householder reflection maps e_last to the frozen signed axis.
    w = np.eye(7)[-1] - u
    mat = np.eye(7) - 2 * np.outer(w, w) / (w @ w)
    np.testing.assert_allclose(mat[:, -1], u, atol=3e-13)
    np.testing.assert_allclose(mat.T @ mat, np.eye(7), atol=3e-13)
    return mat


BASIS = {h:basis(h) for h in (19,31)}


def inputs(path, height, repeats):
    with np.load(path) as data:
        x = data["xx"]
        q = data["query_paid"][:, :, :repeats]
        p = data["pred"][1, 0, :, :, :repeats]
        o = data["pred"][0, 0, :, :, :repeats]
        assert list(data["decoders"][:2]) == ["original", "source_axis"]
    mask = np.isfinite(q).all(axis=-1)
    assert np.array_equal(mask, np.isfinite(q).any(axis=-1))
    np.testing.assert_allclose(p[~mask], o[~mask], atol=3e-12, rtol=0.)
    q = q[mask] * 1000.
    z = q @ BASIS[height]
    np.testing.assert_allclose(z @ BASIS[height].T, q, atol=1e-12)
    # A signed displacement along the discarded axis cannot affect the first six coordinates.
    np.testing.assert_allclose(((q + .137 * BASIS[height][:,-1]) @ BASIS[height])[:,:6], z[:,:6], atol=2e-12)
    x = np.broadcast_to(x[..., None, :], (*mask.shape, 11))[mask]
    pclip = np.clip(p[mask], 1e-4, 1.-1e-4)
    a = np.column_stack((x, np.full(len(x), (height-25)/6), np.log(pclip/(1-pclip)), q, z))
    assert np.isfinite(a).all()
    return a.astype(np.float32), mask, p, o


def labels(path):
    with np.load(path) as data:
        vals = data["labels"]
    assert vals.dtype == bool and vals.shape == (3, 3, 4, 128)
    return vals[1:].mean(axis=(0,1))


def source_data(panel_indices, repeats):
    features, outcomes = [], []
    for panel in panel_indices:
        block = f"panel_{panel:02d}"
        for _, height, cell in cells(SOURCE):
            z, mask, _, _ = inputs(SOURCE/"results"/block/cell, height, repeats)
            y = labels(SOURCE/"labels45"/block/cell)
            features.append(z)
            outcomes.append(np.broadcast_to(y[...,None], mask.shape)[mask])
    return (torch.as_tensor(np.concatenate(features), device=DEVICE),
            torch.as_tensor(np.concatenate(outcomes), device=DEVICE, dtype=torch.float32))


def coordinates(raw, rep):
    profile = raw[:,13:20] if rep == "full" else raw[:,20:27]
    if rep == "dropped":
        profile = torch.cat((profile[:,:6], torch.zeros_like(profile[:,6:7])),1)
    return torch.cat((raw[:,:13], profile),1)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(20, WIDTH), nn.SiLU(), nn.Linear(WIDTH, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, logodds):
        return torch.sigmoid(self.mlp(z).flatten() + logodds)


def risk(p, y):
    return (p*p - 2*p*y + y).mean()


def fit():
    raw, y = source_data(range(8), 2)
    val, vy = source_data(range(8,12), 4)
    summary = {}
    for rep in REPS:
        train = coordinates(raw, rep)
        mean = train.mean(0)
        scale = train.std(0, unbiased=False).clamp_min(1e-7)
        X = (train-mean)/scale
        V = (coordinates(val, rep)-mean)/scale
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = Head().to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
            gen = torch.Generator(device=DEVICE).manual_seed(seed + 10)
            best, record = np.inf, {}
            filename = ROOT / f"exp_fit_{rep}_{seed}.pt"
            for epoch in range(1,17):
                perm = torch.randperm(len(y), device=DEVICE, generator=gen)
                for group in perm.split(4096):
                    pred = net(X[group], raw[group,12])
                    loss = (pred-y[group]).square().mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                if epoch in (4,8,16):
                    with torch.no_grad():
                        score = float(risk(net(V,val[:,12]),vy))
                    record[epoch] = score
                    if score < best:
                        best = score
                        record["best_epoch"] = epoch
                        torch.save(dict(weights=net.state_dict(),mean=mean,scale=scale),filename)
            summary[f"{rep}_{seed}"] = dict(validation=record, sha256=digest(filename))
            print(rep, seed, record, flush=True)
    info = dict(models=summary, training_query_rows=len(y), val_query_rows=len(vy),
                torch=str(torch.__version__),device=torch.cuda.get_device_name(),
                source_axis_sha256=digest(SOURCE/"source_axis45.json"),job=os.environ["SLURM_JOB_ID"])
    (ROOT/"exp_fit.json").write_text(json.dumps(info,indent=2)+"\n")


@torch.inference_mode()
def predict():
    meta = json.loads((ROOT/"exp_fit.json").read_text())
    store = ROOT/"exp_forecasts"
    store.mkdir(exist_ok=True)
    nets = []
    for rep, seed in itertools.product(REPS, SEEDS):
        path = ROOT/f"exp_fit_{rep}_{seed}.pt"
        assert digest(path)==meta["models"][f"{rep}_{seed}"]["sha256"]
        d = torch.load(path, map_location=DEVICE, weights_only=True)
        net = Head().to(DEVICE)
        net.load_state_dict(d["weights"])
        net.eval()
        nets.append((net,rep,d["mean"],d["scale"]))
    out = {}
    for panel in range(12):
        block = f"panel_{panel:02d}"
        prows = []
        for _, height, cell in cells(TARGET):
            a, mask, common, _ = inputs(TARGET/"results"/block/cell, height, 4)
            t = torch.as_tensor(a, device=DEVICE)
            p = np.stack([common.copy() for net in nets]).astype(np.float32)
            for i, (net, rep, mean, scale) in enumerate(nets):
                p[i][mask] = net((coordinates(t,rep)-mean)/scale,t[:,12]).cpu().numpy()
            assert np.isfinite(p).all() and p.min() >= -1.e-8 and p.max() <= 1+1.e-8
            prows.append(p)
        dest = store / f"{block}.npz"
        np.savez_compressed(dest, pred=np.stack(prows), cells=[cell for _,_,cell in cells(TARGET)])
        out[block] = digest(dest)
    (ROOT/"exp_forecasts.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("fit", "predict"))
    args=ap.parse_args()
    assert os.getenv("SLURM_JOB_ID") and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    if args.stage == "fit":
        fit()
    else:
        predict()
