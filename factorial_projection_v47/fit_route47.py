"""Source-only relative-risk calibration. No target results or target family descriptors are loaded."""
import itertools
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
V45 = ROOT.parent / "image_pressure_v45"
V46 = ROOT.parent / "normal_variance_v46"
ALIGN_CUTS = np.array([.125, .375, .625])
RMS_CUTS = np.array([.30, .60])  # millimetres
GRID = (6, 2, 4, 3)


def main():
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Source fitting belongs on an allocated compute node")
    raw = json.loads((V46 / "router46.json").read_text())["visual"]
    axes = json.loads((V45 / "source_axis45.json").read_text())
    grid = np.array(raw["x_grid"])
    thresholds = np.array(raw["thresholds"])
    legacy_risk = np.array(raw["risk_joint"])
    ygrid = np.arange(-3., 4.)
    design = np.stack([np.ones(7), ygrid, ygrid**2], axis=1)
    residual = np.eye(7) - design @ np.linalg.pinv(design)
    direction = np.array([axes[str(h)]["axis"] for h in (19, 31)]) @ residual
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    # Each row is one catalog and one source family. All generator cells share the same sampling weights.
    family_names = ("fold", "ripple", "terrace", "dimple")
    counts = np.zeros((4, 12) + GRID)
    costs = np.zeros_like(counts)
    for fi, family in enumerate(family_names):
        old = fi < 2
        root = V45 if old else V46
        labels = "labels45" if old else "labels46"
        camera = V46 / ("source_camera46" if old else "camera46")
        for p, variant, hi in itertools.product(range(12), ("base", "hollow", "bulge"), range(2)):
            h = (19, 31)[hi]
            prefix = f"panel_{p:02d}"
            with np.load(camera / prefix / f"{family}_{variant}_h{h}.npz") as f:
                x = f["x_hat_mm"]
                v = f["variance_hat"]
            b = np.abs(x[..., None] - grid).argmin(-1)
            q = (v[..., None] > thresholds[b]).sum(-1)
            gate = (legacy_risk[b, q] < 0).astype(int)
            for surface, law in itertools.product(("inlay","woven"), ("rational","gaussian_rate")):
                name = f"{family}_{surface}_{law}_{variant}_h{h}.npz"
                with np.load(root / "results" / prefix / name) as f:
                    obs = f["query_paid"]
                    po, pp = f["pred"][:2, 0]
                with np.load(root / labels / prefix / name) as f:
                    y = f["labels"][1:].mean(axis=0)
                paid = np.isfinite(obs).all(-1)
                rr = np.nan_to_num(obs) @ residual * 1000.
                en = (rr**2).sum(-1)
                cos = (rr @ direction[hi])**2 / np.maximum(en, 1e-20)
                ai = np.searchsorted(ALIGN_CUTS, cos)
                si = np.searchsorted(RMS_CUTS, np.sqrt(en / 7.))
                diff = (pp-po)[None] * (pp[None]+po[None]-2*y[...,None])
                idx = np.ravel_multi_index(np.broadcast_arrays(b[...,None], gate[...,None], ai[None], si[None]), GRID)
                mask = np.broadcast_to(paid, diff.shape)
                ids = idx[mask]
                n = int(np.prod(GRID))
                counts[fi,p] += np.bincount(ids, minlength=n).reshape(GRID)
                costs[fi,p] += np.bincount(ids, weights=diff[mask], minlength=n).reshape(GRID)
        print("source", family, int(counts[fi].sum()), flush=True)

    def calibrate(count, cost):
        # Sparse/unsupported cells inherit the relative risk in their camera stratum.
        n, s = count.sum(axis=(0,1)), cost.sum(axis=(0,1))
        bn, bs = n.sum((2,3), keepdims=True), s.sum((2,3), keepdims=True)
        xrate = bs.sum(1, keepdims=True) / np.maximum(bn.sum(1, keepdims=True), 1)
        parent = np.where(bn > 0, bs / np.maximum(bn, 1), xrate)
        rate = .75 * s / np.maximum(n, 1) + .25 * parent
        return np.where(n >= 128, rate, parent)

    risk = calibrate(counts, costs)
    folds = []
    for f, name in enumerate(family_names):
        mask = np.arange(4) != f
        active = calibrate(counts[mask], costs[mask]) < 0
        total = counts[f].sum(axis=(1,2,3,4))
        folds.append(dict(family=name, frequency=((counts[f]*active).sum((1,2,3,4))/total).tolist(),
                          delta=((costs[f]*active).sum((1,2,3,4))/total).tolist()))
    export = dict(source_families=family_names, residual_operator=residual.tolist(), effective_axis=direction.tolist(),
                  x_grid=grid.tolist(), variance_cuts=thresholds.tolist(), legacy_risk=legacy_risk.tolist(),
                  alignment_cuts=ALIGN_CUTS.tolist(), rms_cuts_mm=RMS_CUTS.tolist(), risk=risk.tolist(),
                  counts=counts.sum((0,1)).tolist(), fit_on="v45/v46 paired Brier (two finite pressure radii)",
                  shrinkage_to_parent=.25, minimum_bin_count=128, leave_one_family_out_on_query_arm=folds)
    dest = ROOT / "policy47.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(export, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, dest)


if __name__ == "__main__":
    main()
