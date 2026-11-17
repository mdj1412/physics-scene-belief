"""Source-matched ablation: aggregate over every residual-feature cell before taking a decision."""
import itertools
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
V45 = ROOT.parent / "image_pressure_v45"
V46 = ROOT.parent / "normal_variance_v46"


def main():
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Requires an allocated worker")
    fit = json.loads((ROOT/"policy47.json").read_text())
    grid, thresholds, d0 = [np.array(fit[k]) for k in ("x_grid", "variance_cuts", "legacy_risk")]
    counts = np.zeros((6,2))
    costs = np.zeros_like(counts)
    for fi, fam in enumerate(("fold","ripple","terrace","dimple")):
        root, lab, cam = (V45,"labels45","source_camera46") if fi<2 else (V46,"labels46","camera46")
        for p, v, h in itertools.product(range(12),("base","hollow","bulge"),(19,31)):
            sub = f"panel_{p:02d}"
            with np.load(V46/cam/sub/f"{fam}_{v}_h{h}.npz") as f:
                x, vv = f["x_hat_mm"], f["variance_hat"]
            b = abs(x[...,None]-grid).argmin(-1)
            q = (vv[...,None] > thresholds[b]).sum(-1)
            ids = 2*b+(d0[b,q]<0)
            for sf, law in itertools.product(("inlay","woven"),("rational","gaussian_rate")):
                stem = f"{fam}_{sf}_{law}_{v}_h{h}.npz"
                with np.load(root/"results"/sub/stem) as f:
                    po, pp = f["pred"][:2,0]
                    paid = np.isfinite(f["query_paid"]).all(-1)
                with np.load(root/lab/sub/stem) as f:
                    y = f["labels"][1:].mean(0)
                diff = (pp-po)[None]*(pp[None]+po[None]-2*y[...,None])
                idx = np.broadcast_to(ids[...,None],diff.shape)
                mask = np.broadcast_to(paid,diff.shape)
                counts += np.bincount(idx[mask],minlength=12).reshape(6,2)
                costs += np.bincount(idx[mask],weights=diff[mask],minlength=12).reshape(6,2)
    np.testing.assert_array_equal(counts,np.array(fit["counts"]).sum((2,3)))
    risk = np.where(counts>0, costs/np.maximum(1,counts), costs.sum(-1,keepdims=True)/counts.sum(-1,keepdims=True))
    result = dict(source="Same v45/v46 paired query draws as main gate; marginalize residual features",
                  n=counts.astype(int).tolist(),risk=risk.tolist(),active=(risk<0).tolist(),
                  empirical_supported_active=int(((risk<0)&(counts>0)).sum()))
    (ROOT/"camera_ablation47.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
