"""Fixed-bin Brier accounting on previously committed common-readout forecasts."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parent / "normal_variance_v46"
METHODS = ("full", "dropped", "restored")
FIELDS = ("brier", "uncertainty", "reliability", "resolution", "within_bin_remainder")


def run():
    if not os.getenv("SLURM_JOB_ID") or os.getenv("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use a Slurm CPU step with CUDA disabled")
    seal = json.loads((ROOT / "exp_forecasts.json").read_text())
    panels = []
    for panel in range(12):
        group = f"panel_{panel:02d}"
        forecasts = ROOT / "exp_forecasts" / f"{group}.npz"
        assert hashlib.sha256(forecasts.read_bytes()).hexdigest() == seal[group]
        with np.load(forecasts) as data:
            pred, cells = data["pred"], data["cells"]
        # count, sum probability, sum squared probability, sum Y, sum pY
        hist = np.zeros((9, 5, 10))
        for i, cell in enumerate(cells):
            with np.load(TARGET / "labels46" / group / cell) as data:
                y = data["labels"][1:].mean((0, 1))
            y = np.broadcast_to(y[..., None], pred[i, 0].shape).ravel()
            for j, probs in enumerate(pred[i]):
                p = probs.astype(float).ravel()
                bins = np.minimum((p * 10).astype(int), 9)
                for k, val in enumerate((None, p, p*p, y, p*y)):
                    hist[j,k] += np.bincount(bins, weights=val, minlength=10)
        counts, sums, sq, positives, prod = hist.transpose(1,0,2)
        total = counts.sum(1)
        base = positives.sum(1)/total
        means = sums/np.maximum(counts, 1)
        freq = positives/np.maximum(counts, 1)
        unc = base*(1-base)
        rel = (counts*(means-freq)**2).sum(1)/total
        res = (counts*(freq-base[:,None])**2).sum(1)/total
        within = (sq - counts*means**2 - 2*(prod - means*positives)).sum(1)/total
        actual = (positives + sq - 2*prod).sum(1)/total
        np.testing.assert_allclose(unc + rel - res + within, actual, atol=2e-13, rtol=0)
        panels.append(np.stack((actual,unc,rel,res,within),axis=-1).reshape(3,3,5).mean(1))
    out = np.array(panels)
    prior = json.loads((ROOT / "exp_results.json").read_text())
    np.testing.assert_allclose(out.mean(0)[:,0], np.array(prior["absolute"])[2:,0,0], atol=2e-12, rtol=0.)
    result = dict(methods=METHODS, fields=FIELDS, mean=out.mean(0).tolist(),
                  restored_minus_dropped=(out[:,2]-out[:,1]).mean(0).tolist(),
                  per_panel=out.tolist(), job=os.environ["SLURM_JOB_ID"])
    (ROOT / "calibration66.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run()
