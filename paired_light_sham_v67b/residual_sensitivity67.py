"""Paired replay on raw archived v64 frames with a specified *post*-correction residual."""
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"paired_light_pilot_v65"))
from operations65 import ARCH, LEGACY, capture, classify, later_diode

# Wave samples at 16 kHz. Prespecified signed residuals on slot two.
LAGS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)


def summarize(v):
    m = v.mean(0)
    half = t.ppf(.975, len(v)-1)*v.std(0, ddof=1)/np.sqrt(len(v))
    return np.stack((m, m-half, m+half), -1).tolist()


def main():
    assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    outcomes = np.zeros((24,len(LAGS),2,4))
    for i in range(24):
        mats = ((np.arange(4)+i)%4)//2
        with np.load(LEGACY/"scored63"/f"panel_{i:02d}.npz") as z:
            pr_ref = np.array([(~z["gate"][:,mats==m].astype(bool)).mean() for m in range(2)])
        with np.load(ARCH/"fixtures"/f"panel_{i:02d}.npz") as z:
            raw=capture.cam.optical_features(z["frames"])
            wave, electronics, reset, episode = z["wave"], z["electronic"][1],z["reset_id"],int(z["episode"])
        def routers(lag):
            signs=np.stack([capture.cam.read_features(tuple(x[d] for x in raw),
                                later_diode(wave[d],lag),2) for d in range(4)])
            codes=np.stack([capture.cam.encode(electronics,s,reset) for s in signs])
            return signs,classify(codes),codes
        ref_sign,ref_upper,ref_codes=routers(0)
        with np.load(ARCH/"pilots"/f"panel_{i:02d}.npz") as z:
            np.testing.assert_array_equal(z["codes"][:,2,1,...,episode,:],ref_codes)
        for j,lag in enumerate(LAGS):
            signs,upper,_=routers(lag)
            for d,k in itertools.product(range(4),range(2)):
                slot=d^k
                l,r=ref_upper[d],upper[slot]
                xor=l!=r
                direction=[(l==a)&xor for a in (0,1)]
                values=[(ref_sign[d]!=signs[slot]).mean(),xor.mean(),
                        (direction[0]*pr_ref).mean(),(direction[1]*pr_ref).mean()]
                outcomes[i,j,k] += np.array(values)/4
    terminal=outcomes[...,2:].sum(-1)
    assert (terminal[:,:,0]<=outcomes[:,:,0,1]+1e-13).all()
    assert (terminal[:,:,1]<=outcomes[:,:,1,1]+1e-13).all()
    assert not np.any(outcomes[:,LAGS.index(0),0])
    data=dict(context="v64 archived 24 catalogs, 2 materials, one saved 8-impact bank per material. Exact same pixels in sham; slots counterbalanced in both arms. No additional error injection except lag.",
              lag_samples=LAGS, ms_per_sample=1/16, arm=["same pixels same lamp","counterbalanced AB and BA"],
              metrics=["clip disagreement including abstentions","upper xor","gate-to-fixed terminal","fixed-to-gate terminal"],
              arm_estimates=outcomes.mean(0).tolist(),terminal=terminal.mean(0).tolist(),
              two_minus_sham_pointwise_catalog_ci=summarize(terminal[:,:,1]-terminal[:,:,0]),
              independent_unit="catalog", job=os.environ["SLURM_JOB_ID"],catalog_outcomes=outcomes.tolist())
    (HERE/"residual_sensitivity67.json").write_text(json.dumps(data,indent=2)+"\n")
if __name__=="__main__":
    main()
