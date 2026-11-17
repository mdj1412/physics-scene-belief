"""Independently cross v66's two observational-law tapes and two archived terminal labels."""
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

OUT = Path(__file__).resolve().parent
ARCH = OUT.parent / "invariance_audit_v66" / "locked"
LAWS = ("rational", "gaussian_rate")
FAMILIES = ("wendland", "dispersive")


def interval(x):
    half = t.ppf(.975, len(x)-1)*x.std(0, ddof=1)/np.sqrt(len(x))
    return [x.mean(0).tolist(), (x.mean(0)-half).tolist(), (x.mean(0)+half).tolist()]


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    forecasts = np.zeros((12, 2, 2, 2, 2))  # catalog, shape, evidence law, label law, decoder
    outcomes = np.zeros((12, 2, 4))  # 0->1, 1->0, outcome attribution on each evidence tape
    evidence = np.zeros((12, 2, 3))
    for i, f, surface, variant, height in product(range(12), range(2), ("inlay","woven"),
                                                  ("base","hollow","bulge"), (19,31)):
        sub = f"panel_{i:02d}"
        stem = f"{FAMILIES[f]}_{variant}_h{height}"
        with np.load(ARCH / "routes" / sub / (stem + ".npz")) as z:
            route = z["route"].astype(bool)
        predictions, labels, measurements = [], [], []
        for law in LAWS:
            path = f"{FAMILIES[f]}_{surface}_{law}_{variant}_h{height}.npz"
            with np.load(ARCH / "results" / sub / path) as z:
                raw = z["pred"][[0,2],0].astype(float)
                p = raw[0][None,None] + route[None,...,None] * (raw[:,None]-raw[0][None,None])
                predictions.append(p)
                measurements.append(tuple(z[k] for k in ("theta","selected","xx","query_paid","probe_paid")))
            with np.load(ARCH / "labels" / sub / path) as z:
                labels.append(z["labels"][1:].astype(float))
        a,b = measurements
        np.testing.assert_array_equal(a[0], b[0])
        for k in (1,2,3,4):
            if k in (1,2,3): evidence[i,f,k-1] += float(not np.array_equal(a[k],b[k],equal_nan=True))/12
        for e,g in product(range(2), repeat=2):
            # [decoder, area, offset, environment, prototype, observation repeat]
            err = predictions[e][:,None] - labels[g][None,...,None]
            forecasts[i,f,e,g] += (err*err).mean(axis=(1,2,3,4,5))/12
        dy = labels[1]-labels[0]
        outcomes[i,f,:2] += np.array([(dy>0).mean(), (dy<0).mean()])/12
        for e,p in enumerate(predictions):
            outcomes[i,f,e+2] += (-2*(p[1]-p[0])[None]*dy[...,None]).mean()/12
    delta = forecasts[...,1]-forecasts[...,0]
    max_error = float(np.max(abs(outcomes[...,2:]-(delta[:,:,:,1]-delta[:,:,:,0]))))
    np.testing.assert_allclose(max_error,0,atol=1e-14)
    historical = np.array(json.loads((ARCH/"locked_results.json").read_text())["panel_family"])[...,[0,2]]
    reaggregate=(forecasts[:,:,0,0]+forecasts[:,:,1,1])/2
    np.testing.assert_allclose(reaggregate,historical,atol=1e-11,rtol=0)
    # Symmetric split averages each factor's contribution over the other factor's two levels.
    label_part = outcomes[...,2:].mean(-1)
    evidence_part = ((delta[:,:,1,0]-delta[:,:,0,0])+(delta[:,:,1,1]-delta[:,:,0,1]))/2
    both=delta[:,:,1,1]-delta[:,:,0,0]
    np.testing.assert_allclose(both,label_part+evidence_part,atol=1e-14)
    data = dict(laws=LAWS,families=FAMILIES,estimated_over="12 v66 fixed held-out catalogs",
         delta_by_evidence_and_label=delta.mean(0).tolist(),
         label_flip_0to1_1to0=outcomes[...,:2].mean(0).tolist(),
         total_law_effect=interval(both), label_component=interval(label_part),
         forecast_component=interval(evidence_part), unequal_file_fractions_sel_xx_q=evidence.mean(0).tolist(),
         max_weighted_flip_identity_error=max_error,max_v66_brier_error=float(np.max(abs(reaggregate-historical))),
         catalog_risks=forecasts.tolist(),job=os.environ["SLURM_JOB_ID"])
    (OUT/"legacy_cross67.json").write_text(json.dumps(data,indent=2)+"\n")

if __name__ == "__main__":
    main()
