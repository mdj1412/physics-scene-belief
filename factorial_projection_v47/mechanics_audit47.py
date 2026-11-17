"""Audit assignment, local contour matching, physical label changes, and quadrature."""
import argparse
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from world47 import (ROOT, FAMILIES, HEIGHTS, VARIANTS, ALIGNMENTS, COUPLINGS, OFFSETS, RADII,
                     COS2, OPTICAL_NORM_MM, camera_shape, source_axis, sensor_delta_mm, loaded_shape)


def geometry():
    cos2, fderr, delta_norm = {}, [], []
    match, wings = [], []
    for family, height in product(FAMILIES, HEIGHTS):
        x = np.linspace(height-2.5, height+2.5, 301)
        f, g = camera_shape(x, family)
        fb, gb = camera_shape(x, FAMILIES[0])
        match.extend([np.max(abs(f-fb)),np.max(abs(g-gb))])
        for al in ALIGNMENTS:
            diff = sensor_delta_mm(family,height,al,"bulge") - sensor_delta_mm(family,height,al,"hollow")
            c = float((diff@source_axis(height))**2/(diff@diff))
            cos2[f"{family}_{height}_{al}"] = c
            delta_norm.append(abs(np.linalg.norm(diff)/2 - OPTICAL_NORM_MM))
            assert abs(c-COS2[al]) < 1.e-12
        # Check analytic under-load slopes using a displacement derivative.
        xx = x[1:-1]
        for va,co in product(VARIANTS,COUPLINGS):
            z1 = loaded_shape(xx+1.e-5,height,family,va,co)[0]
            z0 = loaded_shape(xx-1.e-5,height,family,va,co)[0]
            gg = loaded_shape(xx,height,family,va,co)[1]
            fderr.append(float(abs((z1-z0)/2.e-5-gg).max()))
    # Witness distinct remote contours, including actual queried wing points.
    for i,fa in enumerate(FAMILIES):
        for fb in FAMILIES[:i]:
            x=np.r_[np.arange(-3,4)*2+19.,np.arange(-3,4)*2+31.]
            wings.append(float(abs(camera_shape(x,fa)[0]-camera_shape(x,fb)[0]).max()))
    assert max(fderr)<2e-8 and max(match)==0 and min(wings)>0.1
    return dict(cosine_squared=cos2, half_contrast_norm_mm=OPTICAL_NORM_MM,
                maximum_norm_error_mm=max(delta_norm), maximum_derivative_error=max(fderr),
                maximum_unloaded_profile_or_slope_mismatch_on_site= max(match),
                minimum_pairwise_maximum_wing_height_difference_mm=min(wings),
                low_high_routing_evidence="Only the paid optical query can differ; same noise, lens, physical law and outcomes.",
                null_strong_routing_evidence="Every video frame, factual observable and paid query/probe is shared.")


def ci(samples, multiplicity=1):
    samples=np.asarray(samples, dtype=float)
    mean=samples.mean(0)
    err=samples.std(0,ddof=1)*t.ppf(1-.025/multiplicity,len(samples)-1)/np.sqrt(len(samples))
    return np.stack([mean,mean-err,mean+err],axis=-1).tolist()


def outcomes():
    flip, any_change, prevalence, family_difference = [], [], [], []
    contrasts, swapped, n = [],0,0
    for panel in range(12):
        with np.load(ROOT/"labels"/f"panel_{panel:02d}.npz") as a:
            y=a["labels"]; s=a["slip_speed"]
            assert y.shape==(4,2,2,2,2,3,4,128)
            assert all(a["couplings"]==COUPLINGS)
            assert not np.any(y[:,:,0,0]!=y[:,:,1,0])
            # Each element in these statistics represents a distinct physical recipe;
            # optical alignment and sensor noise never enter label counts.
            f = y[:,:,0] != y[:,:,1]   # [F,H,C,R,X,E,N]
            flip.append(f.mean(axis=(1,3,4,5,6)))
            any_change.append((y[:,:,:,0] != y[:,:,:,1]).mean(axis=(1,2,3,4,5,6)))
            prevalence.append(y.mean(axis=(1,2,4,5,6,7)))
            family_difference.append([(y[i,:,:,1]!=y[j,:,:,1]).mean() for i in range(4) for j in range(i)])
            n += y.size
            if panel in (0,6):
                with np.load(ROOT/"labels_fine"/f"panel_{panel:02d}.npz") as b:
                    fine=b["labels"]
                    contrasts.append(dict(panel=panel,cases=int(y.size),flips=int((fine!=y).sum()),
                                          max_slip_speed_delta_mps=float(abs(b["slip_speed"]-s).max())))
                    swapped+=int((fine!=y).sum())
    return dict(total_factor_combinations_including_exact_null_repeats=n,
                strong_and_null_variant_flip=ci(flip,4), flip_axis_order=["family","coupling","mean_lo_hi"],
                pooled_variant_flips=ci(np.asarray(flip).mean(axis=1),1),
                family_order=FAMILIES, coupling_order=COUPLINGS,
                coupled_vs_null_change_by_family=ci(any_change,4), prevalence=ci(prevalence,8),
                pairwise_strong_family_outcome_disagreement=np.mean(family_difference,axis=0).tolist(),
                fine_quadrature=contrasts, threshold_mps=.020)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("kind",choices=("geometry","outcomes"));a=p.parse_args()
    val=(geometry if a.kind=="geometry" else outcomes)()
    (ROOT/f"{a.kind}_checks47.json").write_text(json.dumps(val,indent=2)+"\n")
    print(a.kind, "complete", flush=True)
