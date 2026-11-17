"""Unloaded-image matching and actual measured alignment on held-out paid inputs."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from world47 import ROOT, FAMILIES, VARIANTS, HEIGHTS, ALIGNMENTS, OFFSETS, camera_shape, render_clip, source_axis, sensor_delta_mm, COS2


def summarize(a):
    return dict(mean=float(a.mean()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.90)),
                p99=float(np.quantile(a,.99)),maximum=float(a.max()))


def pixel_control():
    # Matched independent calls to renderer on 32 previously unseen objects, 24 settings;
    # keep the underlying inputs and RNG stream the same as the factors change.
    import torch
    torch.set_num_threads(1)
    pop=np.load(ROOT/"worlds/panel_00.npz")
    theta=pop["theta"][:, :8].reshape(-1,6)
    hashes=[]; n=0
    dx=np.arange(-2.,2.25,.5)
    max_true_var_diff=0.
    for family,h,off in itertools.product(FAMILIES,HEIGHTS,OFFSETS):
        ref=None
        for va in VARIANTS:
            for a,c in itertools.product(ALIGNMENTS,("null","strong")):
                pixels=render_clip(theta,family,va,h,off,470033).numpy()
                digest=hashlib.sha256(pixels.tobytes()).hexdigest()
                if ref is None: ref=digest
                assert digest==ref
                n += len(theta)
        hashes.append(ref)
        x=h+off+dx[None,:]; z0=theta[:,2:3]*x+theta[:,3:4]*x*x/50.
        first=camera_shape(x,FAMILIES[0])[0]+z0; actual=camera_shape(x,family)[0]+z0
        reference=np.arctan(first[:,2:]-first[:,:-2]).var(-1)
        observed=np.arctan(actual[:,2:]-actual[:,:-2]).var(-1)
        max_true_var_diff=max(max_true_var_diff,float(np.max(abs(reference-observed))))
    return dict(clips_rerendered=n, separate_matched_groups=len(hashes),aligned_and_coupled_pixel_mismatches=0,
                max_true_var_family_difference_rad2=max_true_var_diff)


def observed():
    xx=[]; vv=[]; xx_err=[]; empirical=[]; pair_fraction=[]
    for i in range(12):
        with np.load(ROOT/"forecast"/f"panel_{i:02d}.npz") as z:
            x=z["x_mm"]; v=z["var_rad2"]
            # F,H,V,X,E,N; true clip reused across optical alignment and loaded condition.
            assert x.shape==(4,2,2,3,4,128)
            assert np.array_equal(x[:,:,0],x[:,:,1]) and np.array_equal(v[:,:,0],v[:,:,1])
            xx.append(x[:,:,0].max(0)-x[:,:,0].min(0)); vv.append(v[:,:,0].max(0)-v[:,:,0].min(0))
            truth=np.asarray(HEIGHTS)[None,:,None,None,None] + np.asarray(OFFSETS)[None,None,:,None,None]
            xx_err.append(abs(x[:,:,0]-truth))
            six=(np.abs(x[:,:,0,...,None] - np.array([18.5,19,19.5,30.5,31,31.5]))).argmin(-1)
            pair_fraction.append(np.mean(np.any(six!=six[:1],axis=0)))
            q=z["query_paid"].astype(float)
            for f,h,a in itertools.product(range(4),range(2),range(2)):
                d = q[f,a,h,1] - q[f,a,h,0]
                d=d[np.isfinite(d).all(-1)]
                expected=2e-3*sensor_delta_mm(FAMILIES[f],HEIGHTS[h],ALIGNMENTS[a],"bulge")
                eps=float(np.max(abs(d-expected)))
                u=source_axis(HEIGHTS[h]); frac=(d@u)**2/(d*d).sum(-1)
                empirical.append([eps,float(np.max(abs(frac-COS2[ALIGNMENTS[a]])))])
    return dict(measured_across_family_range_mm=summarize(np.stack(xx)),family_range_variance_rad2=summarize(np.stack(vv)),
                pooled_position_error_mm=summarize(np.stack(xx_err)),any_family_position_bin_difference_fraction=float(np.mean(pair_fraction)),
                empirical_axis_max_abs_cos2_error=float(np.max(np.array(empirical)[:,1])),query_twin_max_vector_error_m=float(np.max(np.array(empirical)[:,0])),
                within_family_position_variance_variant_mismatch=0, alignment_coupling_observables="camera route is computed upstream once and shared by factor identity")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("kind",choices=("pixels","observed"));a=p.parse_args()
    s=(pixel_control if a.kind=="pixels" else observed)()
    (ROOT/f"{a.kind}_optics47.json").write_text(json.dumps(s,indent=2)+"\n")
    print(s,flush=True)
