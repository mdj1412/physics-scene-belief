"""Matched silhouette, signed-deflection and independent quadrature checks."""
from itertools import combinations
from pathlib import Path
import argparse,json
import numpy as np
from scipy.stats import t
from world48 import ROOT,SOURCE,FAMILIES,HEIGHTS,STATES,OFFSETS,ALIGNMENTS,camera_shape,loaded_shape,sensor_delta_mm,source_axis
from sensor48 import readings,gauge_mm


def pre():
    fd_error=[];center=[];energies=[];unloaded=[];wing=[]
    for h in HEIGHTS:
        x=np.linspace(h-2.49,h+2.49,101)
        for f in FAMILIES:
            f0,g=camera_shape(x,f)
            a,b=camera_shape(x,FAMILIES[0])
            unloaded.append(max(abs(f0-a).max(),abs(g-b).max()))
            for s in STATES:
                z_plus=loaded_shape(x+1.e-6,h,f,s)[0]
                z_minus=loaded_shape(x-1.e-6,h,f,s)[0]
                fd_error.append(np.max(abs((z_plus-z_minus)/2e-6-loaded_shape(x,h,f,s)[1])))
            for alignment in ALIGNMENTS:
                q=sensor_delta_mm(f,h,alignment,"bulge")
                energies.append(float((q@source_axis(h))**2/(q@q)))
                assert abs(energies[-1]-{"high":.9,"low":.1}[alignment])<1e-12
        for f,u in combinations(FAMILIES,2):
            xq=h+np.arange(-3,4)*2
            wing.append(float(np.max(abs(camera_shape(xq,f)[0]-camera_shape(xq,u)[0]))))
    assert max(unloaded)==0 and max(fd_error)<1e-7
    # Source and target expose one finite, positively oriented gauge differential.
    amplitudes={f:[gauge_mm(f,h,1) for h in HEIGHTS] for f in (*SOURCE,*FAMILIES)}
    agree=0;total=0
    with np.load(ROOT/"worlds/panel_00.npz") as p:
        for f in FAMILIES:
            for h in HEIGHTS:
                z0=readings(p,f,h,0,h)
                zp=readings(p,f,h,1,h)
                zm=readings(p,f,h,-1,h)
                assert np.max(abs((zm+zp)*.5-z0))<2e-8 and np.all(zp>zm)
                total+=z0.size
    return dict(max_preimpact_local_family_difference=max(unloaded),minimum_pairwise_optical_wing_difference_per_height_mm=min(wing),
        maximum_loaded_slope_derivative_error=max(fd_error), noiseless_positive_gauge_displacement_mm=amplitudes,
        matched_station_control_rows=total,empirical_alignment_range=[min(energies),max(energies)])


def interval(x):
    x=np.array(x);m=x.mean(0);e=x.std(axis=0,ddof=1)*t.ppf(.975,len(x)-1)/np.sqrt(len(x))
    return np.stack((m,m-e,m+e),axis=-1).tolist()


def outcomes():
    variants=[]; statechange=[]; fines=[]; raw=0; prev=[]
    for i in range(12):
        with np.load(ROOT/f"labels/panel_{i:02d}.npz") as a:
            y=a['labels'];sp=a['slip_speed']
        assert list(y.shape)==[4,2,3,2,3,4,128]
        assert all(np.array_equal(y[0,:,0],f[:,0]) for f in y[1:])
        variants.append((y[:,:,1]!=y[:,:,2]).mean(axis=(1,2,3,4,5)))
        statechange.append(np.stack([(y[:,:,0]!=y[:,:,j]).mean() for j in (1,2)]))
        prev.append(y.mean(axis=(0,1,3,4,5,6)))
        raw+=y.size
        if i in (0,6):
            with np.load(ROOT/f"labels_fine/panel_{i:02d}.npz") as b:
                fines.append(dict(panel=i,slots=y.size,changed=int(np.count_nonzero(y!=b["labels"])),
                               max_abs_speed_delta_mps=float(abs(sp-b["slip_speed"]).max())))
    return dict(slots=raw,distinct_recipes_before_sampling_duplicates=raw*9//12,
        strong_signed_flip_by_family=interval(variants),strong_signed_flip_pooled=interval(np.mean(variants,axis=1)),
        negative_or_positive_vs_null_flip=interval(statechange),stick_by_state=interval(prev),fine_grid=fines)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("pre","outcomes"));args=ap.parse_args()
    data=pre() if args.mode=="pre" else outcomes()
    (ROOT/f"{args.mode}_physics48.json").write_text(json.dumps(data,indent=2)+"\n")
