"""Check physical prototype equality and geometric equivalence, without outcome access."""
from pathlib import Path
import sys, json, itertools
import numpy as np
from locked_geometry import profile,FAMILIES,VARIANTS
ROOT=Path(__file__).resolve().parent
PROJ=ROOT.parent.parent
sys.path[:0]=[str(PROJ/name) for name in ('contact_patch_v44','normal_variance_v46','factorial_projection_v47','preload_projection_v48','polarity_calibration_v50')]
import forecast44, geometry46, world47, world48, world50
grid=np.linspace(5,40,701)
funcs=[(k,fn,fs) for k,fn,fs in [('v44-45',forecast44.shape_profile,('fold','ripple')),('v46',geometry46.shape_profile,geometry46.SHAPES),('v47',world47.camera_shape,world47.FAMILIES),('v48',world48.camera_shape,world48.FAMILIES),('v50',world50.camera_shape,world50.FAMILIES)]]
past={}
for ver,fun,families in funcs:
    for fam,var in itertools.product(families,VARIANTS):
        z,g=fun(grid,fam,var)
        past[ver+'/'+fam+'/'+var]=z
dist={}
for fam,var in itertools.product(FAMILIES,VARIANTS):
    z,g=profile(grid,fam,var)
    # Also allow an affine change in pose/tilt before comparing vertical profiles.
    basis=np.column_stack([np.ones(len(grid)),grid])
    errors={}
    for key,v in past.items():
        delta=z-v
        err=delta-basis@np.linalg.lstsq(basis,delta,rcond=None)[0]
        errors[key]=float(np.sqrt(np.mean(err**2)))
    near=min(errors,key=errors.get)
    assert errors[near]>.001
    dist[fam+'/'+var]=[near,errors[near]]
def ids(x):
    return set(map(bytes,np.ascontiguousarray(x,dtype='<f8').reshape(-1,6)))
new=set()
for path in (ROOT/'results/worlds').glob('panel_*.npz'):
    with np.load(path) as z:new.update(ids(z['theta']))
previous={}
for path in sorted(set(PROJ.glob('*/world*/panel_*.npz'))|set(PROJ.glob('*/results/worlds/panel_*.npz'))):
    with np.load(path) as z:
        for key in ['theta','cal_theta','old_cal_theta']:
            if key not in z.files:continue
            x=z[key]
            if x.shape[-1]!=6:continue
            k=path.relative_to(PROJ).parts[0]+'/'+key
            stats=previous.setdefault(k,[0,0])
            stats[0]+=1;stats[1]+=len(new&ids(x))
assert sum(v[1] for v in previous.values())==0
print(json.dumps(dict(prior_optical_profile_variants=len(past),nearest_affine_invariant_rms_mm=dist,
                      actual_overlap_by_archive_field=previous),indent=2))
