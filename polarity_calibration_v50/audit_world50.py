"""Check specimen split, novel geometry and solver sensitivity without policy data."""
import hashlib
import itertools
import json
import os
import sys
import numpy as np
from scipy.stats import t
import world50 as w

ROOT=w.ROOT
sys.path.insert(0,str(ROOT.parent/'preload_projection_v48'))
import world48 as prior


def intervals(x):
    a=np.asarray(x)
    m=a.mean(axis=0)
    e=a.std(axis=0,ddof=1)*t.ppf(.975,len(a)-1)/np.sqrt(len(a))
    return np.stack((m,m-e,m+e),axis=-1).tolist()


def main():
    assert os.getenv('SLURM_JOB_ID')
    x=np.linspace(9,40,801)
    profiles=[w.camera_shape(x,f)[0] for f in w.FAMILIES]
    old=[prior.camera_shape(x,f)[0] for f in (*prior.FAMILIES,*prior.SOURCE)]
    novel=min(np.sqrt(np.mean((p-q)**2)) for p in profiles for q in old)
    pair=min(np.sqrt(np.mean((p-q)**2)) for p,q in itertools.combinations(profiles,2))
    assert novel>.03 and pair>.03
    families_check=[]
    for h in w.HEIGHTS:
        inner=np.linspace(h-2.5,h+2.5,401)
        curves=np.stack([w.loaded_shape(inner,h,f,0)[0] for f in w.FAMILIES])
        families_check.append(np.ptp(curves,axis=0).max().item())
    assert not any(families_check)
    dep_set=set(); cal_set=set()
    diff=[]; primary_stick=[]; ultra_stick=[]; full_flip=[]; fine_sign=[]; ultra_sign=[]
    population_stats=[]; fields={}
    for sf in w.SURFACES:
        fields[sf]={h:w.surface_field_mm(h,sf) for h in (0,19,31)}
    for i in range(12):
        tag=f'panel_{i:02d}.npz'
        with np.load(ROOT/'worlds'/tag) as worlds:
            th,cal=worlds['theta'],worlds['cal_theta']
            dep_set.update(row.tobytes() for row in th.reshape(-1,6))
            cal_set.update(row.tobytes() for row in cal.reshape(-1,6))
            assert worlds['support_panel']==900+i and worlds['cal_support_panel']==920+i
            for key in ['object_index','cal_object_index']:
                assert all(len(set(row))==128 for row in worlds[key])
            population_stats.append((th.mean((0,1)).tolist(),cal.mean((0,1)).tolist()))
            idx=worlds['object_index']
        with np.load(ROOT/'labels50'/tag) as b,np.load(ROOT/'labels50_ultra'/tag) as c:
            y,z=b['labels'],c['labels']
            for f in (b,c):
                assert np.array_equal(f['object_index'],idx)
                assert np.isfinite(f['slip_speed']).all() and np.array_equal(f['labels'],f['slip_speed']<=.020)
                assert f['labels'].shape==(4,2,3,2,3,4,128)
            assert b['dt_ms']==.0125 and c['dt_ms']==.00625
            average=lambda arr: arr.mean((0,1,3,4,5,6))
            fine_sign.append(average(y)); ultra_sign.append(average(z))
            primary_stick.append(w.STATE_WEIGHTS@average(y))
            ultra_stick.append(w.STATE_WEIGHTS@average(z))
            diff.append(w.STATE_WEIGHTS@average(z!=y))
            full_flip.append(np.mean(y[:,:,1]!=y[:,:,2]))
    assert not dep_set.intersection(cal_set)
    assert len(dep_set)==len(cal_set)==6144
    result=dict(catalogs=12,deploy_unique_vectors=len(dep_set),calibration_unique_vectors=len(cal_set),shared_vectors=0,
                draws_per_resolution=663552,slots_per_resolution=884736,
                minimum_old_to_new_contour_rms_mm=novel,minimum_between_new_contour_rms_mm=pair,
                maximum_family_difference_inert_patch_mm=families_check,
                positive_weak_gauge_mm={f:[w.gauge_mm(f,h,1) for h in w.HEIGHTS] for f in w.FAMILIES},
                surface_isotropic_scale_and_direction=fields,primary_stick=intervals(primary_stick),
                ultra_stick=intervals(ultra_stick),fine_ultra_disagreement=intervals(diff),
                negative_positive_full_state_label_change=intervals(full_flip),
                stick_given_full_state_0_negative_positive=intervals(fine_sign),
                ultra_stick_by_state=intervals(ultra_sign),target_hashes={tag:hashlib.sha256((ROOT/tag).read_bytes()).hexdigest()
                                 for tag in ('world50.py','population50.py','solver50.py')})
    (ROOT/'world50_audit.json').write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps(result,indent=1))

if __name__=='__main__':
    main()
