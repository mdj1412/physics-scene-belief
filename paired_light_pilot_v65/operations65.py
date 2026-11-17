"""Device-free cost, missingness and timing diagnostic on fixed v64 evidence."""
from pathlib import Path
import json, math, os, sys
import numpy as np

ROOT=Path(__file__).resolve().parent
ARCH=ROOT.parent/'reflective_field_v64'
sys.path.insert(0,str(ARCH))
import illumination64 as capture
from sim_score64 import base,LEGACY

def later_diode(wave, lag):
    delayed=wave.copy()
    if lag:
        ix=np.clip(np.arange(wave.shape[-1])-lag,0,wave.shape[-1]-1)
        delayed[...,1,:]=wave[...,1,ix]
    return delayed

def classify(codes):
    return base.classify(base.split_codes(codes).reshape(-1,8))[:,0].reshape(codes.shape[:-1])

def main():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lags=(-16,-8,-4,-1,0,1,4,8,16)
    totals={k:[] for k in lags}
    check=[]
    for i in range(24):
        mats=((np.arange(4)+i)%4)//2
        with np.load(LEGACY/'scored63'/f'panel_{i:02d}.npz') as f:
            gate=[f['gate'][:,mats==m].ravel().astype(bool) for m in (0,1)]
        rfrac=np.array([(~a).mean() for a in gate])
        with np.load(ARCH/'pilots'/f'panel_{i:02d}.npz') as f:
            codes=f['codes'][:,2,1]
        with np.load(ARCH/'fixtures'/f'panel_{i:02d}.npz') as f:
            features=capture.cam.optical_features(f['frames'])
            wave,elec,reset,ep=f['wave'],f['electronic'][1],f['reset_id'],int(f['episode'])
        def cam_sign(lag):
            return np.stack([capture.cam.read_features(tuple(x[d] for x in features),
                        later_diode(wave[d],lag),2) for d in range(4)])
        orig=cam_sign(0)
        def encoded(signs):
            return np.stack([capture.cam.encode(elec,s,reset) for s in signs])
        np.testing.assert_array_equal(encoded(orig),codes[...,ep,:])
        upper=classify(codes)
        full=0.0; reduced=0.0; q=0.0
        for d in (0,2):
            for m in (0,1):
                l,r=upper[d,:,:,m].ravel(),upper[d+1,:,:,m].ravel()
                a=gate[m]
                # material has the same held-out gate choices under both lamps.
                same=(l==r)
                # Enumerate counts of actual I and R prefixes without allocating product matrix.
                fl=np.count_nonzero((l|False)!=(r|False))*(~a).sum()
                fl+=np.count_nonzero((l|True)!=(r|True))*a.sum()
                full+=fl/(len(l)*len(a))/4
                reduced+=(~same).sum()*rfrac[m]/len(l)/4
                q+=(~same).mean()/4
        check.append((full,reduced,q))
        for lag in lags:
            signs=cam_sign(lag)
            rc=encoded(signs)
            flip=classify(rc)!=upper[...,ep]
            totals[lag].append([float((signs!=orig).mean()),float(flip.mean()),
                                float((flip*rfrac).mean())])
    full,shortcut,q=np.mean(check,0)
    assert np.max(abs(np.asarray(check)[:,0]-np.asarray(check)[:,1]))<1e-14
    expected=json.loads((ARCH/'sim_score64.json').read_text())['lamp_switch_uniform_width']
    np.testing.assert_allclose([q,full],[v[0] for v in expected[:2]],atol=1e-14,rtol=0)
    assert np.all(np.array(totals[0])==0)
    def zero_required(threshold):return math.ceil(math.log(.05)/math.log1p(-threshold))
    result=dict(source='v64 simulated archived panel / same v63 deployment distribution',
        p=full,q=q,reference_given_discordant=full/q,
        fraction_prefix_with_5pct_agreement_audit=q+.05*(1-q),
        shutter_stress_by_lag={str(k):np.mean(v,0).tolist() for k,v in totals.items()},
        shutter_stress_columns=['clip code change including missing','upper xor','expected purchase xor'],
        same_lamp_fixtures='24 catalogs x 2 surfaces each, one 8-impact bank per surface, replayed over paired mechanisms and angular conditions, 48 distinct physical synthetic banks',
        zero_independent_shams_needed={'half_point_p':zero_required(full/2),'fifth_point_p':zero_required(full/5)},
        zero_shams_upper={str(n):1-.05**(1/n) for n in (48,100,192,1000)},
        max_per_catalog_difference=float(abs(np.array(check)[:,0]-np.array(check)[:,1]).max()),
        job=os.environ['SLURM_JOB_ID'])
    (ROOT/'operations65.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
