"""Validate raw archives, reconstruct source-optimal gain and paired Brier independently."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'probe_speed_v37'))
from gate37 import load_artifacts
from gate36 import log_density, observations

CELLS=('shelf_weakening','shelf_strengthening','ridge_weakening','ridge_strengthening')
REGIMES=('strong','weak','null')
SHRINK=.325
BETA=.8289760825845313


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_reconstruction(art,x,y3,dirs,y4):
    """Direct row-wise log density and independently enumerated knot candidates."""
    z=np.empty(8); selected=np.empty(8); advantage=np.empty(8)
    for i in range(8):
        ll=np.zeros(len(art['sign']))
        for j in range(3):
            obs=observations(x[i:i+1,j],y3[i:i+1,j])
            ll+=log_density(art,obs,int(dirs[i,j]>0))[0]
        z[i]=softmax(ll) @ art['far_sign']
        near=observations(x[i:i+1,3],y4[i:i+1])
        row=softmax(ll+log_density(art,near,2)[0]) @ art['gain']
        grid=art['zgrid']
        if abs(z[i])<1.e-15:
            selected[i],advantage[i]=SHRINK,0.
            continue
        candidates=[0.,SHRINK,1.]
        candidates.extend(k/z[i] for k in grid if 0<k/z[i]<1)
        pairs=[(np.interp(w*z[i],grid,row),-abs(w-SHRINK),w) for w in candidates]
        value,_,selected[i]=max(pairs)
        advantage[i]=value-np.interp(SHRINK*z[i],grid,row)
    return z,selected,advantage


def main():
    out=ROOT/'results'
    run=json.loads((out/'completion.json').read_text())
    for file,expected in run['hashes'].items():
        assert digest(ROOT.parent/file)==expected,file
    assert run['blocks']==96
    art=load_artifacts()[1.1]
    expected_timing=0
    max_error=max_weight=max_z=0.
    lo,hi=1.,0.
    over=0
    seeds=set()
    groups={cell:[] for cell in CELLS}
    for c,cell in enumerate(CELLS):
        for catalog in range(12):
            for b in range(2):
                f=out/cell/f'catalog_{catalog:02d}'/f'block_{b:02d}.npz'
                m=json.loads(f.with_suffix('.json').read_text())
                assert digest(f)==m['sha256'] and m['fingerprint']==run['fingerprint']
                assert m['base_seed']==390_000_000+10_000_000*c+100_000*catalog+10_000*b
                seedlist=m['calibration_seeds']+m['deployment_seeds']+m['pair']
                assert not (set(seedlist)&seeds)
                seeds.update(seedlist)
                with np.load(f) as raw:
                    assert tuple(raw['methods'])==('fixed','optimizer')
                    assert tuple(raw['regimes'])==REGIMES and tuple(raw['signs'])==(1,-1)
                    a=raw['actions']
                    assert np.all((a==1).sum(-1)==115) and np.all((a==3).sum(-1)==13)
                    assert raw['p_base'].shape==(2,8,128,32)
                    assert raw['labels'].shape==(2,3,8,128) and raw['labels'].dtype==bool
                    assert raw['risk'].shape==(2,3,2,8)
                    # Read compressed arrays just once per block.
                    d={key:raw[key] for key in ('p_base','labels','risk','cal_x','first3_y',
                                                'fourth_y','dirs','z','weights')}
                    lo=min(lo,d['p_base'].min());hi=max(hi,d['p_base'].max())
                    over+=int((d['p_base']>1).sum())
                    assert lo>=-1.e-12 and hi<=1.+1.e-12 and np.isfinite(d['p_base']).all()
                    st=(catalog//2+c)%6 if catalog%2==0 and b==0 else -1
                    assert int(raw['timing_stratum'])==st
                    reps=3 if st>=0 else 0
                    assert raw['full_ms'].shape==(reps,2)
                    if reps:
                        assert (raw['full_ms']>0).all() and np.isfinite(raw['full_ms']).all()
                        np.testing.assert_array_equal(np.sort(raw['order'],1),np.tile([0,1],(3,1)))
                        expected_timing+=1
                    for s in range(2):
                        p=d['p_base'][s]
                        for r in range(3):
                            z,w,source_adv=source_reconstruction(art,d['cal_x'][s],d['first3_y'][s],
                                                                   d['dirs'][s],d['fourth_y'][s,r])
                            np.testing.assert_allclose(z,d['z'][s],rtol=0,atol=2.e-12)
                            np.testing.assert_allclose(w,d['weights'][s,r],rtol=0,atol=2.e-10)
                            max_z=max(max_z,np.abs(z-d['z'][s]).max())
                            max_weight=max(max_weight,np.abs(w-d['weights'][s,r]).max())
                            risk=[]
                            for scale in (SHRINK,w):
                                mult=np.exp(BETA*z*scale)[:,None,None]
                                q=p*mult/(1.+p*(mult-1.))
                                risk.append(((q-d['labels'][s,r,:,:,None])**2).mean((1,2)))
                            risk=np.stack(risk)
                            np.testing.assert_allclose(risk,d['risk'][s,r],rtol=0,atol=2.e-12)
                            max_error=max(max_error,np.abs(risk-d['risk'][s,r]).max())
                            groups[cell].append(np.stack((source_adv,risk[0]-risk[1])))
    assert expected_timing==24 and len(seeds)==96*18
    result=dict(blocks=96,regime_orientation_records=576,evaluation_seeds=96*16,
                qmc_bank_seeds=96*2,timing_clusters=expected_timing,
                max_abs_recomputed_Brier=max_error,max_abs_recomputed_weight=max_weight,
                max_abs_recomputed_signed_cue=max_z,base_range=[lo,hi],
                above_one_roundoff=over,mean_source_advantage_versus_target_reduction=
                {key:np.stack(entries).mean((0,2)).tolist() for key,entries in groups.items()})
    (ROOT/'audit39.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('v39 raw audit passed',flush=True)


if __name__=='__main__':
    main()
