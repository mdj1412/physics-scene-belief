"""Moment-based, observation-only attribution of paired-object covariance."""
import json
import os
from pathlib import Path
import numpy as np
from pilot58 import ROOT, N_CATALOG, Q, H, P_SIGN_ERROR

def totals(codes):
    # Dimensions preceding material, replicate, pair enumerate factorial cells.
    first, second = codes//5, codes%5
    complete = (first>0) & (second>0)
    left = (np.maximum(1,first)-1).astype('u1')
    right = (np.maximum(1,second)-1).astype('u1')
    a,b = left%2, left//2
    c,d = right%2, right//2
    def aggregate(v):
        return np.sum(complete*v,axis=(-1,-2,-3),dtype='f8')
    return np.stack([aggregate(1), aggregate(a), aggregate(b), aggregate(c), aggregate(d),
                     aggregate(a*b), aggregate(c*d), aggregate(a*d), aggregate(b*c),
                     aggregate(a*c),aggregate(b*d)],axis=-1)

def estimates(s):
    a,b,c,d,ab,cd,ad,bc,ac,bd=np.moveaxis(s[...,1:]/s[...,:1],-1,0)
    within=((ab-a*b)+(cd-c*d))/2
    cross=((ad-a*d)+(bc-b*c))/2
    same=((ac-a*c)+(bd-b*d))/2
    d_single=2*P_SIGN_ERROR*(1-P_SIGN_ERROR)
    scale=(1-2*d_single)**2
    high=Q*(1-Q)*scale
    attenuation=np.sqrt(within/high)
    sensor=1-attenuation
    # The last two terms sum to the structural model's prediction of C_same.
    phys_same=cross/attenuation
    inst_same=sensor*high
    return np.stack([cross/within,sensor,same,cross,within,phys_same,inst_same,same-phys_same-inst_same],axis=-1)

def main():
    rows={}
    data=[]
    for i in range(N_CATALOG):
        with np.load(ROOT/'pilots'/f'panel_{i:02}.npz') as f:
            data.append(totals(f['codes']))
    data=np.stack(data)
    weights=np.random.default_rng(58077523).multinomial(N_CATALOG,np.ones(N_CATALOG)/N_CATALOG,size=5000)
    for key in np.ndindex(2,2,2,2):
        limited=data[(slice(None),)+key]
        est=estimates(limited.sum(0))
        bs=estimates(weights@limited)
        ci=np.quantile(bs,[.025,.975],axis=0)
        rows[','.join(map(str,key))]=np.column_stack((est,ci.T)).tolist()
    out=dict(names=['latent_icc','inferred_latch','same_sensor_cov','cross_sensor_cov','same_object_cov',
                    'physical_same_cov','instrument_same_cov','model_minus_residual_observed_minus_fit'],
             rows=rows,known_q=Q,known_sign_error=P_SIGN_ERROR,pointwise_catalog_bootstrap=5000)
    (ROOT/'moment_results58.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    main()
