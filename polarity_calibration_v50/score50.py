"""Paired scoring with an independent calibration prefix shared by deployment objects of one environment."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'contact_transfer_v49'))
from score49 import joint_state_weights

RHO=(0.,.5,1.)
KAPPA=(0.,.25,.5,.75,1.)
BUDGET=(0,1,4,8)
METHODS=('hard_local','frozen_local','frozen_fixed','hard_fixed','unsigned_fit','unsigned_sym',
         'soft_local','soft_fixed','known_kappa','original')
PRED=('original','local_plus','local_minus','fixed_plus','fixed_minus','local_unsigned_fit')


def interval(x,n=1):
    x=np.asarray(x)
    m=x.mean(0)
    half=x.std(0,ddof=1)*t.ppf(1-.025/n,len(x)-1)/np.sqrt(len(x))
    return np.stack([m,m-half,m+half],axis=-1)


def components(p,y):
    # p [M,F,H,S,A,V,E,O,N], y [F,H,T,R,X,E,O]
    mean_y=y.mean((3,4))
    pbar=p.mean((4,5,8)) # M,F,H,S,E,O
    p2=(p*p).mean((4,5,7,8)) # M,F,H,S,E
    # Predictions and labels must pair the SAME object, NOT independently average objects.
    kernel=p2[:,:,:,:,None,:]+np.einsum('mfhseo,fhteo->mfhste',1.-2*pbar,mean_y,optimize=True)/p.shape[-2]
    gap=np.stack([((p[a]-p[b])**2).mean((3,4,6,7)) for a,b in ((1,2),(3,4))])
    return kernel,gap # M,F,H,S,T,E and 2,F,H,S,E


def integrate(kernel,gap,rho,k):
    # Reduce family, height and hidden states but retain environment.
    risks=np.zeros((6,4)); gaps=np.zeros((2,4))
    for h in range(2):
        for m in range(6):
            w=joint_state_weights(rho,k,h==0 and m in (3,4))
            risks[m]+=np.einsum('fste,st->e',kernel[m,:,h],w)/8.
        for site in range(2):
            w=joint_state_weights(rho,k,h==0 and site==1).sum(1)
            gaps[site]+=np.einsum('fse,s->e',gap[site,:,h],w)/8.
    return risks,gaps


def blend(plus,minus,g2,alpha,alpha2):
    # E_R Brier for probability alpha*p(+) + (1-alpha)*p(-).
    return minus+(plus-minus)*alpha-g2*(alpha-alpha2)


def score_panel(i,label_path='labels50'):
    name=f'panel_{i:02d}.npz'
    with np.load(ROOT/'forecast50'/name) as d:
        names=[str(v) for v in d['methods']]
        p=d['p'][[names.index(key) for key in PRED]]
    with np.load(ROOT/label_path/name) as lab:
        y=lab['labels']; slots=lab['object_slots'] if 'object_slots' in lab else np.arange(y.shape[-1])
    if len(slots)!=p.shape[-2]: p=p.take(slots,axis=-2)
    with np.load(ROOT/'cal50'/name) as cal:
        hats=cal['khat']
    assert hats.shape[:3]==(5,4,4) and np.all(hats[:,0]==.5)
    assert p.shape[:4]==(6,4,2,3) and y.shape==(4,2,3,2,3,4,p.shape[-2])
    assert np.logical_or(y==0,y==1).all()
    np.testing.assert_allclose(p[1,:,1],p[3,:,1],rtol=0,atol=1e-12)
    np.testing.assert_allclose(p[2,:,1],p[4,:,1],rtol=0,atol=1e-12)
    kernel,g2=components(p,y)
    # Explicit binary squared-error check on rotating condition per panel incl radius and offset.
    m,f,h,s,full=(i%6,i%4,i%2,i%3,(i+1)%3)
    a=p[m,f,h,s]
    actual=np.mean([((a-y[f,h,full,r,x,...,None])**2).mean((0,1,3,4)) for r,x in np.ndindex(2,3)],axis=0)
    np.testing.assert_allclose(actual,kernel[m,f,h,s,full],rtol=0,atol=3e-12)
    # Retain per-env risk for matched analysis, and independently recombine calibration/deployment session laws.
    out=np.empty((3,5,5,4,len(METHODS),4))
    mean=hats.mean(-1); second=(hats*hats).mean(-1); hard=(hats>=.5).mean(-1)
    for r,rho in enumerate(RHO):
        for c,k in enumerate(KAPPA):
            loss,d2=integrate(kernel,g2,rho,k)
            original,lp,lm,fp,fm,unsigned=loss
            result=out[r,c]  # calibration kappa, K, method, environment
            result[:,:,0]=blend(lp,lm,d2[0],hard,hard)
            result[:,:,1]=lp
            result[:,:,2]=fp
            result[:,:,3]=blend(fp,fm,d2[1],hard,hard)
            result[:,:,4]=unsigned
            result[:,:,5]=blend(lp,lm,d2[0],.5,.25)
            result[:,:,6]=blend(lp,lm,d2[0],mean,second)
            result[:,:,7]=blend(fp,fm,d2[1],mean,second)
            result[:,:,8]=blend(lp,lm,d2[0],k,k*k)
            result[:,:,9]=original
            # Compare moment integration with literal session-by-session Brier using only sample prefix decisions.
            sample=hats[c,2,0]
            brute=(sample*lp[0]+(1-sample)*lm[0]-sample*(1-sample)*d2[0,0]).mean()
            np.testing.assert_allclose(result[c,2,6,0],brute,rtol=0,atol=2e-13)
    return out


def main(label_dir,tag):
    assert os.getenv('SLURM_JOB_ID')
    envs=np.stack([score_panel(i,label_dir) for i in range(12)])
    full=envs.mean(-1)
    vals=np.stack([full[:,:,c,c] for c in range(5)],axis=2) # N,rho,k,K,method
    np.testing.assert_allclose(vals[:,:,:,0,0],vals[:,:,:,0,1],atol=1e-12)
    np.testing.assert_allclose(vals[:,:,:,0,5],vals[:,:,:,0,6],atol=1e-12)
    margins=vals.mean((1,2))
    d=margins[:,2,0,None]-margins[:,2,1:6]
    cond=vals[:,:,:,2,0,None]-vals[:,:,:,2,1:5]
    n=vals.shape[0]
    shifts=full[:,:,:,:,2,0]-full[:,:,:,:,2,1]
    soft=margins[:,:,6,None]-margins[:,:,[0,1,3,4,5,7,8]]
    kappa=vals.mean(1)
    results=dict(methods=METHODS,rho=RHO,kappa=KAPPA,budget=BUDGET,catalogs=n,loss=vals.mean(0),
                 uniform_mean=margins.mean(0),kappa_average=kappa.mean(0),
                 primary_diff_hard_K4=interval(d,5),primary_count_lower=(d<0).sum(0),
                 conditional_diff_K4=interval(cond,60),conditional_wins=(cond<0).sum(0),
                 hard_minus_frozen_by_k=interval(kappa[:,:,:,0]-kappa[:,:,:,1],20),
                 hard_minus_unsigned_by_k=interval(kappa[:,:,:,0]-kappa[:,:,:,4],20),
                 matched_soft_effects=interval(soft,28),cross_law_hard_minus_frozen=shifts.mean((0,1)),
                 cross_law_local_losses=full[:,:,:,:,2,:].mean((0,1)),
                 heldout='Independent calibration vs deployment specimen supports; policy never sees deployment Y.',
                 intervals='Paired catalog Student-t, Bonferroni counts: 5 primary, 60 by-cell, 28 secondary soft.')
    np.savez_compressed(ROOT/f'catalog_scores{tag}.npz',matched=vals,cross=full,environment=envs,methods=METHODS)
    (ROOT/f'score{tag}.json').write_text(json.dumps(results,default=lambda x:np.asarray(x).tolist(),indent=1))
    print('equal-cell K=4 mean',margins.mean(0)[2], 'primary diffs',results['primary_diff_hard_K4'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',default='labels50')
    parser.add_argument('--tag',default='50')
    args=parser.parse_args()
    main(args.labels,args.tag)
