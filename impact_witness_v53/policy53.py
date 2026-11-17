"""Frozen estimators for a purchased weak + impact witness after three reference pairs."""
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
V52 = ROOT.parent / 'postswap_adaptation_v52'
sys.path.insert(0, str(V52))
from policy52 import (V50, bank, witnesses, load_model, likelihood_grid, mean_grid,
                      posterior_single, DRIFT_PRIOR, RETENTIONS, PRIORS,
                      DEAD_ZONE, MAGNITUDE, STD, densities, readable)
import world50

METHODS = ('impact_joint', 'impact_selector', 'impact_vote_selector', 'impact_post_vote',
           'impact_joint_masked', 'reference4_joint', 'impact_joint_uniform',
           'impact_joint_noisy', 'impact_joint_fixed_e', 'reference4_uniform')
EPS = (0., .25, .5)
# Moments of .5*Beta(1,4), Uniform(0,.5), or point mass at .10.
ERROR_MOMENTS = {'nominal': (.1, 1/60), 'uniform': (.25, 1/12), 'fixed': (.1, .01)}


@lru_cache(maxsize=12)
def impact_bank(panel):
    """Evaluator-only. Rebuild the actual calibration specimen's signed loaded shape."""
    weak, full, order = bank(panel)
    # Opposite endpoint interventions share specimen, bias and sensor noise. The
    # difference is twice the signed, noise-free reference response at k=1.
    f = full[...,3]
    wstate = np.sign(f[-1] - f[0]).astype(np.int8)
    assert np.all((f==f[0]) | (f==f[-1]))
    active = wstate != 0
    np.testing.assert_array_equal(active, f[-1]!=f[0])
    assert np.all(np.abs(f[-1]-f[0])[active]>.4)
    rstate = np.where(f==f[-1], wstate, -wstate)
    with np.load(V50 / 'cal50' / f'panel_{panel:02d}.npz') as data:
        fam = data['family'][...,3]
        height = data['height'][...,3]
        ids = data['cal_object_id']
        assert np.all(np.diff(np.sort(ids[...,:4],axis=-1),axis=-1)>0)
    generator = np.random.default_rng([5302719, panel])
    impact_flip, read_error = generator.random((2,*wstate.shape))
    table = np.array([[[world50.gauge_mm(family,h,s,load_fraction=1.)
                        for s in (-1,0,1)] for h in world50.HEIGHTS] for family in world50.FAMILIES])
    assert np.all(table[...,0]<0) and np.all(table[...,1]==0) and np.all(table[...,2]>0)
    hindex = np.searchsorted(world50.HEIGHTS, height)
    sign = []
    for e in EPS:
        actual = np.where(impact_flip<e,-rstate,rstate)
        # Two points +/-1.5 mm apart from nominal height yield the signed change
        # in contour slope specified at impact; branch chosen before deployment Y.
        gauge = table[fam,hindex,actual+1]
        np.testing.assert_array_equal(np.sign(gauge),actual)
        sign.append(np.sign(gauge).astype(np.int8))
    sign = np.stack(sign)[:,:,:,order]
    noisy = np.where((read_error<.1)[:,order],-sign,sign).astype(np.int8)
    return sign,noisy


def witnesses53(panel,old_k_idx,new_k_idx,eps_idx):
    old_w,old_ref,new_w,new_ref = witnesses(panel,old_k_idx,new_k_idx,4)
    signs,noisy = impact_bank(panel)
    return old_w,old_ref,new_w,new_ref,signs[eps_idx,new_k_idx],noisy[eps_idx,new_k_idx]


def direction_likelihood(weak,sign,method,sensor_error=0.):
    """Return L(flip), L(keep)-L(flip); observed zero contributes no orientation."""
    x = np.asarray(weak,dtype=float)
    s = np.asarray(sign)
    if x.shape!=s.shape or not np.isin(s,[-1,0,1]).all():
        raise ValueError('impact observation requires matching ternary signs')
    if not 0<=sensor_error<=.5:
        raise ValueError('invalid sign error')
    if method=='likelihood':
        d = densities(x,MAGNITUDE[0],STD[0])
        keep = np.where(s>0,d[...,1],d[...,2])
        flip = np.where(s>0,d[...,2],d[...,1])
        valid = s!=0
    elif method=='vote':
        keep = ((x>0)==(s>0)).astype(float)
        flip = 1-keep
        valid = (s!=0) & (np.abs(x)>=DEAD_ZONE[0])
    else:
        raise ValueError(method)
    noisy_flip = (1-sensor_error)*flip + sensor_error*keep
    return np.where(valid,noisy_flip,1.), np.where(valid,(1-2*sensor_error)*(keep-flip),0.)


def mixed53(ow,of,nw3,nf3,iw,sign,method='likelihood',sensor_error=0.,prior='nominal'):
    """Marginalize over post reference retention, preserve/complement and unknown e."""
    old_lik = likelihood_grid(ow,of,method)
    new_lik = likelihood_grid(nw3,nf3,method)
    mu,mu2 = ERROR_MOMENTS[prior]
    k = RETENTIONS
    b = 1-2*k
    expected_q = k+b*mu
    expected_q2 = k*k+2*k*b*mu+b*b*mu2
    intercept,slope = direction_likelihood(iw,sign,method,sensor_error)
    a = intercept[...,None]
    d = slope[...,None]
    ell = a+d*expected_q
    weighted_q = a*expected_q+d*expected_q2
    mass_preserve = (1-DRIFT_PRIOR)*old_lik*new_lik*PRIORS
    mass_flip = DRIFT_PRIOR*old_lik[...,::-1]*new_lik*PRIORS
    mass = mass_preserve+mass_flip
    denom = (mass*ell).sum(-1)
    q = (mass*weighted_q).sum(-1)/denom
    expected_e_lik = a*mu+d*(k*mu+b*mu2)
    ep = (mass*expected_e_lik).sum(-1)/denom
    new = mean_grid(new_lik*ell)
    return dict(alpha_joint=q,post_k=(mass*ell*k).sum(-1)/denom,
                drift_prob=(mass_flip*ell).sum(-1)/denom,post_e=ep,
                old=mean_grid(old_lik),new=new)


def three_actions(d,model):
    c = model[0]+d['alpha_joint'][...,None]*(model[1]-model[0])
    def risk(a):
        return c[...,0]+a*(c[...,1]+a*c[...,2])
    old,new = d['old'],d['new']
    costs = np.stack((risk(old),risk(new),c[...,3]),axis=-1)
    action = np.argmax(costs<=costs.min(-1,keepdims=True)+1e-14,axis=-1).astype(np.int8)
    return action,np.where(action==0,old,np.where(action==1,new,.5))


def post_vote(nw3,nf3,iw,sign):
    # Identical counts, Laplace pseudocounts and threshold as v52; the fourth
    # reference polarity is replaced by the directly observed impact polarity.
    ok = readable(nw3,nf3)
    same = (ok & ((nw3>0)==(nf3>0))).sum(-1)
    wi = (sign!=0) & (np.abs(iw)>=DEAD_ZONE[0])
    same += (wi & ((iw>0)==(sign>0))).astype(int)
    return (1+same)/(2+ok.sum(-1)+wi)


def choice53(method,ow,of,nw4,nf4,impact_sign,noisy_sign,model):
    if method not in METHODS:
        raise ValueError(method)
    ref4 = method.startswith('reference4')
    nref = 4 if ref4 else 3
    nw,nf = nw4[...,:nref],nf4[...,:nref]
    iw = nw4[...,3]
    sign = noisy_sign if method=='impact_joint_noisy' else impact_sign
    if ref4 or method=='impact_joint_masked':
        sign = np.zeros_like(iw)
    if method=='impact_post_vote':
        alpha = post_vote(nw,nf,iw,sign)
    else:
        prior = 'uniform' if method.endswith('uniform') else 'fixed' if method.endswith('fixed_e') else 'nominal'
        mode = 'vote' if method=='impact_vote_selector' else 'likelihood'
        se = .1 if method=='impact_joint_noisy' else 0.
        d = mixed53(ow,of,nw,nf,iw,sign,mode,se,prior)
        if method in ('impact_selector','impact_vote_selector'):
            return three_actions(d,model)
        alpha = d['alpha_joint']
    return np.zeros(alpha.shape,dtype=np.int8),alpha
