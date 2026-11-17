"""Pre-purchase choice of a fourth sensor, conditioning only on three post-swap readings."""
from functools import lru_cache
from itertools import combinations_with_replacement
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
V53 = ROOT.parent / 'impact_witness_v53'
sys.path.insert(0, str(V53))
from policy53 import (RETENTIONS, PRIORS, DRIFT_PRIOR, ERROR_MOMENTS, STD,
                      MAGNITUDE, DEAD_ZONE, likelihood_grid, load_model)

MODES = ('evsi', 'vote', 'uniform', 'no_old')


def instrument_bins(magnitude, sd, dead, subdivisions):
    """Gaussian cell probabilities for the negative, inactive and positive states."""
    edges = np.unique(np.r_[-np.inf, np.linspace(-magnitude-5*sd, magnitude+5*sd, subdivisions+1),
                            -dead, 0., dead, np.inf])
    centers = (edges[:-1]+edges[1:])*.5
    def cdf(state):
        return np.array([.5*math.erfc(-(v-state*magnitude)/(sd*math.sqrt(2))) for v in edges])
    return np.stack([np.diff(cdf(state)) for state in (-1,0,1)]), centers


def channels(subdivisions):
    """Endpoint observation probabilities for a reference pair (k) and impact pair (q)."""
    w,x = instrument_bins(MAGNITUDE[0], STD[0], DEAD_ZONE[0], subdivisions)
    r,y = instrument_bins(MAGNITUDE[1], STD[1], DEAD_ZONE[1], subdivisions)
    visible = (abs(x[:,None])>=DEAD_ZONE[0]) & (abs(y[None,:])>=DEAD_ZONE[1])
    ref,impact=[],[]
    for orientation in (-1,1):
        a = .5*np.outer(w[1],r[1])
        for s in (-1,1):
            a += .25*np.outer(w[s+1],r[orientation*s+1])
        ref.append(np.r_[a[visible],a[~visible].sum()])
        # Zero impact yields no directional information. Nonzero sign and weak reading are kept.
        impact.append(np.r_[.25*w[1+orientation],.25*w[1-orientation],.5])
    return np.array(ref),np.array(impact)


def information_table(channel, grid):
    valid = channel.sum(axis=0)>1e-16
    p0,p1=channel[:,valid]
    gap=p1-p0
    return np.array([np.sum(gap*gap / ((1-m)*p0+m*p1)) for m in grid])


def hidden_old_histories(reference, thresholds):
    """Marginal likelihood of unordered old four-pair histories given new retention k."""
    valid=reference.sum(axis=0)>0
    p0,p1=reference[:,valid]
    confidence=(p1-p0)/(p1+p0)
    group=np.digitize(confidence,thresholds)
    channels=np.stack([np.bincount(group,weights=a,minlength=len(thresholds)+1)
                       for a in (p0,p1)])
    lik=channels[0]+RETENTIONS[:,None]*(channels[1]-channels[0])
    out=[]
    for seq in combinations_with_replacement(range(lik.shape[1]),4):
        counts=np.bincount(seq,minlength=lik.shape[1])
        freq=math.factorial(4)/math.prod(math.factorial(int(x)) for x in counts)
        row=freq*np.prod(lik**counts,axis=1)
        out.append((1-DRIFT_PRIOR)*row+DRIFT_PRIOR*row[::-1])
    arr=np.array(out)
    np.testing.assert_allclose(arr.sum(axis=0),1.,rtol=0,atol=2e-13)
    return arr


@lru_cache(maxsize=4)
def tables(subdivisions=96, fine_old=False):
    ref,imp=channels(subdivisions)
    np.testing.assert_allclose(np.r_[ref.sum(1),imp.sum(1)],1.,atol=1e-14)
    grid=np.linspace(.0001,.9999,2049)
    cuts=[-.995,-.95,-.7,-.25,.25,.7,.95,.995] if fine_old else [-.98,-.5,.5,.98]
    history=hidden_old_histories(ref,cuts)
    coeff=load_model()
    np.testing.assert_allclose(coeff[0,2],coeff[1,2],atol=1e-12)
    # With alpha=posterior mean(q), expected source Brier improvement is
    # -(c2 + slope of c1 in q) times Var(E[q|new reading]).
    scale=-(coeff[0,2]+coeff[1,1]-coeff[0,1])
    assert scale>0
    return grid,information_table(ref,grid),information_table(imp,grid),history,float(scale)


def values54(weak3,ref3,mode='evsi',subdivisions=96,fine_old=False):
    weak3=np.asarray(weak3)
    if weak3.ndim<1 or weak3.shape[-1]!=3 or np.shape(ref3)!=weak3.shape:
        raise ValueError('Selection requires exactly three measured post-exchange pairs.')
    if mode not in MODES: raise ValueError(mode)
    grid,info_r,info_i,hist,scale=tables(subdivisions,fine_old)
    if mode=='no_old': hist=np.ones((1,len(RETENTIONS)))
    mu,m2=ERROR_MOMENTS['uniform' if mode=='uniform' else 'nominal']
    mass=likelihood_grid(weak3,ref3,'vote' if mode=='vote' else 'likelihood')*PRIORS
    mass=mass/mass.sum(-1,keepdims=True)
    weights=mass@hist.T
    moment=(mass*RETENTIONS)@hist.T/weights
    second=(mass*RETENTIONS**2)@hist.T/weights
    vk=np.maximum(0.,second-moment**2)
    mq=mu+(1-2*mu)*moment
    cov=(1-2*mu)*vk
    vq=(1-2*mu)**2*vk + (m2-mu**2)*(1-4*moment+4*second)
    gain_r=scale*(weights*cov**2*np.interp(moment,grid,info_r)).sum(-1)
    gain_i=scale*(weights*vq**2*np.interp(mq,grid,info_i)).sum(-1)
    return gain_r,gain_i


def select54(weak3,ref3,mode='evsi'):
    ref,impact=values54(weak3,ref3,mode)
    return impact>ref
