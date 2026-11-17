"""Frozen dual-instrument policy with a four-session marginal likelihood."""
import itertools

import numpy as np

Q = 0.29
SECOND = (0.145, 0.0925)
ARTIFACT = 0.26582278481012656
READ_ERROR = 0.0392
SOURCE_LOG_CUTOFF = np.log(8.417543970276335e-5 / 4.175119807325477e-5)
METHODS = ('dual4_map','dual8_map','dual4_risk','single4_naive','gate','fixed','single4_adjusted','oracle')


def encode(reference, impact):
    """Map paid signed readings (...,8,2) to the four pair symbols."""
    reference,impact=np.asarray(reference),np.asarray(impact)
    if reference.shape != impact.shape or reference.shape[-2:] != (8,2):
        raise ValueError("Eight slots and simultaneous readings from two sensors are required.")
    active=(reference!=0)&(impact!=0)
    if np.any(active[...,0]!=active[...,1]):
        raise ValueError("The source instrument model assumes common physical activity.")
    flips=reference!=impact
    obj=np.where(active[...,0],1+flips[...,0].astype(np.uint8)+2*flips[...,1],0)
    return (5*obj[...,::2]+obj[...,1::2]).astype(np.uint8)


def pair_distr(moment, latch):
    """Condition on two active objects. Missing reads are marginalized separately."""
    binary=range(2)
    physical=np.array([[1-2*Q+moment,Q-moment],[Q-moment,moment]])
    state=(('physical',1-latch), (0,latch*(1-Q)), (1,latch*Q))
    observed=np.zeros((4,4))
    bits=np.array(list(itertools.product(binary,repeat=4)),dtype=int)
    first=bits[:,0]+2*bits[:,1]
    second=bits[:,2]+2*bits[:,3]
    for x0,x1 in itertools.product(binary,repeat=2):
        for (s0,p0),(s1,p1) in itertools.product(state,repeat=2):
            noiseless=np.array([x if s=='physical' else s for x in (x0,x1) for s in (s0,s1)])
            errors=(bits!=noiseless).sum(1)
            prob=READ_ERROR**errors*(1-READ_ERROR)**(4-errors)
            observed[first,second]+=physical[x0,x1]*p0*p1*prob
    assert np.isclose(observed.sum(),1)
    return observed


def tables():
    dual=np.zeros((2,2,2,25))
    single=np.zeros_like(dual)
    for regime,rho,hi in itertools.product(range(2),repeat=3):
        obs=pair_distr(SECOND[regime] if rho else Q*Q, ARTIFACT if hi else 0)
        # Indices 1..4 are 1+sensor0+2*sensor1. Index 0 means both unobserved.
        partial=np.ones((5,5))
        partial[1:,1:]=obs
        partial[0,1:]=obs.sum(0)
        partial[1:,0]=obs.sum(1)
        n0=np.array([0,1,0,1])
        observed_single=np.array([[obs[np.ix_(n0==a,n0==b)].sum() for b in (0,1)] for a in (0,1)])
        uni=np.ones((3,3))
        uni[1:,1:]=observed_single
        uni[0,1:]=observed_single.sum(0)
        uni[1:,0]=observed_single.sum(1)
        single_indices=np.array([0,1,2,1,2])
        dual[regime,rho,hi]=np.log(partial.ravel())
        single[regime,rho,hi]=np.log(uni[np.ix_(single_indices,single_indices)].ravel())
    return dual,single


TABLE,SINGLE=tables()
a,b=np.divmod(np.arange(25),5)
n=((a==2)|(a==4)).astype(int)+((b==2)|(b==4))
counts=np.array([[1-2*Q+x,2*(Q-x),x] for x in SECOND])
NAIVE=np.where((a>0)&(b>0),np.log(counts[0,n]/counts[1,n]),0.)


def log_odds(codes,table=TABLE):
    codes=np.asarray(codes)
    if codes.shape[-1:]!=(4,):
        raise ValueError("Four adjacent observed pair symbols required.")
    # Nuisance hypotheses are shared across all sessions in this pilot.
    def log_marginal(regime):
        hyp=table[regime].reshape(4,25)[:,codes].sum(-1)
        return np.logaddexp.reduce(hyp,axis=0)
    return log_marginal(0)-log_marginal(1)


def route(codes,method):
    codes=np.asarray(codes)
    shape=codes.shape[:-1]
    if method in ('dual8_map','gate'):
        return np.zeros(shape,dtype=bool)
    if method=='fixed':
        return np.ones(shape,dtype=bool)
    if method=='single4_naive':
        return NAIVE[codes].sum(-1)>1e-12
    if method=='single4_adjusted':
        return log_odds(codes,SINGLE)>1e-12
    if method in ('dual4_map','dual4_risk'):
        return log_odds(codes)>(SOURCE_LOG_CUTOFF if method=='dual4_risk' else 1e-12)
    raise ValueError("Unknown decision method.")
