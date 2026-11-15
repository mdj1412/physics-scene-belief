"""Post-outcome explanatory counterexample; not part of primary physics tests."""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from common import *
require();labels=np.array([0.,0.,1.,1.]);weights=np.full(4,.25);gx,gw=hermgauss(80);gw=gw/np.sqrt(np.pi)
def evaluate(mu):
    sep=(mu[:,None]-mu[None,:])**2
    # sqrt(2)*binary target, so half squared disagreement equals (label_i-label_j)^2.
    score=np.mean((labels[:,None]-labels[None,:])**2*(-np.expm1(-sep/8)))
    risk=0.
    for i in range(4):
        y=mu[i]+np.sqrt(2)*gx;ll=-.5*(y[:,None]-mu[None,:])**2
        post=np.exp(ll-ll.max(1,keepdims=True));post/=post.sum(1,keepdims=True);p=post@labels
        risk+=weights[i]*np.sum(gw*(p-labels[i])**2)
    covariance=np.sum(weights*(labels-.5)*(mu-np.sum(weights*mu)))
    gain=covariance**2/(1+np.sum(weights*(mu-np.sum(weights*mu))**2))
    return dict(pair_score=float(score),expected_Brier=float(risk),LMMSE_gain=float(gain))
bad=evaluate(np.array([-4.,4.,-4.,4.]));good=evaluate(np.array([-.5,-.5,.5,.5]))
assert abs(bad['expected_Brier']-.25)<1e-12 and bad['pair_score']>good['pair_score'] and good['expected_Brier']<.25
write(R/'mixture_counterexample.json',dict(kind='post_outcome_abstract_mechanism_not_primary_physics',factual_likelihoods_equal=True,prior=[.25]*4,labels=labels.tolist(),noise_sd=1,uninformative_mixture=bad,informative_weak_view=good,quadrature_order=80))
print(read(R/'mixture_counterexample.json'))
