import numpy as np
import torch
from model import *
from acquisition import *
from common import *

def run():
    require();bank=Bank(4096);counts=[int(((bank.theta[:,0]==g)&(bank.theta[:,1]==c)).sum()) for g,c in [(0,0),(0,1),(1,0),(1,1),(0,2),(1,2)]]
    assert counts==[512]*4+[1024]*2,counts
    th=t([[0,1,0,0,0,.6,-1,0]]);v,vt,vn=velocity(th,0,(1,.1),details=True);assert abs(float(vt[0]))<1e-12
    _,vt,_=velocity(th,0,(1,.9),details=True);assert float(vt[0])>.5
    z=targets(th,t([-.001,.1]));assert torch.allclose(z[:,2:4],t([[1,.9]]));assert float(z[0,4])==0
    z=targets(th,t([.001,.1]));assert abs(float(z[0,4])-2**.5)<1e-12
    # Deterministic symmetric measurement: LMMSE covariance cancels although an observation identifies |h|.
    h=t([-2,-.5,.5,2]);event=t([0,1,1,0]);w=t([.25]*4)
    cov=float((w*(h-w@h)*(event-w@event)).sum());assert abs(cov)<1e-12
    rng=np.random.default_rng(982);y=h[:,None]+t(rng.normal(size=(4,2000)))*.1
    lp=-.5*((y[:,:,None]-h)/.1)**2;posterior=torch.softmax(lp,-1);pred=posterior@event
    risk=float(((pred-event[:,None])**2).mean());assert risk<.001
    # This abstract mechanism is NOT a population performance claim or part of the primary data.
    p=np.load(R/'data/pilot/public.npz');q=np.load(R/'data/pilot/private.npz');assert np.array_equal(p['context'][:,0],p['context'][:,1])
    x=t(p['factual'][0,0]);w=bank.fit(x);z=targets(bank.theta,t(p['context'][0,0]));scores=lmmse(bank,w,z)
    assert torch.isfinite(scores).all() and abs(float(w.sum())-1)<1e-12
    same=nested(bank,w,z,bank,w,z,M=512);assert torch.isfinite(same['risk']).all()
    # Direct Gaussian identity verifies matrix likelihood against explicit squared residual (up to shared constant).
    explicit=-.5*((bank.mu[0]-x)/noise(0)).square().sum(-1)
    assert torch.allclose(bank.fit(x),explicit.softmax(-1),atol=1e-11,rtol=1e-10)
    write(R/'validation.json',dict(prior_counts=counts,stick_slip_branch=True,contact_branch=True,public_context_paired=True,likelihood_checked=True,symmetric_covariance=cov,symmetric_no_observation_Brier=.25,symmetric_measured_Brier=risk))
    print('Validation passed',flush=True)
if __name__=='__main__':run()
