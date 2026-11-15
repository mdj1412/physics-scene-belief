"""Independent linear-Gaussian closed-form test of update and tempering."""
import os,json
from pathlib import Path
import numpy as np
import torch
import compact
from compact import Compact
from physics import t,DT,DEV
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(1)
A=t([[1.,.2,-.1],[.3,-.2,.9]]);obs=t([[.2,-.1]])
def linear_measure(theta,action):
    p=torch.stack([(theta[...,2]+.4)/.8,(theta[...,3]+.1)/.2,(theta[...,4]-.35)/.55],-1)
    u=torch.special.ndtri(p)
    offset=torch.stack([.2*theta[...,0],.3*theta[...,1]],-1)
    return u@A.T+offset
original=compact.measure;compact.measure=linear_measure
cov=torch.linalg.inv(torch.eye(3,device=DEV,dtype=DT)+A.T@A/.05**2)
gc=t([[0,0],[0,1],[1,0],[1,1]]);res=obs-torch.stack([.2*gc[:,0],.3*gc[:,1]],1)
expected=res@A@cov/.05**2;S=A@A.T+.05**2*torch.eye(2,device=DEV,dtype=DT)
lp=-.5*torch.einsum('ki,ij,kj->k',res,torch.linalg.inv(S),res);prob=torch.softmax(lp,0)
records=[]
try:
    for steps in [1,4]:
        c=Compact(5,steps);s=c.update(c.prior(1),3,obs)
        record=dict(steps=steps,mean_error=float(abs(s.mean[0]-expected).max()),cov_error=float(abs(s.cov[0]-cov).max()),probability_error=float(abs(s.logp.exp()[0]-prob).max()))
        assert max(record[k] for k in ['mean_error','cov_error','probability_error'])<1e-9,record
        records.append(record)
finally:compact.measure=original
Path('results/linear_validation.json').write_text(json.dumps(records,indent=2));print(records)
