"""Source-population response prior for a state-dependent tangential elastic impulse."""
import csv
import json
import os
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
V70=HERE.parents[1]/'racket_transfer_v70'
GROUPS=json.loads((V70/'solver'/'metrics.json').read_text())['folds']
BUDGETS=(0,1,2,4,8,16,32)
R=.02
K=2/3

def read(cfgs,split):
    arrays=[np.genfromtxt(V70/'data'/'upstream'/split/f'racket_{c:02d}.csv',delimiter=',',names=True) for c in sorted(cfgs)]
    a=np.concatenate(arrays)
    return (np.column_stack([a['v_in_x'],a['v_in_z'],R*a['spin_before_y']]),
            np.column_stack([a['v_out_x'],a['v_out_z'],R*a['spin_after_y']]))
def rawfeat(x):
    return np.column_stack([abs(x[:,1]),abs(x[:,0]-x[:,2])])
def design(x,center,sd):
    z=(rawfeat(x)-center)/sd
    mat=np.zeros((len(x),3,4));s=x[:,0]-x[:,2]
    mat[:,0,2]=-s;mat[:,0,3]=-s*z[:,1]
    mat[:,1,0]=-x[:,1];mat[:,1,1]=-x[:,1]*z[:,0]
    mat[:,2,2]=s/K;mat[:,2,3]=s*z[:,1]/K
    return mat
def base(x):
    return x*np.array([1.,0.,1.])
def source_fit(other):
    merged=np.concatenate([x for x,_ in other])
    center=rawfeat(merged).mean(0); sd=rawfeat(merged).std(0)
    means=[]; noises=[]
    for x,y in other:
        h=design(x,center,sd);target=y-base(x)
        theta=np.linalg.lstsq(h.reshape(-1,4),target.reshape(-1),rcond=None)[0]
        res=target-h@theta
        means.append(theta);noises.append(res.T@res/len(x))
    # Four slope/intercept coefficients; fixed 0.02 spread and 0.05 m/s observation floor.
    prior=np.mean(means,axis=0);cov=np.cov(np.array(means).T)+.02**2*np.eye(4)
    noise=np.mean(noises,axis=0)+.05**2*np.eye(3)
    return center,sd,prior,np.linalg.inv(cov),np.linalg.inv(noise)
def train_tape(n,first,draws=256):
    return np.stack([np.random.default_rng(np.random.SeedSequence([710071,first,i])).choice(n,32,replace=False)
                     for i in range(draws)])
def run():
    records=[];info={}
    for rubber,par in GROUPS.items():
        held=par['held']
        other=[read(g['held'],'train') for n,g in GROUPS.items() if n!=rubber]
        center,sd,prior,pi,precision=source_fit(other)
        x,y=read(held,'train');xx,yy=read(held,'test')
        ids=train_tape(len(x),min(held))
        h=design(x,center,sd); residual=y-base(x)-h@prior
        ht=h.transpose(0,2,1)@precision
        hess=ht@h
        grad=(ht@residual[...,None])[...,0]
        z=(rawfeat(xx)-center)/sd
        s=xx[:,0]-xx[:,2]
        info[rubber]={'prior':prior.tolist(),'source_center':center.tolist(),'source_scale':sd.tolist(),
                      'prior_cov':np.linalg.inv(pi).tolist(),'residual_cov':np.linalg.inv(precision).tolist()}
        for budget in BUDGETS:
            idx=ids[:,:budget]
            theta=prior+np.linalg.solve(pi+hess[idx].sum(1),grad[idx].sum(1)[...,None])[...,0]
            e=np.clip(theta[:,0,None]+theta[:,1,None]*z[:,0],0.,1.)
            alpha=np.clip(theta[:,2,None]+theta[:,3,None]*z[:,1],0.,.8)
            v0=xx[:,0]-alpha*s;vn=-e*xx[:,1];rim=xx[:,2]+alpha*s/K
            ve=np.hypot(v0-yy[:,0],vn-yy[:,1]).mean(1);we=(abs(rim-yy[:,2])/R).mean(1)
            for d in range(len(ids)):
                records.append((rubber,d,budget,'elastic_response',ve[d],we[d]))
    with (HERE/'elastic_draws.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['rubber','draw','budget','method','velocity','spin']);w.writerows(records)
    (HERE/'elastic_fit.json').write_text(json.dumps({'job':os.environ.get('SLURM_JOB_ID'),'folds':info},indent=2))
if __name__=='__main__':run()
