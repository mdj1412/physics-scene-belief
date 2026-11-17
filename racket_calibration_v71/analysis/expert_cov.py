"""Exploratory check: propagate the cross-rubber covariance to sign-specific expert heads."""
import csv
import json
import numpy as np
from elastic import BUDGETS, GROUPS, HERE, R, read, train_tape

def feats(x,ep):
    f=(x-np.array(ep['mean']))/ep['std']
    return np.column_stack([np.ones(len(f)),f])
def matrix(f,branch):
    a=np.concatenate([f*(~branch[:,None]),f*branch[:,None]],axis=1)
    # Coefficient ordering (branches then affine rows, output channels last).
    return (a[:,None,:,None]*np.eye(3)[None,:,None,:]).reshape(len(f),3,24)
def label(y):
    return y[:,0]-y[:,2]>0
def calc():
    summaries=[]; info={}
    for group, spec in GROUPS.items():
        ep=spec['empirical']
        center=np.array([ep['expert_minus'],ep['expert_plus']]).ravel()
        coeff=[];covars=[]
        for source,other in GROUPS.items():
            if source==group: continue
            sx,sy=read(other['held'],'train')
            h=matrix(feats(sx,ep),label(sy))
            c=np.linalg.lstsq(h.reshape(-1,24),sy.ravel(),rcond=None)[0]
            e=sy-h@c
            coeff.append(c);covars.append(e.T@e/len(sy))
        sc=np.cov(np.array(coeff).T)
        cov=.85*sc+.15*np.diag(np.diag(sc))+.03**2*np.eye(24)
        prec=np.linalg.inv(cov); rn=np.linalg.inv(np.mean(covars,axis=0)+.05**2*np.eye(3))
        x,y=read(spec['held'],'train');xx,yy=read(spec['held'],'test')
        f=feats(x,ep);tf=feats(xx,ep)
        b=label(y); wrong=(f@ep['logistic']>0)!=b
        h=matrix(f,b);ht=h.transpose(0,2,1)@rn
        jac=ht@h; grad=(ht@(y-h@center)[...,None])[...,0]
        test_gate=tf@ep['logistic']>0
        tape=train_tape(len(x),min(spec['held']))
        info[group]=dict(max_eigen=float(np.linalg.eigvalsh(cov)[-1]),min_eigen=float(np.linalg.eigvalsh(cov)[0]))
        for budget in BUDGETS:
            ix=tape[:,:budget]; flip=2*wrong[ix].sum(1)>budget
            theta=center+np.linalg.solve(prec+jac[ix].sum(1),grad[ix].sum(1)[...,None])[...,0]
            tc=theta.reshape(256,2,4,3)
            low=tf@tc[:,0];high=tf@tc[:,1]
            pred=np.where((test_gate[None,:]^flip[:,None])[:,:,None],high,low)
            err=pred-yy
            ve=np.linalg.norm(err[...,:2],axis=-1).mean(1);we=abs(err[...,2]).mean(1)/R
            for d in range(256):summaries.append((group,d,budget,'gate_expert_cov',ve[d],we[d]))
    with (HERE/'cov_draws.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['rubber','draw','budget','method','velocity','spin']);w.writerows(summaries)
    (HERE/'cov_fit.json').write_text(json.dumps(info,indent=2))
if __name__=='__main__':calc()
