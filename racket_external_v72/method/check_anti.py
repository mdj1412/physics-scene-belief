"""Independent NumPy replay of primary dimensionless Super Anti estimates."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'racket_calibration_v71'/'analysis'))
import elastic

def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def norm(x,y):
    return tuple(a*8/abs(x[:,1:2]) for a in (x,y))
def f(X,m,s):
    return np.column_stack([np.ones(len(X)),(X[:,[0,2]]-m)/s])
def H(x,c,sd):
    s=x[:,0]-x[:,2]; z=(abs(s)-c)/sd
    h=np.zeros((len(x),3,3));h[:,1,0]=-x[:,1];h[:,0,1]=-s;h[:,0,2]=-s*z
    h[:,2,1]=1.5*s;h[:,2,2]=1.5*s*z
    return h

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('gpu',type=Path);args=ap.parse_args()
    src=[norm(*elastic.read(c,'train')) for c in ((1,3,9),(2,4),(5,6),(7,),(8,),(10,))]
    x,y=map(np.concatenate,zip(*src))
    m=x[:,[0,2]].mean(0);sd=x[:,[0,2]].std(0)
    A=f(x,m,sd);sg=y[:,0]>y[:,2]
    muB=np.linalg.lstsq(A,y,rcond=None)[0]
    beta=np.zeros(3);reg=np.diag([0.,1.,1.])
    for _ in range(55):
        p=1/(1+np.exp(-np.clip(A@beta,-35,35)))
        update=np.linalg.solve(A.T@(A*(p*(1-p))[:,None])+reg,A.T@(p-sg)+reg@beta)
        beta-=update
        if np.linalg.norm(update)<1e-12:break
    heads=[np.linalg.lstsq(A[sg==side],y[sg==side],rcond=None)[0] for side in (False,True)]
    sc=abs(x[:,0]-x[:,2]); c,s=sc.mean(),sc.std()
    es=[];ds=[];ens=[];dns=[]
    for x,y in src:
        a=f(x,m,sd);b=np.linalg.lstsq(a,y,rcond=None)[0];r=y-a@b
        ds.append((b-muB).ravel());dns.append(r.T@r/(len(x)-3))
        h=H(x,c,s);u=y-x*np.array([1,0,1]);e=np.linalg.lstsq(h.reshape(-1,3),u.ravel(),rcond=None)[0];r=u-h@e
        es.append(e);ens.append(r.T@r/len(x))
    ds=np.array(ds);cov=ds.T@ds/len(ds)
    P=.1*np.linalg.inv(.85*cov+.15*np.diag(np.maximum(np.diag(cov),.0001)))
    W=np.linalg.inv(np.mean(dns,axis=0)+.0001*np.eye(3))
    U=np.linalg.inv(np.cov(np.array(es).T)+.0004*np.eye(3));V=np.linalg.inv(np.mean(ens,axis=0)+.0025*np.eye(3));e0=np.mean(es,axis=0)
    rr=read(args.input)
    def col(t):return np.array([[float(r['vx_'+t]),float(r['vz_'+t]),float(r['omega_'+t])*float(r['radius_px'])] for r in rr])
    x,y=norm(col('in'),col('out'));order=np.argsort(x[:,0]-x[:,2],kind='stable');x,y=x[order],y[order];rr=[rr[i] for i in order]
    real=y[:,0]>y[:,2]
    valid=np.array([r.get('sign_resolved','1')!='0' for r in rr])&(y[:,0]!=y[:,2])
    a=f(x,m,sd);route=a@beta>0
    orig=np.where(route[:,None],a@heads[1],a@heads[0]);alt=np.where(route[:,None],a@heads[0],a@heads[1])
    def risk(p):
        delta=p-y
        return np.linalg.norm(delta[:,:2],axis=1)+abs(delta[:,2])
    ah=np.stack([np.kron(row[None],np.eye(3)) for row in a]);eh=H(x,c,s);base=x*np.array([1,0,1])
    pool=np.rint(np.linspace(0,len(rr)-1,8)).astype(int);test=np.setdiff1d(np.arange(len(rr)),pool)
    gpu=read(args.gpu/'draws.csv');maxerr={};count=0
    for k in (4,8):
        for i,inds in enumerate(itertools.combinations(pool,k)):
            paid=np.array(inds);voted=paid[valid[paid]]
            flip=2*np.sum(route[voted]!=real[voted])>len(voted)
            lossflip=risk(alt)[paid].sum()<risk(orig)[paid].sum()
            S=P.copy();rhs=np.zeros(9);T=U.copy();r=np.zeros(3)
            for t in paid:
                a1,a2=ah[t],eh[t]
                S+=a1.T@W@a1;rhs+=a1.T@W@(y[t]-a[t]@muB)
                T+=a2.T@V@a2;r+=a2.T@V@(y[t]-base[t]-a2@e0)
            b=muB+np.linalg.solve(S,rhs).reshape(3,3);e=e0+np.linalg.solve(T,r)
            sl=x[:,0]-x[:,2];alpha=np.clip(e[1]+e[2]*(abs(sl)-c)/s,0,.8)
            py=np.column_stack([x[:,0]-alpha*sl,-np.clip(e[0],0,1)*x[:,1],x[:,2]+1.5*alpha*sl])
            for name,yp in [('majority',alt if flip else orig),('continuous_loss_gate',alt if lossflip else orig),('single_affine_eb',a@b),('elastic_impulse',py)]:
                ref=next(r for r in gpu if r['mode']=='fixed' and int(r['k'])==k and int(r['draw'])==i and r['method']==name)
                delta=yp[test]-y[test]
                metrics=dict(norm_velocity=np.mean(np.linalg.norm(delta[:,:2],axis=1))/8,norm_spin=abs(delta[:,2]).mean()/8)
                v=test[valid[test]]
                if len(v):metrics['output_sign_acc']=np.mean((yp[v,0]>yp[v,2])==real[v])
                for key,z in metrics.items():maxerr[key]=max(maxerr.get(key,0.),abs(z-float(ref[key])))
                count+=1
    assert max(maxerr.values())<1e-8 and count==284
    (args.gpu/'audit.json').write_text(json.dumps(dict(count=count,max_absolute_error=maxerr,job=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID")),indent=2))
if __name__=='__main__':main()
