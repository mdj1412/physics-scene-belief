"""Independent NumPy replay of all primary measured Feint OX folds on CPU."""
import csv
import itertools
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'racket_calibration_v71'/'analysis'))
import elastic

def fromcsv(p):
    with p.open() as f:return list(csv.DictReader(f))

def design(x,mean,sd):
    return np.column_stack([np.ones(len(x)),(x-mean)/sd])

def main():
    groups=[(1,3,9),(2,4),(5,6),(7,),(8,),(10,)]
    sources=[elastic.read(cfg,'train') for cfg in groups]
    x,y=map(np.concatenate,zip(*sources));mean=x.mean(0);sd=x.std(0);a=design(x,mean,sd)
    ysign=y[:,0]>y[:,2]
    beta=np.zeros(4); reg=np.diag([0.,1.,1.,1.])
    for _ in range(50):
        prob=1/(1+np.exp(-np.clip(a@beta,-35,35)))
        db=np.linalg.solve(a.T@((prob*(1-prob))[:,None]*a)+reg,a.T@(prob-ysign)+reg@beta)
        beta-=db
        if np.max(abs(db))<1e-11:break
    heads=[np.linalg.lstsq(a[ysign==s],y[ysign==s],rcond=None)[0] for s in (False, True)]
    B=np.linalg.lstsq(a,y,rcond=None)[0]
    coefficients=[]; residuals=[]
    for xx,yy in sources:
        fa=design(xx,mean,sd)
        c=np.linalg.lstsq(fa,yy,rcond=None)[0];e=yy-fa@c
        coefficients.append((c-B).reshape(-1));residuals.append(e.T@e/(len(e)-4))
    dif=np.stack(coefficients);cov=dif.T@dif/len(dif)
    P=.1*np.linalg.inv(.85*cov+.15*np.diag(np.maximum(np.diag(cov),.0001)))
    noise=np.mean(residuals,axis=0)+.0001*np.eye(3)
    ec,es,mu,Ep,En=elastic.source_fit(sources)
    root=HERE.parent/'data'
    maxerr={}; counts={}; shots=[]
    for file,out,ix in [('feint_fig12_paired.csv','feint',[0,2]),('feint_ox_digitized.csv','feint_derived',[0,1,2])]:
        rows=fromcsv(root/file); idx={r['id']:i for i,r in enumerate(rows)}
        reference=fromcsv(HERE/out/'primary_rows.csv')
        x=np.array([[0.,-30/3.6,.019*math.tau*float(r['n_in_rps'])] for r in rows])
        y0=np.array([[float(r['u_out_mps']),float(r.get('e_normal','0'))*30/3.6,.019*math.tau*float(r['n_out_rps'])] for r in rows]); z=y0[:,ix]
        f=design(x,mean,sd);route=f@beta>0
        pred=np.where(route[:,None],f@heads[1],f@heads[0])[:,ix]
        alt=np.where(route[:,None],f@heads[0],f@heads[1])[:,ix]
        positive=z[:,0]>z[:,-1];valid=z[:,0]!=z[:,-1]
        def epcost(v):
            d=np.abs(v-z);return np.linalg.norm(d[:,:-1],axis=1)+d[:,-1]
        maxerr[out]={m:0. for m in ['velocity','spin','output_sign_acc','route_sign_acc','flip','orientation_acc']}
        for row in reference:
            cal=np.array([idx[s] for s in row['paid'].split('|')]);test=np.array([idx[s] for s in row['held'].split('|')]);m=row['method']
            if m in ('majority','continuous_loss_gate'):
                if m=='majority':
                    voted=cal[valid[cal]]
                    flip=2*(route[voted]!=positive[voted]).sum()>len(voted)
                else:flip=epcost(alt)[cal].sum()<epcost(pred)[cal].sum()
                yh=(alt if flip else pred)[test]
            elif m=='single_affine_eb':
                # Direct posterior assembly over paid samples, independent of CUDA tensor contractions.
                g=P.copy(); score=np.zeros(12);noise_m=np.linalg.inv(noise[np.ix_(ix,ix)])
                for i in cal:
                    J=np.kron(f[i:i+1],np.eye(3))[ix]
                    g+=J.T@noise_m@J
                    score+=J.T@noise_m@(z[i]-J@B.ravel())
                posterior=B+np.linalg.solve(g,score).reshape(4,3)
                yh=(f[test]@posterior)[:,ix]
            else:
                F=elastic.design(x,ec,es)[:,ix]
                base=elastic.base(x)[:,ix]
                g=Ep.copy(); score=np.zeros(4)
                noise_m=np.linalg.inv(np.linalg.inv(En)[np.ix_(ix,ix)])
                for i in cal:
                    g+=F[i].T@noise_m@F[i];score+=F[i].T@noise_m@(z[i]-base[i]-F[i]@mu)
                coef=mu+np.linalg.solve(g,score)
                u=(elastic.rawfeat(x[test])-ec)/es
                alpha=np.clip(coef[2]+coef[3]*u[:,1],0,.8);e=np.clip(coef[0]+coef[1]*u[:,0],0,1)
                slip=x[test,0]-x[test,2]
                yh=np.column_stack([x[test,0]-alpha*slip,-e*x[test,1],x[test,2]+1.5*alpha*slip])[:,ix]
            if row['k']=='8':
                for t,yp in zip(test,yh):
                    truth=z[t]; slip=truth[0]-truth[-1];hat=yp[0]-yp[-1]
                    shots.append(dict(endpoint=out,method=m,id=rows[t]['id'],incoming_rps=rows[t]['n_in_rps'],
                        u_out=float(truth[0]),u_pred=float(yp[0]),omega_out_rad_s=float(truth[-1]/.019),
                        omega_pred_rad_s=float(yp[-1]/.019),release=slip,pred_release=hat,
                        release_sign=('unresolved' if slip==0 else ('positive' if slip>0 else 'negative')),
                        output_sign=('positive' if hat>0 else 'negative'),
                        selected_route_sign=('positive' if route[t]^flip else 'negative') if m in ('majority','continuous_loss_gate') else ''))
            d=np.abs(yh-z[test]);b=valid[test]
            metrics=dict(velocity=np.linalg.norm(d[:,:-1],axis=1).mean(),spin=d[:,-1].mean()/.019,
                         output_sign_acc=((yh[b,0]>yh[b,-1])==positive[test][b]).mean())
            if m in ('majority','continuous_loss_gate'):
                majority_wrong=2*(route[test][b]!=positive[test][b]).sum()>b.sum()
                metrics.update(flip=float(flip),orientation_acc=float(flip==majority_wrong),
                               route_sign_acc=((route[test][b]^flip)==positive[test][b]).mean())
            for name,value in metrics.items():
                maxerr[out][name]=max(maxerr[out][name],abs(float(row[name])-value))
        counts[out]=len(reference)
    assert max(v for rr in maxerr.values() for v in rr.values())<1e-8
    assert len(shots)==40
    with (HERE/'feint_k8_held_predictions.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=shots[0].keys());wr.writeheader();wr.writerows(shots)
    print(maxerr)
    (HERE/'feint_audit.json').write_text(json.dumps(dict(counts=counts,max_absolute_error=maxerr,job=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID")),indent=2))
if __name__=="__main__":main()
