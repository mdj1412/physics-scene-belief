"""Learn a signed decision gain on v29 only, using nested grouped validation."""
from pathlib import Path
import json
import time

import numpy as np

from contact30 import V29, torch
from runner29 import atomic_json, digest, jsonable

ROOT = Path(__file__).resolve().parent
NAMES = ['alpha_c','alpha_f','sd_c','sd_f','abs_alpha_c','abs_alpha_f','alpha_c_x_f',
         'alpha_c_x_abs_f','alpha_f_x_abs_c','sd_f_x_alpha_c','probe_tan_mean',
         'probe_tan_sd','probe_normal_mean','probe_normal_sd','paired_velocity_cov']
LAMBDAS = (3., .3, .03)


def features(est, x, ids, measured):
    ac, af = est['alpha']
    sc, sf = est['sd']
    normal, tangential = measured.T
    f = x[ids, -2]
    return np.asarray([ac,af,sc,sf,abs(ac),abs(af),ac*af,ac*abs(af),af*abs(ac),sf*ac,
                       tangential.mean(),tangential.std(),normal.mean(),normal.std(),
                       np.mean((f-f.mean())*(tangential-tangential.mean()))])


def quota_actions(gap):
    a = np.full(gap.shape,1,dtype=np.int8)
    order = np.argsort(-gap,axis=1,kind='stable')
    np.put_along_axis(a,order[:,:13],3,axis=1)
    return a


def load_training():
    xs,ys,hashes=[],[],[]
    for block in range(24):
        path=V29/'results'/f'block_{block:03d}'
        meta=json.loads((path/'metrics.json').read_text())
        for f,k in [('metrics.npz','npz_sha256'),('choice_lock.npz','choices_sha256')]:
            assert digest(path/f)==meta[k]
        hashes.append(meta['npz_sha256'])
        rows=[]
        with np.load(path/'choice_lock.npz') as locks:
            for e in range(8):
                x=locks['x_cal'][e]; order=locks['probe_order'][e]; y=locks['probe_y'][e]
                rows.append([features(meta['estimates'][e][str(k)], x, order[:k], y[:k]) for k in (2,4)])
        with np.load(path/'metrics.npz') as outcomes:
            costs=outcomes['cost'][:,:,-1,:]
            choice=quota_actions(outcomes['audit_gap'])
            risk=np.take_along_axis(costs[:,:,None,:],(choice==3)[...,None].astype(int),axis=3)[...,0].mean(1)
        xs.append(rows);ys.append(risk[:,:-1]-risk[:,1:])
    return np.asarray(xs), np.asarray(ys), hashes


def fit(x,y,lam):
    xx=torch.as_tensor(x,dtype=torch.float64,device='cuda').reshape(-1,x.shape[-1])
    yy=torch.as_tensor(y,dtype=torch.float64,device='cuda').reshape(-1)
    mean,sd=xx.mean(0),xx.std(0,unbiased=False).clamp_min(1e-8)
    z=(xx-mean)/sd
    beta=torch.linalg.solve(z.T@z+len(xx)*lam*torch.eye(z.shape[-1],device='cuda',dtype=z.dtype),z.T@(yy-yy.mean()))
    return dict(mean=mean.cpu().tolist(),sd=sd.cpu().tolist(),beta=beta.cpu().tolist(),intercept=float(yy.mean()),lam=lam)


def predict(model,x):
    return (x-np.asarray(model['mean']))/model['sd'] @ np.asarray(model['beta']) + model['intercept']


def choose(x,y,groups):
    losses=[]
    for lam in LAMBDAS:
        squared=[]
        for fold in np.unique(groups):
            train=groups!=fold
            model=fit(x[train],y[train],lam)
            squared.extend((predict(model,x[~train])-y[~train]).ravel()**2)
        losses.append(float(np.mean(squared)))
    return LAMBDAS[int(np.argmin(losses))],losses


def train():
    start=time.perf_counter()
    x,y,hashes=load_training()
    groups=np.arange(len(x))%6
    models=[];records=[]
    for stage in range(2):
        xx,yy=x[:,:,stage],y[:,:,stage]
        outer=np.zeros_like(yy); null=np.zeros_like(yy)
        for fold in range(6):
            sel=groups!=fold
            lam,_=choose(xx[sel],yy[sel],groups[sel])
            outer[~sel]=predict(fit(xx[sel],yy[sel],lam),xx[~sel])
            null[~sel]=yy[sel].mean()
        chosen,losses=choose(xx,yy,groups)
        models.append(fit(xx,yy,chosen))
        records.append(dict(grouped_cv_mse=losses,chosen=chosen,n=yy.size,
            actual_mean=yy.mean(),actual_sd=yy.std(),oof_mse=np.square(outer-yy).mean(),
            oof_baseline_mse=np.square(null-yy).mean(),oof_corr=np.corrcoef(outer.ravel(),yy.ravel())[0,1],
            out_of_fold_predictions=outer,targets=yy))
    saved=dict(models=models,training_diagnostics=records,features=NAMES,development_hashes=hashes,
               training_seconds=time.perf_counter()-start,source='v29 Coulomb; 24 independent blocks; grouped nested CV')
    atomic_json(ROOT/'results'/'model_lock.json',jsonable(saved))
    return saved
