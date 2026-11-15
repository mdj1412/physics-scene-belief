"""Factual action-value regression and optional posterior fallback."""
import json
from pathlib import Path
from itertools import combinations_with_replacement
import numpy as np
from physics25 import FY,CALIBRATION

QUAD=combinations_with_replacement
I,J=map(np.array,zip(*QUAD(range(6),2)))
SLOPE=FY/(FY@FY)
quad=FY**2-(FY**2).mean()
CURVE=.05*quad/(quad@quad)
def features(x):
    profile=x[...,:9]
    mean=profile.mean(axis=-1)
    slope=profile@SLOPE
    curvature=profile@CURVE
    residual=profile-mean[...,None]-slope[...,None]*FY-curvature[...,None]*quad/.05
    return np.stack([mean,slope,curvature,x[...,9],x[...,10],
                     np.sqrt(np.mean(residual**2,axis=-1))],axis=-1)

class Predictor:
    def __init__(self,model):
        self.family=model['family'];self.mean=np.array(model['mean']);self.std=np.array(model['std'])
        self.beta=np.array(model['beta']);self.bias=model['bias']
        self.centers=np.array(model['centers']);self.width=model['width']
    def design(self,f):
        h=np.clip((f-self.mean)/self.std,-5,5)
        poly=np.concatenate([h,h[...,I]*h[...,J]],axis=-1)
        if self.family=="quadratic":return poly
        dist=((h[...,None,:]-self.centers)**2).sum(axis=-1)
        return np.concatenate([poly,np.exp(-.5*dist/self.width**2)],axis=-1)
    def score(self,x):return self.design(features(x))@self.beta+self.bias

def learn(x,y,family,penalty):
    f=features(x);mean=f.mean(0);scale=np.maximum(f.std(0),1e-8)
    u=(f-mean)/scale
    rng=np.random.default_rng(2510801)
    centers=u[rng.choice(len(f),size=96,replace=False)] if family=='rbf' else np.empty((0,6))
    model=dict(family=family,mean=mean.tolist(),std=scale.tolist(),centers=centers.tolist(),
               width=1.8,beta=[],bias=0.)
    des=Predictor(model).design(f);ctr=des.mean(0);des-=ctr
    beta=np.linalg.solve(des.T@des+penalty*len(y)*np.eye(des.shape[1]),des.T@(y-y.mean()))
    model.update(beta=beta.tolist(),bias=float(y.mean()-ctr@beta))
    return model

class FrozenGate:
    """V24-B coefficients and transform with the new public sensor calibration."""
    def __init__(self):
        path=Path(__file__).resolve().parent.parent/'calibrated_router_v24/results/selected.json'
        self.cfg=json.loads(path.read_text());m=self.cfg['model']
        self.mean=np.asarray(m['mean']);self.std=np.asarray(m['scale']);self.coef=np.asarray(m['coef'])
        self.bias=m['bias'];self.cut=m['cut']
        self.precision=np.log(CALIBRATION[2]/CALIBRATION[4]);self.noise=CALIBRATION[0]*1000
    def high(self,x):
        f=np.array([x[:9]@SLOPE,x[:9]@CURVE,x[9],x[10],self.precision,self.noise])
        h=np.clip((f-self.mean)/self.std,-4,4)
        return bool(self.bias+h@self.coef[:6]+(h*h)@self.coef[6:12]+
                    (h[:4]*h[4])@self.coef[12:16]+(h[:4]*h[5])@self.coef[16:]>=self.cut)
