"""Portable orientation readouts. Prediction paths use only published coefficients and paid records."""
import numpy as np


def inputs(x, y, direction):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    d=np.asarray(direction,dtype=float)
    if x.ndim!=3 or y.shape!=x.shape[:-1]+(2,) or x.shape[-1]!=11 or d.shape!=x.shape[:-1]:
        raise ValueError("Expected x[N,k,11], y[N,k,2], direction[N,k]")
    if not np.isin(d,[-1,1]).all():
        raise ValueError("Each acquired velocity requires a signed action")
    return x,y,d


def raw_features(x,y,direction):
    x,y,d=inputs(x,y,direction)
    # Horizontal incoming speed is 1. Convert outgoing velocities to measured impulses.
    jx=1.-y[:,:,0]
    jy=1.10*d-y[:,:,1]
    # Central slope of the noisy observed contour, shrunk to limit sensor noise.
    fy=np.linspace(-.010,.010,9)
    slope=.1*(x[:,:,:9]@fy)/(fy@fy)
    mu=d*(jy+slope*jx)/np.maximum(jx-slope*jy,.35)
    fx=x[:,:,-2]
    fixed=[np.ones_like(mu),mu,mu**2,mu*fx,jx,jy*d,fx,x[:,:,-1],slope,fx**2]
    radial=[np.exp(-.5*((mu-center)/.035)**2) for center in np.linspace(.06,.5,12)]
    obs=np.stack(fixed+radial,axis=-1)
    return np.concatenate((obs.mean(1),(obs*d[:,:,None]).mean(1)),axis=1)


def logit(z):
    return 1/(1+np.exp(-np.clip(z,-40,40)))


def predict(model,features):
    x=(np.asarray(features)-model['center'])/model['scale']
    return logit(x@model['beta']+model['intercept'])


def fit(features,labels,ridge=.01):
    y=np.asarray(labels,float); x=np.asarray(features,float)
    center=x.mean(0);scale=x.std(0);scale[scale<1e-7]=1.
    x=(x-center)/scale
    design=np.column_stack([np.ones(len(x)),x])
    w=np.zeros(design.shape[1]); penalty=np.eye(len(w))*ridge;penalty[0,0]=0
    def objective(coef):
        eta=design@coef
        return np.mean(np.logaddexp(0.,eta)-y*eta)+.5*coef@penalty@coef
    for _ in range(60):
        p=logit(design@w)
        grad=design.T@(p-y)/len(y)+penalty@w
        hess=(design.T*(p*(1-p)))@design/len(y)+penalty
        step=np.linalg.solve(hess,grad)
        if np.max(np.abs(grad))<1e-10:break
        t=1.
        while t>1e-7 and objective(w-t*step)>objective(w)+1e-14:t/=2
        w-=t*step
    return dict(center=center.tolist(),scale=scale.tolist(),beta=w[1:].tolist(),intercept=float(w[0]),
                ridge=ridge,n=len(y))


def raw_probability(lock,law_or_family,paid_x,paid_y,directions,schedule):
    family='_'.join(law_or_family.split('_')[:2])
    if schedule not in ('same','alternating'):
        raise ValueError(schedule)
    z=raw_features(paid_x,paid_y,directions)
    return predict(lock['models'][family][schedule][str(paid_x.shape[1])],z)
