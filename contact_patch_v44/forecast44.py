"""Forecast projection signs using only the completed v43 catalog and static shoe geometry."""
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, erf

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'blind_decoder_v43'
sys.path.insert(0,str(PREV))
import world43 as hist

HEIGHTS = (19.,31.)
SHAPES = ('fold','ripple')
SURFACES = ('inlay','woven')
LAWS = ('rational','gaussian_rate')
VARS = ('base','hollow','bulge')
DELTAS = (-.5,0.,.5)
RADII=(.5,2.)
SPANS=(0.,2.)
SCOPES=('all','deformed')
weights={'fold':(.70,-.90,.45,.85),'ripple':(-.65,.50,.88,-.76)}
knots=(11.,19.,23.,31.,39.)


def shape_profile(h, shape, variant):
    """mm, mm/mm, just the family-specific additive profile and its derivative."""
    u=h/25
    if shape=='fold':
        f=.085*25*((np.logaddexp(0.,5*u-3.75)-np.logaddexp(0.,-3.75))/5-expit(-3.75)*u)
        g=.085*(expit(5*u-3.75)-expit(-3.75))
    else:
        f=.061*25*((1-np.cos(1.65*u))/1.65-.43*(1-np.cos(3.3*u))/3.3)
        g=.061*(np.sin(1.65*u)-.43*np.sin(3.3*u))
    sign = {'base':0,'hollow':-1,'bulge':1}[variant]
    for left,right,w in zip(knots,knots[1:],weights[shape]):
        active=(h>left)&(h<right)
        x=np.clip((h-left)/(right-left),0.,1.)
        if shape=='fold':
            psi=np.sin(np.pi*x)**2; dp=np.pi*np.sin(2*np.pi*x)
            c=1+.38*(2*x-1); dc=.76
        else:
            psi=16*x**2*(1-x)**2; dp=32*x*(1-x)*(1-2*x)
            c=.55+.45*np.cos(2*np.pi*x); dc=-.9*np.pi*np.sin(2*np.pi*x)
        f=f+sign*.75*w*active*psi*c
        g=g+sign*.75*w*active*(dp*c+psi*dc)/(right-left)
    return f,g


def site(theta, height, shape, variant, r, span):
    # Grid in mm. Tie averaging on an ideal flat patch keeps its symmetric center.
    edge=span*.5+r
    a=np.linspace(-edge,edge,int(round(2*edge/.025))+1) if edge else np.array([0.])
    z=height+a
    f,g=shape_profile(z,shape,variant)
    _,_,sl,cu,_,_=theta.T
    # Global constant height terms cancel out of the support search.
    surface=-(sl[:,None]*a + cu[:,None]*(height*a+a*a/2)/25 +f[None])
    tool=np.sqrt(np.maximum(0.,r*r-np.maximum(0.,np.abs(a)-span*.5)**2))
    v=surface+tool
    top=np.isclose(v,v.max(1,keepdims=True),rtol=0.,atol=1e-10)
    u=top @ a/top.sum(1)
    local=(top*(sl[:,None]+cu[:,None]*z/25+g[None])).sum(1)/top.sum(1)
    return local,height+u


def predicted_event(theta, geometry, surface, law, sign, regime):
    g,h=geometry
    vt=(.425+g)/np.sqrt(1+g*g)
    demand=np.abs(.425+g)/((1+theta[:,5])*np.maximum(1.-.425*g,1e-12))
    if surface=='inlay':
        k=np.exp(-.5*((h-22.)/5.5)**2)
        scale,orient=1+.079*k,1-1.45*k
    else:
        phase=2*np.pi*h/24.
        scale,orient=1+.047*np.sin(phase),.38+.62*np.cos(phase)
    a=np.abs(vt)
    if law=='rational':
        stat=1.045-.046*(a*a)/(.58**2+a*a)
        orient_speed=.145*vt**3/(.61**3+a**3)
    else:
        band=np.exp(-.5*((a-.64)/.25)**2)
        stat=1.007+.043*band
        orient_speed=.16*erf(vt/.34)*(1.-1.55*band)
    near={'strong':1.,'weak':.3,'null':0.}[regime]
    envelope=near+(1.-near)*np.cos(.5*np.pi*np.minimum(abs(h)/23.,1.))**2
    mu=theta[:,4]*scale*(1.+sign*orient*envelope*orient_speed)*stat
    return demand<=mu


def make():
    dims=(12,2,2,3,2,2,3)
    transfer=np.zeros(dims)
    forecast=np.zeros(dims)
    point_check=0
    for panel in range(12):
        for i,j,k in itertools.product(range(2),range(2),range(3)):
            shape,h,var=SHAPES[i],HEIGHTS[j],VARS[k]
            cached={}
            for surf,law in itertools.product(SURFACES,LAWS):
                path=PREV/'results'/f'panel_{panel:02d}'/f'{shape}_{surf}_{law}_{var}_h{h:.0f}.npz'
                with np.load(path) as a:
                    pred=a['pred'][:,0]
                    truth=a['labels'][[1,4,7]]
                    alphas=a['alphas'];regimes=a['regimes'];signs=a['signs'];seed=int(a['seed'])
                po,pp=pred[0],pred[2]
                moment=(pp**2-po**2).mean(axis=-1)
                change=2*(pp-po).mean(axis=-1)
                prev=(moment[None]-change[None]*truth).mean(axis=(1,2))
                for e in range(len(alphas)):
                    ctx=(shape,var,h*.001,surf,law,int(signs[e]),panel)
                    key=(e,'theta')
                    if key not in cached:
                        cached[key]=hist.draw_object(128,seed+200+e,20,alphas[e],ctx).cpu().numpy()
                    th=cached[key]
                    for d,off in enumerate(DELTAS):
                        point=site(th,h+off,shape,var,0.,0.)
                        ye=predicted_event(th,point,surf,law,int(signs[e]),str(regimes[e]))
                        point_check+=int((ye!=truth[d,e]).sum())
                        for ri,r in enumerate(RADII):
                            for li,span in enumerate(SPANS):
                                key=(e,d,ri,li)
                                if key not in cached:
                                    cached[key]=site(th,h+off,shape,var,r,span)
                                y=predicted_event(th,cached[key],surf,law,int(signs[e]),str(regimes[e]))
                                forecast[panel,i,j,k,ri,li,d]+=(moment[e]-change[e]*y).mean()/16.
                transfer[panel,i,j,k]+=prev/4.
    assert point_check==0, point_check
    assert np.isfinite(forecast).all()
    rows=[]
    for i,j,ri,li,d in itertools.product(range(2),range(2),range(2),range(2),range(3)):
        for scope in SCOPES:
            sl=slice(None) if scope=='all' else slice(1,None)
            args=(slice(None),i,j,sl,ri,li,d)
            rows.append(dict(shape=SHAPES[i],height_mm=HEIGHTS[j],r_mm=RADII[ri],span_mm=SPANS[li],
                             offset_mm=DELTAS[d],scope=scope,pred=float(forecast[args].mean()),
                             carry=float(transfer[args].mean())))
    data=dict(source='v43 only, 12 historical panels; no new solver or new target probabilities',
              metric='privileged_zero minus original_zero Brier; positive = harm',
              point_sanity_mismatched=point_check, source_panels=12,
              rows=rows)
    (ROOT/'forecast44.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    assert os.environ.get('BELIEF_DEVICE','cpu')=='cpu'
    make()
