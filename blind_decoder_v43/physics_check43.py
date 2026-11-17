"""Independent NumPy contour gradients, position-domain forecast and held-out event audit."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erf, expit

ROOT = Path(__file__).resolve().parent
DELTAS = np.array([-1.,-.5,-.25,-.125,0.,.125,.25,.5,1.])
WEIGHTS = {'fold':(.70,-.90,.45,.85), 'ripple':(-.65,.50,.88,-.76)}
KNOTS = (11.,19.,23.,31.,39.)


def displacement(z, shape, variant):
    """Lateral displacement in mm at absolute height z in mm, no analytic slope."""
    u = z/25.
    if shape == 'fold':
        f = .085*25*((np.logaddexp(0.,5*u-3.75)-np.logaddexp(0.,-3.75))/5-expit(-3.75)*u)
    else:
        f = .061*25*((1.-np.cos(1.65*u))/1.65-.43*(1.-np.cos(3.3*u))/3.3)
    total = 0.
    for left,right,weight in zip(KNOTS,KNOTS[1:],WEIGHTS[shape]):
        if left < z < right:
            r = (z-left)/(right-left)
            if shape=='fold':
                total = .75*weight*np.sin(np.pi*r)**2*(1.+.38*(2*r-1.))
            else:
                total = .75*weight*16*(r*(1.-r))**2*(.55+.45*np.cos(2*np.pi*r))
    return f+{'base':0.,'hollow':-1.,'bulge':1.}[variant]*total


def slope(z, shape, variant):
    # Richardson combination cancels the leading bias even at a C1 knot.
    eps = .00004
    def diff(h):
        return (displacement(z+h,shape,variant)-displacement(z-h,shape,variant))/(2*h)
    return 2*diff(eps/2)-diff(eps)


def friction(mu, vt, h, surface, law, sign, regime):
    mm = 1000*h
    u = min(abs(mm)/23., 1.)
    strength = {'strong':1.,'weak':.3,'null':0.,'global_null':0.}[regime]
    envelope = 0. if regime=='global_null' else strength+(1.-strength)*math.cos(.5*math.pi*u)**2
    if surface=='inlay':
        ink = math.exp(-.5*((mm-22.)/5.5)**2)
        scale, orient = 1.+.079*ink, 1.-1.45*ink
    else:
        scale, orient = 1.+.047*math.sin(2*math.pi*mm/24), .38+.62*math.cos(2*math.pi*mm/24)
    v = np.abs(vt)
    if law=='rational':
        r = 1.-.58**2/(.58**2+v**2)
        a = .145*np.sign(vt)*v**3/(v**3+.61**3)
        fs, fk = 1.045-.046*r, .900+.035*r
    else:
        band = np.exp(-.5*((v-.64)/.25)**2)
        a = .16*erf(vt/.34)*(1.-1.55*band)
        fs, fk = 1.007+.043*band, .930-.061*band
    common = mu*scale*(1.+sign*envelope*orient*a)
    return common*fs, common*fk


def score(theta, shape, variant, surface, law, sign, regime, mm):
    g = theta[:,2] + theta[:,3]*(mm/25.) + slope(mm,shape,variant)
    v = .425
    speed = (v+g)/np.sqrt(1.+g**2)
    mu_static,_ = friction(theta[:,4],speed,mm/1000.,surface,law,sign,regime)
    return np.abs(v+g) <= (1.+theta[:,5])*np.maximum(1.-v*g,0.)*mu_static


def forecast():
    rows = []
    for family in WEIGHTS:
        for height in (19.,31.):
            x = np.arange(-6.,7.,2.)
            df = np.array([displacement(height+d,family,'bulge')-displacement(height+d,family,'base') for d in x])
            apparent = x.dot(df)/x.dot(x)
            for d in DELTAS:
                true = slope(height+d,family,'bulge')-slope(height+d,family,'base')
                criterion = 2*true*apparent-apparent**2
                rows.append(dict(shape=family,height_mm=height,delta_mm=d,apparent_slope=apparent,
                                 actual_slope_change=true,geometry_sign=np.sign(criterion)))
    (ROOT/'geometry_forecast43.json').write_text(json.dumps(rows,indent=2)+'\n')


def check_world():
    import world43 as w
    distances = np.unique(np.r_[np.linspace(-.01,.039,170),np.array(KNOTS)*.001])
    worst = 0.
    for family in WEIGHTS:
        for variant in w.VARIANTS:
            actual,_ = w.perturbation(distances,family,variant)
            fitted = np.array([slope(d*1000.,family,variant) for d in distances])
            worst = max(worst,np.max(np.abs(actual.cpu().numpy()-fitted)))
        for h in w.HEIGHTS:
            dg,df = w.deformation(h,family)
            assert float(dg)==0. and float(df)==0.
            np.testing.assert_allclose(np.linalg.norm(w.oracle_direction(family,h)),1.)
    assert worst < 1e-7, worst
    print('maximum slope error',worst)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--forecast',action='store_true')
    p.add_argument('--check-world',action='store_true')
    a=p.parse_args()
    if a.forecast:forecast()
    if a.check_world:check_world()
