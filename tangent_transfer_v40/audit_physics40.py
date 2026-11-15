"""Independently replay paid sensors, query stick labels and matched contour invariants on CPU."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import physics40 as w

ROOT = Path(__file__).resolve().parent
FAMILIES = ('shelf', 'ridge')
SURFACES = ('saturating', 'crossover')
VARIANTS = ('base', 'hollow', 'bulge')
REGIMES = ('strong', 'weak', 'null')
N_PANEL = 12
TOL = 3.e-12


def rng(seed,lane):
    return np.random.default_rng(np.random.SeedSequence([seed,270000,lane]))


def geom(y,family,variant):
    y = np.asarray(y)
    u = y/.025
    if family == 'shelf':
        a = np.pi*u/2
        slope, value = .086*np.sin(a/2)**2, .043*.025*(u-2*np.sin(a)/np.pi)
    else:
        slope, value = -.153*u/(1+u*u)**2, -.0019125*u*u/(1+u*u)
    mult = {'base':0., 'bulge':.0009, 'hollow':-.0009}[variant]
    for left,right in ((.017,.023),(.025,.033)):
        # In centered coordinates the unit shape is (1-z^2)^2 on |z|<1.
        z = (2*y-(right+left))/(right-left)
        inside = (left<y)&(y<right)
        value = value+np.where(inside,mult*(1-z*z)**2,0.)
        slope = slope+np.where(inside,-8*mult*z*(1-z*z)/(right-left),0.)
    return slope,value


def contour(th,y,fam,var):
    return th[:,1,None]+th[:,2,None]*y+th[:,3,None]*y*y/.05+geom(y,fam,var)[1]


def collision(th,h,speed,fam,var,surface,sign,regime):
    m = th[:,2]+th[:,3]*h/.025+geom(h,fam,var)[0]
    den = np.hypot(1,m)
    vn,vt = (1-speed*m)/den,(speed+m)/den
    s = abs(vt)
    strength = {'strong':1.,'weak':.3,'null':0.}[regime]
    envelope = 1.-(1-strength)*np.sin(min(abs(h)/.023,1.)*np.pi/2)**2
    if surface=='saturating':
        a = .18*np.tanh(vt/.62)**3
        b = np.tanh(s/.58)**2
        static,kinetic = 1.015+.025*b,.88+.075*b
    else:
        a = .20*np.tanh(vt/.32)*np.tanh((s-.70)/.16)
        b = np.exp(-((s-.68)/.24)**2)
        static,kinetic = 1.035-.02*b,.94-.055*b
    mu = th[:,4]*(1+sign*envelope*a)
    jn = (1+th[:,5])*np.maximum(vn,0.)
    stick = s<=mu*static*jn
    after_n = vn-jn
    after_t = np.where(stick,0,vt-np.sign(vt)*mu*kinetic*jn)
    if np.any(vn<=0) or np.any(kinetic<=0) or np.any(kinetic>static):
        raise AssertionError('Invalid constitutive range in supported states')
    if np.any(vt*after_t < -1.e-14) or np.any(after_n**2+after_t**2>1+speed**2+1.e-12):
        raise AssertionError('Nondissipative collision')
    return np.column_stack(((after_n+m*after_t)/den,(after_t-m*after_n)/den)),stick


def compare(errors, key, left, right):
    a,b=np.asarray(left),np.asarray(right)
    if a.shape!=b.shape:
        raise ValueError((key,a.shape,b.shape))
    difference=float(abs(a.astype(float)-b.astype(float)).max())
    if not np.isfinite(difference) or difference>TOL:
        raise ValueError((key,difference))
    errors[key]=max(difference,errors.get(key,0.))


def support_audit(errors):
    rows=[]
    count=0
    for k,a,lane in itertools.product(range(N_PANEL),range(8),(10,20)):
        seed=2040100000+10000*k+100*a+lane
        th=w.draw_theta(2048,seed,lane,w.ALPHAS[a])
        theta=th.cpu().numpy()
        rows.append(np.ascontiguousarray(theta,dtype='<f8').view(np.dtype((np.void,48))).ravel())
        for family,variant in itertools.product(FAMILIES,VARIANTS):
            law=(family,variant,'saturating',1,k)
            compare(errors,'support_pairing',w.latent_support(w.ALPHAS[a],law,lane).cpu().numpy(),theta)
            for ys in (w.FY,w.VYS[0],np.array([0.,.023,.025])):
                compare(errors,'visual_profile',contour(theta,ys,family,variant),w.contour(th,ys,law).cpu().numpy())
                local,extra=w.perturbation(ys,family,variant)
                slope,offset=geom(ys,family,variant)
                compare(errors,'slope_of_profile',slope,local.cpu().numpy())
                compare(errors,'shape_offset',offset,extra.cpu().numpy())
        # The three profiles have the same slope at these heights: sufficient to use
        # one actual variant per support. Every variant is inspected in archive replay.
        for family,surface,sgn in itertools.product(FAMILIES,SURFACES,(1,-1)):
            variant=VARIANTS[(k+a+(lane==20))%3]
            law=(family,variant,surface,sgn,k)
            cases=[(0.,s,'strong') for s in (0.,-1.10,1.10)]
            cases += [(h,s,r) for h,s in ((.023,1.10),(.025,.425)) for r in REGIMES]
            for h,s,reg in cases:
                vel,st=collision(theta,h,s,family,variant,surface,sgn,reg)
                v,state=w.impact(th,h,s,law,reg)
                compare(errors,'velocity_basis_transform',vel,v.cpu().numpy())
                compare(errors,'contact_states',st,state.cpu().numpy())
                if h==0 and not np.all(st==(s==0)):
                    raise AssertionError('Unexpected far-probe state')
                count+=len(st)
    rows=np.concatenate(rows)
    unique=len(np.unique(rows))
    if unique!=len(rows): raise ValueError('Overlapping catalog splits')
    return dict(catalog_splits=N_PANEL*8*2,unique_prototypes=unique,
                duplicate_prototypes=0,independent_collision_recalculations=count)


def audit_block(meta,visits,errors):
    m=json.loads(meta.read_text())
    family,variant,surface=m['cell'].split('_')
    k,b=m['catalog'],m['block']
    seed=400000000+100000*k+10000*b
    assert m['base_seed']==seed and m['start']==(-1,1)[b]
    alphas=np.asarray(w.ALPHAS)[np.random.default_rng(seed+401).permutation(8)]
    raw=meta.with_suffix('.npz')
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==m['sha256']
    with np.load(raw,allow_pickle=False) as f:
        data={n:f[n] for n in ('alpha','ids','dirs','first3_y','cal_x','fourth_y','fourth_stick','labels')}
    compare(errors,'alpha_order',alphas,data['alpha'])
    for e,alpha in enumerate(alphas):
        sc,sd=seed+100+e,seed+200+e
        assert sc==m['calibration_seeds'][e] and sd==m['deployment_seeds'][e]
        ci=rng(sc,4010).integers(2048,size=1024); di=rng(sd,4020).integers(2048,size=128)
        ids=rng(sc,98).permutation(1024)[:4]
        dirs=np.array([1,-1,1])*m['start']
        eps=rng(sc,99).normal(size=(4,2))*.027
        law=(family,variant,surface,1,k)
        th=w.latent_support(alpha,law,10).cpu().numpy()[ci[ids]]
        td=w.latent_support(alpha,law,20).cpu().numpy()[di]
        key=(k,tuple(alpha))
        bucket=visits.setdefault(key,[set(),set(),set()])
        for seen,vals in zip(bucket,(ci,ci[ids],di)):
            seen.update(map(int,vals))
        for j,sgn in enumerate((1,-1)):
            compare(errors,'paid_pool_ids',ids,data['ids'][j,e])
            compare(errors,'far_directions',dirs,data['dirs'][j,e])
            vx=collision(th,0.,0.,family,variant,surface,sgn,'strong')[0]
            xf=np.column_stack((contour(th,w.FY,family,variant),vx))
            x=xf+rng(sc,11).normal(size=(1024,11))[ids]*np.array([.0016]*9+[.095]*2)
            compare(errors,'paid_factual',x,data['cal_x'][j,e])
            far,_=collision(th[:3],0.,dirs*1.10,family,variant,surface,sgn,'strong')
            compare(errors,'first_three_velocity',far+eps[:3],data['first3_y'][j,e])
            for ri,reg in enumerate(REGIMES):
                vel,state=collision(th[3:],.023,1.10,family,variant,surface,sgn,reg)
                compare(errors,'fourth_velocity',vel[0]+eps[3],data['fourth_y'][j,ri,e])
                compare(errors,'fourth_state',state[0],data['fourth_stick'][j,ri,e])
                labels=collision(td,.025,.425,family,variant,surface,sgn,reg)[1]
                compare(errors,'archived_stick_label',labels,data['labels'][j,ri,e])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--follow',action='store_true')
    args=p.parse_args()
    if os.environ.get('BELIEF_DEVICE')!='cpu' or not (ROOT/'freeze40.sha256').exists():
        raise RuntimeError('Use CPU on compute node after the v40 freeze')
    w.torch.set_num_threads(1)
    errors={};visits={};seen=set()
    info=support_audit(errors)
    while True:
        for meta in sorted((ROOT/'results').glob('*/catalog_*/block_*.json')):
            if meta not in seen:
                audit_block(meta,visits,errors)
                seen.add(meta)
        if len(seen)==288 or not args.follow: break
        time.sleep(4)
    if len(seen)!=288: raise ValueError(f'{len(seen)} of 288 blocks verified')
    nc,npd,nd=(sum(len(v[j]) for v in visits.values()) for j in range(3))
    info.update(complete_blocks=len(seen),raw_deployment_draws=N_PANEL*2*8*128,
                distinct_deployment_prototypes=nd,distinct_paid_prototypes=npd,
                distinct_calibration_pool_prototypes=nc,archived_stick_labels=288*2*3*8*128,
                max_absolute_errors=errors,tolerance=TOL,host=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'))
    (ROOT/'physics_audit40.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)


if __name__ == '__main__':
    main()
