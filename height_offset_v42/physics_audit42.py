"""Independent physical replay and irreducible error for hidden placement noise."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.stats import t as student

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'estimated_projection_v41'))
import physics_audit41 as independent
import world42 as new

w=new.world
OFFSETS=new.OFFSETS_MM
ZERO=new.ZERO
REGIMES=('strong','weak','null')
SIGNS=(1,-1)
SHAPES=('flare','roll')
HEIGHT_KEYS=('h19','h31')
LAWS=('saturating','crossover')
VARS=('base','hollow','bulge')
N_PANEL=12


def record_error(errors,key,actual,expected,tolerance=3e-12):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape:
        raise ValueError(f'{key} dimensions disagree: {a.shape} vs {b.shape}')
    err=float(np.max(np.abs(a.astype(float)-b.astype(float))))
    if not np.isfinite(err) or err>tolerance: raise ValueError((key,err,tolerance))
    errors[key]=max(errors.get(key,0.),err)


def encoded_rows(theta):
    return np.ascontiguousarray(theta,dtype='<f8').view(np.dtype((np.void,48))).ravel()


def physical_supports(panel):
    prototypes=[]
    for ai,a in enumerate(w.ALPHAS):
        for lane in (10,20):
            seed=w.catalog_seed(panel,ai,lane)
            prototypes.append(encoded_rows(w.draw_theta(2048,seed,lane,a).cpu().numpy()))
    for fam,height in itertools.product(SHAPES,(.019,.031)):
        seed=w.pair_seed(panel,fam,height)
        for j,a in enumerate(w.ALPHAS):
            prototypes.append(encoded_rows(w.draw_theta(4,seed+1000000+100000*j,73,a).cpu().numpy()))
    return prototypes


def verify_splits():
    # 2 signed geometries share supports by design. Check new/old calibration,
    # deployment and passive splits at the level of all exact latent tuples.
    old=np.concatenate([arr for p in range(N_PANEL) for arr in physical_supports(p)])
    fresh=np.concatenate([arr for p in range(N_PANEL) for arr in physical_supports(new.catalog_panel(p))])
    all_unique=len(np.unique(np.concatenate([old,fresh])))
    if all_unique!=len(old)+len(fresh):
        raise ValueError('Previously exposed/repeated prototype in targeted v41/v42 populations')
    return {'historical':len(old),'fresh':len(fresh),'unique_combined':all_unique,
            'contact_support':N_PANEL*8*2*2048,'passive_objects':N_PANEL*4*32}


def verify_views(panel,errors):
    for f,h in itertools.product(SHAPES,HEIGHT_KEYS):
        height=int(h[1:])/1000
        path=ROOT/'results'/'directions'/f'panel_{panel:02d}_{f}_{h}.npz'
        with np.load(path,allow_pickle=False) as z:
            pairs=new.direction_pairs(panel,f,height)
            record_error(errors,'raw_passive_pairs',pairs,z['pairs'])
            d=(pairs[:,1]-pairs[:,0])/.00030
            _,vv=np.linalg.eigh(np.cov(d,rowvar=False))
            record_error(errors,'direction_eigensolver_abs_cos',1.,abs(vv[:,-1]@z['estimated']))
            if 'oracle' in z:
                ys=w.view_heights(height)
                vector=independent.geom(ys,f,'bulge')[1]-independent.geom(ys,f,'base')[1]
                record_error(errors,'analytic_direction_abs_cos',1.,abs((vector/np.linalg.norm(vector))@z['oracle']))


def archived(c,p):
    name='_'.join(c)
    path=ROOT/'results'/name/f'catalog_{p:02d}.npz'
    m=json.loads(path.with_suffix('.json').read_text())
    if 'sha256' in m and hashlib.sha256(path.read_bytes()).hexdigest()!=m['sha256']:
        raise ValueError('Invalid raw '+str(path))
    with np.load(path,allow_pickle=False) as z:
        keys=['alpha','labels','actions','p_base','first3_y','fourth_y','dirs','cal_x','ids','z','weights']
        data={k:z[k] for k in keys}
    return m,data


def replay(m,a,cond,panel,visits,errors):
    fam,hkey,var,law=cond
    h=int(hkey[1:])/1000
    new_panel=new.catalog_panel(panel)
    if m['new_panel']!=new_panel or m['panel']!=panel or m['cell']!='_'.join(cond): raise ValueError(m)
    a_y=a['labels']
    if a_y.dtype!=bool or a_y.shape!=(9,2,3,8,128): raise ValueError('Invalid label tensor')
    # Same RNGs as generator; the collision, geometry and sensor reconstruction
    # use only independent NumPy reference equations from the v41 audit.
    for e,alpha in enumerate(a['alpha']):
        sc=m['calibration_seeds'][e]; sd=m['deployment_seeds'][e]
        cidx=independent.rng(sc,4110).integers(2048,size=1024)
        didx=independent.rng(sd,4120).integers(2048,size=128)
        ids=independent.rng(sc,98).permutation(1024)[:4]
        direction=np.array([1,-1,1])*m['start']
        noise=independent.rng(sc,99).normal(size=(4,2))*.027
        key=(panel,tuple(alpha))
        visits.setdefault(key,[set(),set(),set()])
        for v,vv in zip(visits[key],(didx,cidx[ids],cidx)):v.update(map(int,vv))
        phys=(fam,var,h,law,1,new_panel)
        th=w.latent_support(alpha,phys,10).cpu().numpy()[cidx[ids]]
        td=w.latent_support(alpha,phys,20).cpu().numpy()[didx]
        for si,sgn in enumerate(SIGNS):
            record_error(errors,'paid_ids',a['ids'][si,e],ids)
            record_error(errors,'paid_directions',a['dirs'][si,e],direction)
            velocity,_=independent.collision(th,0.,0.,fam,var,law,sgn,'strong')
            factual=np.column_stack((independent.contour(th,w.FY,fam,var),velocity))
            factual+=independent.rng(sc,11).normal(size=(1024,11))[ids]*np.array([.0016]*9+[.095]*2)
            record_error(errors,'paid_factual',a['cal_x'][si,e],factual)
            far,_=independent.collision(th[:3],0.,direction*1.10,fam,var,law,sgn,'strong')
            record_error(errors,'first_three_velocities',a['first3_y'][si,e],far+noise[:3])
            for ri,reg in enumerate(REGIMES):
                near,_=independent.collision(th[3:],.023,1.10,fam,var,law,sgn,reg)
                record_error(errors,'fourth_velocity',a['fourth_y'][si,ri,e],near[0]+noise[3])
                # Broadcast all nine contact heights; collision() uses scalar envelope,
                # so integrate the normal/tangent reference at each height.
                for oi,mm in enumerate(OFFSETS):
                    _,truth=independent.collision(td,h+mm*.001,.425,fam,var,law,sgn,reg)
                    record_error(errors,'archived_query_event',a_y[oi,si,ri,e],truth)


def realized_events(label,base_label):
    # Arrays use variation/object as matched interventions, never as independent panels.
    axes=(1,2,3,4)
    y=label.astype(float)
    ref=base_label.astype(float)
    signed=y-ref
    relative=y-y[ZERO]
    by_offset=dict(stick=y.mean(axes),contour_flip=np.abs(signed).mean(axes),
                   base_slip_to_stick=(signed>0).mean(axes),base_stick_to_slip=(signed<0).mean(axes),
                   height_flip=np.abs(relative).mean(axes),nominal_slip_to_stick=(relative>0).mean(axes),
                   nominal_stick_to_slip=(relative<0).mean(axes))
    two_sign=[]
    for d in (.125,.25,.5,1.):
        plus=y[OFFSETS.index(d)]
        minus=y[OFFSETS.index(-d)]
        two_sign.append(float(np.not_equal(plus,minus).mean())/4)
    m=y.mean(0)
    return {k:v.tolist() for k,v in by_offset.items()},two_sign,float(np.mean(m*(1-m)))


def decompose(p,y,base_label,decoders):
    # Absolute common y term cancels in a paired Brier difference.
    # p[decoder,sign,env,object,noise]; average repeated p only where linear in p.
    raw=p[decoders.index('original')]
    est=p[decoders.index('estimated')]
    dp=(est-raw).mean(-1)
    offset_free=((est-raw)*(est+raw)).mean()
    axes=(1,2,3,4)
    delta=offset_free-2*(dp[None,:,None]*y).mean(axes)
    common=offset_free-2*(dp[None,:,None]*base_label).mean(axes)
    label_term=-2*(dp[None,:,None]*(y.astype(float)-base_label)).mean(axes)
    np.testing.assert_allclose(delta,common+label_term,atol=2e-15,rtol=0)
    return dict(observed=delta.tolist(),base_label_counterfactual=common.tolist(),
                signed_contour_effect=label_term.tolist())


def parse_group(cond,panel,visits,errors,decoders):
    fam,hkey,law=cond
    group=[]
    ref=None
    for var in VARS:
        c=(fam,hkey,var,law)
        m,a=archived(c,panel)
        replay(m,a,c,panel,visits,errors)
        if ref is None:ref=a
        else:
            for key in ('alpha','z','weights','actions','ids','cal_x','dirs','first3_y','fourth_y'):
                record_error(errors,'invariant_'+key,a[key],ref[key])
            record_error(errors,'variant_label_nominal',a['labels'][ZERO],ref['labels'][ZERO])
        events,sign,floor=realized_events(a['labels'],ref['labels'])
        effects=decompose(a['p_base'],a['labels'],ref['labels'],decoders)
        group.append(dict(family=fam,height=hkey,law=law,variant=var,panel=panel,
                          offsets=events,paired_sign_floor=sign,uniform_offset_floor=floor,effects=effects))
    return group


def interval(values):
    x=np.asarray(values)
    m=x.mean(0)
    radius=x.std(0,ddof=1)*student.ppf(.975,len(x)-1)/np.sqrt(len(x))
    return {'mean':m.tolist(),'lo':(m-radius).tolist(),'hi':(m+radius).tolist()}


def summarize(rows,keys):
    by_panel=[[r for r in rows if r['panel']==k and all(r[n] in val for n,val in keys.items())] for k in range(N_PANEL)]
    resolve=lambda r,n,k: r[n][k] if k else r[n]
    def avg(n,k=None):
        return interval([np.mean([resolve(r,n,k) for r in rr],axis=0) for rr in by_panel])
    return {'offsets':{k:avg('offsets',k) for k in rows[0]['offsets']},
            'effects':{k:avg('effects',k) for k in rows[0]['effects']},'paired_sign_floor':avg('paired_sign_floor'),
            'uniform_offset_floor':avg('uniform_offset_floor')}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--follow',action='store_true')
    opt=ap.parse_args()
    if os.environ.get('BELIEF_DEVICE')!='cpu':raise RuntimeError('Set BELIEF_DEVICE=cpu')
    w.torch.set_num_threads(1)
    split=verify_splits()
    done=set();seen=set(); rows=[];visits={}; errors={}
    idx=list(itertools.product(range(N_PANEL),SHAPES,HEIGHT_KEYS,LAWS))
    with w.torch.inference_mode():
        while idx:
            progress=False
            for case in idx.copy():
                panel,f,h,law=case
                paths=[ROOT/'results'/'_'.join((f,h,v,law))/f'catalog_{panel:02d}.json' for v in VARS]
                if not all(path.exists() for path in paths): continue
                dec=list(json.loads((ROOT/'results'/'run.json').read_text())['decoders'])
                rows.extend(parse_group((f,h,law),panel,visits,errors,dec))
                done.add(case);idx.remove(case);progress=True
                if panel not in seen:
                    verify_views(panel,errors);seen.add(panel)
                print('verified',*case,flush=True)
            if idx:
                if not opt.follow:raise FileNotFoundError(f'{len(idx)} unresolved archive groups')
                if not progress:time.sleep(4)
    assert len(rows)==288 and len(done)==96 and len(seen)==12
    group={'all':{},'deformed':{'variant':('bulge','hollow')},'base':{'variant':('base',)}}
    for f,h in itertools.product(SHAPES,HEIGHT_KEYS):
        group[f+'_'+h]={'family':(f,), 'height':(h,), 'variant':('bulge','hollow')}
    results={name:summarize(rows,keys) for name,keys in group.items()}
    num=[sum(len(v[i]) for v in visits.values()) for i in range(3)]
    out=dict(panels=N_PANEL,offsets_mm=OFFSETS,paired_magnitudes_mm=(.125,.25,.5,1.),run_host=socket.gethostname(),
             split=split,distinct_deployment=num[0],distinct_paid=num[1],distinct_pool=num[2],raw_draws=12*8*128,
             physical_labels_replayed=len(rows)*9*2*3*8*128,max_errors=errors,groups=results,cells=rows)
    (ROOT/'physics_audit42.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
