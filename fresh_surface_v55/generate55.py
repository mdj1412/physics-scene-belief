"""New physical contact populations, sensors and paired interventions; no gate or risk access."""
import argparse
import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V50 = ROOT.parent / 'polarity_calibration_v50'
sys.path.insert(0,str(V50))
import world50
import policy50
from calibration50 import instrument
r = policy50.legacy
N = 24
SLOTS = 256
K = np.linspace(0.,1.,5)
ERROR = (0.,.25,.5)
PAIR = [2,3,0,1]


def dest(group,i):
    return ROOT/group/f'panel_{i:02d}.npz'


def save(group,i,**data):
    path=dest(group,i)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**data)
    tmp.replace(path)


def load(group,i):
    with np.load(dest(group,i)) as a:
        return {name:a[name] for name in a}


def pop55(i):
    return load('worlds',i)


def raw55(i):
    return load('raw',i)


@lru_cache(None)
def material_parameters(i,f):
    rg=np.random.default_rng([551117,i,f,61])
    return rg.uniform(.04,.095),rg.uniform(17.,24.),rg.uniform(4.,7.),rg.uniform(.25,.65),rg.uniform(.25,1.25)


def field_mm(h,surface):
    # Per-catalog spatial frequency, center, width, amplitude and orientation are held out.
    _,family,index=surface.split('_')
    f=('braid','seam').index(family)
    a,c,w,b,phase=material_parameters(int(index),f)
    t=(h-c)/w
    if f==0:
        x=t+phase
        return 1.+a*(.62*math.sin(x)+.38*math.sin(2.3*x+.37)), b*math.cos(x)-.32*math.sin(1.7*x)
    cap=1./(1.+t*t)
    return .995+a*cap*math.tanh(t+phase), b-.83*cap+.19*math.tanh(t)


def install():
    world50.install_world43(r.old.world)
    if not getattr(r.old.world,'_v55_field',False):
        previous=r.old.world.surface_field
        r.old.world.surface_field=lambda h,s: field_mm(float(h)*1000.,s) if s.startswith('v55_') else previous(h,s)
        r.old.world._v55_field=True


def draw_objects(alphas,support,lane):
    theta,indices=[],[]
    for e,row in enumerate(alphas):
        support_seed=int(np.random.SeedSequence([551118,support,lane,e]).generate_state(1)[0])
        bank=r.old.world.w.draw_theta(r.old.world.w.SUPPORT_SIZE,support_seed,lane,list(row))
        ids=np.random.default_rng([551119,support,lane,e]).choice(len(bank),128,replace=False)
        theta.append(bank[ids].cpu().numpy())
        indices.append(ids)
    return np.stack(theta),np.stack(indices)


def make_world(i):
    r.old.BASE=550000000
    seed,alpha,sign,regime=r.old.population(i)
    theta,idx=draw_objects(alpha,5500+i,20)
    cal,calid=draw_objects(alpha,5600+i,10)
    old,oldid=draw_objects(alpha,5700+i,10)
    k=(np.arange(4)+i)%4
    surface=np.asarray([f'v55_{f}_{i:02d}' for f in ('braid','seam')])[k//2]
    law=np.asarray(world50.LAWS)[k%2]
    assert np.all(law==law[PAIR]) and np.all(surface!=surface[PAIR])
    return dict(seed=seed,theta=theta,object_index=idx,cal_theta=cal,cal_object_index=calid,
                old_cal_theta=old,old_cal_object_index=oldid,support_panel=5500+i,cal_support_panel=5600+i,
                old_cal_support_panel=5700+i,alphas=alpha,signs=sign,regimes=regime,surface=surface,law=law,
                material_parameters=np.asarray([material_parameters(i,f) for f in range(2)]))


def calibration(i,pop,stream):
    rng=np.random.default_rng([551120,i,stream])
    e=np.arange(4)[:,None,None]
    chosen=np.argsort(rng.random((4,SLOTS,128)),axis=-1)[...,:4]
    assert np.all(np.diff(np.sort(chosen,axis=-1),axis=-1)>0)
    fam=np.stack([rng.permutation(np.repeat(np.arange(4),32)) for _ in range(4)])[e,chosen]
    h=rng.integers(2,size=chosen.shape)
    amp=np.array([[world50.gauge_mm(f,hh,1,1.) for hh in world50.HEIGHTS] for f in world50.FAMILIES])[fam,h]
    amp*=rng.uniform(.90,1.10,(4,128))[e,chosen]
    gain,bias=np.asarray([instrument(int(pop['seed'])+stream*73001,h) for h in world50.HEIGHTS]).T
    weak_state=rng.choice([0,-1,1],p=(.5,.25,.25),size=chosen.shape)
    weak=gain[h]*.2*amp*weak_state+bias[h]+rng.normal(0,.007,(4,128))[e,chosen]+rng.normal(0,.028,chosen.shape)
    coin=rng.random(chosen.shape)
    ref_sign=np.stack([np.where(coin<k,weak_state,-weak_state) for k in K])
    full=rng.uniform(.94,1.06,(4,2))[e,h]*amp*ref_sign+rng.uniform(-.020,.020,(4,2))[e,h]
    full+=rng.normal(0,.017,(4,128))[e,chosen]+rng.normal(0,.060,chosen.shape)
    err=rng.random((4,SLOTS))
    imp=np.stack([np.where(err<ep,-ref_sign[...,3],ref_sign[...,3]) for ep in ERROR]).astype(np.int8)
    ids=pop['cal_object_index' if stream==0 else 'old_cal_object_index'][e,chosen]
    return np.broadcast_to(weak,full.shape).astype(np.float32),full.astype(np.float32),imp,ids


def prepare(i):
    pop=make_world(i)
    save('worlds',i,**pop)
    nw,nf,imp,ids=calibration(i,pop,0)
    ow,of,_,old_ids=calibration(i,pop,1)
    save('raw',i,nw=nw,nf=nf,imp=imp,ow=ow[:,PAIR],of=of[:,PAIR],new_ids=ids,old_ids=old_ids[PAIR])


@lru_cache(None)
def design(i):
    rng=np.random.default_rng([551121,i])
    def balanced(cells,counts):
        return np.stack([rng.permutation(np.repeat(cells,counts)) for _ in range(4)])
    fh=balanced(np.arange(8),16)
    return dict(f=fh//2,h=fh%2,v=balanced(np.arange(2),64),a=balanced(np.arange(2),64),
                state=balanced(np.array([0,-1,1]),[64,32,32]),rad=balanced(np.arange(2),64),
                off=rng.integers(3,size=(4,128)),meter=balanced(np.arange(4),32))


def measured55(pop,i):
    d=design(i)
    xx=None
    sensors=[]
    disp=np.full((4,128),np.nan)
    eaxis,oaxis=np.indices((4,128))
    keys=(r.r.Q,r.r.P)
    for fi,fam in enumerate(world50.FAMILIES):
        for hi,height in enumerate(world50.HEIGHTS):
            cell=(d['f']==fi)&(d['h']==hi)
            state=np.where(d['state']==0,0,np.where(d['state']==-1,1,2))
            dx=np.stack([world50.readings(pop,fam,height,s) for s in world50.STATES])
            disp[cell]=dx[state,eaxis,oaxis,d['meter']][cell]
            for vi,variant in enumerate(r.VARIANTS):
                x,obs=r.apparatus(pop,fam,height,variant)
                if xx is None:
                    xx=np.empty_like(x)
                    sensors=[{key:np.empty_like(v[key][:,:1]) for key in keys} for v in obs]
                mask=cell&(d['v']==vi)
                xx[mask]=x[mask]
                for ai,al in enumerate(r.ALIGN):
                    measured=policy50.observe(obs,fam,height,al,variant)
                    visible=mask&(d['a']==ai)
                    for e,row in enumerate(measured):
                        ids=np.flatnonzero(visible[e])
                        for key in keys:
                            sensors[e][key][ids,0]=row[key][ids,d['meter'][e,ids]]
    assert np.isfinite(xx).all() and np.isfinite(disp).all()
    return xx,sensors,disp,d['h']


def forward55(observed,engine,templates,coef):
    xx,sensors,disp,height=observed
    actions=r.infer.actions43(xx,engine)
    paid=r.infer.paid_sensors(sensors,actions)
    base=np.empty((4,128))
    env=np.arange(4)[:,None]
    for h in (0,1):
        ids=np.stack([np.flatnonzero(row==h) for row in height])
        assert ids.shape==(4,64)
        read=[{key:val[ids[e]] for key,val in obs.items()} for e,obs in enumerate(paid)]
        base[env,ids]=r.infer.decode43(engine,templates[h],xx[env,ids],actions[env,ids],read)[...,0]
    weights=coef[1,height,(actions==r.r.Q).astype(int)]
    return policy50.update(base,disp,weights),policy50.update(base,-disp,weights),actions


def forecaster():
    install()
    assert r.torch.cuda.device_count()==1
    r.torch.set_num_threads(1)
    with r.torch.inference_mode():
        engine,templates=policy50.engine50()
        coef=policy50.coefficients()[0]
        for i in range(N):
            observed=measured55(pop55(i),i)
            plus,minus,act=forward55(observed,engine,templates,coef)
            assert np.isfinite([plus,minus]).all() and min(plus.min(),minus.min())>=0 and max(plus.max(),minus.max())<=1
            save('forecasts',i,pplus=plus,pminus=minus,actions=act)
            print('forecast',i,flush=True)


def labels(i,ultra):
    import solver50
    engine=solver50.finer if ultra else solver50.mechanics
    engine.loaded_shape=world50.loaded_shape
    sim=engine.Impact(True)
    pop=pop55(i)
    d=design(i)
    th=pop['theta']
    out=np.empty((2,4,128))
    # Group by terrain so Bullet retains only necessary contact manifolds per trial.
    tasks={}
    for e,j in np.ndindex(4,128):
        fam=world50.FAMILIES[d['f'][e,j]]
        h=world50.HEIGHTS[d['h'][e,j]]
        curv=round(th[e,j,3]/.05)*.05
        for side,mult in enumerate((1,-1)):
            tasks.setdefault((fam,h,curv,int(d['state'][e,j])*mult),[]).append((side,e,j))
    for (fam,h,curv,s),rows in sorted(tasks.items()):
        terrain=sim.terrain(fam,h,s,curv)
        for side,e,j in rows:
            off=world50.OFFSETS[d['off'][e,j]]
            rad=world50.RADII[d['rad'][e,j]]
            u,w=engine.velocity(th[e,j,2]+th[e,j,3]*h/25)
            scale,ori=field_mm(h+off,pop['surface'][e])
            rate=1.+.095*np.exp(-abs(u)/.37) if pop['law'][e]=='rational' else .96+.105*np.tanh(abs(u)/.45)
            mu=th[e,j,4]*rate*scale*(1.+.045*ori*pop['signs'][e]*solver50.REGIME[pop['regimes'][e]]*np.tanh(u/.34))
            out[side,e,j],_=sim.trial(terrain[0],sim.site(terrain,off,rad),rad,u,w,mu,th[e,j,5])
    sim.close()
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[0][d['state']==0],out[1][d['state']==0])
    suffix='_ultra' if ultra else ''
    save('solver'+suffix,i,**{'y_keep'+suffix:out[0]<=.020,'y_flip'+suffix:out[1]<=.020,'speed'+suffix:out})
    print('bullet',i,suffix,flush=True)


def assemble():
    dig=hashlib.sha256()
    points=set()
    previous=set()
    for f in (V50/'worlds').glob('panel_*.npz'):
        with np.load(f) as p:
            for key in ('theta','cal_theta'):
                previous.update(map(bytes,np.ascontiguousarray(p[key]).reshape(-1,6)))
    summary=[]
    for i in range(N):
        pop=pop55(i)
        for key in ('theta','cal_theta','old_cal_theta'):
            pts=set(map(bytes,np.ascontiguousarray(pop[key]).reshape(-1,6)))
            assert len(pts)==512 and not pts.intersection(points) and not pts.intersection(previous)
            points.update(pts)
        raw=raw55(i);pred=load('forecasts',i);fine=load('solver',i);ultra=load('solver_ultra',i)
        save('panels',i,**raw,**pred,**fine,**ultra)
        checksum=hashlib.sha256(dest('panels',i).read_bytes()).hexdigest()
        summary.append(dict(catalog=i,surfaces=pop['surface'].tolist(),sha256=checksum))
        dig.update(checksum.encode())
    audit=dict(catalogs=N,clusters=N,new_material_realizations=2*N,surface_families=['braid','seam'],
               laws=list(world50.LAWS),target_calibration_prefixes=N*4*SLOTS*len(K),
               unique_theta_vectors=len(points),old_theta_vectors_checked=len(previous),
               deployment_objects=N*512,calibration_specimens_per_epoch=N*512,exact_collisions=0,
               positive_label='residual tangential speed <= 0.020 m/s',
               tag=dig.hexdigest(),panels=summary)
    (ROOT/'independence55.json').write_text(json.dumps(audit,indent=2))
    print('assembled',dig.hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','solver','forecasts','assemble'))
    parser.add_argument('panels',nargs='*',type=int)
    a=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    if a.stage=='forecasts':
        forecaster()
    elif a.stage=='assemble':
        assemble()
    else:
        assert not r.torch.cuda.is_available()
        for i in a.panels or range(N):
            if a.stage=='prepare':
                prepare(i)
            else:
                labels(i,False)
                labels(i,True)
