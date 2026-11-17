"""Fresh exchange study with a paired surface pilot, inherited physics and untouched v54 routing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
V55 = ROOT.parent / 'fresh_surface_v55'
sys.path.insert(0, str(V55))
import generate55 as v55

OFFSET = 2000
N = v55.N
COMPOSITION = np.array([[.42,0,.58],[0,.84,.16]])
E = np.array([0.,.25,.5])
PILOT_REPS = 4096
PILOT_ATTEMPTS = 8
REF_DEADZONE = .12
world50,policy50,r = v55.world50,v55.policy50,v55.r
install = v55.install
forward57 = v55.forward55

def destination(group, physical_catalog):
    assert OFFSET <= physical_catalog < OFFSET+N
    return ROOT/group/f'panel_{physical_catalog-OFFSET:02d}.npz'

# Every original v55 function (material, object, gauge, solver, decoder) receives
# the fresh, globally disjoint physical catalog key. Only the output location is remapped.
v55.dest = destination

def pop57(i):
    return v55.pop55(OFFSET+i)

def raw57(i):
    return v55.raw55(OFFSET+i)

def measured57(pop,i):
    return v55.measured55(pop,OFFSET+i)

def pilots(i):
    """Counterfactual pilot protocols use identical sacrificial specimens and sign noise.

    Each new object is a separate draw of family, fixture height, deformation
    magnitude, sign and noise. The globally unique pilot specimen index is
    ((catalog * 2 + material) * PILOT_REPS + episode) * 8 + slot.
    Only the session component is repeated, once, in the 4x2 protocol.
    """
    rng=np.random.default_rng([57021913,i,1])
    shape=(2,PILOT_REPS,PILOT_ATTEMPTS)
    family=rng.integers(4,size=shape)
    height=rng.integers(2,size=shape)
    mat=np.arange(2)[:,None,None]
    amp=np.array([[world50.gauge_mm(f,h,1,1.) for h in world50.HEIGHTS] for f in world50.FAMILIES])
    gain=rng.uniform(.94,1.06,(2,2))[mat,height]
    bias=rng.uniform(-.020,.020,(2,2))[mat,height]
    latent_ref=rng.choice([-1,0,1],size=shape,p=[.25,.5,.25]).astype('i1')
    response=gain*amp[family,height]*rng.uniform(.90,1.10,shape)*latent_ref+bias
    response+=rng.normal(0,.017,shape)+rng.normal(0,.060,shape)
    read=np.where(np.abs(response)>=REF_DEADZONE,np.sign(response),0).astype('i1')
    mix=rng.random(shape)
    flip_coin=rng.random(shape)
    ref=np.broadcast_to(read[:,None,None],(2,2,2,PILOT_REPS,PILOT_ATTEMPTS))
    impact=np.empty_like(ref)
    for regime,weights in enumerate(COMPOSITION):
        # searchsorted skips zero-mass atoms in both canonical compositions.
        single=E[np.searchsorted(weights.cumsum(),mix,side='right')]
        for arm,eps in enumerate((single,single[...,::2].repeat(2,axis=-1))):
            impact[:,regime,arm]=np.where(flip_coin<eps,-latent_ref,latent_ref)
    # Every slot is independent conditional on its session e. Only e changes across designs.
    np.testing.assert_array_equal(impact[:,:,0,: ,::2],impact[:,:,1,: ,::2])
    valid=(ref!=0)&(impact!=0)
    mismatch=(ref!=impact)&valid
    left,right=mismatch[...,::2],mismatch[...,1::2]
    both=valid[...,::2]&valid[...,1::2]
    np.testing.assert_array_equal(impact==0,np.broadcast_to(latent_ref[:,None,None]==0,impact.shape))
    v55.save('pilots',OFFSET+i,pilot_ref=ref,pilot_imp=impact)
    axes=(0,3,4)
    v55.save('pilot_audit',OFFSET+i,readable=valid.sum(axes),both=both.sum(axes),
             flips=mismatch.sum(axes),both_flips=(left&right).sum(axes),
             no_complete_pairs=(~both.any(-1)).sum((0,3)),
             raw_ref_error=((ref!=latent_ref[:,None,None])&valid).sum(axes),
             pair_joint_success_count=both.sum(-1).mean((0,3)),
             attempts=8*2*PILOT_REPS,comparisons=4*2*PILOT_REPS)

def forecasts():
    install()
    assert r.torch.cuda.device_count()==1
    r.torch.set_num_threads(1)
    with r.torch.inference_mode():
        engine,templates=policy50.engine50()
        coef=policy50.coefficients()[0]
        for i in range(N):
            plus,minus,act=forward57(measured57(pop57(i),i),engine,templates,coef)
            assert np.isfinite([plus,minus]).all()
            assert min(plus.min(),minus.min())>=0 and max(plus.max(),minus.max())<=1
            v55.save('forecasts',OFFSET+i,pplus=plus,pminus=minus,actions=act)
            print('forecast',i,flush=True)

def panel(i):
    parts=[v55.load(g,OFFSET+i) for g in ('raw','forecasts','solver','solver_ultra')]
    v55.save('panels',OFFSET+i,**parts[0],**parts[1],**parts[2],**parts[3])
    import score55 as score
    score.ROOT=ROOT
    score.REPLAY_SEEDS=[57041971,57045813]
    score.panel_score(i,ROOT/'scored57')
    print('scored',i,flush=True)

def audit():
    history=set()
    for directory in (ROOT.parent/'mismatch_switch_v56',V55,ROOT.parent/'polarity_calibration_v50'):
        for f in sorted((directory/'worlds').glob('panel_*.npz')):
            with np.load(f) as data:
                for key in ('theta','cal_theta','old_cal_theta'):
                    if key in data: history.update(map(bytes,np.ascontiguousarray(data[key]).reshape(-1,6)))
    already=set(); manifest=[]
    pcs=[]; n_change=n_active=n_diff=0
    for i in range(N):
        p=pop57(i)
        for key in ('theta','cal_theta','old_cal_theta'):
            arr=set(map(bytes,np.ascontiguousarray(p[key]).reshape(-1,6)))
            assert len(arr)==512 and not arr.intersection(history) and not arr.intersection(already)
            already.update(arr)
        f=v55.load('panels',OFFSET+i); act=v55.design(OFFSET+i)['state']!=0
        assert not np.any((f['y_keep']!=f['y_flip']) & ~act)
        n_active+=act.sum();n_change+=(f['y_keep']!=f['y_flip']).sum()
        n_diff+=sum((f[k]!=f[k+'_ultra']).sum() for k in ('y_keep','y_flip'))
        pcs.append(v55.load('pilot_audit',OFFSET+i))
        manifest.append({g:hashlib.sha256(destination(g,OFFSET+i).read_bytes()).hexdigest()
                         for g in ('panels','scored57','pilots')})
    audit_counts={k:np.stack([p[k] for p in pcs]).sum(0).tolist() for k in
                  ('readable','both','flips','both_flips','no_complete_pairs','raw_ref_error')}
    result=dict(catalogs=N,materials=2*N,world_key_offset=OFFSET,mixtures=COMPOSITION.tolist(),replay_seeds=[57041971,57045813],
                unique_physics_draws=len(already),verified_prior_draws=len(history),
                pilot_episodes_per_arm_regime=N*2*PILOT_REPS,pilot_attempts_per_arm_regime=N*2*PILOT_REPS*PILOT_ATTEMPTS,
                deployment_objects=N*512,active=int(n_active),changed_labels=int(n_change),
                resolution_disagreements=int(n_diff),solver_endpoints=N*1024,pilot_counts=audit_counts,files=manifest)
    (ROOT/'world_audit57.json').write_text(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','forecasts','solver','panel','audit'])
    parser.add_argument('ids',nargs='*',type=int)
    a=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    if a.stage=='forecasts':
        forecasts();return
    assert not r.torch.cuda.is_available()
    if a.stage=='audit':
        audit();return
    for i in a.ids or range(N):
        if a.stage=='prepare':
            v55.prepare(OFFSET+i)
            pilots(i)
        elif a.stage=='solver':
            v55.labels(OFFSET+i,False);v55.labels(OFFSET+i,True)
        else:panel(i)

if __name__=='__main__':main()
