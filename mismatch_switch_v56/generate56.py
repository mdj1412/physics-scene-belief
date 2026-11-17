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

OFFSET = 1000
N = v55.N
W = np.array([.10, .35, .40, .4348074444131813, .47, .52, .80])
PILOT_REPS = 512
PILOT_ATTEMPTS = 16
REF_DEADZONE = .12
world50,policy50,r = v55.world50,v55.policy50,v55.r
install = v55.install
forward56 = v55.forward55

def destination(group, physical_catalog):
    assert OFFSET <= physical_catalog < OFFSET+N
    return ROOT/group/f'panel_{physical_catalog-OFFSET:02d}.npz'

# Every original v55 function (material, object, gauge, solver, decoder) receives
# the fresh, globally disjoint physical catalog key. Only the output location is remapped.
v55.dest = destination

def pop56(i):
    return v55.pop55(OFFSET+i)

def raw56(i):
    return v55.raw55(OFFSET+i)

def measured56(pop,i):
    return v55.measured55(pop,OFFSET+i)

def pilots(i):
    """One independent prior session per sacrificial pair, on the same target surface."""
    pop=pop56(i)
    amp=np.array([[world50.gauge_mm(f,h,1,1.) for h in world50.HEIGHTS] for f in world50.FAMILIES])
    dimensions=(2,len(W),PILOT_REPS,PILOT_ATTEMPTS)
    rg=np.random.default_rng([56020419,i,1])
    # Paired sensitivity across w: the same fresh fixture/noise uniforms for all w.
    sh=(2,1,PILOT_REPS,PILOT_ATTEMPTS)
    mat=np.arange(2)[:,None,None,None]
    family=rg.integers(4,size=sh); height=rg.integers(2,size=sh)
    gain=rg.uniform(.94,1.06,(2,2))[mat,height]
    bias=rg.uniform(-.020,.020,(2,2))[mat,height]
    latent_ref=rg.choice([-1,0,1],size=sh,p=[.25,.50,.25]).astype('i1')
    ref_mm=gain*amp[family,height]*rg.uniform(.90,1.10,sh)*latent_ref+bias
    ref_mm+=rg.normal(0,.017,sh)+rg.normal(0,.060,sh)
    ref=np.where(abs(ref_mm)>=REF_DEADZONE,np.sign(ref_mm),0).astype('i1')
    high=rg.random(sh)<W[None,:,None,None]
    low_component=rg.random(sh)<.5
    eps=np.where(high,.5,.25*low_component)
    revert=rg.random(sh)<eps
    impact=np.where(revert,-latent_ref,latent_ref).astype('i1')
    ref=np.broadcast_to(ref,dimensions)
    valid=(ref!=0)&(impact!=0)
    assert np.array_equal(np.broadcast_to(latent_ref==0,dimensions),impact==0)
    materials=[f'v55_{f}_{OFFSET+i:02d}' for f in ('braid','seam')]
    assert set(materials)==set(pop['surface'])
    # Estimator and timing receive exclusively the two signed specimen readings.
    v55.save('pilots',OFFSET+i,pilot_ref=ref,pilot_imp=impact)
    # Separate diagnostic observables; no values in this file are available to the switch.
    axes=(0,2,3)
    v55.save('pilot_audit',OFFSET+i,W=W,replicates=PILOT_REPS,attempts=PILOT_ATTEMPTS,
             readable=valid.sum(axes),active=(impact!=0).sum(axes),true_high=high.sum(axes),
             disagree=((ref!=impact)&valid).sum(axes),reference_errors=((ref!=latent_ref)&valid).sum(axes),
             latent_e_sum=eps.sum(axes),unreadable_groups8=(~valid[...,:8].any(-1)).sum((0,2)),
             groups=2*PILOT_REPS,materials=materials)

def forecasts():
    install()
    assert r.torch.cuda.device_count()==1
    r.torch.set_num_threads(1)
    with r.torch.inference_mode():
        engine,templates=policy50.engine50()
        coef=policy50.coefficients()[0]
        for i in range(N):
            plus,minus,act=forward56(measured56(pop56(i),i),engine,templates,coef)
            assert np.isfinite([plus,minus]).all()
            assert min(plus.min(),minus.min())>=0 and max(plus.max(),minus.max())<=1
            v55.save('forecasts',OFFSET+i,pplus=plus,pminus=minus,actions=act)
            print('forecast',i,flush=True)

def panel(i):
    parts=[v55.load(g,OFFSET+i) for g in ('raw','forecasts','solver','solver_ultra')]
    v55.save('panels',OFFSET+i,**parts[0],**parts[1],**parts[2],**parts[3])
    import score55 as score
    score.ROOT=ROOT
    score.REPLAY_SEEDS=[56041971,56045813]
    score.panel_score(i,ROOT/'scored56')
    print('scored',i,flush=True)

def audit():
    history=set()
    for directory in (V55,ROOT.parent/'polarity_calibration_v50'):
        for f in sorted((directory/'worlds').glob('panel_*.npz')):
            with np.load(f) as data:
                for key in ('theta','cal_theta','old_cal_theta'):
                    if key in data: history.update(map(bytes,np.ascontiguousarray(data[key]).reshape(-1,6)))
    already=set(); manifest=[]
    pcs=[]; n_change=n_active=n_diff=0
    for i in range(N):
        p=pop56(i)
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
                         for g in ('panels','scored56','pilots')})
    audit_counts={k:np.stack([p[k] for p in pcs]).sum(0).tolist() for k in
                  ('readable','active','true_high','disagree','reference_errors','latent_e_sum','unreadable_groups8')}
    result=dict(catalogs=N,materials=2*N,world_key_offset=OFFSET,W=W.tolist(),replay_seeds=[56041971,56045813],
                unique_physics_draws=len(already),verified_prior_draws=len(history),
                primary_sessions_per_w=N*2*PILOT_REPS,pilot_attempts_per_w=N*2*PILOT_REPS*PILOT_ATTEMPTS,
                deployment_objects=N*512,active=int(n_active),changed_labels=int(n_change),
                resolution_disagreements=int(n_diff),solver_endpoints=N*1024,pilot_counts=audit_counts,files=manifest)
    (ROOT/'world_audit56.json').write_text(json.dumps(result,indent=2))

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
