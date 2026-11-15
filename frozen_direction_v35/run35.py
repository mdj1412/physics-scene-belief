"""Apply frozen v34 head to fixed-4 deployment and time the paired end-to-end paths."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from target35 import FAMILIES, calibration, deployment_truth, deployment_x, t

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
V34 = BASE / 'bidirectional_probe_v34'
from run34 import (ActionConditionedDetector, History, PosteriorPredictiveEngine,
                   Engine, v33, batch_gap, atomic_json, atomic_npz, Q, P, torch,
                   SCHEDULES, digest)
from direct_head34 import cue_est
from anchored34 import apply

OUT = ROOT/'results'
CONFIG = json.loads((ROOT/'design35.json').read_text())
LOCK = json.loads((V34/'diag_models.json').read_text())
BETAS = np.array(json.loads((V34/'anchored34.json').read_text())['beta'])[:,0]
assert BETAS[0] == CONFIG['primary_beta']

def paid(records):
    return (np.stack([x[ids] for x,ids,y,d in records]), np.stack([r[2] for r in records]),
            np.stack([r[3] for r in records]))

def choices(det, records, pair, xx):
    choices = []
    for rec,x in zip(records,xx):
        h = History(det,rec)
        engine = PosteriorPredictiveEngine(pair,h.fit(4))
        choices.append(v33.decisions(batch_gap(engine,x)))
    choices = np.array(choices)
    assert np.all((choices==P).sum(1)==13)
    return choices

def scores_all(decs,xx,acts,obs):
    """Repeat v33's same single-observation decoder over its paired sensor draws."""
    scores = np.empty((8,128,32))
    for i,(dec,x,ac,sensors) in enumerate(zip(decs,xx,acts,obs)):
        for a in (Q,P):
            ids=np.flatnonzero(ac==a)
            # One world x 32 draws uses v28 potential_costs convention verbatim.
            for j in ids:
                prior=dec.inner.fit(t(x[j]))
                prob=dec.inner.update(prior,a,t(sensors[a][j])) @ dec.inner.label[:,0]
                scores[i,j]=prob.cpu().numpy()/np.sqrt(2.)
    return scores

def full_call(det, rec, pair, decs, xx, sensors, schedule, method):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    ac = choices(det,rec,pair,xx)
    p = np.stack([v33.decode_selected(dec,x,a,y) for dec,x,a,y in zip(decs,xx,ac,sensors)])
    if method:
        z = cue_est(LOCK,0,*paid(rec),schedule)
        p = apply(p,z[:,None],BETAS[0])
    torch.cuda.synchronize()
    return ac,p,(time.perf_counter_ns()-start)*1e-6

def timing(det, recs, pair, decs, xx, obs, matched, seed, timed):
    repeats=3 if timed else 0
    ms=np.empty((2,repeats,2)); order=np.empty((repeats,4),dtype=int)
    sensors=[{a:o[a][:,0] for a in (Q,P)} for o in obs]
    rng=np.random.default_rng(seed)
    if timed:
        for rep in (-1,0,1,2):
            perm=rng.permutation(4)
            if rep>=0: order[rep]=perm
            for ix in perm:
                s,m=divmod(ix,2)
                a,p,dt=full_call(det,recs[s],pair,decs,xx,sensors,SCHEDULES[s],m)
                np.testing.assert_array_equal(a,matched[s]['actions'])
                np.testing.assert_allclose(p,matched[s]['p'+str(m)][:,:,0],atol=1e-9,rtol=0.)
                if rep>=0: ms[s,rep,m]=dt
    return ms,order

def cell(det,fp,family,b,sign):
    obj,surface=FAMILIES[family];law=(obj,surface,sign)
    base=CONFIG['base_seed']+CONFIG['family_step']*family+CONFIG['block_step']*b
    dst=OUT/(obj+'_'+surface+('_plus' if sign==1 else '_minus'))/f'block_{b:03d}'
    dst.mkdir(parents=True,exist_ok=True)
    if (dst/'meta.json').exists():
        saved=json.loads((dst/'meta.json').read_text())
        assert saved['fingerprint']==fp and all(digest(dst/(s+'.npz'))==saved['archives'][s] for s in SCHEDULES)
        return
    alpha=np.array(json.loads((V34/'policy_design34.json').read_text())['alphas'])[np.random.default_rng(base+401).permutation(8)]
    pair=[base+301,base+302];start=(-1,1)[b%2]
    recs=[[calibration(base+100+e,al,law,sc,start) for e,al in enumerate(alpha)] for sc in SCHEDULES]
    for u,v in zip(*recs):
        np.testing.assert_array_equal(u[0],v[0]);np.testing.assert_array_equal(u[1],v[1])
        np.testing.assert_array_equal(u[2][::2],v[2][::2])
    xx=[deployment_x(base+200+e,al,law) for e,al in enumerate(alpha)]
    decs=[Engine(pair,*al) for al in alpha]
    selected=[choices(det,rec,pair,xx) for rec in recs]
    truths=[deployment_truth(base+200+e,al,law) for e,al in enumerate(alpha)]
    labels,theta,obs=zip(*truths)
    packed=[]
    for s,(schedule,rec) in enumerate(zip(SCHEDULES,recs)):
        px,py,pd=paid(rec)
        z=np.stack([cue_est(LOCK,j,px,py,pd,schedule) for j in range(4)])
        p0=scores_all(decs,xx,selected[s],obs)
        p1=apply(p0,z[0,:,None,None],BETAS[0])
        assert np.all(np.isfinite(p1)) and p1.min()>=-1e-12 and p1.max()<1+1e-12
        packed.append(dict(p0=p0,p1=p1,labels=labels,theta=theta,z=z,paid_x=px,paid_y=py,
            paid_direction=pd,paid_ids=[r[1] for r in rec],actions=selected[s],alpha=alpha,betas=BETAS))
    ms,order=timing(det,recs,pair,decs,xx,obs,packed,base+501+int(sign<0),
                    b in CONFIG['timing_blocks'] and surface in CONFIG['timed_surfaces'])
    for s,sch in enumerate(SCHEDULES):
        atomic_npz(dst/(sch+'.npz'),**packed[s],full_ms=ms[s],timing_order=order)
    atomic_json(dst/'meta.json',dict(object=obj,surface=surface,sign=sign,block=b,start=start,base=base,
        pair=pair,calibration_seeds=[base+100+e for e in range(8)],deployment_seeds=[base+200+e for e in range(8)],
        fingerprint=fp,archives={s:digest(dst/(s+'.npz')) for s in SCHEDULES}))

def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    files=[ROOT/name for name in ('run35.py','target35.py','design35.json','run35.sh')]
    files += [V34/name for name in ('diag_models.json','anchored34.json','policy_design34.json','run34.py',
              'direct_head34.py','diag34_model.py','anchored34.py','target34.py','detector34.py')]
    files += [BASE/rel for rel in ('directional_prefix_v33/target33.py','directional_prefix_v33/evaluate33.py',
             'learned_probe_value_v30/contact30.py','probe_gain_reliability_v31/batch31.py','calibration_probe_v28/estimator28.py',
             'population_shift_v27/runner27.py','within_calibration_v25/physics25.py')]
    hashes={str(p.relative_to(BASE)):digest(p) for p in files}
    fp='|'.join(hashes.values())
    OUT.mkdir(exist_ok=True)
    meta=dict(hashes=hashes,job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
              numpy=np.__version__,python=sys.version,cuda=torch.version.cuda)
    atomic_json(OUT/'run.json',meta)
    with torch.inference_mode():
        det=ActionConditionedDetector()
        for block in range(CONFIG['blocks']):
            for f,(obj,sur) in enumerate(FAMILIES):
                for sign in ((1,) if sur=='isotropic' else (1,-1)):
                    tic=time.perf_counter()
                    cell(det,fp,f,block,sign)
                    print(block,obj,sur,sign,round(time.perf_counter()-tic,2),flush=True)
    atomic_json(OUT/'completion.json',dict(**meta,n_archives=160,primary_archives=128,control_archives=32))

if __name__=='__main__': main()
