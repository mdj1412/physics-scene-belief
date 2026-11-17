"""Collect fixed v43-style sensors and four predictions. Target contact events are external."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT/'blind_decoder_v43'))
import inference43 as infer
import model43
import world43 as world

r, torch = infer.r, infer.r.torch
v38 = r.v38
OUT = ROOT/'results'
DECODERS = ('original','privileged')
POLICIES = ('zero','temperature')
METHODS = tuple(f'{d}_{c}' for d in DECODERS for c in POLICIES)
CELLS = tuple(itertools.product(world.TARGET_SHAPES, world.TARGET_SURFACES,
                                world.TARGET_LAWS, world.VARIANTS, world.HEIGHTS))
BASE, SUPPORT_BASE = 440000000, 400
PANELS, ENVS, REPEATS, TIMING_REPS = 12, 4, 4, 3


def path_for(panel, cell):
    stem = '_'.join(cell[:4])+f'_h{round(cell[4]*1000)}'
    return OUT/f'panel_{panel:02d}'/(stem+'.npz')


def population(panel):
    seed = BASE+100000*panel
    alphas = v38.ALPHAS[np.random.default_rng(seed+401).permutation(len(v38.ALPHAS))[:ENVS]]
    signs = np.array([(-1,1)[(panel+e)%2] for e in range(ENVS)])
    regimes = [world.REGIMES[(panel+e)%3] for e in range(ENVS)]
    return seed,alphas,signs,regimes


def context(panel,cell,sign):
    shape,surface,law,variant,height = cell
    return shape,variant,height,surface,law,int(sign),SUPPORT_BASE+panel


def manifest(panel):
    seed,alpha,signs,regimes = population(panel)
    theta=[]; idx=[]
    for e in range(ENVS):
        ctx = context(panel,CELLS[0],signs[e])
        theta.append(world.draw_object(128,seed+200+e,20,alpha[e],ctx).cpu().numpy())
        idx.append(world.w.stream(seed+200+e,4320).integers(world.w.SUPPORT_SIZE,size=128))
    path=OUT/'worlds'/f'panel_{panel:02d}.npz'
    path.parent.mkdir(exist_ok=True,parents=True)
    v38.atomic_npz(path,theta=np.stack(theta),object_index=np.stack(idx),alphas=alpha,
                   signs=signs,regimes=regimes,seed=seed,support_panel=SUPPORT_BASE+panel)


def observations(panel,cell):
    """Legacy experimental apparatus for the factual and paid sensors, never the query label."""
    seed,alphas,signs,regimes=population(panel)
    with np.load(OUT/'worlds'/f'panel_{panel:02d}.npz') as raw:
        theta_np=raw['theta']
    xx, sensors, records=[],[],[]
    height=cell[4]
    for e,alpha in enumerate(alphas):
        ctx=context(panel,cell,signs[e])
        theta=world.t(theta_np[e])
        ds=seed+200+e
        xx.append(world.factual(theta,ds,21,ctx))
        # The observed test-probe height is 0; the counterfactual impact is at 19 or 31 mm.
        response = world.impact(theta,0.,world.w.PROBE_INCOMING,ctx,regimes[e])[0]
        means={r.Q:world.contour(theta,world.w.view_heights(height),ctx),r.P:response}
        row={}
        for kind, mu in means.items():
            eps=world.w.stream(ds,40+kind).normal(size=(len(mu),REPEATS,mu.shape[-1]))
            row[kind]=(mu[:,None]+world.t(eps)*world.w.sigmas()[kind]).cpu().numpy()
        sensors.append(row)
        records.append(world.calibration(seed+100+e,alpha,ctx,regimes[e],(-1,1)[panel%2])[0])
    return theta_np,np.stack(xx),sensors,records


def predict(panel,cell,engine,model,templates,selection_cache,fingerprint):
    path=path_for(panel,cell)
    if r.old.already_complete(path,fingerprint):
        return
    theta,xx,sensors,records=observations(panel,cell)
    key=hashlib.sha256(xx.tobytes()).digest()
    if key not in selection_cache:
        selection_cache[key]=infer.actions43(xx,engine)
    act=selection_cache[key]
    paid=infer.paid_sensors(sensors,act)
    shape,_,_,_,height=cell
    u=world.oracle_direction(shape,height)
    pred=[]
    for d in DECODERS:
        tk=(shape if d=='privileged' else None,height)
        if tk not in templates:
            templates[tk]=infer.make_template(engine,height,u if d=='privileged' else None)
        p=infer.decode43(engine,templates[tk],xx,act,paid)
        pred.append([p,model43.temperature(model,p,d)])
    pred=np.stack(pred)
    if not np.isfinite(pred).all() or pred.min() < -1e-12 or pred.max()>1+1e-12:
        raise ValueError('Invalid probability')
    x,y,dirs=v38.observations(records)
    seed,alphas,signs,regimes=population(panel)
    path.parent.mkdir(parents=True,exist_ok=True)
    v38.atomic_npz(path,theta=theta,xx=xx,selected=act,pred=pred,
                   query_paid=np.stack([m[r.Q] for m in paid]),probe_paid=np.stack([m[r.P] for m in paid]),
                   calibration_x=x,calibration_y=y,calibration_dirs=dirs,
                   signs=signs,regimes=regimes,alphas=alphas,seed=seed,height=height,oracle_direction=u,
                   decoders=DECODERS,policies=POLICIES)
    v38.atomic_json(path.with_suffix('.json'),dict(sha256=v38.digest(path),fingerprint=fingerprint))


def timing(panel,engine,model,fingerprint):
    path=OUT/'timing'/f'panel_{panel:02d}.npz'
    if r.old.already_complete(path,fingerprint): return
    f=panel%8
    cell=(world.TARGET_SHAPES[f//4],world.TARGET_SURFACES[(f//2)%2],
          world.TARGET_LAWS[f%2],world.VARIANTS[(panel//2)%3],world.HEIGHTS[panel%2])
    _,xx,sensors,_=observations(panel,cell)
    one=[{a:row[a][:,:1] for a in (r.Q,r.P)} for row in sensors]
    with np.load(path_for(panel,cell)) as raw:
        expected=raw['pred'][...,:1]
        known_selection=raw['selected']
    u=world.oracle_direction(cell[0],cell[4])
    rng=np.random.default_rng(BASE+100000*panel+4011)
    warm=np.zeros((TIMING_REPS,len(METHODS))); cold=warm.copy(); order=warm.astype(int)
    max_error=0.
    for rep in range(-1,TIMING_REPS):
        seq=rng.permutation(len(METHODS))
        if rep>=0:order[rep]=seq
        for m in seq:
            di,c=divmod(int(m),len(POLICIES))
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            template=infer.make_template(engine,cell[4],u if di else None)
            torch.cuda.synchronize()
            ready=time.perf_counter_ns()
            act=infer.actions43(xx,engine)
            paid=infer.paid_sensors(one,act)
            p=infer.decode43(engine,template,xx,act,paid)
            if c:p=model43.temperature(model,p,DECODERS[di])
            torch.cuda.synchronize()
            end=time.perf_counter_ns()
            if rep>=0:
                warm[rep,m]=(end-ready)/1.e6
                cold[rep,m]=(end-start)/1.e6
            np.testing.assert_array_equal(act,known_selection)
            np.testing.assert_allclose(p,expected[di,c],rtol=0,atol=2.e-9)
            max_error=max(max_error,float(abs(p-expected[di,c]).max()))
    path.parent.mkdir(exist_ok=True,parents=True)
    v38.atomic_npz(path,warm_ms=warm,cold_ms=cold,order=order,methods=METHODS,max_error=max_error)
    v38.atomic_json(path.with_suffix('.json'),dict(sha256=v38.digest(path),fingerprint=fingerprint))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('even','odd','timing'))
    mode=ap.parse_args().mode
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expected one allocated GPU')
    torch.set_num_threads(1)
    hashes=json.loads((PROJECT/'blind_decoder_v43/results/run_even.json').read_text())['hashes']
    for line in (ROOT/'gpu_freeze44.sha256').read_text().splitlines():
        val,name=line.split(maxsplit=1)
        hashes[ROOT.name+'/'+name]=val
    for name,val in hashes.items():
        if v38.digest(PROJECT/name)!=val:raise ValueError('Broken artifact hash: '+name)
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    meta=dict(job=os.getenv('SLURM_JOB_ID'),mode=mode,hashes=hashes,fingerprint=fingerprint,
              gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
              methods=METHODS,base=BASE,support_start=SUPPORT_BASE,panels=PANELS,noise=REPEATS)
    OUT.mkdir(exist_ok=True);v38.atomic_json(OUT/f'run_{mode}.json',meta)
    with torch.inference_mode():
        panels=list(range(PANELS) if mode=='timing' else range(int(mode=='odd'),PANELS,2))
        if mode!='timing':
            for panel in panels:manifest(panel)
        engine=infer.make_bank(); model=model43.load_model(); templates={}
        for panel in panels:
            if mode=='timing':
                timing(panel,engine,model,fingerprint)
            else:
                cache={}
                for i in np.random.default_rng(BASE+100000*panel+707).permutation(len(CELLS)):
                    predict(panel,CELLS[i],engine,model,templates,cache,fingerprint)
            print('complete',mode,panel,flush=True)
    v38.atomic_json(OUT/f'complete_{mode}.json',meta)


if __name__=='__main__':main()
