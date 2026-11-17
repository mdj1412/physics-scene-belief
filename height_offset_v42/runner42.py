"""Frozen hidden-contact offset experiment. A prediction is shared across all hidden heights."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'estimated_projection_v41'))
import runner41 as r
import selector42
import world42 as w

v38=r.v38
torch=r.torch
OUT=ROOT/'results'
PANELS=12
BASE=420_000_000
REPEATS=8
REPS_TIMING=4
POLICIES=('zero','fixed','optimizer','selector')
TIMING_METHODS=('original_zero','estimated_zero','estimated_fixed','estimated_optimizer','estimated_selector')


def directions(panel,fam,hkey):
    paired=w.direction_pairs(panel,fam,r.hvalue(hkey))
    est=r.world.estimate_direction(paired)
    oracle=r.world.oracle_direction(fam,r.hvalue(hkey))
    path=OUT/'directions'/f'panel_{panel:02d}_{fam}_{hkey}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        with np.load(path) as a:
            for key,value in (('pairs',paired),('estimated',est),('oracle',oracle)):
                np.testing.assert_allclose(a[key],value,atol=1e-12,rtol=0)
    else:
        v38.atomic_npz(path,pairs=paired,estimated=est,oracle=oracle)
    return paired,oracle,est


def cost(path,records,detector,artifact,scorer,pair,banks,height,paired,xx,obs,actions,bases,z,wopt,choice,seed,fingerprint):
    """Measure each end-to-end policy on one noise realization; randomized complete crossover."""
    rng=np.random.default_rng(seed)
    one=[{a:s[a][:,:1] for a in (r.Q,r.P)} for s in obs]
    order=np.empty((REPS_TIMING,len(TIMING_METHODS)),np.int8)
    warm=np.empty_like(order,dtype=float)
    setup=np.empty_like(warm)
    maximum=0.
    for j in range(-1,REPS_TIMING):
        perm=rng.permutation(len(TIMING_METHODS))
        if j>=0:order[j]=perm
        for m in perm:
            decoder=0 if m==0 else 2
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            u=None if decoder==0 else r.world.estimate_direction(paired)
            templates=r.templates(banks,height,u)
            torch.cuda.synchronize()
            ready=time.perf_counter_ns()
            # Sensor acquisition reads the first three paid contacts. Only optimizer/selector read the fourth outcome.
            actual=v38.prefix_actions(detector,records,pair,xx)
            pred=r.decode(banks,templates,xx,actual,one)[:,:,0]
            if m==2:
                zz,ww=v38.policy(artifact,records,'fixed',True)
                pred=v38.apply(pred,(zz*ww)[:,None],v38.BETA)
            elif m in (3,4):
                x,y,dirs=v38.observations(records)
                zz,ww=r.continuous.source_optimum(artifact,x,y[:,:3],dirs[:,:3],y[:,3])
                if m==4:
                    selected=selector42.select(scorer,pred,zz,ww,actual,y[:,:3],y[:,3],dirs[:,:3])
                    ww=np.choose(selected,[0.,.325,ww])
                pred=v38.apply(pred,(zz*ww)[:,None],v38.BETA)
            torch.cuda.synchronize()
            end=time.perf_counter_ns()
            if j>=0:
                warm[j,m]=(end-ready)/1.e6
                setup[j,m]=(ready-start)/1.e6
            base=bases[decoder,:,:,0]
            expected=base
            if m>=2:
                expected_weight=(.325 if m==2 else wopt if m==3 else np.choose(choice[:,:,0][2],[0.,.325,wopt]))
                expected=v38.apply(base,(z*expected_weight)[:,None],v38.BETA)
            if m==4:np.testing.assert_array_equal(selected,choice[2,:,0])
            np.testing.assert_array_equal(actual,actions)
            np.testing.assert_allclose(pred,expected,atol=2.e-10,rtol=0)
            maximum=max(maximum,float(np.abs(pred-expected).max()))
    path.parent.mkdir(exist_ok=True)
    v38.atomic_npz(path,full_ms=warm,setup_ms=setup,cold_ms=warm+setup,order=order,methods=TIMING_METHODS)
    v38.atomic_json(path.with_suffix('.json'),dict(sha256=v38.digest(path),fingerprint=fingerprint,max_error=maximum))


def block(panel,c,banks,items,paired,detector,artifact,scorer,alpha,pair,seed,start,cache,fingerprint):
    fam,hkey,var,law=r.CONDITIONS[c]
    path=OUT/r.CELLS[c]/f'catalog_{panel:02d}.npz'
    timing_path=OUT/'timing'/f'panel_{panel:02d}.npz'
    tc,_,ts=r.schedule(panel)
    if r.old.already_complete(path,fingerprint) and (c!=tc or r.old.already_complete(timing_path,fingerprint)):
        return
    path.parent.mkdir(exist_ok=True)
    h=r.hvalue(hkey)
    new_panel=w.catalog_panel(panel)
    bases=np.empty((3,2,8,128,REPEATS))
    n_offsets=len(w.OFFSETS_MM)
    labels=np.empty((n_offsets,2,3,8,128),bool)
    risk=np.empty((n_offsets,3,4,2,3,8))
    choices=np.empty((3,2,3,8,REPEATS),np.int8)
    actions=np.empty((2,8,128),int)
    cue=np.empty((2,8))
    weights=np.empty((2,3,8))
    cal_x=[]; first3=[]; dirs3=[]; ids=[]; fourth=[]
    for si,sign in enumerate(r.SIGNS):
        context=(fam,var,h,law,sign,new_panel)
        dep=[r.world.deployment(seed+200+e,a,context,r.REGIMES[0],REPEATS) for e,a in enumerate(alpha)]
        xx,nominal,objects,obs=zip(*dep)
        xx=np.stack(xx)
        theta=np.concatenate(objects)
        y4=[]
        for ri,reg in enumerate(r.REGIMES):
            targets=w.offset_labels(theta,context,reg).reshape(n_offsets,8,128)
            labels[:,si,ri]=targets
            cal=[r.world.calibration(seed+100+e,a,context,reg,start)[0] for e,a in enumerate(alpha)]
            x,y,dirs=v38.observations(cal)
            if ri==0:
                np.testing.assert_array_equal(targets[w.OFFSETS_MM.index(0.)],np.stack(nominal))
                key=r.old.evidence_hash(cal,xx)
                if key not in cache:cache[key]=v38.prefix_actions(detector,cal,pair,xx)
                actions[si]=cache[key]
                bases[:,si]=np.stack([r.decode(banks,item,xx,actions[si],obs) for item in items])
                cal_x.append(x);first3.append(y[:,:3]);dirs3.append(dirs[:,:3]);ids.append(np.stack([rec[1] for rec in cal]))
            else:
                np.testing.assert_array_equal(x,cal_x[-1])
                np.testing.assert_array_equal(y[:,:3],first3[-1])
                np.testing.assert_array_equal(dirs[:,:3],dirs3[-1])
            z,ww=r.continuous.source_optimum(artifact,x,y[:,:3],dirs[:,:3],y[:,3])
            if ri==0:cue[si]=z
            else:np.testing.assert_array_equal(z,cue[si])
            weights[si,ri]=ww
            y4.append(y[:,3])
            for di,base in enumerate(bases[:,si]):
                pick=selector42.select(scorer,base,z,ww,actions[si],y[:,:3],y[:,3],dirs[:,:3])
                if pick.shape!=(8,REPEATS) or not np.isin(pick,[0,1,2]).all():raise ValueError('Invalid selector output')
                choices[di,si,ri]=pick
                fixed=v38.apply(base,(z*.325)[:,None,None],v38.BETA)
                opt=v38.apply(base,(z*ww)[:,None,None],v38.BETA)
                ps=(base,fixed,opt,np.choose(pick[:,None,:],[base,fixed,opt]))
                for pi,pred in enumerate(ps):
                    risk[:,di,pi,si,ri]=np.square(pred[None]-targets[:,:,:,None]).mean((2,3))
            if tc==c and ts==si*3+ri:
                cost(timing_path,cal,detector,artifact,scorer,pair,banks,h,paired,xx,obs,actions[si],bases[:,si],
                     z,ww,choices[:,si,ri],seed+313+c,fingerprint)
        fourth.append(np.stack(y4))
    v38.atomic_npz(path,labels=labels,risk=risk,p_base=bases,choice=choices,actions=actions,z=cue,weights=weights,
        cal_x=np.stack(cal_x),first3_y=np.stack(first3),fourth_y=np.stack(fourth),dirs=np.stack(dirs3),ids=np.stack(ids),
        alpha=alpha,offsets_mm=w.OFFSETS_MM,methods=POLICIES,decoders=r.DECODERS)
    v38.atomic_json(path.with_suffix('.json'),dict(sha256=v38.digest(path),fingerprint=fingerprint,panel=panel,new_panel=new_panel,
        seed=seed,start=start,c=c,cell=r.CELLS[c],model_bank_seed=pair,calibration_seeds=[seed+100+e for e in range(8)],
        deployment_seeds=[seed+200+e for e in range(8)]))
    print('completed',panel,r.CELLS[c],flush=True)


def main():
    if not (os.getenv('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1):raise RuntimeError('One assigned GPU required')
    torch.set_num_threads(2)
    prior=json.loads((PROJECT/'estimated_projection_v41/results/run.json').read_text())['hashes']
    locked=dict(prior)
    for line in (ROOT/'freeze42.sha256').read_text().splitlines():
        val,name=line.split(maxsplit=1);locked['height_offset_v42/'+name]=val
    for name,val in locked.items():
        if v38.digest(PROJECT/name)!=val:raise ValueError('Dependency mismatch '+name)
    fingerprint=hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest()
    meta=dict(fingerprint=fingerprint,hashes=locked,job=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
        cuda=torch.version.cuda,base=BASE,panels=PANELS,repeats=REPEATS,offsets_mm=w.OFFSETS_MM,cells=r.CELLS,beta=v38.BETA,
        methods=POLICIES,decoders=r.DECODERS,regimes=r.REGIMES,signs=r.SIGNS,timing_methods=TIMING_METHODS,timing_repetitions=REPS_TIMING)
    OUT.mkdir(exist_ok=True)
    v38.atomic_json(OUT/'run.json',meta)
    with torch.inference_mode():
        detector=v38.Detector();artifact=v38.load_artifacts()[v38.SPEED];scorer=selector42.load_selector()
        for panel in range(PANELS):
            seed=BASE+100_000*panel
            start=(-1,1)[panel%2]
            pair=[seed+301,seed+302]
            alpha=v38.ALPHAS[np.random.default_rng(seed+401).permutation(8)]
            banks=[v38.Engine(pair,*a).inner for a in alpha]
            data={(f,k):directions(panel,f,k) for f in r.world.FAMILIES for k in r.HEIGHTS}
            items={k:[r.templates(banks,r.hvalue(k[1]),u) for u in (None,ora,est)] for k,(_,ora,est) in data.items()}
            cache={}
            for c in np.random.default_rng(seed+707).permutation(len(r.CELLS)):
                fam,k,_,_=r.CONDITIONS[c]
                block(panel,int(c),banks,items[fam,k],data[fam,k][0],detector,artifact,scorer,alpha,pair,seed,start,cache,fingerprint)
    v38.atomic_json(OUT/'completion.json',dict(**meta,blocks=PANELS*len(r.CELLS),timing_clusters=PANELS))


if __name__=='__main__':main()
