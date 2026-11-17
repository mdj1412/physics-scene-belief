"""Matched contact transfer with independent, label-free geometry estimation."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'tangent_transfer_v40'))
import run40 as old
import world41 as world
import selector41
from physics25 import profile, impact, QUERY_INCOMING

v38=old.previous
continuous=old.continuous
torch=v38.torch
Q,P=v38.Q,v38.P
DECODERS=('original','oracle','estimated')
POLICIES=('fixed','optimizer','selector')
HEIGHTS=('h19','h31')
CONDITIONS=tuple(itertools.product(world.FAMILIES,HEIGHTS,world.CONTOURS,world.SURFACES))
CELLS=tuple('_'.join(cell) for cell in CONDITIONS)
REGIMES=old.REGIMES
SIGNS=old.SIGNS
OUT=ROOT/'results'
BASE=410_000_000
PANELS=12
REPEATS=32


def hvalue(key):
    return int(key[1:])/1000


def rawpath(c,panel,block):
    return OUT/CELLS[c]/f'catalog_{panel:02d}'/f'block_{block:02d}.npz'


def direction_file(panel,family,hkey):
    return OUT/'directions'/f'panel_{panel:02d}_{family}_{hkey}.npz'


def inputs(panel,family,hkey):
    height=hvalue(hkey)
    paired=world.direction_pairs(panel,family,height,n=32)
    estimated=world.estimate_direction(paired)
    oracle=world.oracle_direction(family,height)
    for vector in (estimated,oracle):
        np.testing.assert_allclose(np.linalg.norm(vector),1.,atol=1.e-10)
    path=direction_file(panel,family,hkey)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        v38.atomic_npz(path,pairs=paired,estimated=estimated,oracle=oracle)
    else:
        with np.load(path) as a:
            for k,v in [('pairs',paired),('estimated',estimated),('oracle',oracle)]:
                np.testing.assert_allclose(a[k],v,rtol=0.,atol=1.e-12)
    return paired,estimated,oracle


def templates(banks,height,unit=None):
    """Recompute candidate profile and Coulomb impulse label at actual query height."""
    out=[]
    heights=world.view_heights(height)
    for bank in banks:
        white=profile(bank.theta,heights)/bank.sigma[Q]
        label=impact(bank.theta,height,QUERY_INCOMING)[1].to(bank.theta.dtype)
        if unit is not None:
            u=v38.t(unit)
            white=white-(white@u)[:,None]*u
        out.append((white,.5*white.square().sum(-1),label))
    return out


def decode(banks,items,xx,actions,sensors):
    """Use the actual chosen sensor, with the same likelihood for one or 32 noise draws."""
    r=sensors[0][Q].shape[1]
    result=np.empty((len(banks),xx.shape[1],r))
    for e,(bank,(white,norm,label)) in enumerate(zip(banks,items)):
        for a in (Q,P):
            selected=np.flatnonzero(actions[e]==a)
            # Large repeated-noise arrays are bounded; warm online evaluation is batched per sensor.
            batches=[selected] if r==1 else np.array_split(selected,max(1,(len(selected)+15)//16))
            for ids in batches:
                prior=bank.fit(v38.t(xx[e,ids]))[:,None,:]
                y=v38.t(sensors[e][a][ids])
                if a==Q:
                    ll=(y/bank.sigma[a])@white.T-norm
                    posterior=torch.softmax(prior.clamp_min(1.e-300).log()+ll,-1)
                else:
                    posterior=bank.update(prior,a,y)
                result[e,ids]=(posterior@label).cpu().numpy()
    return result


def corrected(model,base,z,w,actions,y,dirs,index):
    if index==0:
        gain=.325
    elif index==1:
        gain=w[:,None,None]
    else:
        pick=selector41.select(model,base,z,w,actions,y[:,:3],y[:,3],dirs[:,:3])
        gain=np.where(pick[:,None,:],w[:,None,None],.325)
    return v38.apply(base,z[:,None,None]*gain,v38.BETA)


def schedule(panel):
    key=(world.FAMILIES[panel%2],HEIGHTS[(panel//2)%2],world.CONTOURS[(panel//4)%3],world.SURFACES[(panel//3)%2])
    return CONDITIONS.index(key),panel%2,panel%6


def costs(detector,artifact,model,banks,records,xx,obs,actions,reference,z,w,y,dirs,
          pair,paired,oracle,family,height,seed):
    rng=np.random.default_rng(seed)
    warm=np.empty((3,3,3));setup=np.empty_like(warm);fit_ms=np.empty_like(warm)
    trial_order=[]
    one=[{a:row[a][:,:1] for a in (Q,P)} for row in obs]
    for rep in (-1,0,1,2):
        order=rng.permutation(9)
        if rep>=0:trial_order.append(order)
        for m in order:
            di,pi=divmod(int(m),3)
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            unit=(None if di==0 else oracle if di==1 else world.estimate_direction(paired))
            fit_end=time.perf_counter_ns()
            item=templates(banks,height,unit)
            torch.cuda.synchronize()
            setup_end=time.perf_counter_ns()
            choice=v38.prefix_actions(detector,records,pair,xx)
            p=decode(banks,item,xx,choice,one)[:,:,0]
            x,ys,ds=v38.observations(records)
            if pi==0:
                zz,ww=v38.policy(artifact,records,'fixed',True)
            else:
                zz,ww=continuous.source_optimum(artifact,x,ys[:,:3],ds[:,:3],ys[:,3])
            if pi==2:
                choose=selector41.select(model,p,zz,ww,choice,ys[:,:3],ys[:,3],ds[:,:3])
                ww=np.where(choose,ww,.325)
            answer=v38.apply(p,(zz*ww)[:,None],v38.BETA)
            torch.cuda.synchronize()
            end=time.perf_counter_ns()
            np.testing.assert_array_equal(choice,actions)
            expected=corrected(model,reference[di],z,w,actions,y,dirs,pi)[:,:,0]
            np.testing.assert_allclose(answer,expected,rtol=0.,atol=2.e-10)
            if rep>=0:
                warm[rep,di,pi]=(end-setup_end)/1e6
                setup[rep,di,pi]=(setup_end-start)/1e6
                fit_ms[rep,di,pi]=(fit_end-start)/1e6
    return warm, setup, fit_ms, np.array(trial_order)


def run_block(c,panel,b,seed,pair,alpha,banks,items,di_data,detector,artifact,model,fingerprint,cache):
    path=rawpath(c,panel,b)
    path.parent.mkdir(parents=True,exist_ok=True)
    if old.already_complete(path,fingerprint):
        return
    fam,hkey,variant,law=CONDITIONS[c]
    height=hvalue(hkey)
    start=(-1,1)[b]
    timing_cell,timing_block,timing_stratum=schedule(panel)
    rows={key:[] for key in ('risk','p_base','actions','z','weights','labels','chosen_optimizer','cal_x',
                             'first3_y','fourth_y','fourth_stick','dirs','ids')}
    warm=setup=fit=np.empty((0,3,3));trial=np.empty((0,9),int)
    for si,sign in enumerate(SIGNS):
        context=(fam,variant,height,law,sign,panel)
        grouped={key:[] for key in ('weights','labels','fourth_y','fourth_stick','risk','chosen_optimizer')}
        for ri,reg in enumerate(REGIMES):
            cal=[world.calibration(seed+100+e,a,context,reg,start) for e,a in enumerate(alpha)]
            records=[v[0] for v in cal]
            x,y,dirs=v38.observations(records)
            deployment=[world.deployment(seed+200+e,a,context,reg,REPEATS) for e,a in enumerate(alpha)]
            xx,target,_,obs=zip(*deployment)
            xx,target=np.stack(xx),np.stack(target)
            if ri==0:
                key=old.evidence_hash(records,xx)
                if key not in cache:cache[key]=v38.prefix_actions(detector,records,pair,xx)
                actions=cache[key]
                bases=np.stack([decode(banks,item,xx,actions,obs) for item in items])
                rows['p_base'].append(bases)
                rows['actions'].append(actions);rows['cal_x'].append(x)
                rows['first3_y'].append(y[:,:3]);rows['dirs'].append(dirs[:,:3]);rows['ids'].append(np.stack([v[1] for v in records]))
                old_x,old_xx,old_y,old_obs=x,xx,y[:,:3],obs
            else:
                for now,ref in [(x,old_x),(xx,old_xx),(y[:,:3],old_y)]:
                    np.testing.assert_array_equal(now,ref)
                for now,ref in zip(obs,old_obs):
                    for a in (Q,P):np.testing.assert_array_equal(now[a],ref[a])
            z,w=continuous.source_optimum(artifact,x,y[:,:3],dirs[:,:3],y[:,3])
            if ri==0:rows['z'].append(z)
            else:np.testing.assert_array_equal(rows['z'][-1],z)
            take=[];loss=[]
            for base in bases:
                chosen=selector41.select(model,base,z,w,actions,y[:,:3],y[:,3],dirs[:,:3])
                if chosen.shape!=(len(alpha),REPEATS):raise ValueError('Unexpected selector shape')
                take.append(chosen)
                fixed=v38.apply(base,(z*.325)[:,None,None],v38.BETA)
                opt=v38.apply(base,(z*w)[:,None,None],v38.BETA)
                ps=[fixed,opt,np.where(chosen[:,None,:],opt,fixed)]
                loss.append([np.square(p-target[:,:,None]).mean((1,2)) for p in ps])
            if (c,b,si*3+ri)==(timing_cell,timing_block,timing_stratum):
                paired,_,oracle=di_data
                warm,setup,fit,trial=costs(detector,artifact,model,banks,records,xx,obs,actions,bases,z,w,y,dirs,
                                          pair,paired,oracle,fam,height,seed+609+c)
            for k,v in [('labels',target),('weights',w),('fourth_y',y[:,3]),('risk',loss),
                        ('fourth_stick',[item[3] for item in cal]),('chosen_optimizer',take)]:
                grouped[k].append(v)
        for k,vs in grouped.items():rows[k].append(np.stack(vs))
    fields={name:np.stack(value) for name,value in rows.items()}
    fields['p_base']=np.swapaxes(fields['p_base'],0,1)
    v38.atomic_npz(path,**fields,alpha=alpha,full_ms=warm,setup_ms=setup,cold_ms=setup+warm,
                   fit_ms=fit,trial_order=trial,methods=POLICIES,decoders=DECODERS)
    v38.atomic_json(path.with_suffix('.json'),dict(cell=CELLS[c],panel=panel,block=b,base_seed=seed,
                    start=start,pair=pair,calibration_seeds=[seed+100+e for e in range(8)],
                    deployment_seeds=[seed+200+e for e in range(8)],fingerprint=fingerprint,sha256=v38.digest(path)))
    print('completed',CELLS[c],panel,b,flush=True)


def main():
    if not (os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1):
        raise RuntimeError('One assigned GPU is required')
    torch.set_num_threads(2)
    legacy=json.loads((PROJECT/'tangent_transfer_v40/results/run.json').read_text())['hashes']
    locked={k:v for k,v in legacy.items() if not k.startswith('tangent_transfer_v40/')}
    for path in ('run40.py','physics40.py'):
        name='tangent_transfer_v40/'+path
        locked[name]=legacy[name]
    for line in (ROOT/'freeze41.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1);locked['estimated_projection_v41/'+name]=expected
    for name,expected in locked.items():
        if v38.digest(PROJECT/name)!=expected:raise RuntimeError(f'Frozen input changed: {name}')
    fingerprint=hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest()
    metadata=dict(fingerprint=fingerprint,hashes=locked,job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),
                torch=str(torch.__version__),cuda=torch.version.cuda,cells=CELLS,methods=POLICIES,decoders=DECODERS,
                regimes=REGIMES,signs=SIGNS,beta=v38.BETA,base=BASE,catalogs=PANELS,blocks_per_catalog=2)
    OUT.mkdir(exist_ok=True)
    v38.atomic_json(OUT/'run.json',metadata)
    with torch.inference_mode():
        model=selector41.load_selector();detector=v38.Detector();artifact=v38.load_artifacts()[v38.SPEED]
        for panel in range(PANELS):
            directions={(f,h):inputs(panel,f,h) for f,h in itertools.product(world.FAMILIES,HEIGHTS)}
            for b in range(2):
                seed=BASE+100_000*panel+10_000*b
                pair=[seed+301,seed+302]
                alpha=v38.ALPHAS[np.random.default_rng(seed+401).permutation(8)]
                banks=[v38.Engine(pair,*a).inner for a in alpha]
                items={key:[templates(banks,hvalue(key[1]),u) for u in (None,ori,est)]
                       for key,(_,est,ori) in directions.items()}
                cache={}
                for c in np.random.default_rng(seed+707).permutation(len(CELLS)):
                    fam,hkey,_,_=CONDITIONS[c];key=(fam,hkey)
                    run_block(int(c),panel,b,seed,pair,alpha,banks,items[key],directions[key],
                              detector,artifact,model,fingerprint,cache)
    v38.atomic_json(OUT/'completion.json',dict(**metadata,blocks=576,timing_clusters=12))


if __name__=='__main__':main()
