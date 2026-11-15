"""Run speed-paired four-probe policies on a single reserved GPU."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import target37 as world
from detector37 import SpeedDetector
from run36 import (Engine, PosteriorPredictiveEngine, batch_gap, Q, P, v33, t, torch,
                   atomic_json, atomic_npz, digest, apply, potentials, select_prob, observations)
from gate37 import load_artifacts,predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
CFG=json.loads((ROOT/'design37.json').read_text())
SPEEDS=CFG['speeds']
BETA=CFG['beta']

def one_action(det,record,pair,x):
    posterior=det.fit_actions(record,True)
    acquisition=PosteriorPredictiveEngine(pair,posterior)
    return v33.decisions(batch_gap(acquisition,x))

def choose(det,records,pair,xx,speed,cache=None):
    det.use_speed(speed)
    output=[]
    for e,(record,x) in enumerate(zip(records,xx)):
        if cache is None:
            output.append(one_action(det,record,pair,x))
        else:
            # Cache only exact same paid responses within a fixed environment,
            # motion and bank. The 1024-row factual pool and xx share their seed.
            key=(e,speed,record[2].tobytes())
            if key not in cache: cache[key]=one_action(det,record,pair,x)
            output.append(cache[key])
    return np.stack(output)

def estimate(artifact,records,include_fourth):
    x,y,d=observations(records)
    return predict(artifact,x,y[:,:3],d[:,:3],y[:,3] if include_fourth else None,'alternating')

def online(det,artifact,method,records,pair,decs,xx,sensors,speed):
    torch.cuda.synchronize()
    start=time.perf_counter_ns()
    actions=choose(det,records,pair,xx,speed)
    p=np.stack([v33.decode_selected(dec,x,a,y) for dec,x,a,y in zip(decs,xx,actions,sensors)])
    result=estimate(artifact,records,method=='gate')
    weight=result['accept4'] if method=='gate' else CFG['shrink']
    p=apply(p,(result['z']*weight)[:,None],BETA)
    torch.cuda.synchronize()
    return actions,p,(time.perf_counter_ns()-start)*1.e-6

def timing(det,artifacts,by_speed,pair,decs,xx,obs,raw,seed,timed):
    ms=np.empty((3 if timed else 0,3,2)); order=np.empty((len(ms),6),dtype=int)
    sensors=[{a:o[a][:,0] for a in (Q,P)} for o in obs]
    rng=np.random.default_rng(seed)
    if timed:
        for rep in (-1,0,1,2):
            perm=rng.permutation(6)
            if rep>=0: order[rep]=perm
            for item in perm:
                j,m=divmod(item,2)
                speed=SPEEDS[j];record=by_speed[j]
                a,p,dt=online(det,artifacts[speed],CFG['timed_methods'][m],record,pair,decs,xx,sensors,speed)
                np.testing.assert_array_equal(a,raw[j]['actions'])
                np.testing.assert_allclose(p,raw[j]['timed_pred'][m][:,:,0],atol=1e-8,rtol=0.)
                if rep>=0: ms[rep,j,m]=dt
    return ms,order

def block(det,artifacts,fp,cell_index,block_index):
    cell=CFG['cells'][cell_index]
    obj,sur=cell.split('_')
    base=CFG['base_seed']+CFG['cell_stride']*cell_index+CFG['block_stride']*block_index
    dst=OUT/cell/f'block_{block_index:03d}'
    dst.mkdir(parents=True,exist_ok=True)
    if (dst/'meta.json').exists():
        old=json.loads((dst/'meta.json').read_text())
        assert old['fingerprint']==fp and all(digest(dst/p)==v for p,v in old['sha256'].items())
        return
    alpha=np.array(json.loads((ROOT.parent/'bidirectional_probe_v34/policy_design34.json').read_text())['alphas'])
    alpha=alpha[np.random.default_rng(base+401).permutation(8)]
    start=(-1,1)[block_index%2]
    pair=[base+301,base+302]; decs=[Engine(pair,*al) for al in alpha]
    action_cache={}; potential_cache={}; files={}
    for sign in (1,-1):
        for reg in CFG['regimes']:
            if reg=='global_null' and sign<0: continue
            law=(obj,sur,sign)
            cal=[world.calibration(base+100+i,al,law,reg,start,SPEEDS,CFG['height']) for i,al in enumerate(alpha)]
            rec,theta4,clean,stick=zip(*cal)
            by_speed=list(zip(*rec))
            xx,labels,theta,obs=zip(*[world.deployment(base+200+i,al,law,reg) for i,al in enumerate(alpha)])
            xx=np.stack(xx); labels=np.stack(labels)
            key=(sign,reg=='global_null')
            if key not in potential_cache:
                potential_cache[key]=(potentials(decs,xx,obs),xx.copy(),[o[P].copy() for o in obs],[o[Q].copy() for o in obs])
            pot,xx_ref,pr,qr=potential_cache[key]
            np.testing.assert_array_equal(xx_ref,xx)
            for i,o in enumerate(obs):
                np.testing.assert_array_equal(o[P],pr[i]);np.testing.assert_array_equal(o[Q],qr[i])
            raw=[]
            for j,speed in enumerate(SPEEDS):
                records=by_speed[j]
                actions=choose(det,records,pair,xx,speed,action_cache)
                assert ((actions==P).sum(1)==13).all()
                prob=select_prob(pot,actions)
                g=estimate(artifacts[speed],records,True)
                z=g['z']
                ps=[prob,apply(prob,z[:,None,None],BETA),
                    apply(prob,(CFG['shrink']*z)[:,None,None],BETA),
                    apply(prob,(g['accept4']*z)[:,None,None],BETA)]
                risk=np.stack([((p-labels[:,:,None])**2).mean((1,2)) for p in ps])
                raw.append(dict(p_base=prob,risk=risk,actions=actions,timed_pred=ps[2:],**g))
            for other in raw[1:]:
                for k in ('z','score3','near_strength3','accept3'):np.testing.assert_array_equal(other[k],raw[0][k])
            name=reg+('_plus' if sign>0 else '_minus')
            timed=block_index<2 and name==CFG[f'timed_cell_block{block_index}'][cell_index]
            ms,order=timing(det,artifacts,by_speed,pair,decs,xx,obs,raw,base+601+int(sign<0),timed)
            far_x,y,dirs=observations(by_speed[0])
            path=dst/(name+'.npz')
            stacked={k:np.stack([s[k] for s in raw]) for k in ('p_base','risk','actions','accept4','score4','near_strength4')}
            atomic_npz(path,**stacked,**{k:raw[0][k] for k in ('z','accept3','score3','near_strength3')},
                       matched_clean=np.swapaxes(clean,0,1),matched_stick=np.swapaxes(stick,0,1),
                       fourth_y=np.stack([observations(records)[1][:,3] for records in by_speed]),
                       far_x=far_x,first3_y=y[:,:3],dirs=dirs[:,:3],ids=[r[1] for r in by_speed[0]],
                       theta4=theta4,theta=theta,labels=labels,alpha=alpha,methods=CFG['methods'],beta=BETA,
                       speeds=SPEEDS,full_ms=ms,timing_order=order)
            files[path.name]=digest(path)
    atomic_json(dst/'meta.json',dict(cell=cell,block=block_index,start=start,base=base,pair=pair,
       calibration_seeds=[base+100+i for i in range(8)],deployment_seeds=[base+200+i for i in range(8)],fingerprint=fp,sha256=files))
    print('done',cell,block_index,flush=True)

def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    old=json.loads((ROOT.parent/'query_matched_probe_v36/results/run.json').read_text())['hashes']
    for p,h in old.items():assert digest(ROOT.parent/p)==h
    names=('run37.py','run37.sh','design37.json','target37.py','detector37.py','gate37.py','gate37_frozen.npz')
    hashes={**old,**{str((ROOT/p).relative_to(ROOT.parent)):digest(ROOT/p) for p in names}}
    meta=dict(hashes=hashes,job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda)
    fp='|'.join(hashes.values())
    OUT.mkdir(exist_ok=True)
    atomic_json(OUT/'run.json',meta)
    with torch.inference_mode():
        det=SpeedDetector(SPEEDS,CFG['height']); artifacts=load_artifacts()
        for b in range(8):
            for c in range(4):block(det,artifacts,fp,c,b)
    atomic_json(OUT/'completion.json',dict(**meta,clusters=32,archives=224))
if __name__=='__main__':main()
