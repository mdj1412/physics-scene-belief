"""Fixed-budget comparison of reference and near-query calibration contact."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import target36 as world
from detector36 import Detector
from run35 import (v33, Engine, PosteriorPredictiveEngine, batch_gap, Q, P, atomic_json,
                   atomic_npz, digest, apply, cue_est, LOCK, BETAS, t, torch)
from gate36 import load_artifact, predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
DESIGN=json.loads((ROOT/'design36.json').read_text())
METHODS=DESIGN['methods']
REGIMES=DESIGN['regimes']
BETA=DESIGN['beta']
assert BETA==BETAS[0]

def observations(records):
    return (np.stack([x[idx] for x,idx,y,d in records]),
            np.stack([r[2] for r in records]), np.stack([r[3] for r in records]))

def choices(det,records,pair,xx,near):
    selected=[]
    for rec,x in zip(records,xx):
        belief=det.fit_actions(rec,near)
        acquisition=PosteriorPredictiveEngine(pair,belief)
        selected.append(v33.decisions(batch_gap(acquisition,x)))
    selected=np.stack(selected)
    assert np.all((selected==P).sum(1)==13)
    return selected

def potentials(decs,xx,obs):
    # All 32 paired realizations for both candidate actions. Processing independent
    # world/draw rows together uses exactly the same likelihood as online decoding.
    result=np.empty((8,128,2,32))
    for e,(dec,x,sen) in enumerate(zip(decs,xx,obs)):
        for j in range(0,128,8):
            base=dec.inner.fit(t(x[j:j+8]))[:,None,:]
            for idx,a in enumerate((Q,P)):
                probes=t(sen[a][j:j+8])
                out=dec.inner.update(base,a,probes)@dec.inner.label[:,0]
                result[e,j:j+8,idx]=out.cpu().numpy()/np.sqrt(2.)
    return result

def select_prob(potential,actions):
    return np.take_along_axis(potential,(actions==P)[:,:,None,None].astype(int),axis=2)[:,:,0]

def decision_head(artifact, records, method):
    x,y,d=observations(records)
    if method=='far_v35':
        return cue_est(LOCK,0,x,y,d,'alternating')
    near=y[:,-1] if method=='near_gate4' else None
    g=predict(artifact,x,y[:,:3],d[:,:3],near,'alternating')
    return np.asarray(g['z'])*(np.asarray(g['accept4']) if method=='near_gate4'
                                else np.asarray(g['accept3']) if method=='near_gate3' else 1.)

def online(det,artifact,method,far,near,pair,decs,xx,sensors):
    torch.cuda.synchronize()
    tic=time.perf_counter_ns()
    use_near=method.startswith('near_')
    rec=near if use_near else far
    actions=choices(det,rec,pair,xx,use_near)
    p=np.stack([v33.decode_selected(dec,x,a,yy) for dec,x,a,yy in zip(decs,xx,actions,sensors)])
    if not method.endswith('_original'):
        p=apply(p,decision_head(artifact,rec,method)[:,None],BETA)
    torch.cuda.synchronize()
    return actions,p,(time.perf_counter_ns()-tic)/1.e6

def timing(det,model,far,near,pair,decs,xx,obs,ref,ref_actions,seed,timed):
    repeats=3 if timed else 0
    ms=np.empty((repeats,len(METHODS)))
    orders=np.empty_like(ms,dtype=int)
    sensors=[{a:s[a][:,0] for a in (Q,P)} for s in obs]
    rng=np.random.default_rng(seed)
    if timed:
        for r in (-1,0,1,2):
            order=rng.permutation(len(METHODS))
            if r>=0: orders[r]=order
            for k in order:
                a,p,elapsed=online(det,model,METHODS[k],far,near,pair,decs,xx,sensors)
                np.testing.assert_array_equal(a,ref_actions[int(k>=2)])
                np.testing.assert_allclose(p,ref[k][:,:,0],rtol=0,atol=1e-8)
                if r>=0: ms[r,k]=elapsed
    return ms,orders

def block(det,artifact,fingerprint,cell_id,block_id):
    cell=DESIGN['cells'][cell_id]
    obj,sur=cell.split('_')
    base=DESIGN['target_base_seed']+cell_id*DESIGN['cell_stride']+block_id*DESIGN['block_stride']
    dst=OUT/cell/f'block_{block_id:03d}'
    dst.mkdir(parents=True,exist_ok=True)
    done=dst/'meta.json'
    if done.exists():
        saved=json.loads(done.read_text())
        assert saved['fingerprint']==fingerprint
        assert all(digest(dst/n)==h for n,h in saved['sha256'].items())
        return
    alphas=np.array(json.loads((ROOT.parent/'bidirectional_probe_v34/policy_design34.json').read_text())['alphas'])
    alphas=alphas[np.random.default_rng(base+401).permutation(8)]
    pair=[base+301,base+302]
    start=(-1,1)[block_id%2]
    decs=[Engine(pair,*alpha) for alpha in alphas]
    snapshot={}
    files={}
    for sign in (1,-1):
        for reg in REGIMES:
            if reg=='global_null' and sign<0: continue
            law=(obj,sur,sign)
            name=reg+('_plus' if sign>0 else '_minus')
            cals=[world.calibration(base+100+i,alpha,law,reg,start) for i,alpha in enumerate(alphas)]
            far,near,private=map(list,zip(*cals))
            xx,labels,theta,obs=map(list,zip(*[world.deployment(base+200+i,alpha,law,reg) for i,alpha in enumerate(alphas)]))
            xx=np.stack(xx)
            cache_key=(sign,reg=='global_null')
            far_x,far_y,dirs=observations(far)
            near_x,near_y,near_dirs=observations(near)
            np.testing.assert_array_equal(near_x,far_x)
            np.testing.assert_array_equal(near_y[:,:3],far_y[:,:3])
            if cache_key not in snapshot:
                far_actions=choices(det,far,pair,xx,False)
                pot=potentials(decs,xx,obs)
                snapshot[cache_key]=(far_actions,pot,far_y.copy(),xx.copy(),[o[P].copy() for o in obs],[o[Q].copy() for o in obs])
            far_actions,pot,y_ref,x_ref,p_ref,q_ref=snapshot[cache_key]
            np.testing.assert_array_equal(far_y,y_ref);np.testing.assert_array_equal(xx,x_ref)
            for i,o in enumerate(obs):
                np.testing.assert_array_equal(o[P],p_ref[i]);np.testing.assert_array_equal(o[Q],q_ref[i])
            near_actions=choices(det,near,pair,xx,True)
            p_far=select_prob(pot,far_actions);p_near=select_prob(pot,near_actions)
            z_far=cue_est(LOCK,0,far_x,far_y,dirs,'alternating')
            g=predict(artifact,near_x,near_y[:,:3],dirs[:,:3],near_y[:,3],'alternating')
            z=np.asarray(g['z'])
            weights=[None,z_far,None,z,z*np.asarray(g['accept3']),z*np.asarray(g['accept4'])]
            preds=[(p_far if k<2 else p_near) if cue is None else apply((p_far if k<2 else p_near),cue[:,None,None],BETA)
                   for k,cue in enumerate(weights)]
            labels=np.stack(labels)
            risks=np.stack([np.square(p-labels[:,:,None]).mean((1,2)) for p in preds])
            timed=(block_id<2 and name==DESIGN[f'timed_cell_block{block_id}'][cell_id])
            ms,order=timing(det,artifact,far,near,pair,decs,xx,obs,preds,(far_actions,near_actions),base+601+int(sign<0),timed)
            theta4,clean,stick=map(np.asarray,zip(*private))
            path=dst/(name+'.npz')
            atomic_npz(path,methods=METHODS,risk=risks,p_far=p_far,p_near=p_near,labels=labels,theta=theta,theta4=theta4,
                       matched_clean=clean,matched_stick=stick,far_x=far_x,far_y=far_y,near_y=near_y,dirs=dirs,
                       ids=[r[1] for r in far],far_actions=far_actions,near_actions=near_actions,z_far=z_far,
                       z=z,accept3=g['accept3'],accept4=g['accept4'],score3=g['score3'],score4=g['score4'],
                       near_strength3=g['near_strength3'],near_strength4=g['near_strength4'],
                       full_ms=ms,timing_order=order,alpha=alphas,beta=BETA)
            files[path.name]=digest(path)
    atomic_json(done,dict(cell=cell,block=block_id,start=start,base=base,pair=pair,calibration_seeds=[base+100+i for i in range(8)],
                          deployment_seeds=[base+200+i for i in range(8)],sha256=files,fingerprint=fingerprint))
    print('done',cell,block_id,flush=True)

def main():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(2)
    files=[ROOT/n for n in ('design36.json','detector36.py','target36.py','run36.py','run36.sh','gate36.py','gate36_frozen.npz')]
    old=json.loads((ROOT.parent/'frozen_direction_v35/results/run.json').read_text())['hashes']
    for path,expected in old.items():
        assert digest(ROOT.parent/path)==expected
    hashes={**old,**{str(p.relative_to(ROOT.parent)):digest(p) for p in files}}
    fp='|'.join(hashes.values())
    OUT.mkdir(exist_ok=True)
    meta=dict(hashes=hashes,gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],torch=str(torch.__version__),cuda=torch.version.cuda)
    atomic_json(OUT/'run.json',meta)
    with torch.inference_mode():
        det=Detector();artifact=load_artifact()
        for b in range(DESIGN['blocks_per_cell']):
            for c in range(len(DESIGN['cells'])): block(det,artifact,fp,c,b)
    atomic_json(OUT/'completion.json',dict(**meta,clusters=32,archives=224))

if __name__=='__main__': main()
