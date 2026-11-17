"""Post-freeze arithmetic audit of all primary predictions, timings and matched comparisons."""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import selector41

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
sys.path.insert(0, str(ROOT.parent / 'probe_transfer_v39'))
from audit39 import load_artifacts, source_reconstruction

TOL = 2.e-10


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(stats, name, a, b, tol=TOL):
    a,b = np.asarray(a),np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError((name,a.shape,b.shape))
    error = float(np.max(np.abs(a.astype(float)-b.astype(float))))
    if error > tol:
        raise AssertionError((name,error,tol))
    stats[name] = max(error,stats.get(name,0.))


def audit_pack(path, run, model, source, cache, stats):
    m = json.loads(path.with_suffix('.json').read_text())
    assert digest(path) == m['sha256'] and m['fingerprint'] == run['fingerprint']
    with np.load(path,allow_pickle=False) as f:
        for name in ('full_ms','setup_ms','cold_ms','fit_ms'):
            assert np.isfinite(f[name]).all() and (f[name]>=0.).all()
        if len(f['full_ms']):
            assert f['full_ms'].shape == (3,3,3)
            assert (f['full_ms']>0).all() and (f['setup_ms']>0).all()
            for order in f['trial_order']:
                assert np.array_equal(np.sort(order),np.arange(9))
            check(stats,'cold_clock_identity_ms',f['cold_ms'],f['full_ms']+f['setup_ms'],1.e-8)
        rr = int(len(f['full_ms'])>0)
        assert np.array_equal(f['methods'],['fixed','optimizer','selector'])
        assert np.array_equal(f['decoders'],['original','oracle','estimated'])
        a = {k:f[k] for k in ('p_base','z','weights','actions','cal_x','first3_y','dirs','fourth_y',
                               'labels','risk','chosen_optimizer')}
    p = a['p_base']
    assert p.shape == (3,2,8,128,32) and p.min() >= -1.e-12 and p.max()<=1+1.e-12
    assert a['labels'].dtype==np.bool_ and a['risk'].shape==(2,3,3,3,8)
    for s,reg in itertools.product(range(2),range(3)):
        prefix = [a['cal_x'][s],a['first3_y'][s],a['dirs'][s],a['fourth_y'][s,reg]]
        key=hashlib.sha256(b''.join(np.ascontiguousarray(v).tobytes() for v in prefix)).digest()
        if key not in cache:
            cache[key] = source_reconstruction(source,*prefix)[:2]
        z,w = cache[key]
        check(stats,'frozen_signed_cue',z,a['z'][s])
        check(stats,'frozen_optimizer_weight',w,a['weights'][s,reg])
        observed=a['labels'][s,reg,:,:,None]
        for d in range(3):
            base=p[d,s]
            chosen = selector41.select(model,base,a['z'][s],a['weights'][s,reg],a['actions'][s],
                                       prefix[1],prefix[3],prefix[2])
            assert np.array_equal(chosen,a['chosen_optimizer'][s,reg,d])
            fixed=np.full((8,1,1),.325)
            optimized=w[:,None,None]
            scales=(fixed,optimized,np.where(chosen[:,None,:],optimized,fixed))
            for policy,scale in enumerate(scales):
                odds=np.exp(.8289760825845313*z[:,None,None]*scale)
                prob=base*odds/(1+base*(odds-1))
                loss=np.mean((prob-observed)**2,axis=(1,2))
                check(stats,'recomputed_Brier',loss,a['risk'][s,reg,d,policy],3.e-12)
    action=a['actions']
    assert np.all((action==1).sum(-1)==115) and np.all((action==3).sum(-1)==13)
    for d in (1,2):
        check(stats,'probe_projection_invariance',p[0][action==3],p[d][action==3],3.e-12)
    return a,rr


def main():
    run=json.loads((OUT/'run.json').read_text())
    for path,expected in run['hashes'].items():
        assert digest(ROOT.parent/path)==expected,path
    stats,cache={},{}
    timing=0
    done=0
    model=selector41.load_selector()
    source=load_artifacts()[1.1]
    pending=list(itertools.product(range(12),range(2),('flare','roll'), ('h19','h31'),('saturating','crossover')))
    while pending:
        progressed=False
        for group in pending.copy():
            panel,block,fam,height,law=group
            paths=[OUT/f'{fam}_{height}_{v}_{law}'/f'catalog_{panel:02d}'/f'block_{block:02d}.npz'
                   for v in ('base','hollow','bulge')]
            if not all(path.with_suffix('.json').exists() for path in paths):
                continue
            ref=None
            for path in paths:
                a,t=audit_pack(path,run,model,source,cache,stats)
                if ref is None:
                    ref=a
                else:
                    for key in ('actions','z','weights','labels','cal_x','first3_y','fourth_y','dirs'):
                        check(stats,'same_contour_'+key,a[key],ref[key],1.e-12)
                    check(stats,'oracle_contour_invariance',a['p_base'][1],ref['p_base'][1],TOL)
                timing+=t
                done+=1
            pending.remove(group)
            progressed=True
        if pending and not progressed:
            time.sleep(8)
    while not (OUT/'completion.json').exists():
        time.sleep(1)
    assert json.loads((OUT/'completion.json').read_text())['fingerprint']==run['fingerprint']
    assert timing==12 and done==576
    result=dict(blocks=done,independent_timing_clusters=timing,unique_paid_prefix_condition_keys=len(cache),
                reconstructed_expected_losses=done*2*3*3*3*8,max_absolute_error=stats,
                run_fingerprint=run['fingerprint'])
    (ROOT/'audit41.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
