"""Make fresh v63 deployment worlds, counterfactual labels, and common GPU forecasts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'mismatch_grouping_v57'))
import generate57 as previous

v55 = previous.v55
OFFSET, N = 8300, 24
REPLAY_SEEDS = [63041729, 63084591]

def destination(group, key):
    assert OFFSET <= key < OFFSET + N
    return ROOT / group / f'panel_{key-OFFSET:02d}.npz'

v55.dest = destination

def population(i):
    return v55.pop55(OFFSET+i)

def raw(i):
    return v55.raw55(OFFSET+i)

def observe(i):
    return v55.measured55(population(i), OFFSET+i)

def forecast():
    previous.install()
    torch = previous.r.torch
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    with torch.inference_mode():
        engine, templates = previous.policy50.engine50()
        beta = previous.policy50.coefficients()[0]
        for i in range(N):
            plus, minus, actions = v55.forward55(observe(i),engine,templates,beta)
            assert np.isfinite([plus,minus]).all()
            assert min(plus.min(),minus.min())>=0 and max(plus.max(),minus.max())<=1
            v55.save('forecasts',OFFSET+i,pplus=plus,pminus=minus,actions=actions)
            print('forecast',i,flush=True)

def panel(i):
    bundle = {}
    for group in ('raw','forecasts','solver','solver_ultra'):
        part=v55.load(group,OFFSET+i)
        assert not bundle.keys() & part.keys()
        bundle.update(part)
    v55.save('panels',OFFSET+i,**bundle)
    import score55 as scorer
    scorer.ROOT=ROOT
    scorer.REPLAY_SEEDS=REPLAY_SEEDS
    scorer.panel_score(i,ROOT/'scored63')

def preaudit():
    keys=('theta','cal_theta','old_cal_theta')
    prior=set()
    archives={}
    for where in sorted(ROOT.parent.glob('*/worlds')) + [ROOT.parent/'switch_diode_v61'/'excluded_5100'/'worlds']:
        if where.parent==ROOT:
            continue
        matched=0
        for path in sorted(where.glob('panel_*.npz')):
            with np.load(path) as d:
                for key in keys:
                    if key in d and d[key].shape[-1]==6:
                        prior.update(map(bytes,np.ascontiguousarray(d[key]).reshape(-1,6)))
                        matched+=1
        if matched: archives[str(where.relative_to(ROOT.parent))]=matched
    assert 'exposure_abstention_v62/worlds' in archives and 'switch_diode_v61/worlds' in archives and 'switch_diode_v61/excluded_5100/worlds' in archives
    fresh=set()
    for i in range(N):
        pop=population(i)
        for key in keys:
            unit=set(map(bytes,np.ascontiguousarray(pop[key]).reshape(-1,6)))
            assert len(unit)==512 and not unit.intersection(prior) and not unit.intersection(fresh)
            fresh.update(unit)
    verified=dict(catalog_keys=[OFFSET,OFFSET+N-1],previous_archives=archives,
                  verified_previous_draws=len(prior),unique_fresh_draws=len(fresh))
    (ROOT/'support63.json').write_text(json.dumps(verified,indent=2)+'\n')
    return verified

def audit():
    verified=preaudit()
    active=changed=diff=0
    manifest={}
    for i in range(N):
        pp=v55.load('panels',OFFSET+i)
        act=v55.design(OFFSET+i)['state']!=0
        flip=pp['y_keep']!=pp['y_flip']
        assert not (flip & ~act).any()
        active+=int(act.sum());changed+=int(flip.sum())
        diff+=sum(int((pp[key]!=pp[key+'_ultra']).sum()) for key in ('y_keep','y_flip'))
        for group in ('worlds','raw','solver','solver_ultra','forecasts','panels','scored63'):
            path=destination(group,OFFSET+i)
            manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'world63.json').write_text(json.dumps(dict(catalog_keys=[OFFSET,OFFSET+N-1],catalogs=N,
         surfaces=2*N,laws=2,objects=512*N,unique_fresh_draws=verified["unique_fresh_draws"],verified_previous_draws=verified["verified_previous_draws"],
         replay_seeds=REPLAY_SEEDS,active=active,changed_labels=changed,refinement_different=diff,
         solver_endpoints=N*1024),indent=2)+'\n')
    (ROOT/'deployment63.sha256').write_text(''.join(f'{h}  {p}\n' for p,h in manifest.items()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','solver','panel','forecast','audit','preaudit'))
    ap.add_argument('ids',nargs='*',type=int);args=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    if args.stage=='forecast': forecast()
    else:
        assert not previous.r.torch.cuda.is_available()
        if args.stage=='audit':audit()
        if args.stage=='preaudit':preaudit()
        for i in args.ids or range(N):
            if args.stage=='prepare': v55.prepare(OFFSET+i)
            if args.stage=='solver':
                v55.labels(OFFSET+i,False);v55.labels(OFFSET+i,True)
            if args.stage=='panel':panel(i)
