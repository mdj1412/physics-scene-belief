"""Audit main GPU detector and gate-only losses from raw CSV and v70 coefficients."""
import csv
import json
import os
import platform
import numpy as np
from finite_reference import BUDGETS, PREV, ROOT, evaluate_row, critical


def run():
    fits = json.loads((PREV/'solver'/'metrics.json').read_text())['folds']
    ref = json.loads((ROOT/'review'/'finite_reference.json').read_text())['rubbers']
    with (ROOT/'core'/'detection.csv').open() as f:
        rows = {(r['rubber'], int(r['draw']), int(r['budget'])): r for r in csv.DictReader(f)}
    with (ROOT/'core'/'draw_metrics.csv').open() as f:
        scores = {(r['rubber'],int(r['draw']),int(r['budget'])):r for r in csv.DictReader(f) if r['method']=='gate_only'}
    checked=0
    max_error=np.zeros(2)
    for name,params in fits.items():
        flags=[]
        for config in sorted(params['held']):
            with (PREV/'data'/'upstream'/'train'/f'racket_{config:02}.csv').open() as f:
                flags.extend(evaluate_row(row,params['empirical'])[0] for row in csv.DictReader(f))
        pool=np.asarray(flags,dtype=int)
        settings=ref[name]
        test_fraction=settings['test_disagreements']/settings['test_n']
        for d in range(256):
            rng=np.random.default_rng(np.random.SeedSequence([710071,min(params['held']),d]))
            selected=rng.choice(len(pool),size=32,replace=False)
            for k in BUDGETS:
                key=(name,d,k)
                row=rows[key]
                mismatch=int(pool[selected[:k]].sum())
                flip=2*mismatch>k
                certified=mismatch>=critical(k,.05)
                assert (mismatch,int(flip),int(certified))==tuple(int(row[q]) for q in ('mismatch','flip','certified')), key
                accuracy=test_fraction if flip else 1.-test_fraction
                assert abs(accuracy-float(row['held_sign_accuracy']))<1.e-10, key
                reference=np.array(settings['loss_flipped' if flip else 'loss_original'])
                measured=np.array([float(scores[key][q]) for q in ('velocity','spin')])
                err=np.abs(reference-measured)
                assert np.max(err)<1.e-8, (key,err)
                max_error=np.maximum(max_error,err)
                checked+=1
    assert checked==len(rows)==len(scores)
    result=dict(checked=checked,max_abs_velocity_error=float(max_error[0]),max_abs_spin_error=float(max_error[1]),
                node=platform.node(),job=os.getenv('SLURM_JOB_ID'),step=os.getenv('SLURM_STEP_ID'))
    (ROOT/'review'/'draw_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':run()
