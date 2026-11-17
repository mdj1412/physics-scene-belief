"""Audit archived v72 retrospective metrics against the v71 CUDA run and draw schedules."""
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
V71=HERE.parents[1]/'racket_calibration_v71'
OUT=HERE/'retrospective'

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def main():
    v72=rows(OUT/'draws.csv')
    keyed={(r['rubber'],int(r['draw']),int(r['k']),r['method']):r for r in v72}
    assert len(keyed)==len(v72)==6*256*2*4
    spec=[('core/draw_metrics.csv','gate_only','majority'),('baselines/draw_metrics.csv','single_affine_eb','single_affine_eb'),
          ('analysis/elastic_draws.csv','elastic_response','elastic_impulse')]
    differences={}
    for path,legacy,method in spec:
        delta=[]
        for r in rows(V71/path):
            if r['method']!=legacy or int(r['budget']) not in (4,8):continue
            now=keyed[r['rubber'],int(r['draw']),int(r['budget']),method]
            delta.append([abs(float(r[c])-float(now[c])) for c in ('velocity','spin')])
        assert len(delta)==6*256*2
        mx=np.max(delta,axis=0)
        assert max(mx)<1e-8,(method,mx)
        differences[method]=mx.tolist()
    reference=np.load(V71/'core/draw_indices.npz')
    tape=np.load(OUT/'draw_indices.npz')
    assert set(reference.files)==set(tape.files)
    for g in tape:np.testing.assert_array_equal(reference[g][:,:8],tape[g])
    # Both deployment gates must agree on every held shot whenever paid picks the same routing.
    worst_gate_comparison=0.
    for (g,d,k,m),r in keyed.items():
        if m!='majority':continue
        other=keyed[g,d,k,'continuous_loss_gate']
        if r['flip']==other['flip']:
            for c in ('velocity','spin','output_sign_acc','route_sign_acc'):
                worst_gate_comparison=max(worst_gate_comparison,abs(float(r[c])-float(other[c])))
    assert worst_gate_comparison==0
    raw=json.loads((OUT/'run.json').read_text())
    buckets=defaultdict(list)
    for r in v72:buckets[r['rubber'],int(r['k']),r['method']].append(r)
    out={}
    for (g,k,m),rr in buckets.items():
        assert len(rr)==256
        s={c:stats.mean(float(r[c]) for r in rr) for c in ('velocity','spin','output_sign_acc','flip','route_sign_acc','choose_min_held_loss') if rr[0][c]!=''}
        if 'flip' in s:
            flip=raw['folds'][g]['held_flip_by_frequency']
            s['orientation_acc']=None if flip is None else (s['flip'] if flip else 1-s['flip'])
        out.setdefault(g,{}).setdefault(k,{})[m]=s
    macro={}
    for k in (4,8):
        for m in out[g][k]:
            macro.setdefault(k,{})[m]={a:stats.mean(out[g][k][m][a] for g in out) for a in out[g][k][m]}
    result=dict(v71_reference_abs_max_error=differences,gate_same_choice_max_error=worst_gate_comparison,
                paired_draw_keys=len(keyed),macro=macro,by_product=out)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for g,group in [('macro',macro)]+[(g,out[g]) for g in ['grass_dtecs','abs_ii_soft']]:
        print(g)
        for k,ms in group.items():
            for m,r in ms.items():
                print(k,m,' '.join(f'{r[c]:.4f}' for c in ['velocity','spin','output_sign_acc','flip','route_sign_acc','choose_min_held_loss'] if c in r))
    print('reference errors',differences)

if __name__=='__main__':main()
