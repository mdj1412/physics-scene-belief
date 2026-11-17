"""Pair methods by actual target shot. Only an explicitly privileged audit sees exit sign."""
import csv
import json
import itertools
from math import comb
from pathlib import Path

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R=.02

def stats(x):
    return np.array([np.mean(x[:,0]),np.mean(x[:,1])])

def inspect():
    sol=ROOT/'solver'
    with (sol/'predictions.csv').open() as f:
        prim=[r for r in csv.DictReader(f) if r['split']=='test']
    with (ROOT/'analysis'/'rbf_predictions.csv').open() as f:
        rb=list(csv.DictReader(f))
    def table(rows):
        d={}
        for r in rows:
            i=(int(r['config']),int(r['row']))
            v=(float(r['error_v']),float(r['error_w']))
            assert r['method'] not in d.setdefault(i,{})
            d[i][r['method']]=v
        return d
    pred=table(prim)
    extra=table(rb)
    assert set(pred)==set(extra)
    for i in pred: pred[i].update(extra[i])
    within={(int(r['config']),int(r['row'])):bool(int(r['within_support'])) for r in rb}
    raw={}
    for i in range(1,11):
        a=np.genfromtxt(ROOT/'data'/'upstream'/'test'/f'racket_{i:02}.csv',delimiter=',',names=True)
        for j,r in enumerate(a):raw[i,j]=r
    fit=json.loads((sol/'metrics.json').read_text())['folds']
    methods=list(next(iter(pred.values())))
    foldstat={}; gate={}; strata={}; config_results={}
    for g,par in fit.items():
        idx=sorted(i for i in pred if i[0] in par['held'])
        x=np.array([[raw[i]['v_in_x'],raw[i]['v_in_z'],raw[i]['spin_before_y']*R] for i in idx])
        y=np.array([[raw[i]['v_out_x'],raw[i]['v_out_z'],raw[i]['spin_after_y']*R] for i in idx])
        p=par['empirical']
        a=np.column_stack([np.ones(len(idx)),(x-p['mean'])/p['std']])
        est=(a@p['logistic'])>0
        measured=(y[:,0]-y[:,2])>0
        minus=a@p['expert_minus'];plus=a@p['expert_plus']
        def err(select):
            delta=np.where(select[:,None],plus,minus)-y
            return np.column_stack([np.linalg.norm(delta[:,:2],axis=1),abs(delta[:,2])/R])
        observed=np.array([pred[i]['two_regime'] for i in idx])
        np.testing.assert_allclose(err(est),observed,atol=1e-8)
        hold=np.array([within[i] for i in idx])
        privileged=err(measured)
        gate[g]=dict(n=len(idx),acc=float(np.mean(est==measured)),observed_positive=float(measured.mean()),
                     predicted_positive=float(est.mean()),oracle=stats(privileged).tolist(),
                     real=stats(observed).tolist())
        foldstat[g]={}
        strata[g]={'n':len(idx),'n_within':int(sum(hold))}
        for m in methods:
            v=np.array([pred[i][m] for i in idx])
            foldstat[g][m]=stats(v).tolist()
            strata[g][m]={lbl:stats(v[mask]).tolist() if mask.any() else None for lbl,mask in [('near',hold),('far',~hold)]}
    for cfg in range(1,11):
        idx=sorted(i for i in pred if i[0]==cfg)
        config_results[cfg]={m:stats(np.array([pred[i][m] for i in idx])).tolist() for m in methods}
    macro={m:np.mean([r[m] for r in foldstat.values()],axis=0).tolist() for m in methods}
    micro={m:stats(np.array([d[m] for d in pred.values()])).tolist() for m in methods}
    contrast={}
    for ma,mb in [('two_regime','inclined'),('two_regime','single_affine'),('pooled_rbf','two_regime'),
                  ('known_rubber_rbf','pooled_rbf'),('known_rubber_rbf','matched_count_source_rbf')]:
        d=np.array([np.array(v[ma])-v[mb] for v in foldstat.values()])
        leave=np.array([np.delete(d,j,axis=0).mean(0) for j in range(len(d))])
        wins=(d<0).sum(0)
        contrast[f'{ma} minus {mb}']={'macro':d.mean(0).tolist(),'range_omit_one':[leave.min(0).tolist(),leave.max(0).tolist()],
            'products_better':wins.tolist(),'sign_p':[min(1.,2*sum(comb(6,n) for n in range(min(int(w),6-int(w))+1))/64) for w in wins]}
    return dict(fold=foldstat,macro=macro,micro=micro,contrasts=contrast,support=strata,regime=gate,config=config_results)
if __name__=='__main__':
    (ROOT/'analysis'/'audit.json').write_text(json.dumps(inspect(),indent=2)+'\n')
