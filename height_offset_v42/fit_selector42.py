"""Fit on v41 nominal-height archives, without generating or opening a v42 world."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from selector42 import ROOT, FIXED, features, moments, METHODS

V41=ROOT.parent/'estimated_projection_v41'
DRAWS=(0,16)
RIDGES=(.01,.1)
THRESHOLDS=(0.,.00025)
FEATURES={'occupancy':list(range(14)),'contact':list(range(21))}


def collect():
    chunks=[]
    for p in sorted((V41/'results').glob('*/catalog_*/block_*.npz')):
        family,height,variant,law=p.parent.parent.name.split('_')
        panel=int(p.parent.name.split('_')[-1])
        with np.load(p) as a:
            base=np.take(a['p_base'][[0,2]],DRAWS,axis=-1)
            vals={key:a[key] for key in ('z','weights','actions','first3_y','fourth_y','dirs','labels')}
        for s in range(2):
            for reg in range(3):
                z=vals['z'][s];w=vals['weights'][s,reg];y=vals['labels'][s,reg]
                for dec in range(2):
                    pb=base[dec,s]
                    x=features(pb,z,w,vals['actions'][s],vals['first3_y'][s],
                               vals['fourth_y'][s,reg],vals['dirs'][s])
                    parts=[x]
                    for sc in (FIXED,w):
                        step,sq,r=moments(pb,z,sc,y)
                        parts.extend(v[...,None] for v in (step,sq,r))
                    group=[panel, int(height=='h31'),int(family=='roll'),int(law=='crossover'),dec]
                    parts.append(np.broadcast_to(group,(*x.shape[:2],len(group))))
                    chunks.append(np.concatenate(parts,-1).reshape(-1,32))
    rows=np.concatenate(chunks)
    assert rows.shape==(576*2*3*2*8*len(DRAWS),32)
    return rows


def fit(rows,which,penalty):
    x=rows[:,which]
    center=x.mean(0)
    scale=np.maximum(x.std(0),.01)
    xx=np.column_stack([np.ones(len(x)),np.clip((x-center)/scale,-5,5)])
    weights=[];bias=[]
    for a in range(2):
        i=21+3*a
        absstep=np.abs(rows[:,i])
        wt=absstep/absstep.sum()
        ridge=np.diag([0.]+[penalty]*len(which))
        coef=np.linalg.solve(xx.T@(wt[:,None]*xx)+ridge,xx.T@(wt*rows[:,i+2]))
        weights.append(coef[1:]);bias.append(coef[0])
    return dict(mode='learned',indices=which,center=center.tolist(),scale=scale.tolist(),
                intercept=bias,coef=np.stack(weights,axis=-1).tolist())


def model_scores(model,rows):
    x=np.clip((rows[:,model['indices']]-model['center'])/model['scale'],-5,5)
    residual=x@model['coef']+model['intercept']
    return np.column_stack([rows[:,22]-2*rows[:,21]*residual[:,0],
                            rows[:,25]-2*rows[:,24]*residual[:,1]])


def evaluate(scores,threshold,actual):
    scores=np.column_stack([np.zeros(len(actual)),scores+threshold])
    chosen=np.argmin(scores,axis=1)
    loss=np.take_along_axis(actual,chosen[:,None],axis=1)[:,0]
    return {'delta_vs_zero':float(loss.mean()),'fraction':(np.bincount(chosen,minlength=3)/len(chosen)).tolist()}


def train(rows):
    actual=np.column_stack([np.zeros(len(rows)),rows[:,22]-2*rows[:,21]*rows[:,23],
                            rows[:,25]-2*rows[:,24]*rows[:,26]])
    panel,heights,families,laws,dec=rows[:,-5:].astype(int).T
    # Repeated variants, noise draws, decoders and starts stay in the held panel.
    splits={'panel':None,'height':heights,'shape':families,'law':laws}
    records=[dict(set='zero',penalty=0.,threshold=0.,objective=0.,folds={})]
    for name,which in FEATURES.items():
        for penalty in RIDGES:
            cv={cut:{} for cut in THRESHOLDS}
            for axis,groups in splits.items():
                strata=[None] if groups is None else [0,1]
                pred=np.empty((len(rows),2))
                for parity in (0,1):
                    for g in strata:
                        tr=panel%2!=parity; ev=~tr
                        if g is not None: tr=tr&(groups!=g);ev=ev&(groups==g)
                        m=fit(rows[tr],which,penalty)
                        pred[ev]=model_scores(m,rows[ev])
                for cut in THRESHOLDS:
                    cv[cut][axis]={}
                    for d,tag in enumerate(('original','estimated')):
                        mask=dec==d
                        cv[cut][axis][tag]=evaluate(pred[mask],cut,actual[mask])
            for cut in THRESHOLDS:
                quality=np.mean([cv[cut][a]['estimated']['delta_vs_zero'] for a in ('panel','height','shape')])
                records.append(dict(set=name,penalty=penalty,threshold=cut,objective=float(quality),folds=cv[cut]))
    conservative=min(records,key=lambda r:r['objective'])
    winner=min(records[1:],key=lambda r:r['objective'])
    model=fit(rows,FEATURES[winner['set']],winner['penalty'])
    model.update(threshold=winner['threshold'],source='v41 historical original and estimated decoders only',
                 selected=winner['set'],penalty=winner['penalty'],actions=METHODS)
    archive_info=V41/'results'/'completion.json'
    model['source_sha256']=hashlib.sha256(archive_info.read_bytes()).hexdigest()
    (ROOT/'selector_model42.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    info=dict(n_rows=len(rows),noise_indices=DRAWS,candidates=records,winner=winner,including_constant=conservative,source_hash=model['source_sha256'],
              static_delta=actual[dec==1].mean(0).tolist())
    (ROOT/'selector_source42.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    print(json.dumps(winner),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,required=True)
    args=ap.parse_args()
    if not args.cache.exists(): np.save(args.cache,collect())
    train(np.load(args.cache))
