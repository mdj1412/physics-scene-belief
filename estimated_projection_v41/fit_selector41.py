"""Fit using v39/v40 only. One measured observation per deployment object per forecast."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from selector41 import design,ROOT

PROJECT=ROOT.parent
V39=PROJECT/'probe_transfer_v39'
V40=PROJECT/'tangent_transfer_v40'
NOISE=[0,8,16,24]
RIDGE=(.01,.1,1.)
SETS={'occupancy':[0,1,2,3,4,5,6,7,8,9,15,16,17,18,21,22],
      'contact_and_occupancy':list(range(23))}
FAMILIES=['shelf','ridge']
LAWS=['weakening','strengthening','saturating','crossover']


def block_rows(arc,pb,which,family,law,panel):
    chunks=[]
    for s in range(2):
        for r in range(3):
            x,a,h,mid=design(pb[s],arc['z'][s],arc['weights'][s,r],arc['actions'][s],
                          arc['first3_y'][s],arc['fourth_y'][s,r],arc['dirs'][s])
            y=arc['labels'][s,r,:,:,None]
            resp=(h*(y-mid)).sum(1)/np.maximum(h.sum(1),1e-15)
            actual=-2*a*resp
            rest=np.broadcast_to(np.array([which,family,law,panel]),x.shape[:2]+(4,))
            chunks.append(np.concatenate((x,resp[:,:,None],a[:,:,None],actual[:,:,None],rest),axis=2).reshape(-1,x.shape[-1]+7))
    return np.concatenate(chunks)


def archived_rows():
    rows=[]
    for root,which in [(V39,0),(V40,1)]:
        for path in sorted((root/'results').glob('*/catalog_*/block_*.npz')):
            name=path.parent.parent.name
            family=FAMILIES.index(name.split('_')[0]);law=LAWS.index(name.split('_')[-1])
            panel=int(path.parent.name[8:])
            with np.load(path) as a:
                arc={k:a[k] for k in ('z','weights','actions','first3_y','fourth_y','dirs','labels')}
                pb=a['p_base'][:,:,:,NOISE]
            rows.append(block_rows(arc,pb,which,family,law,panel))
    return np.concatenate(rows)


def projected_rows():
    # Only historical laws, catalogs, truth, measurement heights and K256 archives are replayed.
    os.environ['BELIEF_DEVICE']='cpu'
    sys.path.insert(0,str(V40))
    import projection40 as control
    v38=control.v38;world=control.world
    v38.torch.set_num_threads(1)
    d=world.perturbation(world.VYS[0],'shelf','bulge')[1]-world.perturbation(world.VYS[0],'shelf','base')[1]
    rows=[]
    for panel in range(12):
        b=panel%2
        for f,family in enumerate(FAMILIES):
            for law in LAWS[2:]:
                path=V40/'results'/f'{family}_base_{law}'/f'catalog_{panel:02d}'/f'block_{b:02d}.npz'
                meta=json.loads(path.with_suffix('.json').read_text())
                with np.load(path) as a:
                    arc={k:a[k] for k in ('z','weights','actions','first3_y','fourth_y','dirs','labels','alpha')}
                    pb=a['p_base'][:,:,:,NOISE].copy()
                for e,al in enumerate(arc['alpha']):
                    inner=v38.Engine(meta['pair'],*al).inner
                    for s,sign in enumerate((1,-1)):
                        ctx=(family,'base',law,sign,panel)
                        factual,y,_,obs=world.deployment(meta['deployment_seeds'][e],al,ctx,'strong',32)
                        np.testing.assert_array_equal(y,arc['labels'][s,0,e])
                        ix=np.flatnonzero(arc['actions'][s,e]==v38.Q)
                        for chunk in np.array_split(ix,8):
                            pb[s,e,chunk]=control.decode(inner,factual[chunk],obs[v38.Q][chunk][:,NOISE],d)
                rows.append(block_rows(arc,pb,2,f,LAWS.index(law),panel))
        print('source projection',panel,flush=True)
    return np.concatenate(rows)


def fit(rows,feat,penalty):
    # Correct unequal multiplicities: v39 has 1 contour, v40 raw has 3, projected
    # is invariant across those 3 and uses 1 of 2 blocks. The v39 domain gets equal
    # weight per historical law to the combined raw/projected v40 law.
    source=rows[:,-4].astype(int)
    density=np.choose(source,[3.,.5,3.])
    weights=density*np.abs(rows[:,-6])
    weights=weights/weights.sum()
    x=rows[:,feat]
    center=np.sum(x*weights[:,None],axis=0)
    scale=np.maximum(np.sqrt(np.sum((x-center)**2*weights[:,None],axis=0)),.01)
    x=np.clip((x-center)/scale,-5,5)
    x=np.column_stack((np.ones(len(x)),x))
    y=rows[:,-7]
    gram=x.T@(weights[:,None]*x)
    penalty_diagonal=np.diag([0.]+[penalty]*(x.shape[1]-1))
    coef=np.linalg.solve(gram+penalty_diagonal,x.T@(weights*y))
    return dict(feature_index=feat,center=center.tolist(),scale=scale.tolist(),coef=coef[1:].tolist(),intercept=float(coef[0]))


def decisions(rows,m):
    x=np.clip((rows[:,m['feature_index']]-m['center'])/m['scale'],-5,5)
    return (-2*rows[:,-6]*(x@np.array(m['coef'])+m['intercept']))<0


def fit_all(rows):
    source,fam,law,panel=rows[:,-4:].astype(int).T
    actual=rows[:,-5]
    # Exclude the destination historical family/law AND the evaluation panel parity.
    # Candidate score equally weights raw and projected historical v40 and two tests.
    records=[]
    for name,feat in SETS.items():
        for penalty in RIDGE:
            cv={}
            for axis,groups in [('family',fam),('law',law)]:
                pred=np.empty(len(rows),bool)
                for group in np.unique(groups[source>0]):
                    for parity in (0,1):
                        tr=(groups!=group)&(panel%2!=parity)
                        ev=(groups==group)&(panel%2==parity)&(source>0)
                        if not np.any(ev):continue
                        model=fit(rows[tr],feat,penalty)
                        pred[ev]=decisions(rows[ev],model)
                for k,s in [('raw',1),('projected',2)]:
                    use=source==s
                    v=(pred[use]*actual[use]).mean()
                    cv[axis+'_'+k]=dict(vs_fixed=v,vs_optimizer=v-actual[use].mean(),fraction=pred[use].mean())
            quality=np.mean([q['vs_fixed'] for q in cv.values()])
            records.append(dict(set=name,penalty=penalty,objective=quality,folds=cv))
    winner=min(records,key=lambda u:u['objective'])
    model=fit(rows,SETS[winner['set']],winner['penalty'])
    model.update(set=winner['set'],penalty=winner['penalty'],tie='fixed',source='v39 raw, v40 raw and computed v40 oracle projection',
                 domain_sha256={v.name:hashlib.sha256((v/'results/completion.json').read_bytes()).hexdigest() for v in (V39,V40)})
    (ROOT/'selector_model41.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    (ROOT/'selector_source41.json').write_text(json.dumps(dict(winner=winner,candidates=records,
            n_rows=len(rows),noise=NOISE,rows_by_domain=np.bincount(source).tolist()),indent=2,allow_nan=False)+'\n')
    print(winner)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['raw','projection','fit'])
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    if args.stage=='fit':
        fit_all(np.concatenate([np.load(args.cache/(k+'.npy')) for k in ['raw','projection']]))
    else:
        np.save(args.cache/(args.stage+'.npy'), archived_rows() if args.stage=='raw' else projected_rows())
