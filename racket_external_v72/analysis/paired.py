"""Paired descriptive contrasts over fixed product/shot sets and a loss-weight sensitivity."""
import csv
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
old = root.parent / 'racket_transfer_v70'
run = root / 'method' / 'retrospective'
radius = .02
folds = json.loads((old/'solver'/'metrics.json').read_text())['folds']
draws = np.load(run/'draw_indices.npz')
with (run/'draws.csv').open() as f:
    raw = list(csv.DictReader(f))

def metrics(product,method,k):
    sub = [r for r in raw if r['rubber']==product and r['method']==method and int(r['k'])==k]
    sub.sort(key=lambda r:int(r['draw']))
    assert [int(r['draw']) for r in sub]==list(range(256))
    return np.array([[float(r[c]) for c in ('velocity','spin','output_sign_acc')] for r in sub])

def two_routes(spec,split):
    files = [old/'data'/'upstream'/split/f'racket_{i:02}.csv' for i in sorted(spec['held'])]
    a = np.concatenate([np.genfromtxt(p,delimiter=',',names=True) for p in files])
    def state(key):
        v,w=('in','before') if key=='in' else ('out','after')
        return np.stack([a[f'v_{v}_x'],a[f'v_{v}_z'],radius*a[f'spin_{w}_y']],axis=-1)
    x,y=state('in'),state('out')
    fit=spec['empirical']
    z=np.column_stack([np.ones(len(x)),(x-fit['mean'])/fit['std']])
    gate=z@fit['logistic']>0
    plus,minus=z@fit['expert_plus'],z@fit['expert_minus']
    def error(flip):
        pred=np.where((gate^flip)[:,None],plus,minus)
        residual=pred-y
        return np.stack([np.linalg.norm(residual[:,:2],axis=1),abs(residual[:,2])/radius,
                ((pred[:,0]-pred[:,2]>0)==(y[:,0]-y[:,2]>0)).astype(float)],axis=1)
    return error(False),error(True)

def main():
    methods=['majority','continuous_loss_gate','single_affine_eb','elastic_impulse']
    out={'fold_order':list(folds),'deltas':{},'weights':{},'max_replay_error':[0.,0.,0.]}
    for k in (4,8):
        vals={m:np.stack([metrics(g,m,k) for g in folds]) for m in methods}
        for a,b in [(methods[1],methods[0]),(methods[3],methods[2]),(methods[1],methods[3])]:
            delta=vals[a]-vals[b]
            grp=delta.mean(1)
            macro_draw=delta.mean(0)
            loo=np.stack([np.delete(grp,i,axis=0).mean(0) for i in range(len(folds))])
            out['deltas'].setdefault(f'{a} minus {b}',{})[k]=dict(
                mean=grp.mean(0).tolist(),by_product=grp.tolist(),
                paired_draw_p10_p90=np.quantile(macro_draw,[.1,.9],axis=0).tolist(),
                leave_one_out_range=[loo.min(0).tolist(),loo.max(0).tolist()])
    for g,spec in folds.items():
        a,b=two_routes(spec,'train')
        test0,test1=(v.mean(0) for v in two_routes(spec,'test'))
        for k in (4,8):
            idx=draws[g][:,:k]
            for name,vel,spin in [('velocity_only',1,0),('quarter_R',1,.25*radius),
                                   ('R',1,radius),('four_R',1,4*radius),('spin_only',0,1)]:
                diff=b[:,:2]-a[:,:2]
                decisions=(vel*diff[:,0]+spin*diff[:,1])[idx].sum(1)<0
                m=np.where(decisions[:,None],test1,test0)
                out['weights'].setdefault(g,{}).setdefault(k,{})[name]=dict(flip=float(decisions.mean()),metrics=m.mean(0).tolist())
                if name=='R':
                    err=np.max(np.abs(m-metrics(g,'continuous_loss_gate',k)),axis=0)
                    assert np.max(err)<1e-8
                    out['max_replay_error']=np.maximum(err,out['max_replay_error']).tolist()
    (root/'analysis'/'paired.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    main()
