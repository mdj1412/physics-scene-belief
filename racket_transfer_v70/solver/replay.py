"""CPU replay of sealed CUDA fits on the publication's complete original recordings."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
from collections import defaultdict

import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data'/'upstream'
R=.020
K=2/3


def read(split,c):
    with (DATA/split/f'racket_{c:02}.csv').open(newline='') as f: return list(csv.DictReader(f))


def state(rows,prefix):
    phase='before' if prefix=='in' else 'after'
    return np.array([[float(row[f'v_{prefix}_x']),float(row[f'v_{prefix}_z']),R*float(row[f'spin_{phase}_y'])] for row in rows])


def key(row):
    return tuple(float(row[k]) for k in row if k.startswith(('spin_','v_')))


def rigid(x,d):
    a=np.deg2rad(d['deg']); co,si=np.cos(a),np.sin(a)
    vx,vz,rim=x.T; vt=co*vx+si*vz; vn=-si*vx+co*vz
    e,mu=d['e'],d['mu']
    cap=np.maximum(-vn,0)*mu*(1+e)
    jt=np.clip((rim-vt)/(1+1/K),-cap,cap)
    vn_new=np.where(vn<0,-e*vn,vn)
    return np.column_stack([co*(vt+jt)-si*vn_new,si*(vt+jt)+co*vn_new,rim-jt/K])


def predict(x,ymean,params):
    ep=params['empirical']; mp=params['physical']
    a=np.column_stack([np.ones(len(x)),(x-ep['mean'])/ep['std']])
    branch=a@ep['logistic']>0
    s=x[:,0]-x[:,2]; alpha=mp['elastic_impulse']['alpha']; e=mp['elastic_impulse']['e']
    el=np.stack([x[:,0]-alpha*s,-e*x[:,1],x[:,2]+alpha*s/K],1)
    return {'inclined':rigid(x,mp['inclined']), 'inclined_signed':rigid(x,mp['inclined_signed']),
            'flat':rigid(x,mp['flat']),'single_affine':a@ep['single'],
            'two_regime':np.where(branch[:,None],a@ep['expert_plus'],a@ep['expert_minus']),
            'elastic_impulse':el,'specular':x*np.array([1,-1,1]),'mean':np.broadcast_to(ymean,x.shape)}


def main():
    fits=json.loads((HERE/'metrics.json').read_text())['folds']
    with (HERE/'predictions.csv').open() as f:
        lookup={(int(r['config']),r['split'],int(r['row']),r['method']):[float(r['vx_pred']),float(r['vz_pred']),R*float(r['wy_pred'])] for r in csv.DictReader(f)}
    batch=defaultdict(list); maxerr=0.; covered=0; omitted=0
    with gzip.open(HERE/'original_predictions.csv.gz','wt') as f:
        writer=csv.writer(f); writer.writerow(['rubber','config','original_row','official_split','split_row','traj_dir',
            'angle','method','vx_pred','vz_pred','wy_pred','vx_obs','vz_obs','wy_obs','error_v','error_w'])
        for name,params in fits.items():
            source_y=np.concatenate([state(read('train',c),'out') for c in params['source']])
            for c in params['held']:
                orig=read('original',c); x,y=state(orig,'in'),state(orig,'out')
                ids={key(r):(split,i) for split in ('train','test') for i,r in enumerate(read(split,c))}
                assert len(ids)==sum(len(read(split,c)) for split in ('train','test'))
                marks=[ids.get(key(r),('omitted',-1)) for r in orig]
                assert len(marks)==len(set(key(r) for r in orig))
                covered+=sum(sp!='omitted' for sp,_ in marks); omitted+=sum(sp=='omitted' for sp,_ in marks)
                for m,yp in predict(x,source_y.mean(0),params).items():
                    v=np.linalg.norm((yp-y)[:,:2],axis=1); w=abs(yp[:,2]-y[:,2])/R
                    errors=np.column_stack([v,w])
                    for label,mask in [('all',np.ones(len(x),dtype=bool)),('omitted',np.array([sp=='omitted' for sp,_ in marks]))]:
                        if mask.any():batch[name,m,label].append(errors[mask])
                    for i,(r,(sp,j)) in enumerate(zip(orig,marks)):
                        if sp!='omitted':
                            ref=lookup[c,sp,j,m]
                            maxerr=max(maxerr,float(np.max(abs(yp[i]-ref))))
                        writer.writerow([name,c,i,sp,j,r['traj_dir'],r['racket_angle'],m,yp[i,0],yp[i,1],yp[i,2]/R,y[i,0],y[i,1],y[i,2]/R,v[i],w[i]])
    assert (covered,omitted)==(8113,71) and maxerr<1e-9,(covered,omitted,maxerr)
    report=dict(verified_gpu_rows=covered,max_gpu_diff_m_s=maxerr,omitted=omitted,groups={},all={})
    for (name,m,key_name),lists in batch.items():
        a=np.concatenate(lists)
        report['groups'].setdefault(name,{}).setdefault(m,{})[key_name]=dict(n=len(a),velocity_mae=a[:,0].mean(),spin_mae=a[:,1].mean())
    for m in sorted(set(k[1] for k in batch)):
        report['all'][m]={}
        for level in ('all','omitted'):
            vals=[d[m][level] for d in report['groups'].values() if level in d[m]]
            w=[v['n'] for v in vals]
            report['all'][m][level]=dict(n=sum(w),v_micro=np.average([v['velocity_mae'] for v in vals],weights=w),w_micro=np.average([v['spin_mae'] for v in vals],weights=w),v_macro=np.mean([v['velocity_mae'] for v in vals]),w_macro=np.mean([v['spin_mae'] for v in vals]))
    (HERE/'original_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(covered,omitted,maxerr)

if __name__=='__main__':main()
