"""Catalog-paired expected binary Brier with the preregistered source calibration contrast."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import t
H=Path(__file__).resolve().parent
LOCK=H.parent/'invariance_audit_v66'/'locked'
B=('prescribed','compliant')
F=('wendland','dispersive')
V=('base','hollow','bulge')
L=('strengthening','weakening')
S=('inlay','woven')
POLS=('raw','prefix3','cal4')
STAGE=('full','soft075','hard','dropped','restored')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def summarize(x,m):
    avg=x.mean(0);width=x.std(0,ddof=1)*t.isf(.025/m,len(x)-1)/np.sqrt(len(x))
    return np.stack((avg,avg-width,avg+width),axis=-1)

def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    seal=json.loads((H/'FORECASTS_SEALED').read_text())
    a=np.zeros((12,2,2,2,3,4,5))
    shifts=np.zeros((12,2,2,3));replay=0.
    visibility=np.zeros((12,2,2,2)); change_actions=np.zeros((12,2,2))
    for k,b in itertools.product(range(12),B):
        sub=f'panel_{k:02}';bi=B.index(b)
        path=H/'forecasts'/b/(sub+'.npz')
        assert digest(path)==seal['outputs'][str(path.relative_to(H))]
        with np.load(path) as z:
            pred=z['p'].astype(float); names=z['cells'].tolist(); s0=z['shift']; s3=z['shift3'];acts=z['actions']
        for fi,f in enumerate(F):
            for li,l in enumerate(L):
                for s,v,h in itertools.product(S,V,(19,31)):
                    fn=f'{f}_{s}_{l}_{v}_h{h}.npz'
                    ix=names.index(fn)
                    raw=pred[ix]
                    shifts[k,bi,li] += [np.abs(s0[ix]).mean()/24,np.abs(s0[ix]-s3[ix]).mean()/24,np.mean(abs(s0[ix]-s3[ix])>1e-8)/24]
                    with np.load(LOCK/'routes'/sub/f'{f}_{v}_h{h}.npz') as z:gate=z['route']
                    with np.load(H/f'labels_{b}'/sub/fn) as z:labels=z['labels'].astype(float)
                    assert labels.shape==(2,3,4,128)
                    y=labels.mean(0) # outcomes are binary; never square p minus the mean label
                    # 1==query in the old apparatus; derive code mask by known per-env 13 probes
                    u,ns=np.unique(acts[ix],return_counts=True); pid=u[np.argmin(ns)]
                    assert (acts[ix]==pid).sum()==52
                    visibility[k,bi,fi,li]+=(gate*(acts[ix]!=pid)[None]).mean()/12
                    new=acts[ix]
                    with np.load(LOCK/'results'/sub/f'{f}_{s}_rational_{v}_h{h}.npz') as z: old=z['selected']
                    change_actions[k,bi,li]+=(new!=old).mean()/24
                    for j,cue in enumerate([np.zeros(4),s3[ix],s0[ix]]):
                        shift=np.exp(cue[None,None,:,None,None])
                        adj=raw*shift/(1.-raw+raw*shift)
                        mixed=np.broadcast_to(adj[:,:,None],(4,5,3,*adj.shape[2:])).copy()
                        mixed[:,1:3]=np.where(gate[None,None,...,None],mixed[:,1:3],mixed[:,:1])
                        risk=(mixed*mixed+y[None,None,...,None]*(1.-2.*mixed)).mean((2,3,4,5))
                        direct=np.mean([(mixed-tmp[None,None,...,None])**2 for tmp in labels],axis=(0,3,4,5,6)) if k==0 and ix==0 else None
                        if direct is not None:replay=max(replay,float(np.abs(direct-risk).max()))
                        a[k,bi,fi,li,j]+=risk/12
    assert replay<1e-12
    summary={}
    for bi,b in enumerate(B):
        for li,l in enumerate(L):
            for cj,c in enumerate(POLS):
                val=a[:,bi,:,li,cj]
                diff=val[...,1]-val[...,0]
                gain=val[...,4]-val[...,3]
                all3=np.column_stack((diff.mean(-1),gain.mean((1,2))))
                summary[f'{b}_{l}_{c}']={'contrasts':summarize(all3,6).tolist(),
                    'risk_by_family':val.mean((0,2)).tolist(),'per_family_rescue':summarize(gain.mean(2),1).tolist(),
                    'seed_W_D_rescue':np.vstack((diff.mean(0),gain.mean((0,1)))).tolist(),
                    'catalog_signs':[(all3[:,0]<0).sum().item(),(all3[:,1]>0).sum().item(),(all3[:,2]<0).sum().item()]}
    out=dict(solver=B,law=L,family=F,policy=POLS,stage=STAGE,source_seeds=list(range(670101,670105)),
       absolute_axes='12 catalog x solver x family x law x policy x axis x stage',
       catalog_absolute=a.tolist(),summaries=summary,
       shift_columns=['mean absolute 4-probe logodds update','mean abs 3-to-4 change','fraction changed >1e-8'],
       shift_solver_law=shifts.mean(0).tolist(), valid_query_route_solver_family_law=visibility.mean(0).tolist(),
       change_vs_archived_acquisition_solver_law=change_actions.mean(0).tolist(),
       brier_replay_absmax=replay,forecast_seal_sha=digest(H/'FORECASTS_SEALED'),job=os.environ['SLURM_JOB_ID'])
    (H/'score_main68.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
