"""Paired exploration using the signed-release threshold policy frozen in FROZEN_KO.md."""
from pathlib import Path
import csv
import json
import numpy as np

HERE=Path(__file__).resolve().parent
V74=HERE.parents[1]
ROOT=V74.parent
R=.020
MARGIN=.2

def release(a):
    return a[...,0]-a[...,2]

def coherent(p,desired):
    delta=np.where((release(p)>0)!=desired,np.where(desired,MARGIN,-MARGIN)-release(p),0)
    new=p.copy()
    new[...,0]+=.4*delta
    new[...,2]-=.6*delta
    assert np.array_equal(release(new)>0,desired)
    return new

def train():
    with (ROOT/'racket_transfer_v70/data/upstream/train/racket_10.csv').open() as f:
        rows=list(csv.DictReader(f))
    def state(v,w):
        return np.array([[float(a[v+'x']),float(a[v+'z']),R*float(a[w+'y'])] for a in rows])
    idx={f"{int(a['racket_angle'])}/{int(a['traj_dir']):03d}":j for j,a in enumerate(rows)}
    return state('v_in_','spin_before_'),state('v_out_','spin_after_'),idx

def fit_each_draw(s,so):
    y=(s>0)!=(so>0)
    if np.unique(y).size<2:
        return np.inf,False,None,'no_vote' if not len(y) else ('reverse_only' if y[0] else 'retain_only')
    u=np.unique(abs(s))
    cuts=np.r_[-np.inf,(u[:-1]+u[1:])/2,np.inf]
    losses=np.sum((abs(s[:,None])<=cuts)!=y[:,None],axis=0)
    best=cuts[losses==min(losses)][-1]
    gap=float(min(abs(s[~y]))-max(abs(s[y]))) if min(losses)==0 else None
    return best,True,gap,'separated' if gap is not None else 'conflict'

def evaluate(name,cut):
    bank=np.load(V74/'model/result'/name/'predictions.npz')
    x,y,j=train()
    ids=[j[v] for v in bank['calibration_ids']]
    tx,ty=bank['incident'],bank['target']
    mask=abs(tx[:,1])>=cut
    tx,ty=tx[mask],ty[mask]
    de=release(ty); eligible=abs(de)>MARGIN
    out={}; logs=[]
    for k in (4,8):
        ids_k=np.array(ids)[bank['tape'][:,:k]]
        th=[]; both=[]; gaps=[]; stats={}
        for i,r in enumerate(ids_k):
            a,b=release(x[r]),release(y[r])
            ok=(abs(x[r,1])>=cut)&(abs(b)>MARGIN)&(a!=0)
            c,both_classes,gap,tag=fit_each_draw(a[ok],b[ok])
            stats[tag]=stats.get(tag,0)+1
            th.append(c);both.append(both_classes)
            if gap is not None:gaps.append(gap)
            logs.append(dict(split=name,K=k,draw=i,paid_fast_resolved=int(sum(ok)),category=tag,
                             cutoff_mps=c,calibration_gap_mps=gap))
        b=bank[f'elastic_k{k}'][:,mask,:]
        baseline=(release(tx)[eligible]<0)==(de[eligible]>0)
        rng=np.random.default_rng(740123)
        resamples=rng.integers(0,int(eligible.sum()),size=(4096,int(eligible.sum())))
        for m,c in [('fewshot_threshold',np.array(th)),('always_reverse',np.full(len(th),np.inf))]:
            s=release(tx)
            intended=(s>0) ^ (abs(s)<=c[:,None])
            p=coherent(b,intended)
            h=(release(p[:,eligible])>0)==(de[eligible]>0)
            v=np.linalg.norm((p-ty)[...,:2],axis=-1)
            w=abs((p-ty)[...,2])/R
            base=(release(bank[f'majority_k{k}'][:,mask,:][:,eligible])>0)==(de[eligible]>0)
            out.setdefault(str(k),{})[m]=dict(fast=len(de),eligible=int(sum(eligible)),accuracy=float(h.mean()),
                velocity=float(v.mean()),spin=float(w.mean()),sign_hits=int(h.sum()),sign_pairs=h.size,
                majority_acc=float(base.mean()),paired_gain_vs_reverse_percentile=np.percentile((h.mean(axis=0)-baseline)[resamples].mean(axis=1),[2.5,97.5]).tolist(),rev=float((abs(s)<=c[:,None]).mean()),coherent_changed=float(((release(b)>0)!=intended).mean()),
                subset_identified_fast_accuracy=float(h[both].mean()) if any(both) else None,
                subset_unidentified_fast_accuracy=float(h[~np.array(both)].mean()) if not all(both) else None)
        out[str(k)]['evidence']=dict(counts=stats,draws=len(both),both_class_draws=sum(both),
                                     consistent_gap_median_mps=float(np.median(gaps)) if gaps else None,
                                     identified_finite_cutoff_median_mps=float(np.median([v for v in th if np.isfinite(v)])) if np.isfinite(th).any() else None)
    return out,logs

def main():
    conf=json.loads((ROOT/'racket_video_v73/gpu/control/run.json').read_text())
    out={};logs=[]
    for name in ('official','secondary'):
        out[name],ll=evaluate(name,conf['fast_cutoff_mps']); logs+=ll
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    with (HERE/'draws.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=logs[0]);w.writeheader();w.writerows(logs)
if __name__=='__main__':main()
