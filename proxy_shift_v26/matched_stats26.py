"""Exploratory known-prior vs frozen-prior evaluation on exactly the same held-out subset."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from stats26 import (contrast_tables, crossed_means, flags_by_margin, interval, paired_time_quantiles,
                     pathways, simultaneous_band, transitions, verify_records)

R = Path('results')
M = R/'matched'
def read(path):
    return json.loads(path.read_text())


def main():
    os.chdir(Path(__file__).resolve().parent)
    assert os.environ.get('SLURM_JOB_ID')
    verify_records()
    for name in ('sources','choices_freeze'):
        manifest = read(M/(name+'.json'))
        for p,h in manifest.get('files',manifest).items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    with np.load(M/'cohort.npz') as f:
        ids,cell,variants,pairs,tids,tcond = [f[k] for k in ('ids','cell','variant','conditions','time_ids','timed_cond')]
        conf = f['confident']
    n,c = variants.shape
    assert n == 256 and c == 36 and np.array_equal(np.bincount(cell),[64]*4)
    assert np.all(variants[:,0]==0) and tuple(pairs[0])==(1.,1.)
    assert np.array_equal(variants,np.load(R/'scenarios.npz')['variants'][:,ids].T)
    matched=[];stale=[];times=[];change=[];conf_changes=[]
    for b in range(3):
        ma=np.load(M/f'choices_{b}.npz')['acts']
        m=np.load(M/f'selected_loss_{b}.npz')['loss']
        ta=np.load(M/f'times_{b}.npz')['times']
        assert np.isin(ma,(1,3)).all() and np.isfinite(m).all()
        with np.load(R/f'choices_{b}.npz') as f:
            raw=f['acts'][ids,:,:2]
        with np.load(R/f'eval_{b}.npz') as f:
            raw_loss=f['loss'][ids].mean(-1)
        row=np.arange(n)[:,None]
        aa=raw[row,variants]
        cost=raw_loss[row,variants]
        st=np.take_along_axis(cost,(aa==3).astype(int),axis=-1)
        assert np.array_equal(ma[:,0],aa[:,0]),'Original setting must recover v25 numerics'
        assert np.allclose(m[:,0],st[:,0],rtol=0,atol=2e-11)
        assert np.isfinite(ta).all() and np.all(ta>0)
        changed=ma!=aa
        assert not changed[...,0][conf].any(), 'External prior must not change confidential/direct actions'
        matched.append(m);stale.append(st);times.append(ta);change.append(changed)
    values=np.concatenate([np.stack(matched),np.stack(stale)],axis=-1)
    point=values.mean(axis=(0,1))
    boot=crossed_means(values.reshape(3,n,-1),cell,n=10000).reshape(-1,c,4)
    mband,crit=contrast_tables(point,boot,0,1)
    sband,stcrit=contrast_tables(point,boot,2,3)
    delta=point[:,0]-point[:,1]-point[:,2]+point[:,3]
    dboot=boot[:,:,0]-boot[:,:,1]-boot[:,:,2]+boot[:,:,3]
    lb,ub,crit_delta=simultaneous_band(dboot,delta)
    ratios=paired_time_quantiles(np.stack(times),cell[tids],n=5000)
    # Important: pooled p90s are over banks, worlds and repeats, whereas bootstrap quantiles
    # are over fresh paired draws and used exclusively for interval estimation.
    p50,p90=np.quantile(np.stack(times).reshape(-1,len(tcond),2),[.5,.9],axis=0)
    rows=[];changes=np.stack(change)
    for i,cond in enumerate(pairs):
        row=dict(idx=i,condition=cond.tolist(),matched_loss=point[i,:2].tolist(),stale_loss=point[i,2:].tolist(),
                 matched_bands=mband[i],stale_bands=sband[i],flag=flags_by_margin(mband[i],.003),
                 stale_flag=flags_by_margin(sband[i],.003),changed_action_fraction=changes[:,:,i].mean((0,1)).tolist(),
                 matched_minus_stale_gap=dict(**interval(dboot[:,i],delta[i]),sim95=[float(lb[i]),float(ub[i])]),
                 known_minus_stale_gate=interval(boot[:,i,0]-boot[:,i,2],point[i,0]-point[i,2]),
                 known_minus_stale_k256=interval(boot[:,i,1]-boot[:,i,3],point[i,1]-point[i,3]))
        if i in tcond:
            j=int(np.flatnonzero(tcond==i)[0])
            row['timing']=dict(p50=p50[j].tolist(),p90=p90[j].tolist(),ratio=p90[j,0]/p90[j,1],
                ratio_ci95=np.quantile(ratios[:,j,0]/ratios[:,j,1],[.025,.975]).tolist())
        rows.append(row)
    crossing={}
    for name,ids_path in pathways(pairs).items():
        crossing[name] = {kind:{key:transitions(ids_path,[r[kind][key] for r in rows])
             for key in ('relative','absolute','gap_only')} for kind in ('flag','stale_flag')}
    metrics=dict(n_worlds=n,n_banks=3,rows=rows,crossing=crossing,
        simultaneous_critical_matched=crit,simultaneous_critical_stale=stcrit,simultaneous_critical_gap_difference=crit_delta)
    output=M/'summary.json';tmp=M/'summary.tmp'
    tmp.write_text(json.dumps(metrics,indent=2,allow_nan=False));tmp.replace(output)
    print('Matched cohort relative onsets', {n:row['flag']['relative']['first'] for n,row in crossing.items()},flush=True)
if __name__=='__main__':main()
