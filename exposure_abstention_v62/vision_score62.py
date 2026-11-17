"""Readout coverage, paired error decomposition and crossed uncertainty in the held-out bank."""
import json
import os
from pathlib import Path
import numpy as np
import vision62 as v

R=v.ROOT
# catalog, board, cell, lag, reader, stat
STATS=['all_opportunities','active','active_read','active_wrong','inactive_read','all_read','all_correct_active']
def summary(c):
    total,act,read,wrong,false,all_read,correct=map(int,c)
    return dict(total=total,active=act,read=read,wrong=wrong,false_read=false,coverage=read/act,
                error_on_read=wrong/read if read else None,correct_yield=correct/act,
                coverage_all=all_read/total)
def ci(d):
    return np.quantile(d,[.025,.975],axis=0).T.tolist()

def main():
    assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and os.getenv('SLURM_JOB_ID')
    cell=np.zeros((24,8,4,4,3,len(STATS)),dtype='int64')
    multiplicity=np.zeros((4,3,9),dtype='i8')
    physical_pair=np.zeros((4,3),dtype='i8')
    raw_coverage=np.zeros((4,3),dtype='int64') # unique clips, optically identifiable, valid dual time register
    for i in range(24):
        with np.load(R/'vision_audit'/f'panel_{i:02}.npz') as a, np.load(R/'pilots'/f'panel_{i:02}.npz') as p:
            truth=a['truth']; camera=a['camera']; lag=a['lag_id']
            assert np.all((camera[:,2]==0)|(camera[:,2]==camera[:,1]))
            for d,r in np.ndindex(4,2):
                np.testing.assert_array_equal(p['codes'][d,2,r],v.encode(a['electronic'][r],camera[d,2],a['reset_id']))
            active=truth[...,0]!=0
            mism=truth[...,0]!=truth[...,1]
            legal=(camera[...,0]!=0)&(camera[...,1]!=0)
            error=(camera[...,0]!=camera[...,1])!=mism[None,None]
            for d,r in np.ndindex(4,3):
                multiplicity[d,r]+=np.bincount(legal[d,r].sum((-1,-2)).ravel(),minlength=9)
                physical_pair[d,r]+=np.any(legal[d,r],axis=(0,1)).sum()
            for b in range(8):
                sl=slice(b,None,8)
                act=active[:,:,:,sl]
                good=legal[:,:,:,:,:,sl]
                bad=error[:,:,:,:,:,sl]
                for h in range(4):
                    mask=(lag[:,sl]==h)[None,None]
                    eligible=act&mask
                    accepted=eligible[None,None]&good
                    false=(~act&mask)[None,None]&good
                    num=accepted.sum((2,3,4,5,6,7)); wrong=(bad&accepted).sum((2,3,4,5,6,7))
                    nf=false.sum((2,3,4,5,6,7))
                    cell[i,b,:,h,:,0]=mask.sum()*4
                    cell[i,b,:,h,:,1]=eligible.sum()
                    cell[i,b,:,h,:,2:]=np.stack([num,wrong,nf,num+nf,num-wrong],-1)
        with np.load(R/'fixtures'/f'panel_{i:02}.npz') as a:
            for d in range(4):
                feat=v.optical_features(a['frames'][d,0,0]);wv=a['wave'][d]
                _,ok,_,_=v.prev.estimate(feat,wv,2)
                flash=(feat[2]>38).sum(-1)
                raw_coverage[d]+=[ok.size,np.count_nonzero(flash==1),ok.sum()]
            # New board parameter vectors differ from every v61 board.
    param=np.array(list(v.BOARDS.values())).T
    old=np.array(list(v.prev.BOARDS.values())).T
    assert np.unique(param,axis=0).shape[0]==8 and not np.any((param[:,None]==old).all(-1))
    total=cell.sum((0,1,3))
    by_board=cell.sum((0,2,3))
    by_lag=cell.sum((0,1))
    per_cell=[[summary(c) for c in row] for row in total]
    per_board=[[summary(c) for c in row] for row in by_board]
    per_lag=[[[summary(c) for c in row] for row in d] for d in by_lag]
    rng=np.random.default_rng(62079081)
    cw=rng.multinomial(24,np.full(24,1/24),5000)
    bw=rng.multinomial(8,np.full(8,1/8),5000)
    agg=np.einsum('qc,qb,cbdms->qdms',cw,bw,cell.sum(3),optimize=True)
    point=np.concatenate([total,total.sum(0,keepdims=True)])
    agg=np.concatenate([agg,agg.sum(1,keepdims=True)],axis=1)
    def metrics(a):
        base=a[...,1,:];sel=a[...,2,:]
        b=base[...,2];e=base[...,3];k=sel[...,2];f=sel[...,3]
        def ratio(num,den):
            return np.divide(num,den,out=np.full_like(den,np.nan,dtype=float),where=den>0)
        return np.stack([k/sel[...,1],b/base[...,1],ratio(k,b),ratio(f,k),ratio(e,b),
                         ratio(e-f,b-k),ratio(e-f,e),
                         (b-k-e+f)/base[...,1]],-1)
    names=['selective_coverage','baseline_coverage','fraction_retained','selective_conditional_error',
           'baseline_conditional_error','error_on_abstained','fraction_of_errors_removed','correct_reads_removed_per_active']
    est=metrics(point);rep=metrics(agg)
    intervals=np.full(est.shape+(2,),np.nan)
    for d,k in np.ndindex(est.shape):
        x=rep[:,d,k]
        if np.isfinite(x).any():intervals[d,k]=np.nanquantile(x,[.025,.975])
    def clean(x):
        if isinstance(x,list):return [clean(y) for y in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    keep=total[:,2,2];base=total[:,1,2];wrong=total[:,1,3];fail=total[:,2,3]
    out=dict(cells=v.CELLS,methods=['nominal','v61_dual','selective'],per_cell=per_cell,pooled=[summary(c) for c in total.sum(0)],
             board=per_board,per_cell_lag=per_lag,lag_values=v.LAG_CELLS,board_parameters={k:a.tolist() for k,a in v.BOARDS.items()},
             abstained_per_cell=(base-keep).tolist(),wrong_among_abstained=(wrong-fail).tolist(),
             interaction_coverage=((total[3,:,2]/total[3,:,1])-(total[2,:,2]/total[2,:,1])-
                                   (total[1,:,2]/total[1,:,1])+(total[0,:,2]/total[0,:,1])).tolist(),
             paid_optical_pairs_per_pilot_hist=multiplicity.tolist(),unique_slots_read_in_any_counterfactual=physical_pair.tolist(),
             crossed_stats=names,crossed_estimates_cell_then_pool=clean(est.tolist()),
             crossed_CI=clean(intervals.tolist()),raw_fixture_unique_clips_optically_unique_dual_valid=raw_coverage.tolist())
    (R/'vision62.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(R/'vision62_cells.npz',counts=cell,names=STATS)
if __name__=='__main__':main()
