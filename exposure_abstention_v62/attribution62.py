"""Privileged audit of the rejected-camera mismatch channels, with all digitization held fixed."""
import os
import json
import numpy as np
from score62 import (ROOT,C,B,MIX,board_average,log_odds,SOURCE_LOG_CUTOFF,target_assign,
                     load_deployment,make_weights)

def get_pilot(i):
    name=f'panel_{i:02}.npz'
    with np.load(ROOT/'pilots'/name) as p:
        x=p['codes']
        original=x[:,1,1]
        selective=x[:,2,1]
    with np.load(ROOT/'vision_audit'/name) as f:
        t=f['truth']
    physical=t[...,0]!=t[...,1]
    symbols=np.stack([original//13,original%13],axis=-1)
    dest=np.stack([selective//13,selective%13],axis=-1)
    removed=(symbols>=5)&(dest<5)
    wrong=((symbols>=9)!=physical[None])&removed
    correct=removed&~wrong
    assert np.all((~removed)|(1+(symbols-1)%4==dest))
    def encode(mask):
        s=np.where(mask,dest,symbols)
        return 13*s[...,0]+s[...,1]
    ranks=[];decisions=[]
    for code in (original,encode(wrong),encode(correct),selective):
        odds,_=log_odds(code,1,True)
        choose=odds>SOURCE_LOG_CUTOFF
        ranks.append(board_average(choose)); decisions.append(choose)
    assert np.array_equal(encode(wrong|correct),selective)
    removed_counts=np.stack([board_average(z.sum((-2,-1))) for z in (wrong,correct)],axis=-1)
    switch=np.stack([board_average(decisions[0]!=v) for v in decisions],axis=-1)
    return np.stack(ranks,-1),removed_counts,switch

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    inputs=[get_pilot(c) for c in range(C)]
    rate,removed,switch=map(np.asarray,zip(*inputs))
    fine=np.stack([load_deployment(c)[0] for c in range(C)])
    bank=np.einsum('cfea,ze->cfza',fine,MIX)
    gap=bank[...,1]-bank[...,0]
    _,_,prod=make_weights(6000,6204133)
    out={}
    for rel in ['independent','matched','reversed']:
        a=target_assign(rate,rel).mean((2,4))
        risks=(a*gap[:,None,:,:,None]).mean((2,3))
        dif=risks-risks[...,:1]
        sampled=prod@dif.reshape(C*B,4)
        w=.5*(sampled[:,1]+sampled[:,3]-sampled[:,2])
        k=.5*(sampled[:,2]+sampled[:,3]-sampled[:,1])
        assert np.max(np.abs(w+k-sampled[:,3]))<1e-14
        def ci(data):
            return [data[0],*np.quantile(data[1:],[.025,.975])]
        out[rel]=dict(mean_Brier_minus_original=[ci(col) for col in sampled.T],
                      Shapley_wrong=ci(w),Shapley_correct=ci(k),
                      n_rejected_per_8=target_assign(removed,rel).mean((0,1,2,3,4,5)).tolist(),
                      switch_probability=target_assign(switch,rel).mean((0,1,2,3,4,5)).tolist())
    score=json.loads((ROOT/'score62.json').read_text())
    for rel,d in out.items():
        expected=score['uniform'][rel]['vs_v61reset_v61sham_gate']['mean_Brier'][0][0]
        assert abs(expected-d['mean_Brier_minus_original'][3][0])<1e-13
    result=dict(methods=['original','remove_wrong','remove_correct','remove_both'],
                rejection_types=['wrong','correct'],by_relation=out)
    (ROOT/'attribution62.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
