"""Paired waveform interventions and the ambiguity of the optical flash index."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import vision61 as v
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'sensor_reset_v59'))
from route59 import log_odds, SOURCE_LOG_CUTOFF

def inject(w, channels):
    out=w.copy()
    for c in channels:
        out[...,c,:16]=w[...,c,:1]
        out[...,c,16:]=w[...,c,:-16]
    return out

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    hist=np.zeros((2,9),dtype=int)
    # Per-class unambiguous hits, abstentions and number of clips.
    decisions=np.zeros((2,3),dtype=int)
    counts=np.zeros((4,4,6),dtype=int)
    cases={'original':(), 'both_plus_1ms':(0,1), 'switch_plus_1ms':(0,), 'diode_plus_1ms':(1,)}
    for i in range(24):
        with np.load(ROOT/'fixtures'/f'panel_{i:02}.npz') as f:
            fr,w,truth,e,idx=[f[k] for k in ('frames','wave','truth','electronic','reset_id')]
        # One copy of each impact. The lamp patch and waveform are shared across P/C, A/B.
        optical=v.optical_features(fr[:,0,0,:,:,:,1])
        led=optical[2]>38
        flash=np.where(led.sum(-1)==1,led.argmax(-1)+1,0)
        _,_,lh,th=v.estimate(optical,w[:,:,:,:,1],2)
        for k,d in enumerate((0,3)):
            hist[k]+=np.bincount(flash[d].ravel(),minlength=9)
            late=(lh[d]>1)&(th[d]>1)
            early=(lh[d]<=1)&(th[d]<=1)
            hits=late if k else early
            decisions[k]+=[hits.sum(),(~(early|late)).sum(),hits.size]
        # This readout audit keeps the four paired sign counterfactuals explicitly.
        feat=v.optical_features(fr)
        baseline=None
        for a,chs in enumerate(cases.values()):
            signs=v.read_features(feat,inject(w,chs)[:,None,None],2)
            active=truth[...,0]!=0
            valid=(signs!=0).all(-1)&active[None]
            mistaken=(signs[...,0]!=signs[...,1])!=(truth[...,0]!=truth[...,1])[None]
            codes=v.encode(e[1],signs,idx)
            action=log_odds(codes,1,True)[0]>SOURCE_LOG_CUTOFF
            if baseline is None: baseline=action
            for d in range(4):
                counts[a,d]+=[active.sum(),valid[d].sum(),(mistaken[d]&valid[d]).sum(),
                              action[d].size,(action[d]!=baseline[d]).sum(),
                              np.count_nonzero((signs[d]!=0).all(-1)&~active)]
    p=hist/hist.sum(1,keepdims=True)
    tv=.5*abs(p[0]-p[1]).sum()
    correct,abstain,n=decisions.T
    out=dict(optical_hist_missing_then_frame_0_to_7=hist.tolist(),optical_empirical_BA=.5*(1+tv),
             dual_by_class_correct_abstain_total=decisions.tolist(),
             dual_all_BA=((correct+.5*abstain)/n).mean(),dual_decided_BA=(correct/(n-abstain)).mean(),
             dual_decided_fraction=(1-abstain/n).mean(),cases=list(cases),factor_order=v.CELLS,
             names=['active_pairs','read_pairs','wrong_read_pairs','paired_policy_comparisons','changed_policy','false_active'],
             counts=counts.tolist())
    (ROOT/'clock61.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    main()
