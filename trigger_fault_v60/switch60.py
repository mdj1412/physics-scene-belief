"""Pairwise routing transitions and their material-specific economic consequences."""
import json
import os
import numpy as np
from score60 import ROOT, C, assign, MIX, log_odds, SOURCE_LOG_CUTOFF

def run():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    all_trans=[]
    for c in range(C):
        with np.load(ROOT/'pilots'/f'panel_{c:02d}.npz') as f:
            x=f['codes'][:,1]
        actions=[]
        for reset in (0,1):
            p,_=log_odds(x[:,reset],bool(reset),True)
            actions.append(p>SOURCE_LOG_CUTOFF)
        before,after=actions
        all_trans.append(np.stack([(~before)&after,before&(~after)],axis=-1).mean(-2).transpose(0,3,1,2,4))
    with np.load(ROOT/'score60_cells.npz') as f:
        cost=np.einsum('cfea,ze->cfza',f['fine'],MIX)
        source_rates=f['rates']
    gaps=cost[...,1]-cost[...,0] # fixed action minus gate action
    table=np.stack(all_trans)
    ref=json.loads((ROOT/'score60.json').read_text())['uniform_delay']
    out={}
    for label in ['matched','independent','reversed','common_low','common_high']:
        trans=assign(table,label).mean(axis=(1,3)) # C,M,Z,2
        gain_by_z=(trans[...,1]-trans[...,0])*gaps
        gain_cat=gain_by_z.mean(axis=(1,2))
        target=ref[label]['primary_vs_sham_gate']['mean_Brier'][0][0]
        np.testing.assert_allclose(-gain_cat.mean(),target,atol=1e-13)
        rf=assign(source_rates,label)[:,:,:,:,:,0].mean(axis=(1,3))
        gain_gate=(-rf*gaps).mean(axis=(1,2))
        loo=(C*gain_cat.mean()-gain_cat)/(C-1)
        out[label]={'sham_gate_to_reset_fixed':trans[...,0].mean(),'sham_fixed_to_reset_gate':trans[...,1].mean(),
                    'gross_action_changed':trans.sum(axis=-1).mean(),
                    'relative_gain_by_env_A_B':gain_by_z.mean(axis=(0,1)).tolist(),
                    'weighted_true_A_savings':(-gaps[:,:,0]*trans[:,:,0,0]).mean()/2,
                    'weighted_true_A_losses':(-gaps[:,:,0]*trans[:,:,0,1]).mean()/2,
                    'weighted_true_B_savings':(gaps[:,:,1]*trans[:,:,1,1]).mean()/2,
                    'weighted_true_B_losses':(gaps[:,:,1]*trans[:,:,1,0]).mean()/2,
                    'catalog_wins_reset_vs_sham':int((gain_cat>0).sum()),'catalog_wins_reset_vs_gate':int((gain_gate>0).sum()),
                    'leave_one_out_reset_gain_over_sham':[loo.min(),loo.max()]}
    (ROOT/'switch60.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    run()
