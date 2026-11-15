"""Audit public priorities with outcomes of paying every eligible and ineligible prefix."""
import json,os
from pathlib import Path
import numpy as np
from analysis29 import R,CONFIG,record,paired_rank,clean,ci
def main():
    assert os.environ.get('SLURM_JOB_ID')
    ds=[record(i) for i in range(len(CONFIG['bank_pairs']))]
    rng=np.random.default_rng(29004112);draw=rng.integers(len(ds),size=(10000,len(ds)))
    result={}
    for h in (0,1):
        rows=[]
        for d in ds:
            s=d['m']['stages']['adaptive'][h];v=d['v'];ix=s['active'];kept=np.isin(ix,s['retained'])
            score=np.array([q['priority'] for q in s['details']]);prob=np.array([q['flip_probability'] for q in s['details']])
            actions=v['audit_action'][ix]
            flip=(actions[:,:,h]!=actions[:,:,h+1]).mean(1)
            gain=(v['audit_common'][ix,:,h]-v['audit_common'][ix,:,h+1]).mean(1)
            qgain=(d['qaudit'][ix,:,h]-d['qaudit'][ix,:,h+1]).mean(1)
            values=dict(score_flip_rank=paired_rank(score,flip),prob_flip_rank=paired_rank(prob,flip),score_gain_rank=paired_rank(score,gain),
                score_quota_gain_rank=paired_rank(score,qgain),pred_flip=prob.mean(),true_flip=flip.mean(),pred_gain=score.mean(),
                ess=np.mean([q['hypothetical_ess'] for q in s['details']]),retained_flip=flip[kept].mean(),stopped_flip=flip[~kept].mean(),
                retained_gain=gain[kept].mean(),stopped_gain=gain[~kept].mean(),retained_quota_gain=qgain[kept].mean(),
                stopped_quota_gain=qgain[~kept].mean(),delta_gain=gain[kept].mean()-gain[~kept].mean(),
                delta_quota_gain=qgain[kept].mean()-qgain[~kept].mean())
            rows.append(values)
        stats={}
        for k in rows[0]:
            x=np.array([v[k] for v in rows]);stats[k]=dict(mean=np.nanmean(x),interval=ci(np.nanmean(x[draw],axis=1)),blocks=np.isfinite(x).sum())
        result[['2to4','4to8'][h]]=stats
    (R/'diagnostics.json').write_text(json.dumps(clean(result),indent=2))
if __name__=='__main__':main()
