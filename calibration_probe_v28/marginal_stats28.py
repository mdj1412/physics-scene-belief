"""Supplementary comparison on the same 24 cohorts as pairing controls."""
import json,os
import numpy as np
from analyze28 import R,cfg,sha,world_draws,bnds
def main():
    assert os.environ.get('SLURM_JOB_ID')
    cases=[e for e in cfg['environments'] if e['target'] and e['rep']<3]
    values=[];sign=[];switch=[]
    for e in cases:
        base=R/e['id'];alt=R/'marginal'/e['id'];m=json.loads((alt/'metrics.json').read_text())
        assert m['npz_sha256']==sha(alt/'metrics.npz') and m['choices_sha256']==sha(alt/'choice_lock.npz')
        with np.load(base/'metrics.npz') as a,np.load(alt/'metrics.npz') as b:
            values.append(np.column_stack((a['loss'][:,3],b['loss'],a['common_known'][:,3],b['common_known'])))
            switch.append(float(np.mean(a['action'][:,3]!=b['action'])))
        p=m['estimate']['posterior_prob_negative'][1]
        sign.append(float((p>.5)==(e['alpha'][1]<0)))
    x=np.array(values).reshape(3,8,128,4).transpose(1,0,2,3)
    d=x[...,::2]-x[...,1::2];pt=d.mean((0,1,2));bb=world_draws(d,2877613).mean(1)
    (R/'marginal_summary.json').write_text(json.dumps(dict(means=x.mean((0,1,2)).tolist(),paired_minus_marginal=pt.tolist(),
        sim95=bnds(bb,pt).tolist(),sign=np.mean(sign),changed_actions=np.mean(switch)),indent=2))
if __name__=='__main__':main()
