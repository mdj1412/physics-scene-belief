"""Compare two privileged selectors on the exact same public X. Estimated policies need no labels here."""
import json,os
import numpy as np
from runner27 import ROOT,Engine,atomic_json,t,torch
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    config=json.loads((ROOT/'design.json').read_text())
    rows=[]
    with torch.inference_mode():
        for env in config['environments']:
            if env['family'] not in ('neg_neg','neg_pos','pos_neg','pos_pos','weak_neg','weak_pos','f_positive','f_negative'):
                continue
            result=ROOT/'results'/env['id']
            ac,af=env['alpha']
            pair=config['bank_pairs'][env['bank_index']]
            opposite=Engine(pair,ac,-af)
            with np.load(result/'choice_lock.npz') as f:
                x=f['x_test'];correct=f['action'][:,3]
            counter=np.array([opposite.choose_pair(opposite.fit(t(v)),256) for v in x])
            # Private evaluation reads the real-law Brier table only after both sets of choices are fixed.
            with np.load(result/'metrics.npz') as f:cost=f['cost'][:,2,:]
            i=np.arange(len(x));loss_delta=cost[i,(counter==3).astype(int)]-cost[i,(correct==3).astype(int)]
            rows.append(dict(env=env['id'],alpha=[ac,af],n=len(x),action_disagreement=float(np.mean(correct!=counter)),
                opposite_minus_actual_common_loss=float(loss_delta.mean())))
    atomic_json(ROOT/'results/twin_actions.json',rows)
if __name__=='__main__':main()
