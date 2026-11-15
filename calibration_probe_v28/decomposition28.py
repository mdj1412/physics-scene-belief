"""Descriptive action/decoder attribution for native paired decision loss on the target panel."""
import os,json
import numpy as np
from analyze28 import R,cfg,world_draws,bnds
def main():
    assert os.environ.get('SLURM_JOB_ID')
    rows=[]
    for e in cfg['environments']:
        if not e['target']:continue
        with np.load(R/e['id']/'metrics.npz') as f:
            c=f['cost'];a=(f['action']==3).astype(int)
            x=[]
            for j in (0,4):
                for i in (1,2,3):
                    w=np.arange(len(a));v00=c[w,j,a[:,j]];v10=c[w,j,a[:,i]];v01=c[w,i,a[:,j]];v11=c[w,i,a[:,i]]
                    action=.5*((v10-v00)+(v11-v01));decoder=.5*((v01-v00)+(v11-v10))
                    assert np.max(abs(v11-v00-action-decoder))<1.e-12
                    x.extend([action,decoder])
            rows.append(np.stack(x,-1))
    x=np.stack(rows).reshape(8,8,128,12).transpose(1,0,2,3)
    pt=x.mean((0,1,2));bb=world_draws(x,2818401).mean(1)
    (R/'decomposition.json').write_text(json.dumps(dict(deltas=pt.tolist(),sim95=bnds(bb,pt).tolist(),order='compare each of 1,4,16 to 0 then each of 1,4,16 to neutral; action,decoder'),indent=2))
if __name__=='__main__':main()
