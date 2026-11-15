import json,os
import numpy as np
from analyze28 import R,cfg,sha,world_draws,bnds
def main():
    assert os.environ.get('SLURM_JOB_ID')
    es=[e for e in cfg['environments'] if e['target'] and e['rep']<3]
    vals=[];sig=[]
    for e in es:
        base=R/e['id'];aux=R/'controls'/e['id']
        m=json.loads((aux/'metrics.json').read_text());main=json.loads((base/'metrics.json').read_text())
        assert sha(aux/'metrics.npz')==m['npz_sha256'] and sha(aux/'choice_lock.npz')==m['choices_sha256'] and m['main_npz_sha256']==main['npz_sha256']
        with np.load(aux/'metrics.npz') as y,np.load(base/'metrics.npz') as x:
            v=[]
            for key in ['loss','common_known','action']:
                a=np.column_stack([x[key][:,0],x[key][:,3],y[key],x[key][:,4],x[key][:,5]])
                v.append((a==3).astype(float) if key=='action' else a)
            vals.append(np.column_stack(v))
        hats=[main['estimates']['0'],main['estimates']['16'],m['estimates']['friction_only'],m['estimates']['deranged_pairs']]
        p=np.array([q['posterior_prob_negative'][1] for q in hats]);success=p if e['alpha'][1]<0 else 1-p
        sig.append((success>.5)+.5*(success==.5))
    x=np.stack(vals).reshape(3,8,128,18).transpose(1,0,2,3)
    z=np.stack([x[...,o+i]-x[...,o+j] for o in (0,6) for i,j in ((1,0),(1,2),(1,3),(2,0))],-1)
    pt=z.mean((0,1,2))
    out=dict(order=['probe0','probe16','friction_only','deranged','neutral','known'],native=x.mean((0,1,2))[:6].tolist(),
        common=x.mean((0,1,2))[6:12].tolist(),probepct=x.mean((0,1,2))[12:].tolist(),
        sign=np.mean(sig,axis=0).tolist(),delta=pt.tolist(),sim95=bnds(world_draws(z,2857301).mean(1),pt).tolist())
    (R/'controls_summary.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
