"""Secondary replay with an actual conventional fourth likelihood or matched fourth likelihood."""
import json
from pathlib import Path
import numpy as np
from gate36 import load_artifact, observations, log_density, posterior, expected_gain, SIGNS

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'design36.json').read_text())
BETA=CFG['beta']

def head(artifact,x,y,dirs):
    env=len(x)
    ll=np.zeros((env,len(artifact['sign'])))
    def incorporate(i):
        obs=observations(x[:,i],y[:,i])
        for act,sign in enumerate(SIGNS):
            picked=dirs[:,i]==sign
            if picked.any(): ll[picked]+=log_density(artifact,obs[picked],act)
    for i in range(3): incorporate(i)
    z=posterior(ll)@artifact['far_sign']
    incorporate(3)
    score=expected_gain(artifact,posterior(ll),z)
    return z,score

def correction(p,z):
    w=np.exp(BETA*z[:,None,None])
    return p*w/(1-p+p*w)

def replay():
    artifact=load_artifact()
    risk=np.full((4,8,4,2,4),np.nan)
    for f,cell in enumerate(CFG['cells']):
        for b in range(8):
            folder=ROOT/'results'/cell/f'block_{b:03d}'
            for r,regime in enumerate(CFG['regimes']):
                for o,sgn in enumerate(('plus','minus') if r!=3 else ('plus',)):
                    with np.load(folder/(regime+'_'+sgn+'.npz')) as a:
                        z,score=head(artifact,a['far_x'],a['far_y'],a['dirs'])
                        np.testing.assert_allclose(z,a['z'],atol=1e-12)
                        accepted=score>0
                        stacked=[correction(a['p_far'],z*accepted),correction(a['p_near'],z*accepted),
                                 correction(a['p_near'],z*a['accept4'])]
                        err=[np.mean((p-a['labels'][:,:,None])**2) for p in stacked]
                        np.testing.assert_allclose(err[2],a['risk'][5].mean(),atol=1e-12)
                        risk[f,b,r,o]=[*err,accepted.mean()]
                if r==3: risk[f,b,r,1]=risk[f,b,r,0]
    base=risk.mean(axis=3)
    per=np.stack((base[:,:,:3].mean(axis=2),base[:,:,3]),axis=2)
    dif=per[:,:,:,2:3]-per[:,:,:,:2]
    point=dif.mean(axis=(0,1))
    rng=np.random.default_rng(360510)
    idx=(rng.integers(0,4,(12000,4,4,2))*2+np.arange(2)).reshape(-1,4,8)
    reps=dif[np.arange(4)[None,:,None],idx].mean((1,2))
    se=reps.std(axis=0,ddof=1)
    radius=np.quantile(np.abs((reps-point)/np.maximum(se,1e-15)).max(axis=(1,2)),.95)
    ci=np.stack((point-radius*se,point+radius*se),axis=-1)
    return dict(arms=['far_gate_actual_actions','far_gate_on_near_actions','near_gate_actual_actions','far_gate_apply_rate'],
                groups=['local_equal','global_null'],level_groups=CFG['regimes'],group_means=per.mean((0,1)).tolist(),
                level_means=base.mean((0,1)).tolist(),near_minus_far=point.tolist(),simultaneous_ci95=ci.tolist(),
                nclusters=32,iterations=12000)

if __name__=='__main__':
    d=replay()
    (ROOT/'far_fourth36.json').write_text(json.dumps(d,indent=2)+'\n')
    print(d['near_minus_far'],d['simultaneous_ci95'])
