"""Secondary feature/action ablation, run after the pre-specified endpoints."""
import json
import os

import numpy as np
from score53 import ROOT,V52,N,G,quadratic,pair_ci,quantiles,bootstrap

if __name__ == '__main__':
    assert os.getenv('SLURM_JOB_ID')
    methods = ['impact_joint','impact_vote_joint','impact_vote_selector']
    with np.load(ROOT/'catalog53.npz') as f:
        archived = f['exact'][:,0]
        names = list(f['methods'].astype(str))
    with np.load(ROOT/'session53.npz') as f:
        sessions = f['realized'][...,[names.index('impact_joint'),names.index('impact_vote_selector')]]
    cs=[]; ss=[]
    for i in range(N):
        file=f'panel_{i:02d}.npz'
        with np.load(ROOT/'votes53'/file) as vf, np.load(V52/'risk52'/file) as risk:
            a=vf['alpha']
            ex=quadratic(risk['exact'][0,...,None,:],a,False).mean(-1)
            rr=quadratic(risk['realized'],a,False)
        cs.append(np.stack([archived[i,...,names.index('impact_joint')],ex,archived[i,...,names.index('impact_vote_selector')]],-1))
        ss.append(np.stack([sessions[i,...,0],rr,sessions[i,...,1]],-1).astype(np.float32))
    cs=np.stack(cs);ss=np.stack(ss)
    by_catalog=((1-G.mean())*cs[:,:,0]+G.mean()*cs[:,:,1]).mean((1,2,3))
    q=quantiles(ss,G.mean()); boot=bootstrap(ss)
    def contrast(i,j):
        return dict(mean=pair_ci(by_catalog[:,i]-by_catalog[:,j],1),p90=[q[i]-q[j],*np.quantile(boot[:,i]-boot[:,j],[.025,.975])])
    out=dict(methods=methods,mean=by_catalog.mean(0),p90=q,
             magnitude_inside_continuous=contrast(0,1),continuous_inside_vote=contrast(1,2),
             scope='Secondary, nominal 95% intervals; same old/new witnesses, same prior. Endpoint contrast of marginals.')
    (ROOT/'vote_score53.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))
    print(out)
