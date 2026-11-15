import json,time
from pathlib import Path
import numpy as np
import torch
from bridge import sample,normalize,physics_likelihood
from belief import Belief
from engine import tensor

root=Path('../boundary_belief_v5/results');rows=[]
for case in [1,25,66]:
    d=json.loads((root/'main'/f'case_{case:03d}.json').read_text());bank=np.load(root/'main'/f'factual_{case:03d}.npz')
    theta=tensor(bank['theta']);b=tensor(bank['rounding']);x0=normalize(theta,b);model=Belief(d['family'])
    for action in ['view','left']:
        torch.manual_seed(51000+case);mu,cov=model.mean_cov(theta,b,action);ids=torch.randint(len(theta),(8,),device='cuda')
        yy=mu[ids]+torch.randn((8,mu.shape[1]),device='cuda',dtype=torch.float64)@torch.linalg.cholesky(cov).T
        ll=physics_likelihood(model,d['observations']['factual'],action,yy)
        for method in ['bridge','full']:
            xx=[];zz=[];ss=[]
            for safe in [False,True]:
                x,z,s=sample(ll,8,512,method,51100+case,initial=x0,moves=12,safe_invalid=safe)
                xx.append(x);zz.append(z);ss.append(s)
            error=float((xx[0]-xx[1]).abs().max());ze=float((zz[0]-zz[1]).abs().max())
            assert error<1e-12 and ze<1e-12,(case,action,method,error,ze)
            row=dict(case=case,action=action,method=method,max_particle_error=error,max_logz_error=ze,
                seconds_before=ss[0]['seconds'],seconds_after=ss[1]['seconds'])
            rows.append(row);print(json.dumps(row),flush=True)
Path('results/validation/invalid_proposal_equivalence.json').write_text(json.dumps(rows,indent=2))
