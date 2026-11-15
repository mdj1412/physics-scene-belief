import json
from pathlib import Path
import numpy as np
import torch
from bridge import normalize,sample,physics_likelihood
from belief import Belief
from engine import tensor
from grouped import prepare

rows=[]
for case in [1,25,66]:
    d=prepare(case,8);model=Belief(d['family']);mu,cov=model.mean_cov(d['theta'],d['b'],'view')
    yy=mu[d['ids']]+d['noise']@torch.linalg.cholesky(cov).T
    yf=tensor(d['yf'][:1])[None,:].expand(8,-1)
    for method in ['bridge','full']:
        x1,z1,s1=sample(physics_likelihood(model,d['yf'],'view',yy),8,512,method,281000+case,initial=d['x0'],moves=12)
        x2,z2,s2=sample(physics_likelihood(model,yf,'view',yy),8,512,method,281000+case,initial=d['x0'][None,:,:].expand(8,-1,-1),moves=12)
        assert torch.equal(x1,x2) and torch.equal(z1,z2),(case,method)
        rows.append(dict(case=case,method=method,bitwise_particles=True,bitwise_logz=True))
    torch.manual_seed(100+case);x=torch.rand((8,512,5),device='cuda',dtype=torch.float64)
    yf=yf.clone();yf[:,0]+=torch.linspace(-.05,.05,8,device='cuda')
    grouped=physics_likelihood(model,yf,'view',yy)(x,torch.arange(8,device='cuda'))
    singles=[physics_likelihood(model,yf[i].cpu().tolist(),'view',yy[i:i+1])(x[i:i+1],tensor([0]).long()) for i in range(8)]
    for k in range(2):assert torch.equal(grouped[k],torch.cat([s[k] for s in singles]))
    print(json.dumps(dict(case=case,bitwise=True,varying_factual_likelihood=True)),flush=True)
Path('results/validation/grouped_equivalence.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
