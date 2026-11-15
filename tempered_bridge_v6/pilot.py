import json,sys
from pathlib import Path
import numpy as np
import torch
from belief import Belief
from bridge import sample,normalize,physics_likelihood,summaries
from engine import tensor

case=int(sys.argv[1]) if len(sys.argv)>1 else 66
root=Path('../boundary_belief_v5/results');d=json.loads((root/'main'/f'case_{case:03d}.json').read_text())
bank=np.load(root/'main'/f'factual_{case:03d}.npz');print('bank',bank.files,flush=True)
theta=tensor(bank['theta']);b=tensor(bank['rounding']);x0=normalize(theta,b)
model=Belief(d['family']);torch.manual_seed(85000+case)
mu,cov=model.mean_cov(theta,b,'view');ids=torch.randint(len(theta),(8,),device='cuda')
yy=mu[ids]+torch.randn((8,3),device='cuda',dtype=torch.float64)@torch.linalg.cholesky(cov).T
output={}
for method in ['bridge','full']:
    x,z,s=sample(physics_likelihood(model,d['observations']['factual'],'view',yy),8,512,method,1800+case,initial=x0)
    summary=summaries(x,d['family']);output[method]=dict(stat=s,summary={k:v.tolist() for k,v in summary.items()})
    print(json.dumps({k:v for k,v in s.items() if k!='history'}),flush=True)
Path('results/pilot').mkdir(exist_ok=True)
Path(f'results/pilot/case_{case:03d}.json').write_text(json.dumps(output))
