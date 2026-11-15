"""Non-physics target validation against dense bounded Gaussian quadrature."""
import json
from pathlib import Path
import numpy as np
import torch
from bridge import sample
from engine import tensor

grid=(np.arange(1000)+.5)/1000
xx,yy=np.meshgrid(grid,grid,indexing='ij')
lf=-.5*((xx+yy-1.)/.045)**2
ys=np.array([.12,.50,.87])
truth=[]
for y in ys:
    ll=lf-.5*((xx-y)/.022)**2;w=np.exp(ll-ll.max());w/=w.sum()
    truth.append([float((w*xx).sum()),float((w*yy).sum())])
torch.manual_seed(6300)
# Exact rejection sampling of the bounded factual Gaussian.
accepted=[]
while sum(len(a) for a in accepted)<32768:
    x=torch.rand((100000,5),device='cuda',dtype=torch.float64)
    p=torch.exp(-.5*((x[:,0]+x[:,1]-1.)/.045)**2)
    accepted.append(x[torch.rand_like(p)<p])
bank=torch.cat(accepted)[:32768]
def likelihood(x,ids):
    return -.5*((x[:,:,0]+x[:,:,1]-1.)/.045)**2,-.5*((x[:,:,0]-tensor(ys)[ids,None])/.022)**2
rows=[]
for method in ['bridge','full']:
    for rep in range(2):
        x,z,s=sample(likelihood,3,4096,method,6400+rep,initial=bank,moves=18)
        estimate=x[:,:,:2].mean(1).cpu().numpy();err=float(np.max(np.abs(estimate-np.array(truth))))
        assert err<.008,(method,err)
        assert min(s['min_stage_ess'])>=.7499*4096
        rows.append(dict(method=method,rep=rep,estimate=estimate.tolist(),max_abs_error=err,stat=s))
out=dict(reference=truth,rows=rows,passed=True)
Path('results/validation').mkdir(exist_ok=True)
Path('results/validation/gaussian.json').write_text(json.dumps(out))
print(json.dumps(dict(passed=True,max_abs_error=max(r['max_abs_error'] for r in rows))),flush=True)
