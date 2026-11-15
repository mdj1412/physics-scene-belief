"""Private simulator, invoked separately from policy selection."""
import numpy as np
import torch
from model import *
from common import *

def generate(n=512,seed=4100101,dest='test'):
    public=[];contexts=[];truth=[];theta=[];observations={a:[] for a in [1,2,3]}
    for i in range(n):
        rng=np.random.default_rng(seed+137*i);th=from_unit(t(rng.uniform(size=(1,7))))[0];theta.append(th.cpu().numpy())
        eps0=t(rng.normal(size=11));eps={a:t(rng.normal(size=(16,len(noise(a))))) for a in [1,2,3]}
        xs=[];zs=[];cs=[];ys={a:[] for a in [1,2,3]}
        for law in ['nominal','shifted']:
            x=measure(th,0,law)+eps0*noise(0);ctx=context(x);xs.append(x.cpu().numpy());cs.append(ctx.cpu().numpy());zs.append(targets(th,ctx,law).cpu().numpy())
            for a in [1,2,3]:ys[a].append((measure(th,a,law)+eps[a]*noise(a)).cpu().numpy())
        public.append(xs);contexts.append(cs);truth.append(zs)
        for a in [1,2,3]:observations[a].append(ys[a])
    save(R/'data'/dest/'public.npz',factual=public,context=contexts)
    save(R/'data'/dest/'private.npz',theta=theta,target=truth,**{f'obs{a}':v for a,v in observations.items()})

if __name__=='__main__':
    require();generate(8,4000101,'pilot');generate();print('Generated public/private files',flush=True)
