"""Replace wrong optical mismatch bits at the existing coverage, for diagnosis only."""
import json
import os
import numpy as np
from thinning62 import ROOT, B, MIX, classify, split_codes

def collect(c):
    with np.load(ROOT/'pilots'/f'panel_{c:02d}.npz') as f:
        s=split_codes(f['codes'][:,1:,1])
    with np.load(ROOT/'vision_audit'/f'panel_{c:02d}.npz') as f:
        y=f['truth']
    true_bit=(y[...,0]!=y[...,1]).reshape((2,2,2,512,8))
    corrected=np.where(s>4,5+(s.astype(int)-1)%4+4*true_bit,s)
    route=[]
    for x in (s,corrected):
        point=classify(x)[...,0].astype(float)
        # D,method,cause,target,material,episode -> D,method,cause,target,material,board
        point[:,:,1]=point[:,:,1].mean(2,keepdims=True)
        route.append(point.reshape(4,2,2,2,2,64,B).mean(5))
    with np.load(ROOT/'scored62'/f'panel_{c:02d}.npz') as f:
        si=((np.arange(4)+c)%4)//2
        base=np.stack([f['exact_cells'][:,:,si==m,:2].mean((1,2)) for m in range(2)])
    risk=np.einsum('mea,ze->mza',base,MIX)
    opt_gap=(risk[:,:,1]-risk[:,:,0]).T
    return (np.asarray(route)*opt_gap[None,None,None,None,:,:,None]).mean((3,4,5))

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    # catalog, original/repaired, cell, v61/selective, board
    x=np.array([collect(i) for i in range(24)])
    rng=np.random.default_rng(6209559)
    wc=rng.multinomial(24,np.full(24,1/24),size=8000)/24
    wb=rng.multinomial(8,np.full(8,1/8),size=8000)/8
    data={'actual_selective_minus_v61':x[:,0,:,1]-x[:,0,:,0],
          'repaired_selective_minus_v61':x[:,1,:,1]-x[:,1,:,0],
          'v61_repaired_minus_actual':x[:,1,:,0]-x[:,0,:,0],
          'selective_repaired_minus_actual':x[:,1,:,1]-x[:,0,:,1]}
    report={}
    for name,d in data.items():
        y=d.mean(1)
        boot=np.einsum('dc,db,cb->d',wc,wb,y,optimize=True)
        report[name]=dict(uniform=[float(y.mean()),*np.quantile(boot,[.025,.975]).tolist()],
                          optical_cells=d.mean((0,2)).tolist())
    (ROOT/'sign_repair62.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
