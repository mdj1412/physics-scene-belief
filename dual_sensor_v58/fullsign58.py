"""Exploratory audit of information discarded by reducing four raw signs to two mismatches."""
from itertools import product
from pathlib import Path
import json
import os
import numpy as np
from route58 import TABLE,Q,SECOND,ARTIFACT,SOURCE_LOG_CUTOFF

ROOT = Path(__file__).resolve().parent
ERROR = .02

def observed_object(m0,m1):
    # Relative deformation mismatch each instrument would report before read noise.
    patterns=(np.arange(16)[:,None]>>np.arange(4))&1
    p=np.zeros(16)
    for ref in (0,1):
        noiseless=np.array([ref,ref,ref^m0,ref^m1])
        p += .5*np.prod(np.where(patterns==noiseless,1-ERROR,ERROR),axis=1)
    return p

def make_table(regime,rho,hi):
    h=hi*ARTIFACT
    modes=[(-1,1-h),(0,h*(1-Q)),(1,h*Q)]
    m=SECOND[regime] if rho else Q**2
    phys=np.array([[1-2*Q+m,Q-m],[Q-m,m]])
    p=np.zeros((17,17))
    emissions=[[observed_object(a,b) for b in (0,1)] for a in (0,1)]
    for x,y in product(range(2),repeat=2):
        for a,pa in modes:
            for b,pb in modes:
                first=emissions[x if a<0 else a][x if b<0 else b]
                second=emissions[y if a<0 else a][y if b<0 else b]
                p[1:,1:] += phys[x,y]*pa*pb*np.outer(first,second)
    p[0,1:]=p[1:,1:].sum(0)
    p[1:,0]=p[1:,1:].sum(1)
    p[0,0]=1.
    # Summing raw patterns that produce the same pair of mismatches must recover the primary instrument.
    bits=(np.arange(16)[:,None]>>np.arange(4))&1
    code=np.r_[0,1+(bits[:,0]^bits[:,2])+2*(bits[:,1]^bits[:,3])]
    collapsed=np.zeros((5,5))
    np.add.at(collapsed,(code[:,None],code[None,:]),p)
    assert np.allclose(collapsed.ravel(),np.exp(TABLE[regime,rho,hi]),rtol=1e-11,atol=1e-13)
    return np.log(p.ravel())

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    tables=np.array([make_table(*p) for p in product(range(2), repeat=3)]).reshape(2,4,289)
    rates=[]
    for file in sorted((ROOT/'pilot_audit').glob('panel_*.npz')):
        with np.load(file) as f:
            raw=f['raw_signed'][1].astype(np.int32) # randomized two-object design
        # In the file 0 denotes missing, 16..31 are the four paid signs.
        sym=np.where(raw==0,0,(raw&15)+1)
        code=sym[...,0]*17+sym[...,1]
        def marginal(z):
            return np.logaddexp.reduce(tables[z][:,code].sum(-1),axis=0)
        llr=marginal(0)-marginal(1)
        rates.append(np.stack((llr>1e-12,llr>SOURCE_LOG_CUTOFF),axis=-1).mean(-2).transpose(3,0,1,2,4))
    rates=np.array(rates).mean((2,3))
    with np.load(ROOT/'score58_cells.npz') as f:
        reduced=f['route'].mean((2,3))[...,[0,2]]
        data=f['exact_by_e']
    mix=np.array([[.42,0,.58],[0,.84,.16]])
    cell_risk=np.einsum('cfea,ze->cfza',data,mix)
    def score(r):
        accuracy=.5*(r[:,:,0]+1-r[:,:,1]).mean(1)
        loss=(cell_risk[...,0,None]+r*(cell_risk[...,1,None]-cell_risk[...,0,None])).mean((1,2))
        return np.stack((accuracy,loss),axis=-1)
    x,y=score(rates),score(reduced)
    draws=np.random.default_rng(58061703).integers(24,size=(5000,24))
    delta=x-y
    out=dict(actions=['MAP','source-risk'],metrics=['balanced_accuracy','mean_Brier'],full=x.mean(0).tolist(),
             reduced=y.mean(0).tolist(),delta=delta.mean(0).tolist(),
             pointwise95=np.quantile(delta[draws].mean(1),[.025,.975],axis=0).tolist())
    (ROOT/'fullsign58.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    main()
