"""Compare access to the cause label, paid observations, and surface-level action values."""
from pathlib import Path
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent / 'sensor_reset_v59'))
from route59 import TABLE, SOURCE_LOG_CUTOFF, log_odds

MIX = np.array([[.42,0,.58],[0,.84,.16]])
NAMES = ['source_frozen','fault_marginalized','cause_and_paid','cause_alone','full_target']
CATALOGS, BOARDS = 24,8

def compute(c):
    # All methods receive the same corrected reset recording. Axes are timing, P/C,
    # source A/B, material, episode, and the four sessions.
    with np.load(ROOT/'pilots'/f'panel_{c:02d}.npz') as p:
        code=p['codes'][:,2,1]
    with np.load(ROOT/'scored61'/f'panel_{c:02d}.npz') as f:
        split=((np.arange(4)+c)%4)//2
        base=np.stack([f['exact_cells'][:,:,split==j,:2].mean((1,2)) for j in range(2)])
    risk=np.einsum('mea,ze->mza',base,MIX)
    # All three log-odds use the original likelihood to avoid refitting on targets.
    l=TABLE[1,1][:,:,code].sum(-1)
    shared=np.logaddexp(l[1,0],l[1,1])-np.log(2.)
    marginalized=np.logaddexp(l[0,0],shared)-np.logaddexp(l[0,1],shared)
    full=log_odds(code,1,True)[0]
    true_cause=l[0,0]-l[0,1]
    true_cause[:,1]=0
    methods=np.stack([full>SOURCE_LOG_CUTOFF,marginalized>SOURCE_LOG_CUTOFF,
                      true_cause>SOURCE_LOG_CUTOFF,np.zeros_like(full,dtype=bool)])
    by_board=[]
    for board in range(BOARDS):
        # method, cause, source regime, material
        rates=methods[...,board::BOARDS].mean(axis=(1,-1))
        rates[:,1]=rates[:,1].mean(axis=1,keepdims=True)
        # A posteriori optimal target action is allowed to differ for each material.
        oracle=(risk[:,:,1]<risk[:,:,0]).T
        rates=np.concatenate([rates,np.broadcast_to(oracle,(1,2,2,2))])
        gate=risk[:,:,0].T
        gap=(risk[:,:,1]-risk[:,:,0]).T
        regret=(rates*gap).mean(axis=(2,3))
        by_board.append(np.stack([gate.mean()+regret.mean(1),regret.mean(1),
                                  rates.mean(axis=(1,2,3)),regret[:,0],regret[:,1]]))
    return np.asarray(by_board)

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    x=np.stack([compute(c) for c in range(CATALOGS)])
    rng=np.random.default_rng(6108179)
    w_c=rng.multinomial(CATALOGS,np.full(CATALOGS,1/CATALOGS),size=8000)/CATALOGS
    w_b=rng.multinomial(BOARDS,np.full(BOARDS,1/BOARDS),size=8000)/BOARDS
    boot=np.einsum('dc,db,cbim->dim',w_c,w_b,x,optimize=True)
    cols=['mean_Brier','delta_gate','fixed_rate','delta_in_P','delta_in_C']
    out={'methods':{},'comparisons':{}}
    def summary(point,v):
        return [float(point),*np.quantile(v,[.025,.975]).tolist()]
    for n,name in enumerate(NAMES):
        out['methods'][name]={col:summary(x[:,:,i,n].mean(),boot[:,i,n]) for i,col in enumerate(cols)}
    for i,j in ((1,0),(2,1),(2,0)):
        out['comparisons'][NAMES[i]+'_minus_'+NAMES[j]]=summary(
            (x[:,:,1,i]-x[:,:,1,j]).mean(),boot[:,1,i]-boot[:,1,j])
    (ROOT/'value61.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ROOT/'value61_cells.npz',catalog_board_metrics=x,metrics=cols,methods=NAMES)
if __name__=='__main__':
    main()
