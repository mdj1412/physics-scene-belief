"""Conditional confidence-vs-lottery contrast when some but not all valid pairs are kept."""
from pathlib import Path
import os,json
import numpy as np
assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
root=Path(__file__).resolve().parent
with np.load(root/'thinning62_cells.npz') as f:
    x=f['catalog_cell_board_method_metrics']
    r=f['readout_counts']
# Column 7 is any rejection, column 6 is the loss of the last valid read.
p=(r[...,7]-r[...,6]).mean(1)
d=(x[:,:,:,1,1]-x[:,:,:,2,1]).mean(1)
assert (p>=0).all()
rng=np.random.default_rng(6209533)
wc=rng.multinomial(24,np.full(24,1/24),size=8000)/24
wb=rng.multinomial(8,np.full(8,1/8),size=8000)/8
def boot(v):return np.einsum('dc,db,cb->d',wc,wb,v,optimize=True)
bp=boot(p);bd=boot(d)
out={'fraction_with_0_lt_k_lt_n':[p.mean(),*np.quantile(bp,[.025,.975])],
     'conditional_mean_delta':[d.mean()/p.mean(),*np.quantile(bd/bp,[.025,.975])]}
(root/'conditional62.json').write_text(json.dumps(out,indent=2)+'\n')
