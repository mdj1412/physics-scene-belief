"""Paired uncertainty for the diagnostic best reachable same-k selector vs the fixed gate."""
import os,json
from pathlib import Path
import numpy as np
assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
p=Path(__file__).resolve().parent
with np.load(p/'headroom63_cells.npz') as f:
    x=f['catalog_cell_board']; keys=list(f['metrics'])
d=x[...,keys.index('oracle_Brier')]-x[...,keys.index('gate_Brier')]
rng=np.random.default_rng(630771)
cw=rng.multinomial(24,np.full(24,1/24),size=6000)/24
bw=rng.multinomial(8,np.full(8,1/8),size=6000)/8
boot=np.einsum('nc,nb,cb->n',cw,bw,d.mean(1),optimize=True)
(p/'gate_bound63.json').write_text(json.dumps(dict(oracle_minus_gate=float(d.mean()),
    pointwise_ci=np.quantile(boot,[.025,.975]).tolist(),cell_points=d.mean((0,2)).tolist()),indent=2)+'\n')
