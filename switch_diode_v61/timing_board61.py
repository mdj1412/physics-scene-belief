"""Cluster sensitivity for the observed nested board/catalog timing schedule."""
from pathlib import Path
import json
import os
import numpy as np

root=Path(__file__).resolve().parent
assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
ms=[]
boards=[]
for i in range(24):
    with np.load(root/'timing61'/f'panel_{i:02d}.npz') as f:
        ms.append(f['ms'])
        unique=np.unique(f['labels'][:,1])
        assert unique.shape==(1,) and unique[0]==i%8
        boards.append(int(unique[0]))
ms=np.stack(ms)
boards=np.array(boards)
assert np.isfinite(ms).all() and (ms>0).all()
by_board=np.stack([np.flatnonzero(boards==b) for b in range(8)])
rng=np.random.default_rng(6109539)
selected_boards=rng.integers(8,size=(8000,8))
catalogs=by_board[selected_boards[:,:,None],rng.integers(3,size=(8000,8,3))].reshape(-1,24)
p90=np.quantile(ms,.9,axis=(0,1))
boot=np.quantile(ms[catalogs],.9,axis=(1,2))
diff=boot[:,0:1]-boot[:,1:3]
blocks=ms[...,:1]-ms[...,1:3]
# A median paired difference is a distinct, supplementary endpoint.
out={'p90_ms':p90.tolist(),'p90_difference_vs_sham_gate_simultaneous95':
     np.quantile(diff,[.0125,.9875],axis=0).T.tolist(),
     'median_block_difference_vs_sham_gate':np.median(blocks,axis=(0,1)).tolist(),
     'board_p90_difference_vs_sham_gate':[],
     'leave_one_board_out_delta_range':None}
drop=[]
for b in range(8):
    q1=np.quantile(ms[boards==b],.9,axis=(0,1))
    q2=np.quantile(ms[boards!=b],.9,axis=(0,1))
    out['board_p90_difference_vs_sham_gate'].append((q1[0]-q1[1:3]).tolist())
    drop.append(q2[0]-q2[1:3])
out['leave_one_board_out_delta_range']=np.array([np.min(drop,axis=0),np.max(drop,axis=0)]).T.tolist()
(root/'timing_board61.json').write_text(json.dumps(out,indent=2)+'\n')
