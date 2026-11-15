"""Descriptive stability of v31 point estimates to omission of a block or one law."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def contrast(c,r):
    return np.stack((c[...,1]-c[...,0],r[...,2]),axis=-1)


def main():
    j=json.loads((OUT/'crossed_stats31.json').read_text())
    c=np.array(j['block_rows']['components'])
    r=np.array(j['block_rows']['ranks'])
    perblock=contrast(c,r)
    permutations=[]
    for law in range(len(j['laws'])):
        for block in range(c.shape[1]):
            rows=[perblock[i].mean(0) if i!=law else np.delete(perblock[i],block,axis=0).mean(0)
                  for i in range(len(j['laws']))]
            permutations.append(np.mean(rows,axis=0))
    leave_one_law=np.stack([np.delete(perblock,j,axis=0).mean((0,1)) for j in range(len(c))])
    # Applies the same alpha/prefix panel conditions as the primary analysis.
    output=dict(order=['batch_minus_future_variance','independent_rank_concordance'],
        leave_one_block_min=np.min(permutations,axis=0).tolist(),
        leave_one_block_max=np.max(permutations,axis=0).tolist(),
        leave_one_law=leave_one_law.tolist(),omitted_law=j['laws'],
        note='Descriptive sensitivity to sampled blocks and prespecified law groups; leave-one-law has only three cases.')
    (OUT/'sensitivity31.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(output)
if __name__=='__main__':
    main()
