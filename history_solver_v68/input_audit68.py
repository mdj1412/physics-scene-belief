"""Audit which velocity measurements actually change under paired law replacement."""
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    results={}
    for backend in ('prescribed','compliant'):
        agg=defaultdict(lambda:np.zeros(5))
        for k in range(12):
            folder=H/'tapes'/backend/f'panel_{k:02}'
            for a in folder.glob('*_strengthening_*_h19.npz'):
                b=folder/a.name.replace('strengthening','weakening')
                with np.load(a) as f,np.load(b) as g:
                    for name,z in [('xx',np.s_[...,-2:]),('cx',np.s_[...,-2:]),('cy',np.s_[:]),
                                   ('probe',np.s_[...,:1,:]),('query',np.s_[...,:1,:])]:
                        d=f[name][z]-g[name][z]
                        vec=d.reshape(-1,d.shape[-1])
                        agg[name]+=np.array([np.sum(vec**2),vec.size,
                            np.count_nonzero(np.max(np.abs(vec),axis=-1)>1e-8),len(vec),abs(vec).sum()])
                    np.testing.assert_array_equal(f['xx'][...,:9],g['xx'][...,:9]);np.testing.assert_array_equal(f['cx'][...,:9],g['cx'][...,:9])
                    np.testing.assert_array_equal(f['query'],g['query']);np.testing.assert_array_equal(f['dirs'],g['dirs'])
        results[backend]={name:dict(velocity_component_rms=float(np.sqrt(ss/n)),fraction_changed=float(c/nn),n_rows=int(nn)) for name,(ss,n,c,nn,_ab) in agg.items()}
    (H/'input_audit68.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
