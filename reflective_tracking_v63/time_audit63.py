"""Sensor and routing wall-time on fresh raw fixtures; separate from the GPU benchmark."""
import json
import os
import socket
import time
from pathlib import Path
import numpy as np

from bench63 import choice,plan,verify,METHODS,ROOT

def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    m=np.arange(2)
    assignment=plan()
    out=np.empty((24,4,7,3))
    for i in range(24):
        with np.load(ROOT/'fixtures'/f'panel_{i:02d}.npz') as file:
            pix,w,elec,ids=(file[k] for k in ('frames','wave','electronic','reset_id'))
            ep=int(file['episode'])
        with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as file:
            codes=file['codes'][:,:,:,:,:,:,ep]
        for row in range(4):
            d,cause,target,z=assignment[i,row]
            frames=pix[d,cause,z,m]
            adc=w[d]
            pulse=np.stack([elec[:,cause[j],z[j],j] for j in m])
            for r in range(8):
                gen=np.random.default_rng([6371941,i,d,r])
                for arm in gen.permutation(3):
                    start=time.perf_counter_ns()
                    dec,code=choice(frames,adc,pulse,ids,arm,gen)
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    fixed=verify(code,arm,codes[d,1,1,cause,z,m],codes[d,2,1,cause,z,m])
                    np.testing.assert_array_equal(dec,fixed)
                    if r: out[i,d,r-1,arm]=elapsed
    np.savez_compressed(ROOT/'reader_times63.npz',ms=out)
    val=np.quantile(out,.9,axis=(0,1,2))
    rng=np.random.default_rng(6370213)
    # Chunk the bootstrap to fit a single-CPU job's memory limit.
    boot=[]
    for _ in range(60):
        idx=rng.integers(24,size=(100,24))
        boot.append(np.quantile(out[idx],.9,axis=(1,2,3)))
    boot=np.concatenate(boot);delta=boot[:,[0]]-boot[:,1:]
    outcome=dict(node=socket.gethostname(),methods=METHODS,cpu_p90_ms=val.tolist(),
        confidence_minus_others_ms=(val[0]-val[1:]).tolist(),two_contrasts_simultaneous95=
        np.quantile(delta,[.0125,.9875],axis=0).T.tolist(),calls_per_method=24*4*7)
    (ROOT/'reader_cost63.json').write_text(json.dumps(outcome,indent=2)+'\n')

def nested():
    ms=np.array([np.load(ROOT/'timing63'/f'panel_{i:02d}.npz')['ms'] for i in range(24)])
    gen=np.random.default_rng(6390218)
    b=gen.integers(8,size=(16000,8,1))
    cat=(b+8*gen.integers(3,size=(len(b),8,3))).reshape(len(b),24)
    vals=np.quantile(ms[cat],.9,axis=(1,2))
    diff=vals[:,[0]]-vals[:,1:]
    outcome=dict(unit='boards and catalogs within sampled boards',replications=len(b),
        two_contrasts_simultaneous95=np.quantile(diff,[.0125,.9875],axis=0).T.tolist())
    (ROOT/'gpu_board63.json').write_text(json.dumps(outcome,indent=2)+'\n')

if __name__=='__main__':
    import sys
    main() if sys.argv[1]=='reader' else nested()
