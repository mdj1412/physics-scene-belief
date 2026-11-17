"""Release target outcomes only after source/forecast seal."""
import itertools
import os
import sys
from pathlib import Path
import numpy as np
from mechanics68 import simulate, RADII, OFF
H=Path(__file__).resolve().parent
SRC=H.parent/'invariance_audit_v66'/'locked'
BACK=('compliant','prescribed')
FAM=('wendland','dispersive')
LAWS=('strengthening','weakening')

def main(indices):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert (H/'FORECASTS_SEALED').exists()
    for i in indices:
        sub=f'panel_{i:02d}'
        with np.load(SRC/'results'/'worlds'/(sub+'.npz')) as z:
            th=z['theta'].reshape(512,6)
            sig=np.repeat(z['signs'],128)
            reg=np.repeat(z['regimes'],128)
        fine_indices=np.arange(512).reshape(4,128)[:,:32].ravel()
        for back in BACK:
            for f,s,l,v,h in itertools.product(FAM,('inlay','woven'),LAWS,('base','hollow','bulge'),(19,31)):
                fn=f'{f}_{s}_{l}_{v}_h{h}.npz'
                where=H/('labels_'+back)/sub
                where.mkdir(exist_ok=True,parents=True)
                path=where/fn
                y=np.empty((2,3,4,128),dtype=np.float32)
                yf=np.empty((2,3,4,32),dtype=np.float32)
                for ir,io in itertools.product(range(2),range(3)):
                    _,speed=simulate(th,h+OFF[io],.425,f,v,s,l,sig,reg,RADII[ir],back)
                    _,fine=simulate(th[fine_indices],h+OFF[io],.425,f,v,s,l,
                                    sig[fine_indices],reg[fine_indices],RADII[ir],back,True)
                    y[ir,io]=speed.reshape(4,128)
                    yf[ir,io]=fine.reshape(4,32)
                tmp=path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,labels=y<=.020,slip_speed=y,fine_labels=yf<=.020,fine_slip_speed=yf,
                                    centers_mm=h+np.array(OFF),radii_mm=RADII,fine_prototype_indices=np.arange(32))
                os.replace(tmp,path)
        print('completed labels',i,flush=True)
if __name__=='__main__':
    main([int(s) for s in sys.argv[1:]])
