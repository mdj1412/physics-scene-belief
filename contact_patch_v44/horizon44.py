"""Post-primary diagnostic; same two fixed objects, dt unchanged, horizon doubled."""
import itertools
import os
from pathlib import Path
import numpy as np
import solver44 as s

def main():
    root=s.ROOT; s.HORIZON=.0008
    s.b.connect(s.b.DIRECT)
    for panel,shape,variant,h in itertools.product(range(12),s.SHAPES,s.VARIANTS,(19,31)):
        with np.load(root/'results'/'worlds'/f'panel_{panel:02d}.npz') as a:
            th=a['theta'][0,:2];sg=np.repeat(a['signs'][0],2);reg=np.repeat(a['regimes'][0],2)
        cells={}
        for i,off in enumerate(s.OFFSETS):
            surface=s.Surface(th,h+off,shape,variant,s.DT,s.GRID_MM)
            for ir,r in enumerate(s.RADII):
                for iw,w in enumerate(s.SPANS):
                    _,site,g=surface.support(r,w)
                    for sf,law in itertools.product(s.SURFACES,s.LAWS):
                        mu=s.material_mu(th,h+off,site,g,sf,law,sg,reg)
                        row=surface.simulate(r,w,mu,th[:,5])
                        arr=cells.setdefault((sf,law),np.empty((2,2,3,2)))
                        assert row['valid'].all()
                        arr[ir,iw,i]=row['slip_speed']
        d=root/'labels44_horizon'/f'panel_{panel:02d}'
        d.mkdir(parents=True,exist_ok=True)
        for (sf,law),arr in cells.items():
            np.savez_compressed(d/f'{shape}_{sf}_{law}_{variant}_h{h}.npz',slip_speed=arr)
    s.b.disconnect()

if __name__=='__main__':
    main()
