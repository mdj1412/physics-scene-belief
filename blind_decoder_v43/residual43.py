"""Supplementary controls for clipping, composition with T and permutation of environment cues."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from stats43 import interval, NONZERO
from model43 import load_model

ROOT = Path(__file__).resolve().parent


def outcome(p,y):
    return (p*p - 2*p*y + y).mean(axis=(-3,-2,-1))


def main():
    model=load_model()
    source=json.loads((ROOT/'source_validation43.json').read_text())
    rows=[]
    for panel in range(12):
        cells=[]
        files=sorted((ROOT/'results'/f'panel_{panel:02d}').glob('*.npz'))
        assert len(files)==48
        for file in files:
            with np.load(file) as a:
                y=a['labels'][NONZERO].mean(axis=0)[...,None]
                p,pt,po=a['pred'][:,0],a['pred'][:,1],a['pred'][:,3]
                tau=np.array([model['temperature'][k] for k in ('original','robust','privileged')])[:,None,None,None]
                z=source['beta']*a['z']*a['weights']
                unclip=expit(logit(np.clip(p,0,1))/tau)
                t_then_opt=expit(logit(pt)+z[:,None,None])
                # Multiplicative odds allows p=0 or p=1 in the original policy.
                mixed=[]
                for j in (1,2,3):
                    scale=np.exp(np.roll(z,j))[:,None,None]
                    mixed.append(outcome(p*scale/(1-p+p*scale),y))
                cells.append(np.stack((
                    ((p<1e-5)|(p>1-1e-5)).mean(axis=(1,2,3)),
                    outcome(unclip,y)-outcome(pt,y),
                    outcome(t_then_opt,y)-outcome(pt,y),
                    outcome(po,y)-np.mean(mixed,axis=0),
                ),axis=-1))
        rows.append(np.mean(cells,axis=0))
    rez=dict(names=['outside_clip_fraction','unclipped_T_minus_T','T_plus_opt_minus_T',
                    'opt_minus_permuted_opt'], decoders=['original','robust','privileged'],
             paired_ci=interval(rows).tolist())
    (ROOT/'residual43.json').write_text(json.dumps(rez,indent=2)+'\n')


if __name__=='__main__':
    main()
