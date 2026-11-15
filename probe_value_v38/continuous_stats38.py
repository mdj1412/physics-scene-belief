"""Jointly resample estimated information and continuous-policy gain."""
import json
import numpy as np
from stats38 import ROOT, CELLS, samples, draw, limits

s=json.loads((ROOT/'continuous38.json').read_text())
d=np.stack([[np.load(ROOT/'oracle_results'/c/f'block_{b:03d}/oracle_summary.npz')['gain'].mean()
             for b in range(8)] for c in CELLS])
idx=samples()
out={}
for k in ('block_gain','block_rb_gain'):
    v=np.array(s[k])
    out[k]=dict(fraction=v.mean()/d.mean(),ci=limits(draw(v,idx)/draw(d,idx)))
(ROOT/'continuous_recovery38.json').write_text(json.dumps(out,indent=2,default=lambda a:a.tolist())+'\n')
