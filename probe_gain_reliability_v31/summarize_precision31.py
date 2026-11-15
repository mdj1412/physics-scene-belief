"""Replicate count implied by an RMS precision target, keeping the sampling model explicit."""
from pathlib import Path
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parent


def required_equal_axes(v, target):
    vf,vb,vi=v
    # Seek first integer rather than relying on signs of small unrestricted estimates.
    for n in range(1,2049):
        variance=(vf+vb)/n+vi/n**2
        if 0<=variance<=target**2:
            return n
    return None


def main():
    by_panel={}
    for name in ['results','results_pool']:
        j=json.loads((ROOT/name/'crossed_stats31.json').read_text())
        comps=np.array(j['pooled']['components']['estimate'])
        out=[]
        for v in comps:
            out.append(dict(balanced_n={str(t):required_equal_axes(v,t) for t in [.003,.002,.001]},
                rms_at_four=math.sqrt((v[0]+v[1])/4+v[2]/16)))
        by_panel[name]=out
    by_panel['assumption']='Plug-in projection from 4×4 random-effects expectation for conditionally independent futures and batches. Conditional RMS of one prefix gain estimate, averaged as variance over all prefixes. Not an interval, policy reliability guarantee, or proposal to spend that budget on a deployed robot.'
    (ROOT/'results/precision31.json').write_text(json.dumps(by_panel,indent=2))
if __name__=='__main__':
    main()
