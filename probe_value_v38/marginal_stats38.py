"""Average known/hidden reference comparisons on the prespecified first sensor draw."""
import json
import numpy as np
from stats38 import ROOT,CELLS,samples,draw,limits

def main():
    data=[]
    rb=[]
    for b in range(8):
        f=json.loads((ROOT/'marginal_blocks'/f'block_{b:03d}.json').read_text())
        assert tuple(f['cells'])==CELLS and f['replicate']==0
        bycell=[];brb=[]
        for cell in CELLS:
            items=f['cases'][cell]
            assert len(items)==48
            bycell.append([np.mean([r['known_gain'] for r in items]),
                           np.mean([r['hidden_gain'] for r in items])])
            brb.append(np.stack([r['rb_policy_gain'] for r in items]).mean(0))
        data.append(bycell);rb.append(brb)
    v=np.swapaxes(np.asarray(data),0,1)
    b=draw(v,samples())
    g=v.mean((0,1))
    rb=np.swapaxes(np.asarray(rb),0,1)
    brep=draw(rb,samples())
    ratio=rb.mean((0,1))/g[:,None]
    obj=dict(gain=g,ci=limits(b),difference=g[1]-g[0],difference_ci=limits(b[:,1]-b[:,0]),
             policy_rb_gain=rb.mean((0,1)),policy_rb_gain_ci=limits(brep),
             rb_fraction=ratio,rb_fraction_ci=limits(brep/b[:,:,None]))
    (ROOT/'marginal_stats38.json').write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist())+'\n')

if __name__=='__main__':
    main()
