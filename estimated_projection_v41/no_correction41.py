"""Paired post-hoc check against a decoder that applies zero correction."""
import json
import numpy as np
import stats41 as s

def main():
    primary=json.loads((s.ROOT/'stats41.json').read_text())
    corrected=np.array(primary['panel_loss'])
    base=np.zeros((s.N,3))
    for n in range(s.N):
        for cell in s.CELLS:
            for b in range(2):
                with np.load(s.block_path(cell,n,b)) as raw:
                    p=raw['p_base'];yb=raw['labels'].mean(1)[None,:,:,:,None]
                    base[n]+=((p-yb)**2+yb*(1-yb)).mean((1,2,3,4))/(2*len(s.CELLS))
    np.testing.assert_allclose(base.mean(0),json.loads((s.ROOT/'contour_breakdown41.json').read_text())['uncorrected']['mean'],atol=1e-14)
    delta=corrected-base[:,:,None]
    i=s.interval(delta)
    cross=s.interval(base[:,2]-base[:,0])
    out={'panel_uncorrected':base.tolist(),'paired_correction_minus_zero':{k:v.tolist() for k,v in i.items()},
         'wins_selector_vs_zero':(delta[:,:,2]<0).sum(0).tolist(),
         'estimated_minus_original_at_zero':{k:float(v) for k,v in cross.items()}}
    (s.ROOT/'no_correction41.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
