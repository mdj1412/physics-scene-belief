"""Partial identification over every binary completion of the unconfirmed contact labels."""
import itertools,json,os
import numpy as np
from scipy.stats import t
from crossover68 import H,ROUTES,FAMS,LAWS,archive,routed,loss

def envelope(lo,hi):
    # The lower CI endpoint is concave and the upper CI endpoint convex in catalog scores.
    # Their respective extrema over a hyperrectangle occur at its vertices.
    bits=np.array(list(itertools.product((False,True),repeat=len(lo))))
    x=np.where(bits,lo,hi); half=t.ppf(1-.025/6,len(lo)-1)*x.std(1,ddof=1)/np.sqrt(len(lo))
    return [float(lo.mean()),float(hi.mean()),float((x.mean(1)-half).min()),float((x.mean(1)+half).max())]


def main():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=np.zeros((12,2,2,2,3))
    counts=np.zeros((2,2,2),dtype=int)
    for k in range(12):
        sub=f'panel_{k:02d}';rec=archive(H/'forecasts'/'compliant'/(sub+'.npz'))
        for name,ix in rec['ix'].items():
            f,_,l,v,h=name.removesuffix('.npz').split('_')
            a,b=FAMS.index(f),LAWS.index(l)
            with np.load(H/'labels_compliant'/sub/name) as z:
                y=z['labels'].astype(float); unknown=z['unresolved'].astype(bool)
            with np.load(ROUTES/sub/f'{f}_{v}_{h}.npz') as z: p=routed(rec['p'][ix],rec['shift'][ix],z['route'])
            baseline=loss(p,y)
            counts[a,b]+=[unknown.sum(),unknown.size]
            for i,(x,w) in enumerate([(1,0),(4,3)]):
                nominal=baseline[x]-baseline[w]
                slope=-2*(p[:,x]-p[:,w]).mean((0,4))[None]
                lower=nominal+np.mean(unknown*(np.minimum(0.,slope)-slope*y))
                upper=nominal+np.mean(unknown*(np.maximum(0.,slope)-slope*y))
                out[k,a,b,i]+=[n/12 for n in (nominal,lower,upper)]
    cases={}
    for li,l in enumerate(LAWS):
        rows=out[:,:,li]
        for title,scale in [('wendland_soft',rows[:,0,0]),('dispersive_soft',rows[:,1,0]),('pooled_rescue',rows[:,:,1].mean(1))]:
            cases[l+'_'+title]=envelope(scale[:,1],scale[:,2])
    data=dict(bounds=cases,columns=['lowest possible mean','highest possible mean','lowest simultaneous interval endpoint','highest simultaneous interval endpoint'],
              order=[FAMS,LAWS],unresolved_and_total=counts.tolist(),catalog_bounds=out.tolist(),job=os.environ['SLURM_JOB_ID'])
    (H/'unresolved_bounds68.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
