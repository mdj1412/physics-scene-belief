"""Independent, uncertainty-aware scoring of post-primary numerical refinements."""
from itertools import product
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import t
from crossover68 import H,ROUTES,FAMS,LAWS,archive,routed,loss

def ci(x,m=6):
    a=x.mean(0);d=t.isf(.025/m,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
    return np.stack((a,a-d,a+d),-1).tolist()

def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert (H/'REFINED_READY').is_file() and (H/'fine_history'/'FORECASTS_SEALED').is_file()
    # Catalog, resolution/sample, family, law, contrast, provisional / min / max
    tab=np.zeros((12,4,2,2,2,3));nc=np.zeros((4,2,2,2),np.int64)
    for k in range(12):
        sub=f'panel_{k:02d}'
        rec=archive(H/'forecasts'/'compliant'/(sub+'.npz'))
        for fn,ix in rec['ix'].items():
            f,s,l,v,hs=fn.removesuffix('.npz').split('_')
            fi=FAMS.index(f);li=LAWS.index(l)
            with np.load(ROUTES/sub/f'{f}_{v}_{hs}.npz') as z:
                allp=routed(rec['p'][ix],rec['shift'][ix],z['route'])
            with np.load(H/'labels_compliant'/sub/fn) as c, np.load(H/'labels_compliant_refined'/sub/fn) as a:
                source=[(c,'',128),(a,'',128),(a,'',8),(a,'super_',8)]
                for j,(z,pref,ns) in enumerate(source):
                    y=z[pref+'labels'][...,:ns].astype(float)
                    u=z[pref+'unresolved'][...,:ns]
                    p=allp[...,:ns,:]
                    r=loss(p,y)
                    for m,(ma,mb) in enumerate(((1,0),(4,3))):
                        d=r[ma]-r[mb]
                        # Shared target label for all restarts and noise replicates of that impact.
                        move=(-2*(p[:,ma]-p[:,mb]).mean((0,4))[None])*(1-2*y)*u
                        tab[k,j,fi,li,m]+=np.array([d,d+np.minimum(move,0).mean(),d+np.maximum(move,0).mean()])/12
                    nc[j,fi,li]+=[y.size,int(u.sum())]
    methods=['standard_all','fine_all','fine_same8','super_same8']
    ro={}
    for j,m in enumerate(methods):
        ro[m]=dict(soft=ci(tab[:,j,:,:,0,0]),pooled_rescue=ci(tab[:,j,:,:,1,0].mean(1)),
         rescue_per_family=ci(tab[:,j,:,:,1,0],1),arbitrary_label_means=tab[:,j].mean(0).tolist(),
         counts=nc[j].tolist())
    obj=dict(families=FAMS,laws=LAWS,methods=methods,contrasts=['soft_minus_full','restored_minus_dropped'],
       summary=ro,paired_change_all=ci(tab[:,1,...,0]-tab[:,0,...,0],1),paired_change_same8=ci(tab[:,3,...,0]-tab[:,2,...,0],1),
       by_catalog=tab.tolist(),job=os.environ['SLURM_JOB_ID'])
    (H/'refined68.json').write_text(json.dumps(obj,indent=2)+'\n')
    print({k:(v['soft'],v['pooled_rescue']) for k,v in ro.items()})
if __name__=='__main__':main()
