"""Post-primary: spatial/temporal refinement without fitting or changing any forecasts."""
from functools import lru_cache
import hashlib
from itertools import product
import json
import os
import sys
import ctypes
from pathlib import Path
import numpy as np
import mechanics68 as m
from solver_labels68 import H,SRC,FAM,LAWS,RADII,OFF

def install():
    standard=m.kernel()
    dlls=list(H.glob('extended_*.so'))
    assert len(dlls)==1
    dll=ctypes.CDLL(str(dlls[0]))
    longer=dll.compliant68
    longer.argtypes=standard.argtypes;longer.restype=None
    flag={'level':2,'unresolved':None}
    counts={'calls':0,'retries':0,'unresolved':0,'super_calls':0,'super_unresolved':0}
    def ker(n,sites,res,gaps,g,mu,e,v,gc,out):
        superlevel=flag['level']==3
        res=96 if superlevel else 48
        assert sites==(33 if superlevel else 17)
        standard(n,sites,res,gaps,g,mu,e,v,gc,out)
        retry=out[:,5]==0
        counts['calls']+=n;counts['retries']+=int(retry.sum())
        if retry.any():
            long=np.empty((int(retry.sum()),6))
            longer(len(long),sites,res,*[np.ascontiguousarray(z[retry]) for z in (gaps,g,mu,e,v,gc)],long)
            out[retry,:3]=long[:,:3];out[retry,5]=long[:,5]
        flag['unresolved']=out[:,5]==0
        counts['unresolved']+=int(flag['unresolved'].sum())
        if superlevel:
            counts['super_calls']+=n;counts['super_unresolved']+=int(flag['unresolved'].sum())
        out[:,5]=1.
    basegrid=m.grid
    @lru_cache(None)
    def thirds(h,r,f,v):
        dx=np.linspace(-r,r,33)
        yy=h+dx
        dh,deriv=m.profile(yy,f,v);f0,center=m.profile(h,f,v)
        return dx,yy,None,-dh+f0+center*dx,deriv,float(center)
    def grid(backend,h,r,f,v,fine):
        return thirds(h,r,f,v) if flag['level']==3 else basegrid(backend,h,r,f,v,True)
    m.grid=grid;m.kernel=lambda:ker
    return flag,counts,hashlib.sha256(dlls[0].read_bytes()).hexdigest()

def main(indices):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert (H/'LABELS_READY').is_file()
    flag,counts,dll=install()
    for k in indices:
        sub=f'panel_{k:02d}'
        with np.load(SRC/'results'/'worlds'/(sub+'.npz')) as z:
            th=z['theta'].reshape(512,6);sg=np.repeat(z['signs'],128);rg=np.repeat(z['regimes'],128)
        ids0=np.arange(512).reshape(4,128)
        folder=H/'labels_compliant_refined'/sub;folder.mkdir(exist_ok=True,parents=True)
        for f,s,l,v,h in product(FAM,('inlay','woven'),LAWS,('base','hollow','bulge'),(19,31)):
            fn=f'{f}_{s}_{l}_{v}_h{h}.npz'
            rows={}
            for level,ns,prefix in ((2,128,''),(3,8,'super_')):
                flag['level']=level;ids=ids0[:,:ns].ravel()
                a=np.empty((2,3,4,ns),dtype=np.float32);u=np.empty_like(a,bool)
                for ir,io in product(range(2),range(3)):
                    _,vt=m.simulate(th[ids],h+OFF[io],.425,f,v,s,l,sg[ids],rg[ids],RADII[ir],'compliant',True)
                    a[ir,io]=vt.reshape(4,ns);u[ir,io]=flag['unresolved'].reshape(4,ns)
                rows.update({prefix+'slip_speed':a,prefix+'labels':a<=.020,prefix+'unresolved':u})
            with np.load(H/'labels_compliant'/sub/fn) as z:
                for fld in ('labels','slip_speed','unresolved'):
                    np.testing.assert_array_equal(z['fine_'+fld],rows[fld][...,:32])
            tmp=folder/(fn+'.tmp.npz');np.savez_compressed(tmp,**rows);os.replace(tmp,folder/fn)
        print('refined',k,flush=True)
    (H/f'refined_worker_{indices[0]}.json').write_text(json.dumps(dict(counts=counts,library_sha=dll),indent=2)+'\n')

if __name__=='__main__':main(list(map(int,sys.argv[1:])))
