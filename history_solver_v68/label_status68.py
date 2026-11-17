"""Return finite-horizon speed and a separate confirmed-separation status."""
import itertools
import json
import os
import sys
import numpy as np
from pathlib import Path
from solver_labels68 import H,SRC,BACK,FAM,LAWS,RADII,OFF
import mechanics68
import ctypes, fcntl, subprocess, hashlib
import mechanics68 as mechanics
LATEST={}
def setup():
    original=mechanics.kernel()
    src=(H/'compliant68.cpp').read_text()
    assert src.count('step<30*resolution')==1
    # Extra output slot is reused to distinguish a later contact episode from slow continuous loading.
    alt=src.replace('step<30*resolution','step<300*resolution')
    alt=alt.replace('bool started=false, released=false;','bool started=false, released=false; int loaded_steps=0, idle_after_contact=0, last=0;')
    alt=alt.replace('started=true;', 'started=true; ++loaded_steps; last=step;')
    alt=alt.replace('bool separating=true;', '++idle_after_contact; bool separating=true;')
    alt=alt.replace('out[6*i+3]=jtotal; out[6*i+4]=maxd;', 'out[6*i+3]=double(last+1)/resolution/omega; out[6*i+4]=idle_after_contact;')
    code=H/('extended_'+hashlib.sha256(alt.encode()).hexdigest()[:12]+'.cpp')
    lib=code.with_suffix('.so')
    with (H/'compiler.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not lib.exists():
            code.write_text(alt)
            tmp=lib.with_suffix('.tmp')
            subprocess.run(['g++','-O3','-std=c++11','-shared','-fPIC',str(code),'-o',str(tmp)],check=True)
            tmp.replace(lib)
    altfun=ctypes.CDLL(str(lib)).compliant68
    altfun.argtypes=original.argtypes;altfun.restype=None
    stats={k:dict(total=0,retry=0,still_open=0,with_free_steps=0,longest_last_force_s=0.,max_idle_steps=0) for k in (24,48)}
    def robust(n,sites,res,gaps,g,mu,e,v,gc,out):
        original(n,sites,res,gaps,g,mu,e,v,gc,out); retry=out[:,5]==0
        rec=stats[res]; rec['total']+=n;rec['retry']+=int(retry.sum())
        if retry.any():
            nr=int(retry.sum())
            longer=np.zeros((nr,6),float)
            altfun(nr,sites,res,*[np.ascontiguousarray(a[retry]) for a in (gaps,g,mu,e,v,gc)],longer)
            rec['still_open']+=int((longer[:,5]==0).sum())
            rec['with_free_steps']+=int((longer[:,4]>1).sum())
            rec['longest_last_force_s']=max(rec['longest_last_force_s'],float(longer[:,3].max()))
            rec['max_idle_steps']=max(rec['max_idle_steps'],int(longer[:,4].max()))
            out[retry,:3]=longer[:,:3];out[retry,5]=longer[:,5]
            if (longer[:,5]==0).any():
                opened=longer[longer[:,5]==0]
                rec['open_last_force_max_s']=max(rec.get('open_last_force_max_s',0.),float(opened[:,3].max()))
                rec['open_last_force_beyond_3ms']=rec.get('open_last_force_beyond_3ms',0)+int((opened[:,3]>.003).sum())
                rec['open_force_near_end']=rec.get('open_force_near_end',0)+int((opened[:,3]>.032).sum())
        LATEST['unresolved']=out[:,5]==0
        LATEST['extended']=retry
        out[:,5]=1.
    mechanics.kernel=lambda:robust
    return stats

def run(ix,stats):
    for i in ix:
        sub=f'panel_{i:02d}'
        with np.load(SRC/'results'/'worlds'/(sub+'.npz')) as z:
            th=z['theta'].reshape(512,6);sg=np.repeat(z['signs'],128);rg=np.repeat(z['regimes'],128)
        fine_ids=np.arange(512).reshape(4,128)[:,:32].ravel()
        for back in BACK:
            for f,s,l,v,h in itertools.product(FAM,('inlay','woven'),LAWS,('base','hollow','bulge'),(19,31)):
                fn=f'{f}_{s}_{l}_{v}_h{h}.npz'
                where=H/('labels_'+back)/sub
                where.mkdir(exist_ok=True,parents=True)
                rows={}
                for fine,ids in [(False,np.arange(512)),(True,fine_ids)]:
                    prefix='fine_' if fine else ''
                    dims=(2,3,4,len(ids)//4)
                    y=np.empty(dims,dtype='f4');u=np.zeros(dims,bool);ex=u.copy()
                    for ir,io in itertools.product(range(2),range(3)):
                        _,speed=mechanics68.simulate(th[ids],h+OFF[io],.425,f,v,s,l,sg[ids],rg[ids],RADII[ir],back,fine)
                        y[ir,io]=speed.reshape(4,-1)
                        if back=='compliant':
                            u[ir,io]=LATEST['unresolved'].reshape(4,-1)
                            ex[ir,io]=LATEST['extended'].reshape(4,-1)
                    rows.update({prefix+'labels':y<=.020,prefix+'slip_speed':y,prefix+'unresolved':u,prefix+'extended':ex})
                path=where/fn;tmp=path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,**rows,centers_mm=h+np.array(OFF),radii_mm=RADII,fine_prototype_indices=np.arange(32))
                os.replace(tmp,path)
        print('completed labels',i,flush=True)
        (H/f'status_worker_{ix[0]}.json').write_text(json.dumps(stats,indent=2)+'\n')
if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert (H/'FORECASTS_SEALED').is_file()
    run(list(map(int,sys.argv[1:])),setup())
