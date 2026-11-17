"""For failed releases only, extend the maximum time without changing any force calculation."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import fcntl
import sys
import numpy as np
import mechanics68 as mechanics
from solver_labels68 import main,H

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
            if (longer[:,5]==0).any():
                ids=np.flatnonzero(retry)[longer[:,5]==0]
                fn=H/f'nonrelease_{os.getpid()}.npz'
                np.savez(fn,grade=g[ids],gap=gaps[ids],mu=mu[ids],restitution=e[ids],incoming=v[ids],normal=gc[ids],retry_out=longer[longer[:,5]==0])
                print('unresolved',rec,fn,flush=True)
            out[retry,:3]=longer[:,:3];out[retry,5]=longer[:,5]
    mechanics.kernel=lambda:robust
    return stats

if __name__=='__main__':
    assert (H/'FORECASTS_SEALED').exists()
    ids=list(map(int,sys.argv[1:]))
    data=setup()
    try:main(ids)
    finally:(H/f'extension_worker_{ids[0]}.json').write_text(json.dumps(data,indent=2)+'\n')
